import re

with open("momentum_engine.py", "r", encoding="utf-8") as f:
    text = f.read()

start_marker = "    def optimize("
end_marker = "        return full_w_opt"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker, start_idx) + len(end_marker)

if start_idx == -1 or end_idx == -1 or text.find(end_marker, start_idx) == -1:
    print("Could not find markers.")
    exit(1)

new_methods = """    def _preprocess_optimization_inputs(
        self,
        expected_returns: np.ndarray,
        historical_returns: pd.DataFrame,
        adv_shares: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
        prev_w: Optional[np.ndarray],
        exposure_multiplier: float,
        sector_labels: Optional[np.ndarray],
        execution_date: Optional[pd.Timestamp],
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int, int]:
        original_m = len(expected_returns)
        if execution_date is not None and not historical_returns.empty:
            if historical_returns.index.max() > pd.Timestamp(execution_date):
                raise OptimizationError(
                    "T-1 violation: historical_returns include execution_date.",
                    OptimizationErrorType.DATA,
                )

        _validate_optimizer_input_shapes(
            expected_returns=expected_returns,
            historical_returns=historical_returns,
            adv_shares=adv_shares,
            prices=prices,
            prev_w=prev_w,
            sector_labels=sector_labels,
            portfolio_value=portfolio_value,
        )

        raw_rets = historical_returns.replace([np.inf, -np.inf], np.nan)
        clean_rets, kept_indices, excluded_symbols, lookback, required_count = _apply_history_gate(raw_rets, self.cfg)

        if excluded_symbols:
            logger.info(
                "[OptimizerGuard] excluded_symbols=%d reason=insufficient_history "
                "lookback=%d required_non_nan=%d symbols=%s",
                len(excluded_symbols),
                lookback,
                required_count,
                excluded_symbols,
            )

        if len(kept_indices) == 0:
            raise OptimizationError(
                "No symbols passed optimizer minimum-history gate.",
                OptimizationErrorType.DATA,
            )

        clean_rets = _fill_missing_returns(clean_rets)

        col_stds = clean_rets.std()
        valid_vol_mask = col_stds >= 1e-10
        if not valid_vol_mask.all():
            zero_cols = valid_vol_mask[~valid_vol_mask].index.tolist()
            logger.warning(
                "[Optimizer] Detected %d zero-volatility asset(s): %s",
                len(zero_cols), zero_cols,
            )
        clean_rets, kept_indices = _drop_zero_volatility_columns(clean_rets, kept_indices)

        expected_returns = expected_returns[kept_indices]
        prices = prices[kept_indices]
        adv_shares = adv_shares[kept_indices]
        if prev_w is not None:
            prev_w = prev_w[kept_indices]
        if sector_labels is not None:
            sector_labels = np.asarray(sector_labels)[kept_indices]

        m = len(kept_indices)
        T          = len(clean_rets)
        min_rows   = self.cfg.DIMENSIONALITY_MULTIPLIER * m
        if T < min_rows:
            raise OptimizationError(
                f"Insufficient history: {T} rows for {m} assets.", OptimizationErrorType.DATA
            )

        return clean_rets, expected_returns, prices, adv_shares, prev_w, sector_labels, kept_indices, m, T

    def _build_optimization_constraints(
        self,
        clean_rets: pd.DataFrame,
        expected_returns: np.ndarray,
        adv_shares: np.ndarray,
        portfolio_value: float,
        prev_w: Optional[np.ndarray],
        exposure_multiplier: float,
        sector_labels: Optional[np.ndarray],
        m: int,
        T: int,
    ) -> tuple[Any, np.ndarray, Any, list, list, int, float, float, float, np.ndarray]:
        import scipy.sparse as sp
        from sklearn.covariance import LedoitWolf

        simple_rets = np.expm1(clean_rets)
        lw = LedoitWolf()
        lw.fit(simple_rets)
        Sigma_reg = lw.covariance_
        ridge     = 0.0

        gamma = float(np.clip(exposure_multiplier, self.cfg.MIN_EXPOSURE_FLOOR, 1.0))

        adv_w           = adv_shares / np.maximum(adv_shares.sum(), 1e-9)
        adv_w_series    = pd.Series(adv_w, index=clean_rets.columns)
        aligned_w       = adv_w_series.reindex(clean_rets.columns).fillna(0.0).values
        aligned_w       = aligned_w / np.maximum(aligned_w.sum(), 1e-9)
        adv_weighted_rets = pd.Series(
            simple_rets.values.dot(aligned_w), index=simple_rets.index
        )
        var_95   = adv_weighted_rets.quantile(1 - self.cfg.CVAR_ALPHA)
        ew_cvar  = (
            -float(adv_weighted_rets[adv_weighted_rets <= var_95].mean())
            if not adv_weighted_rets.empty else 0.0
        )
        sentinel = self.cfg.CVAR_DAILY_LIMIT * self.cfg.CVAR_SENTINEL_MULTIPLIER

        if ew_cvar > sentinel + EPSILON:
            logger.warning(
                "Selection ADV-weighted CVaR %.2f%% exceeds sentinel %.2f%%. "
                "Forcing 50%% exposure reduction.",
                ew_cvar * 100, sentinel * 100,
            )
            gamma *= 0.5
            gamma = max(self.cfg.MIN_EXPOSURE_FLOOR, gamma)

        adv_limit = np.clip((adv_shares * self.cfg.MAX_ADV_PCT) / portfolio_value, 1e-9, 0.40)
        adv_limit = np.minimum(adv_limit, self.cfg.MAX_SINGLE_NAME_WEIGHT)

        gamma, l_gamma, u_gamma = _compute_exposure_bounds(
            self.cfg,
            gamma,
            adv_limit,
            sector_labels,
        )

        impact     = np.clip(
            self.cfg.IMPACT_COEFF * portfolio_value
            / np.maximum(adv_shares, 1.0),
            0.0, 1e4,
        )
        T_cvar     = min(T, self.cfg.CVAR_LOOKBACK)
        losses     = -simple_rets.iloc[-T_cvar:].values
        n_vars     = 2 * m + 1 + T_cvar + 1
        prev_w_arr = prev_w if prev_w is not None else np.zeros(m)

        P_w   = 2.0 * (self.cfg.RISK_AVERSION * Sigma_reg + np.diag(impact))
        P_aux = sp.eye(n_vars - m, format="csc") * 1e-6
        P     = sp.block_diag([sp.csc_matrix(P_w), P_aux], format="csc")

        target_weight_hint = np.clip(expected_returns, 0.0, None)
        total_hint = float(np.sum(target_weight_hint))
        if total_hint > 0:
            target_weight_hint = target_weight_hint / total_hint
        else:
            target_weight_hint = np.zeros_like(expected_returns, dtype=float)
        scaled_target_weight_hint = target_weight_hint * float(u_gamma)
        trade_estimate_notionals = np.abs(scaled_target_weight_hint - prev_w_arr) * float(portfolio_value)

        turnover_costs = _compute_one_way_slip_rate_vectorized(
            cfg=self.cfg,
            portfolio_value=portfolio_value,
            adv_notional=np.asarray(adv_shares, dtype=float),
            trade_notional=np.asarray(trade_estimate_notionals, dtype=float),
        )

        q        = np.zeros(n_vars)
        q[:m]    = -expected_returns - 2.0 * impact * prev_w_arr
        q[m:2*m] = turnover_costs
        q[-1]    = self.cfg.SLACK_PENALTY

        builder = _ConstraintBuilder(n_vars)

        budget_row = sp.csc_matrix(
            (np.ones(m), (np.zeros(m, int), np.arange(m))), shape=(1, n_vars)
        )
        builder.add_constraint(budget_row, [l_gamma], [u_gamma])

        A_scen = sp.lil_matrix((T_cvar, n_vars))
        A_scen[:, :m]  = losses
        A_scen[:, 2*m] = -1.0
        for i in range(T_cvar):
            A_scen[i, 2*m + 1 + i] = -1.0
        builder.add_constraint(A_scen.tocsc(), [-np.inf] * T_cvar, [0.0] * T_cvar)

        scen_c = 1.0 / (T_cvar * (1.0 - self.cfg.CVAR_ALPHA))
        lim    = sp.lil_matrix((1, n_vars))
        lim[0, 2*m]                 = 1.0
        lim[0, 2*m+1:2*m+1+T_cvar] = scen_c
        lim[0, -1]                  = -1.0
        builder.add_constraint(lim.tocsc(), [-np.inf], [self.cfg.CVAR_DAILY_LIMIT])

        lb, ub = np.full(n_vars, -np.inf), np.full(n_vars, np.inf)
        lb[:m], ub[:m] = 0.0, adv_limit
        lb[m:2*m], lb[2*m+1:2*m+1+T_cvar], lb[-1] = 0.0, 0.0, 0.0
        builder.add_constraint(sp.eye(n_vars, format="csc"), lb.tolist(), ub.tolist())

        if sector_labels is not None:
            labels = np.asarray(sector_labels, dtype=int)
            for sec_id in np.unique(labels):
                if sec_id == -1:
                    continue
                mask = labels == sec_id
                sec_row = sp.lil_matrix((1, n_vars))
                sec_row[0, np.where(mask)[0]] = 1.0
                builder.add_constraint(sec_row.tocsc(), [0.0], [self.cfg.MAX_SECTOR_WEIGHT])

        tc = sp.lil_matrix((2 * m, n_vars))
        for i in range(m):
            tc[2 * i, i] = 1.0
            tc[2 * i, m + i] = -1.0
            tc[2 * i + 1, i] = -1.0
            tc[2 * i + 1, m + i] = -1.0

        tc_u = []
        for p in prev_w_arr:
            tc_u.extend([p, -p])
        builder.add_constraint(tc.tocsc(), [-np.inf] * (2 * m), tc_u)

        A, lower, upper = builder.build()
        P_upper = sp.triu(P, format="csc")

        return P_upper, q, A, lower, upper, T_cvar, gamma, l_gamma, u_gamma, adv_limit

    def _invoke_solver(
        self,
        P_upper: Any,
        q: np.ndarray,
        A: Any,
        lower: list,
        upper: list,
        m: int,
        prev_w: Optional[np.ndarray],
        portfolio_value: float,
        adv_shares: np.ndarray,
        T_cvar: int,
    ) -> Any:
        current_shape = (m, T_cvar)
        current_nnz   = (P_upper.nnz, A.nnz)

        with self._solver_lock:
            is_same_structure = False
            if (self._solver is not None
                    and self._solver_shape == current_shape
                    and self._solver_nnz == current_nnz
                    and self._solver_struct is not None):
                P_ind, P_ptr, A_ind, A_ptr = self._solver_struct
                is_same_structure = (
                    np.array_equal(P_upper.indices, P_ind)
                    and np.array_equal(P_upper.indptr, P_ptr)
                    and np.array_equal(A.indices, A_ind)
                    and np.array_equal(A.indptr, A_ptr)
                )

            if not is_same_structure:
                self._solver = osqp.OSQP()
                setup_kwargs = dict(
                    verbose=False,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    adaptive_rho=True,
                    max_iter=50000,
                )
                try:
                    self._solver.setup(
                        P_upper, q, A, lower, upper,
                        polishing=True,
                        warm_starting=True,
                        **setup_kwargs,
                    )
                except TypeError as exc:
                    msg = str(exc)
                    if ("polishing" not in msg) and ("warm_starting" not in msg):
                        raise
                    self._solver.setup(
                        P_upper, q, A, lower, upper,
                        polish=True,
                        warm_start=True,
                        **setup_kwargs,
                    )
                self._solver_shape = current_shape
                self._solver_nnz = current_nnz
                self._solver_struct = (
                    P_upper.indices.copy(), P_upper.indptr.copy(),
                    A.indices.copy(), A.indptr.copy(),
                )
            else:
                assert self._solver is not None
                self._solver.update(
                    q=q, l=lower, u=upper,
                    Px=P_upper.data, Ax=A.data,
                )

            res = self._handle_solver_fallback(lambda: self._solver.solve(), "first-pass")
            
            # Turnover iteration
            w_opt = np.maximum(res.x[:m], 0.0)
            prev_w_arr = prev_w if prev_w is not None else np.zeros(m)
            actual_deltas = np.abs(w_opt - prev_w_arr) * float(portfolio_value)
            turnover_costs = _compute_one_way_slip_rate_vectorized(
                cfg=self.cfg,
                portfolio_value=portfolio_value,
                adv_notional=np.asarray(adv_shares, dtype=float),
                trade_notional=np.asarray(actual_deltas, dtype=float),
            )
            q[m:2*m] = turnover_costs

            assert self._solver is not None
            self._solver.update(q=q)
            self._solver.warm_start(x=res.x)

            res = self._handle_solver_fallback(lambda: self._solver.solve(), "second-pass")
        return res

    def _handle_solver_fallback(self, solve_func: callable, stage: str) -> Any:
        try:
            res = solve_func()
        except Exception as exc:
            logger.error(
                "[Optimizer] OSQP %s solve() raised an exception: %s — "
                "invalidating solver cache to force fresh setup on next call.", stage, exc
            )
            self._solver = None
            self._solver_shape = None
            self._solver_nnz = None
            self._solver_struct = None
            raise OptimizationError(
                f"OSQP {stage} solve() failed with exception: {exc}",
                OptimizationErrorType.NUMERICAL,
            ) from exc
        
        if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
            self._solver = None
            self._solver_shape = None
            self._solver_nnz = None
            self._solver_struct = None
            raise OptimizationError(f"OSQP status: {res.info.status}", OptimizationErrorType.NUMERICAL)
        return res

    def _extract_optimization_results(
        self,
        res: Any,
        clean_rets: pd.DataFrame,
        m: int,
        T_cvar: int,
        gamma: float,
        l_gamma: float,
        u_gamma: float,
        adv_limit: np.ndarray,
        Sigma_reg: Any,
        ridge: float,
        kept_indices: np.ndarray,
        original_m: int,
    ) -> np.ndarray:
        if res.info.status in ("solved inaccurate", "solved_inaccurate"):
            logger.warning(
                "[Optimizer] OSQP returned '%s' — KKT conditions not strictly satisfied. "
                "Proceeding to physical CVaR verification.",
                res.info.status,
            )

        w_opt = np.maximum(res.x[:m], 0.0)

        simple_rets = np.expm1(clean_rets)
        losses     = -simple_rets.iloc[-T_cvar:].values
        portfolio_losses  = losses @ w_opt
        sorted_losses     = np.sort(portfolio_losses)
        tail_cutoff       = int(np.floor(T_cvar * (1.0 - self.cfg.CVAR_ALPHA)))
        tail_cutoff       = max(1, tail_cutoff)
        tail_losses       = sorted_losses[-tail_cutoff:]
        physical_cvar     = float(np.mean(tail_losses)) if tail_losses.size else 0.0

        eta           = res.x[2*m]
        z_vec         = res.x[2*m+1: 2*m+1+T_cvar]
        solver_cvar   = float(eta + np.sum(z_vec) / (T_cvar * (1.0 - self.cfg.CVAR_ALPHA)))
        slack_value   = float(res.x[-1])

        adv_binding_count = int(np.sum(w_opt >= adv_limit - 1e-6))

        self.last_diag = SolverDiagnostics(
            status            = res.info.status,
            gamma_intent      = gamma,
            actual_weight     = float(np.sum(w_opt)),
            l_gamma           = l_gamma,
            u_gamma           = u_gamma,
            cvar_value        = physical_cvar,
            slack_value       = slack_value,
            sum_adv_limit     = float(np.sum(adv_limit)),
            adv_binding_count = adv_binding_count,
            ridge_applied     = ridge,
            cond_number       = float(np.linalg.cond(Sigma_reg)),
            t_cvar            = T_cvar,
        )

        POST_SOLVE_TOL = 1e-4

        if physical_cvar > self.cfg.CVAR_DAILY_LIMIT + POST_SOLVE_TOL:
            raise OptimizationError(
                f"Physical CVaR {physical_cvar:.4%} exceeds hard limit "
                f"{self.cfg.CVAR_DAILY_LIMIT:.4%} (solver reported {solver_cvar:.4%}, "
                f"slack={slack_value:.6f}). Refusing to deploy.",
                OptimizationErrorType.NUMERICAL,
            )

        lower_hard = float(np.min(w_opt)) < -POST_SOLVE_TOL
        upper_hard = bool(np.any(w_opt > (adv_limit + POST_SOLVE_TOL)))
        gross = float(np.sum(w_opt))
        gross_low_hard = gross < (l_gamma - POST_SOLVE_TOL)
        gross_high_hard = gross > (u_gamma + POST_SOLVE_TOL)

        near_tol = 1e-7
        if float(np.min(w_opt)) < near_tol:
            logger.warning("[Optimizer] Post-check near lower bound: min(w)=%.9f", float(np.min(w_opt)))
        if bool(np.any(w_opt > (adv_limit - near_tol))):
            logger.warning("[Optimizer] Post-check near ADV bound for one or more names.")
        if abs(gross - l_gamma) < near_tol or abs(gross - u_gamma) < near_tol:
            logger.warning(
                "[Optimizer] Post-check near gross boundary: gross=%.9f l=%.9f u=%.9f",
                gross,
                float(l_gamma),
                float(u_gamma),
            )

        if lower_hard or upper_hard or gross_low_hard or gross_high_hard:
            raise OptimizationError(
                "Post-solve constraint verification failed: "
                f"min_w={float(np.min(w_opt)):.9g}, "
                f"max_excess={float(np.max(w_opt - adv_limit)):.9g}, "
                f"gross={gross:.9g}, "
                f"bounds=[{float(l_gamma):.9g}, {float(u_gamma):.9g}]",
                OptimizationErrorType.NUMERICAL,
            )

        full_w_opt = np.zeros(original_m)
        full_w_opt[kept_indices] = np.round(w_opt, 10)
        return full_w_opt

    def optimize(
        self,
        expected_returns:    np.ndarray,
        historical_returns:  pd.DataFrame,
        adv_shares:          np.ndarray,
        prices:              np.ndarray,
        portfolio_value:     float,
        prev_w:              Optional[np.ndarray] = None,
        exposure_multiplier: float                = 1.0,
        sector_labels:       Optional[np.ndarray] = None,
        execution_date:      Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        original_m = len(expected_returns)
        if original_m == 0:
            return np.array([])

        clean_rets, expected_returns_sub, prices_sub, adv_shares_sub, prev_w_sub, sector_labels_sub, kept_indices, m, T = self._preprocess_optimization_inputs(
            expected_returns, historical_returns, adv_shares, prices, portfolio_value,
            prev_w, exposure_multiplier, sector_labels, execution_date
        )

        P_upper, q, A, lower, upper, T_cvar, gamma, l_gamma, u_gamma, adv_limit = self._build_optimization_constraints(
            clean_rets, expected_returns_sub, adv_shares_sub, portfolio_value, prev_w_sub,
            exposure_multiplier, sector_labels_sub, m, T
        )

        res = self._invoke_solver(
            P_upper, q, A, lower, upper, m, prev_w_sub, portfolio_value, adv_shares_sub, T_cvar
        )

        import sklearn.covariance
        lw = sklearn.covariance.LedoitWolf()
        lw.fit(np.expm1(clean_rets))
        Sigma_reg = lw.covariance_
        
        full_w_opt = self._extract_optimization_results(
            res, clean_rets, m, T_cvar, gamma, l_gamma, u_gamma, adv_limit,
            Sigma_reg, 0.0, kept_indices, original_m
        )

        return full_w_opt
"""

new_text = text[:start_idx] + new_methods + text[end_idx:]

with open("momentum_engine.py", "w", encoding="utf-8") as f:
    f.write(new_text)

print("Replaced optimize successfully.")
