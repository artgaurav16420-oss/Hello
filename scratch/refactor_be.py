import re

with open("backtest_engine.py", "r", encoding="utf-8") as f:
    text = f.read()

# Define the start and end of the block to replace
start_marker = "    def _run_rebalance("
end_marker = "# ─── Helpers "

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Could not find markers.")
    exit(1)

new_methods = """    def _filter_rebalance_universe(
        self,
        symbols: List[str],
        prices_t: np.ndarray,
        member_universe: Optional[set[str]],
    ) -> tuple[List[str], np.ndarray, dict[str, int]]:
        sym_to_global_idx = {sym: i for i, sym in enumerate(symbols)}
        active_symbols = symbols
        active_prices = prices_t
        if member_universe is not None:
            member_set = {str(sym) for sym in member_universe}
            member_set.update(self.state.shares.keys())
            active_symbols = [sym for sym in symbols if sym in member_set]
            if not active_symbols:
                return [], np.array([]), sym_to_global_idx
            active_positions = [sym_to_global_idx[sym] for sym in active_symbols]
            active_prices = prices_t[active_positions]
        return active_symbols, active_prices, sym_to_global_idx

    def _correct_adv_glitches(
        self,
        active_symbols: List[str],
        adv_vector: np.ndarray,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        close_notional: pd.DataFrame,
        date: pd.Timestamp,
        cfg: UltimateConfig,
    ) -> None:
        _adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg is not None else 20
        for _adv_i, _adv_sym in enumerate(active_symbols):
            if np.isclose(float(adv_vector[_adv_i]), 0.0, rtol=1e-9, atol=1e-12) and self.state.shares.get(_adv_sym, 0) > 0:
                if _adv_sym in close.columns and _adv_sym in volume.columns:
                    try:
                        _trail = close_notional[_adv_sym].iloc[-_adv_lookback:].clip(lower=0).dropna()
                        if not _trail.empty:
                            _fallback_adv = float(_trail.mean())
                            if _fallback_adv > 0:
                                adv_vector[_adv_i] = _fallback_adv
                                logger.debug(
                                    "[Backtest] ADV glitch: using trailing fallback "
                                    "ADV=%.0f for held position %s on %s.",
                                    _fallback_adv, _adv_sym, date,
                                )
                    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
                        logger.debug(
                            "[Backtest] ADV fallback failed for %s on %s; keeping ADV=0.",
                            _adv_sym,
                            date,
                        )
                        pass

    def _check_cvar_breach(
        self,
        date: pd.Timestamp,
        valuation_prices: np.ndarray,
        active_symbols: List[str],
        hist_log_rets: pd.DataFrame,
        cfg: UltimateConfig,
    ) -> tuple[bool, bool, bool]:
        apply_decay = False
        _force_full_cash = False
        soft_cvar_breach = False

        if self.state.shares:
            book_cvar = compute_book_cvar(self.state, valuation_prices, active_symbols, hist_log_rets, cfg)
            hard_multiplier = getattr(cfg, "CVAR_HARD_BREACH_MULTIPLIER", 1.5)
            hard_breach_threshold = cfg.CVAR_DAILY_LIMIT * hard_multiplier

            if book_cvar > hard_breach_threshold:
                logger.warning(
                    "[Backtest] Book CVaR %.4f%% exceeds HARD limit %.4f%% (%.1fx) on %s — "
                    "skipping optimization, forcing immediate liquidation.",
                    book_cvar * 100, hard_breach_threshold * 100, hard_multiplier, date,
                )
                self.state.consecutive_failures += 1
                apply_decay      = True
                _force_full_cash = True
                activate_override_on_stress(self.state, cfg)

            elif book_cvar > cfg.CVAR_DAILY_LIMIT + 1e-6:
                soft_cvar_breach = True
                logger.info(
                    "[Backtest] Book CVaR soft breach %.4f%% (limit %.4f%%, hard %.4f%%) on %s — "
                    "running optimizer with CVaR constraint active.",
                    book_cvar * 100, cfg.CVAR_DAILY_LIMIT * 100, hard_breach_threshold * 100, date,
                )
        return apply_decay, _force_full_cash, soft_cvar_breach

    def _generate_rebalance_signals(
        self,
        date: pd.Timestamp,
        active_symbols: List[str],
        hist_log_rets: pd.DataFrame,
        adv_vector: np.ndarray,
        valuation_prices: np.ndarray,
        pv: float,
        prev_w_dict: Dict[str, float],
        sector_map: Optional[dict],
        soft_cvar_breach: bool,
        apply_decay: bool,
        cfg: UltimateConfig,
    ) -> tuple[np.ndarray, bool, bool, List[int], bool]:
        target_weights = np.zeros(len(active_symbols))
        optimization_succeeded = False
        sel_idx: List[int] = []

        try:
            raw_daily, adj_scores, sel_idx, _gate_counts = generate_signals(
                hist_log_rets,
                adv_vector,
                cfg,
                prev_weights=prev_w_dict,
            )
        except SignalGenerationError as ve:
            logger.debug(
                "[Backtest] generate_signals raised ValueError on %s: %s — "
                "treating as empty universe for this bar.",
                date, ve,
            )
            self.state.decay_rounds         = 0
            self.state.consecutive_failures = 0
            return target_weights, optimization_succeeded, apply_decay, sel_idx, True

        if sel_idx:
            sel_syms      = [active_symbols[i] for i in sel_idx]
            sector_labels = _build_sector_labels(sel_syms, sector_map)
            prev_weights  = np.array([prev_w_dict.get(sym, 0.0) for sym in active_symbols])

            try:
                weights_sel = self.engine.optimize(
                    expected_returns    = raw_daily[sel_idx],
                    historical_returns  = hist_log_rets[[active_symbols[i] for i in sel_idx]],
                    execution_date      = date,
                    adv_shares          = adv_vector[sel_idx],
                    prices              = valuation_prices[sel_idx],
                    portfolio_value     = pv,
                    prev_w              = prev_weights[sel_idx],
                    exposure_multiplier = self.state.exposure_multiplier,
                    sector_labels       = sector_labels,
                )
                target_weights[sel_idx]  = weights_sel
                self.state.consecutive_failures = 0
                self.state.decay_rounds  = 0
                optimization_succeeded   = True

            except OptimizationError as oe:
                if oe.error_type != OptimizationErrorType.DATA:
                    self.state.consecutive_failures += 1
                    logger.debug(
                        "[Backtest] Solver failure #%d on %s: %s",
                        self.state.consecutive_failures, date, oe,
                    )
                    if soft_cvar_breach:
                        logger.warning(
                            "[Backtest] Solver failure during active soft CVaR "
                            "breach on %s — bypassing 3-failure wait, "
                            "triggering immediate decay.",
                            date,
                        )
                        apply_decay = True
                    elif self.state.consecutive_failures >= 3:
                        logger.debug(
                            "[Backtest] 3 consecutive solver failures on %s — "
                            "triggering gate-filtered pro-rata liquidation.",
                            date,
                        )
                        apply_decay = True
        elif self.state.shares:
            apply_decay = True
        else:
            self.state.decay_rounds         = 0
            self.state.consecutive_failures = 0

        return target_weights, optimization_succeeded, apply_decay, sel_idx, False

    def _compute_position_decay(
        self,
        date: pd.Timestamp,
        active_symbols: List[str],
        valuation_prices: np.ndarray,
        pv: float,
        target_weights: np.ndarray,
        sel_idx: List[int],
        _force_full_cash: bool,
        cfg: UltimateConfig,
    ) -> tuple[np.ndarray, bool]:
        _exhaust_decay = False
        if _force_full_cash or self.state.decay_rounds >= cfg.MAX_DECAY_ROUNDS:
            target_weights = np.zeros(len(active_symbols), dtype=float)
            logger.warning(
                "[Backtest] %s on %s — forcing full liquidation to cash.",
                "Book CVaR breach" if _force_full_cash else
                f"MAX_DECAY_ROUNDS={cfg.MAX_DECAY_ROUNDS} exhausted",
                date,
            )
            _exhaust_decay = True
            activate_override_on_stress(self.state, cfg)
        else:
            target_weights = compute_decay_targets(self.state, sel_idx, active_symbols, cfg, current_prices=valuation_prices, pv=pv)
            sel_idx_set = set(sel_idx)
            sym_to_pos  = {s: i for i, s in enumerate(active_symbols)}
            n_gated = sum(
                1 for s in self.state.shares
                if s in sym_to_pos and sym_to_pos[s] not in sel_idx_set
            )
            logger.debug(
                "[Backtest] Decay round %d/%d: scaling %d gate-passing, "
                "force-closing %d gated positions.",
                self.state.decay_rounds + 1, cfg.MAX_DECAY_ROUNDS,
                len(sel_idx), n_gated,
            )
        return target_weights, _exhaust_decay

    def _select_execution_prices(
        self,
        date: pd.Timestamp,
        active_symbols: List[str],
        active_prices: np.ndarray,
        close: pd.DataFrame,
        open_px: Optional[pd.DataFrame],
        high_px: Optional[pd.DataFrame],
        low_px: Optional[pd.DataFrame],
        active_col_indices: np.ndarray,
        prev_idx: int,
        date_pos: int,
        target_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        exec_prices, open_fallback_mask = _execution_prices(
            active_symbols, date, active_prices, open_px, high_px, low_px, return_open_fallback_mask=True
        )
        if open_fallback_mask.any():
            sig_px_arr = close.values[prev_idx, active_col_indices]
            cur_px_arr = close.values[date_pos, active_col_indices]
            first_day_mask = (
                open_fallback_mask
                & np.isfinite(sig_px_arr)
                & np.isfinite(cur_px_arr)
                & (sig_px_arr == cur_px_arr)
            )
            if first_day_mask.any():
                skipped_syms = [sym for sym, bad in zip(active_symbols, first_day_mask, strict=True) if bad]
                logger.warning(
                    "[Backtest] Skipping first-day symbols with NaN open fallback on %s: %s",
                    date,
                    skipped_syms,
                )
                target_weights[first_day_mask] = 0.0
        return exec_prices, target_weights

    def _execute_and_log_trades(
        self,
        date: pd.Timestamp,
        target_weights: np.ndarray,
        exec_prices: np.ndarray,
        active_symbols: List[str],
        adv_vector: np.ndarray,
        apply_decay: bool,
        _exhaust_decay: bool,
        soft_cvar_breach: bool,
        _force_full_cash: bool,
        hist_log_rets: pd.DataFrame,
        regime_score: float,
        realised_cvar: float,
        cfg: UltimateConfig,
    ) -> None:
        _T = min(len(hist_log_rets), cfg.CVAR_LOOKBACK)
        _L = -(hist_log_rets.iloc[-_T:].reindex(columns=active_symbols, fill_value=0.0).values)

        execute_rebalance(
            self.state, target_weights, exec_prices, active_symbols, cfg,
            adv_shares     = adv_vector,
            date_context   = date,
            trade_log      = self.trades,
            apply_decay    = apply_decay and not _exhaust_decay,
            scenario_losses = None if _exhaust_decay else _L,
            force_rebalance_trades = soft_cvar_breach,
        )
        if _exhaust_decay:
            self.state.decay_rounds = 0
            self.state.consecutive_failures = 0

        self._rebal_rows.append({
            "date":               date,
            "regime_score":       round(regime_score, 4),
            "realised_cvar":      round(realised_cvar, 6),
            "exposure_multiplier":round(self.state.exposure_multiplier, 4),
            "override_active":    self.state.override_active,
            "n_positions":        len(self.state.shares),
            "apply_decay":        apply_decay,
            "forced_to_cash":     bool(_force_full_cash or _exhaust_decay),
            "force_cash_reason":  (
                "book_cvar_breach" if _force_full_cash else
                "max_decay_rounds" if _exhaust_decay else
                ""
            ),
        })

    def _run_rebalance(
        self,
        date:       pd.Timestamp,
        close:      pd.DataFrame,
        volume:     pd.DataFrame,
        returns:    pd.DataFrame,
        symbols:    List[str],
        prices_t:   np.ndarray,
        idx_df:     Optional[pd.DataFrame],
        sector_map: Optional[dict],
        open_px:    Optional[pd.DataFrame] = None,
        high_px:    Optional[pd.DataFrame] = None,
        low_px:     Optional[pd.DataFrame] = None,
        member_universe: Optional[set[str]] = None,
        date_pos: Optional[int] = None,
        log_rets_arr: Optional[np.ndarray] = None,
    ) -> None:
        cfg = self.engine.cfg

        active_symbols, active_prices, _ = self._filter_rebalance_universe(symbols, prices_t, member_universe)
        if not active_symbols:
            return

        if date_pos is None:
            _loc = close.index.get_loc(date)
            if not isinstance(_loc, (int, np.integer)):
                raise ValueError(
                    f"Duplicate timestamp {date} detected in close index. "
                    "Deduplicate the index in build_precomputed_matrices before running backtest."
                )
            date_pos = int(_loc)
        if log_rets_arr is None:
            log_rets_arr = np.log1p(returns).replace([np.inf, -np.inf], np.nan).values

        col_to_idx = {sym: i for i, sym in enumerate(close.columns)}
        prev_idx = date_pos - 1
        if prev_idx < 0:
            return
        signal_date = close.index[prev_idx]
        active_col_indices = np.array([col_to_idx[sym] for sym in active_symbols], dtype=int)
        hist_log_rets = pd.DataFrame(
            log_rets_arr[:prev_idx + 1, active_col_indices],
            index=returns.index[:prev_idx + 1],
            columns=active_symbols,
        )

        adv_vector, close_notional = _build_adv_vector(
            active_symbols, close, volume, date, cfg=cfg, return_notional=True
        )

        self._correct_adv_glitches(active_symbols, adv_vector, close, volume, close_notional, date, cfg)

        valuation_close = close.loc[signal_date]

        valuation_prices = np.array([
            float(valuation_close[sym]) if (sym in valuation_close.index and pd.notna(valuation_close[sym]))
            else _ffill_price(self.state, sym, cfg)
            for sym in active_symbols
        ])

        pv, gross_exposure = self._compute_portfolio_value_and_gross_exposure(
            valuation_close=valuation_close,
            cfg=cfg,
        )
        prev_w_dict = _build_prev_weights(self.state, active_symbols, pv)

        idx_slice    = idx_df.loc[:signal_date] if idx_df is not None and not getattr(idx_df, "empty", False) else None
        regime_score = compute_regime_score(idx_slice, cfg=cfg, universe_close_hist=close.loc[:signal_date])

        if len(self.state.equity_hist) >= cfg.CVAR_MIN_HISTORY:
            realised_cvar = self.state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY)
        else:
            realised_cvar = 0.0

        self.state.update_exposure(regime_score, realised_cvar, cfg, gross_exposure=gross_exposure)

        apply_decay, _force_full_cash, soft_cvar_breach = self._check_cvar_breach(
            date, valuation_prices, active_symbols, hist_log_rets, cfg
        )

        optimization_succeeded = False
        target_weights = np.zeros(len(active_symbols))
        sel_idx: List[int] = []

        if not _force_full_cash:
            target_weights, optimization_succeeded, apply_decay, sel_idx, abort = self._generate_rebalance_signals(
                date, active_symbols, hist_log_rets, adv_vector, valuation_prices,
                pv, prev_w_dict, sector_map, soft_cvar_breach, apply_decay, cfg
            )
            if abort:
                return

        _exhaust_decay = False
        if apply_decay and not optimization_succeeded:
            target_weights, _exhaust_decay = self._compute_position_decay(
                date, active_symbols, valuation_prices, pv, target_weights, sel_idx, _force_full_cash, cfg
            )

        if optimization_succeeded or apply_decay:
            exec_prices, target_weights = self._select_execution_prices(
                date, active_symbols, active_prices, close, open_px, high_px, low_px,
                active_col_indices, prev_idx, date_pos, target_weights
            )
            
            self._execute_and_log_trades(
                date, target_weights, exec_prices, active_symbols, adv_vector, apply_decay,
                _exhaust_decay, soft_cvar_breach, _force_full_cash, hist_log_rets,
                regime_score, realised_cvar, cfg
            )
\n"""

new_text = text[:start_idx] + new_methods + text[end_idx:]

with open("backtest_engine.py", "w", encoding="utf-8") as f:
    f.write(new_text)

print("Replaced _run_rebalance successfully.")
