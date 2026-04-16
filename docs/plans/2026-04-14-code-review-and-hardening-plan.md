# Ultimate Momentum Code Review & Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use 
the `Task` tool for isolation.

This plan outlines the steps for a comprehensive code review and hardening process for the Ultimate Momentum project, focusing on correctness, numeric stability, test coverage, and documentation.

## **Phase 1: Environment & Tooling Setup**

**Goal:** Establish a robust development environment with necessary linting and static analysis tools.

---

- [x] **Task 1: Project Initialization & Dependency Check**
    *   **Action:** Verify Python 3.11 environment. Check `requirements.txt` for missing critical libraries (OSQP, NumPy 2.x, Pandas 2.x, Optuna).
    *   **Run:** `pip install -r requirements.txt`

- [x] **Task 2: Configure Static Analysis**
    *   **Files:**
        *   Create: `.flake8`
        *   Create: `mypy.ini`
        *   Create: `pytest.ini`
    *   **Content for `.flake8`:**
        ```ini
        [flake8]
        max-line-length = 120
        exclude = .git,__pycache__,docs,data,scratch
        ignore = E203, W503
        ```
    *   **Content for `mypy.ini`:**
        ```ini
        [mypy]
        python_version = 3.11
        ignore_missing_imports = True
        ```

- [x] **Task 3: Run Initial Static Analysis**
    *   **Action:** Run the linters across the entire codebase to get a baseline of issues. We will not fix them yet, but this will inform our review.
    *   **Run:** `flake8 . > flake8_report.txt`
    *   **Run:** `mypy . > mypy_report.txt`
    *   **Expected:** Two report files (`flake8_report.txt`, `mypy_report.txt`) are created containing a list of issues.

### **Phase 2: Unit Testing Core Logic**

**Goal:** Write comprehensive unit tests for the most critical, money-handling components of the engine.

---

- [x] **Task 4: Enhance `test_momentum_engine.py`**
    *   **Files:**
        *   Modify: `test_momentum_engine.py`
    *   **Action:** Review `momentum_engine.py` and add targeted unit tests to `test_momentum_engine.py` covering:
        *   `execute_rebalance`: Test with various scenarios, including empty trades, large trades, and trades that should be rejected by risk limits. Mock external dependencies.
        *   `compute_book_cvar`: Test with known inputs and expected CVaR outputs.
        *   `activate_override_on_stress`: Verify that the override is correctly triggered and cooled down.

- [x] **Task 5: Enhance `test_optimizer.py`**
    *   **Files:**
        *   Modify: `test_optimizer.py`
    *   **Action:** Review `optimizer.py` and add unit tests to `test_optimizer.py` for:
        *   The objective function used in Optuna study, ensuring it correctly calculates the metric to be optimized.
        *   The handling of different `OptimizationError` types.

- [x] **Task 6: Enhance `test_backtest_engine.py`**
    *   **Files:**
        *   Modify: `test_backtest_engine.py`
    *   **Action:** Review `backtest_engine.py` and add unit tests to `test_backtest_engine.py` for:
        *   The main `run_backtest` function, using a small, fixed dataset to verify that the backtest produces expected portfolio values and trades.
        *   Edge cases like date ranges with no trading days.

- [x] **Task 7: Create `test_signals.py`**
    *   **Files:**
        *   Create: `test_signals.py`
    *   **Action:** Create a new test file for `signals.py` and add unit tests for:
        *   `generate_signals`: Test with sample dataframes to ensure momentum signals are calculated correctly.
        *   `compute_adv` and `compute_regime_score`: Test with known data to verify correct calculations.

### **Phase 3: Integration & Workflow Testing**

**Goal:** Ensure that the components tested in Phase 2 work together correctly within the main application workflow.

---

- [x] **Task 8: Enhance `test_daily_workflow.py`**
    *   **Files:**
        *   Modify: `test_daily_workflow.py`
    *   **Action:** Add integration tests to `test_daily_workflow.py` for the `_run_scan` function. This will involve:
        *   Mocking data-fetching functions (`load_or_fetch`, `fetch_nse_equity_universe`).
        *   Mocking `run_rebalance_pipeline` to verify it's called with the correct arguments based on the scan results.
        *   Testing the interactive prompts by mocking `input()`.

### **Phase 4: Data Integrity & Validation**

**Goal:** Verify the correctness of data loading, caching, and historical data construction.

---

- [x] **Task 9: Enhance `test_data_cache.py`**
    *   **Files:**
        *   Modify: `test_data_cache.py`
    *   **Action:** Add tests to `test_data_cache.py` to verify:
        *   `load_or_fetch` correctly caches and retrieves data.
        *   The cache invalidation logic works as expected.

- [x] **Task 10: Enhance `test_historical_builder.py`**
    *   **Files:**
        *   Modify: `test_historical_builder.py`
    *   **Action:** Add tests to `test_historical_builder.py` to ensure the historical universe files are created correctly and with the expected schema.

### **Phase 5: Final Review & Documentation**

**Goal:** Address issues found during static analysis and improve documentation for future maintenance.

---

- [x] **Task 11: Address Static Analysis Issues**
    *   **Files:** All `.py` files.
    *   **Action:** Systematically go through `flake8_report.txt` and `mypy_report.txt`. Fix all reported issues, applying type hints and correcting style violations.

- [x] **Task 12: Improve Code Documentation**
    *   **Files:** All `.py` files.
    *   **Action:** Review all functions and add or improve docstrings, especially for complex functions in the core logic files. Explain *why* the code is doing what it's doing.
