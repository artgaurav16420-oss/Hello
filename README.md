![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/artgaurav16420-oss/Hello?utm_source=oss&utm_medium=github&utm_campaign=artgaurav16420-oss%2FHello&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

# Ultimate Momentum

Ultimate Momentum is an automated Indian equity portfolio engine focused on NSE-listed stocks. It combines momentum-based security selection with portfolio construction controls designed for realistic execution workflows and daily operations.

The project includes both live/paper workflow orchestration and historical research tooling. Strategy evaluation emphasizes risk-aware behavior, including CVaR-oriented constraints and out-of-sample validation for robust parameter selection.

## Prerequisites

- Python 3.10+ (required for modern typing syntax such as `str | None` and compatibility with dependencies including `pandas-market-calendars>=4.0`)
- System dependencies required by Python packages in `requirements.txt` (for example, compiler/build tooling as needed by your platform)

## Installation

```bash
pip install -r requirements-lock.txt
```

## Configuration

- Create a `.env` file and set required credentials, including:
  - `GROWW_API_TOKEN=<your_token>`
- Configure key environment variables for research/live behavior, including:
  - `OPTIMIZER_OOS_CUTOFF` (optional override for optimizer Period-2 OOS end-date cutoff)
- Ensure the local data directories used by the project are present/writable (for caches, downloaded market data, and generated artifacts under `data/`)

## Running the Daily Workflow

Use `daily_workflow.py` for the end-to-end daily run. Use `--paper` mode for paper-trading/simulation workflows.
When you reject previewed trades in the CLI, the workflow still persists the
latest risk metadata (for example: consecutive failure counters, override
cooldowns, decay rounds, and absent-symbol tracking) so risk controls continue
from the most recent scan state rather than reverting to older values.

## Running Backtests

Use `backtest_engine.py` for backtest execution and `optimizer.py` for parameter search / OOS validation workflows.

## Disclaimer

This is a research/personal project and is **not** financial advice.
