"""
log_config.py — Structured Logging Infrastructure v11.49
=========================================================
Provides:
  - JsonFormatter: emits every log record as a single JSON line with
    standard fields (timestamp, level, logger, message) plus arbitrary
    extra kwargs attached by callers.
  - ScanContext: a thread-local context manager that injects a
    per-scan-run correlation_id into every log record emitted during
    the scan.  Multiple concurrent scans (e.g. nifty500 + nse_total
    running in parallel threads) each get their own ID so log entries
    can be correlated back to their scan in the aggregator.
  - DeadLetterTracker: accumulates symbols with stale / missing prices
    during a scan run and emits a single structured CRITICAL record at
    the end of the run if the threshold is exceeded, rather than
    producing one WARNING per symbol that can be lost in log volume.
  - configure_logging: one-call setup that installs JsonFormatter on
    the root logger (stdout) and optionally on a rotating file handler.
    Call this once at process startup (e.g. in daily_workflow.__main__).

PROD-FIX-5: Structured logging with correlation IDs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional


# ─── Thread-local scan context ────────────────────────────────────────────────

_local = threading.local()


def current_correlation_id() -> Optional[str]:
    """Return the correlation ID for the current thread's active scan, or None."""
    return getattr(_local, "correlation_id", None)


class ScanContext:
    """
    Context manager that sets a per-thread correlation ID for the duration of
    a scan run.  All log records emitted inside the `with` block carry the same
    correlation_id field, allowing the full scan trace to be extracted from a
    mixed log stream.

    Usage::

        with ScanContext(label="NIFTY500") as ctx:
            logger.info("Scan started", extra={"universe_size": 500})
            # ... scan logic ...
        # correlation_id is cleared after the block

    Note: this is intentional context propagation (set/restore around a block),
    not per-object state isolation. That contrasts with ad-hoc threading.local()
    reads/writes elsewhere that may outlive a logical operation if not paired.
    """

    def __init__(self, label: str = "", correlation_id: Optional[str] = None):
        self.label = label
        self.correlation_id = str(uuid.uuid4())[:8] if correlation_id is None else correlation_id
        self._prev_id: Optional[str] = None
        self._prev_label: Optional[str] = None
        self.start_time: float = 0.0

    def __enter__(self) -> "ScanContext":
        if self.start_time != 0.0:
            raise RuntimeError("ScanContext is not reentrant; create a new instance for each scan.")
        self._prev_id    = getattr(_local, "correlation_id", None)
        self._prev_label = getattr(_local, "scan_label", None)
        _local.correlation_id = self.correlation_id
        _local.scan_label     = self.label
        self.start_time = time.monotonic()
        logging.getLogger(__name__).info(
            "Scan started",
            extra={
                "event":          "scan_start",
                "scan_label":     self.label,
                "correlation_id": self.correlation_id,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.monotonic() - self.start_time
        status  = "error" if exc_type else "ok"
        logging.getLogger(__name__).info(
            "Scan finished (%.1fs) status=%s",
            elapsed,
            status,
            extra={
                "event":          "scan_end",
                "scan_label":     self.label,
                "correlation_id": self.correlation_id,
                "elapsed_s":      round(elapsed, 3),
                "status":         status,
            },
        )
        _local.correlation_id = self._prev_id
        _local.scan_label     = self._prev_label


# ─── Dead-letter tracker ─────────────────────────────────────────────────────

class DeadLetterTracker:
    """
    Accumulates symbols with stale / missing prices across a scan run and
    emits a single structured log record at flush() time if any were found.

    Instead of one WARNING per missing symbol (which floods logs during a
    provider outage), this collects them all and reports the aggregate with
    a count, a sample of affected symbols, and the correlation_id.

    Usage::

        tracker = DeadLetterTracker(threshold=5)
        for sym in universe:
            px = prices.get(sym)
            if px is None or not np.isfinite(px):
                tracker.add(sym, reason="no_price")
        tracker.flush()   # emits CRITICAL if len >= threshold, else WARNING
    """

    def __init__(self, threshold: int = 10):
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self.threshold = threshold
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, symbol: str, reason: str, detail: str = "") -> None:
        with self._lock:
            self._entries.append({
                "symbol": symbol,
                "reason": reason,
                "detail": detail,
            })

    def flush(self, logger_name: str = "dead_letter") -> None:
        with self._lock:
            entries_snapshot = list(self._entries)
            self._entries.clear()

        if not entries_snapshot:
            return
        log = logging.getLogger(logger_name)
        n   = len(entries_snapshot)
        sample = [e["symbol"] for e in entries_snapshot[:10]]
        reasons = list({e["reason"] for e in entries_snapshot})
        extra = {
            "event":          "dead_letter_flush",
            "n_affected":     n,
            "sample_symbols": sample,
            "reasons":        reasons,
            "correlation_id": current_correlation_id(),
        }
        if n >= self.threshold:
            log.critical(
                "Dead-letter flush: %d symbol(s) had stale/missing prices "
                "(threshold=%d). Sample: %s. Reasons: %s.",
                n, self.threshold, sample, reasons,
                extra=extra,
            )
        else:
            log.warning(
                "Dead-letter flush: %d symbol(s) had stale/missing prices. "
                "Sample: %s.",
                n, sample,
                extra=extra,
            )


# ─── JSON formatter ───────────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """
    Emits each log record as a single JSON line with the fields:
      ts          — ISO-8601 UTC timestamp
      level       — DEBUG / INFO / WARNING / ERROR / CRITICAL
      logger      — logger name
      msg         — formatted message string
      correlation_id — current scan's correlation ID (None outside a scan)
      scan_label  — human-readable scan name (None outside a scan)
      pid         — process ID (useful when multiple processes share a log sink)
      thread      — thread name

    Any kwargs passed via extra={...} are merged into the JSON object.
    The formatter never raises — a serialisation error produces a fallback
    plain-text line prefixed with [JSON-ENCODE-ERROR] so no record is silently
    dropped.
    """

    _RESERVED = frozenset({
        "ts", "level", "logger", "msg", "correlation_id",
        "scan_label", "pid",
        # Standard LogRecord attributes we don't want duplicated
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message",
        "module", "msecs", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "taskName",
        "thread", "thread_id", "thread_name", "threadName",
    })

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """
        Produce a UTC ISO-8601 timestamp with microsecond precision.

        FIX-BUG-15: two problems in the original implementation:

        1. The datefmt ``"%Y-%m-%dT%H:%M:%S.%fZ"`` was passed to
           ``logging.Formatter.formatTime``, which calls ``time.strftime()``.
           ``time.strftime`` does not support ``%f`` (microseconds) — that
           directive belongs to ``datetime.strftime``.  Every log line therefore
           contained the literal string ``.%fZ`` instead of real microseconds,
           making timestamps schema-invalid for any downstream parser.

        2. ``logging.Formatter.converter`` defaults to ``time.localtime``, so
           the timestamp was in local time (IST, UTC+5:30) while the hardcoded
           ``Z`` suffix implied UTC — a silent timezone lie.

        Fix: derive the timestamp directly from ``record.created`` (a Unix
        epoch float) and ``record.msecs`` (milliseconds portion), both of which
        are available on every ``LogRecord`` without calling ``time.strftime``.
        The result is always UTC and always carries sub-millisecond precision.
        """
        import datetime as _dt
        epoch_s  = record.created
        # microseconds: msecs is the fractional-second portion in milliseconds;
        # multiply by 1000 to get microseconds, then modulo to stay in [0, 999999].
        us       = int(record.msecs * 1000) % 1_000_000
        base     = _dt.datetime.fromtimestamp(epoch_s, tz=_dt.timezone.utc)
        return base.strftime("%Y-%m-%dT%H:%M:%S") + f".{us:06d}Z"

    def format(self, record: logging.LogRecord) -> str:
        try:
            # Base fields — ts is now a genuine UTC ISO-8601 string with μs.
            obj: Dict[str, Any] = {
                "ts":             self.formatTime(record),
                "level":          record.levelname,
                "logger":         record.name,
                "msg":            record.getMessage(),
                "correlation_id": getattr(_local, "correlation_id", None),
                "scan_label":     getattr(_local, "scan_label",     None),
                "pid":            record.process,
                "thread_id":      record.thread,
                "thread_name":    record.threadName,
            }

            # Merge caller-supplied extra fields (skip reserved / LogRecord internals).
            # BUG-FIX-DOUBLE-SER: The previous implementation called json.dumps(val)
            # on every extra field to check serialisability, then discarded the result
            # and serialised the whole dict again at the end — O(N) double work.
            # Fix: copy all extra fields unconditionally, then serialise once with
            # default=str as the fallback.  Non-serialisable objects become their
            # repr() string automatically, with no extra pass required.
            for key, val in record.__dict__.items():
                if key not in self._RESERVED and not key.startswith("_"):
                    obj[key] = val

            # Exception info
            if record.exc_info:
                obj["exc"] = self.formatException(record.exc_info)

            return json.dumps(obj, ensure_ascii=False, default=str)

        except Exception as encode_err:  # pragma: no cover
            # Never drop a record — fall back to plain text
            return (
                f"[JSON-ENCODE-ERROR: {encode_err}] "
                f"{record.levelname} {record.name} {record.getMessage()}"
            )



def load_dotenv_safe(dotenv_path: Optional[Path] = None) -> None:
    """Best-effort .env loader that never overrides existing environment vars."""
    env_path = dotenv_path or (Path.cwd() / ".env")
    if not env_path.exists():
        return

    log = logging.getLogger(__name__)
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if value and value[0] in ("\"", "'"):
                q = value[0]
                close_i = value.find(q, 1)
                if close_i != -1:
                    value = value[1:close_i]
                elif value.endswith(q) and len(value) >= 2:
                    value = value[1:-1]
            else:
                # BUG-FIX-DOTENV-DC: strip inline comments for unquoted values.
                hash_pos = value.find("#")
                if hash_pos > 0:
                    value = value[:hash_pos].rstrip()

            os.environ.setdefault(key, value)
    except (OSError, UnicodeDecodeError, ValueError, TypeError, IndexError, AttributeError) as exc:
        log.debug("[Env] Could not parse .env at %s: %s", env_path, exc)

# ─── One-call setup ───────────────────────────────────────────────────────────

def configure_logging(
    level: int = logging.INFO,
    json_stdout: bool = True,
    log_file: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,   # 50 MB
    backup_count: int = 5,
    force: bool = True,
) -> None:
    """
    Configure the root logger for production use.

    Parameters
    ----------
    level:        Root log level (default INFO).
    json_stdout:  If True, install JsonFormatter on stdout.  Set False in
                  interactive/development mode to keep human-readable output.
    log_file:     Optional path for a rotating JSON log file.  If None, only
                  stdout is used.
    max_bytes:    Rotating file max size in bytes (default 50 MB).
    backup_count: Number of rotated files to keep (default 5).
    force:        If True (default), remove all existing root handlers before
                  installing new ones — the normal production behaviour.  Set to
                  False in test environments where pytest installs its own log-
                  capture handler that must not be dropped.  When False, this
                  function is a no-op if the root logger already has handlers.

    Call once at process entry-point::

        from log_config import configure_logging
        configure_logging(log_file="logs/momentum.log")
    """
    root = logging.getLogger()
    root.setLevel(level)

    # LC-01: honour the force flag so test environments can retain their own
    # capture handlers.  In production (force=True, the default) we unconditionally
    # replace any handlers added by basicConfig, a previous configure_logging call,
    # or pytest.  When force=False, skip setup entirely if handlers already exist.
    if not force and root.handlers:
        return

    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = JsonFormatter() if json_stdout else logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # optional rotating file handler (always JSON)
    if log_file:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
