"""
shared_utils.py — Shared utility functions and constants for the engine.
"""
from __future__ import annotations

import errno
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping, Any

import pandas as pd
import requests

NSE_URL_NIFTY500_CSV = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_URL_NIFTY500_INDEX_CONSTITUENT_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
NSE_URL_EQUITY_MASTER_CSV = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

NSE_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,text/html;q=0.1,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


_NULL_LIKE_SYMBOLS: frozenset[str] = frozenset({
    "NAN", "NONE", "NULL", "NA", "N/A", "NAT", "<NA>", "NIL",
})


def normalize_ns_ticker(sym: str) -> str:
    """Return symbol normalized to uppercase NSE format with a `.NS` suffix.

    Returns an empty string for None, NaN-like, or suffix-only inputs so
    callers can filter invalid results with a simple truthiness check.

    - Preserves index symbols beginning with `^`.
    - Removes trailing `.NS`, `.BO`, `.BSE` first, then appends `.NS`.
    """
    if sym is None:
        return ""
    symbol = str(sym).strip().upper()
    if not symbol or symbol in _NULL_LIKE_SYMBOLS:
        return ""
    if symbol.startswith("^"):
        return symbol
    for sfx in (".NS", ".BO", ".BSE"):
        if symbol.endswith(sfx):
            symbol = symbol[: -len(sfx)]
            break
    if not symbol:
        return ""
    return f"{symbol}.NS"


def compute_notional_volume(
    close_or_df: pd.Series | pd.DataFrame,
    volume: pd.Series | None = None,
) -> pd.Series:
    """Compute clipped daily notional volume (`Close * Volume`)."""
    if isinstance(close_or_df, pd.DataFrame):
        close = close_or_df["Close"]
        volume_series = close_or_df["Volume"]
    else:
        if volume is None:
            raise ValueError("volume must be provided when close_or_df is a Series")
        close = close_or_df
        volume_series = volume
    return (close * volume_series).clip(lower=0)


def fetch_nse_csv(
    url: str,
    *,
    timeout: float = 15.0,
    headers: Mapping[str, str] | None = None,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    encoding: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch and parse NSE CSV endpoint with standard NSE-friendly headers."""
    merged_headers = dict(NSE_DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)

    last_error: Exception | None = None
    response_content: bytes | None = None

    # Retry loop - execute at least once even when retries=0
    for attempt in range(max(1, retries)):
        try:
            client = session or requests
            response = client.get(url, headers=merged_headers, timeout=timeout)
            response.raise_for_status()
            response_content = response.content
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max(1, retries) - 1:
                time.sleep(backoff_seconds * (2 ** attempt))

    # If all retry attempts failed, re-raise the transport exception
    if response_content is None:
        if last_error is not None:
            raise last_error
        raise requests.RequestException("Failed to fetch CSV: no content retrieved")

    # Guard against HTML/XML/challenge pages before attempting CSV parsing
    # Check first ~2KB for common HTML/XML markers (case-insensitive)
    sample = response_content[:2048].lower()
    html_markers = (b'<!doctype', b'<html', b'<script', b'<meta', b'<?xml')
    if any(marker in sample for marker in html_markers):
        raise ValueError(
            "Received HTML/XML content instead of CSV. "
            "The endpoint may have returned an error page or challenge."
        )

    # Parse CSV outside retry loop to avoid retrying parse errors
    if encoding is not None:
        return pd.read_csv(io.BytesIO(response_content), encoding=encoding)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(response_content), encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # Final fallback: decode as text and let pandas handle it
    return pd.read_csv(io.StringIO(response_content.decode("utf-8", errors="replace")))


def atomic_write_file(
    path: str | Path,
    writer: Callable[[Path], Any],
    *,
    suffix: str = ".tmp",
    fsync_file: bool = False,
    fsync_dir: bool = False,
) -> Path:
    """Atomically write to `path` via a temp file and rename.

    The `writer` callback receives the temporary path and must write content to it.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=suffix)
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        writer(tmp_path)

        if fsync_file and os.name == "posix":
            # Running fsync on a file handle opened in text mode can fail on
            # some platforms. Open in binary mode ('rb') for maximum portability.
            with tmp_path.open("rb") as fh:
                os.fsync(fh.fileno())

        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # fsync the directory after a successful rename so the new name is durable.
    # This runs outside the cleanup try/except: tmp_path no longer exists at
    # this point and there is nothing to unlink even if fsync_dir fails.
    if fsync_dir and os.name == "posix":
        dir_fd = os.open(str(dst.parent), getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                os.fsync(dir_fd)
            except OSError as e:
                # Ignore unsupported-filesystem errors
                if e.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                    raise
        finally:
            os.close(dir_fd)
    return dst
