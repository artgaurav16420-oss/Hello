import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from log_config import JsonFormatter, load_dotenv_safe


def test_json_formatter_format_time_is_iso_utc_with_microseconds():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.created = 1700000000.123456
    out = fmt.formatTime(record)

    assert out.endswith("Z")
    assert ".%f" not in out
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$", out)

    parsed = datetime.fromisoformat(out.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.tzinfo.utcoffset(parsed) == timezone.utc.utcoffset(parsed)


def test_json_formatter_format_time_uses_utc_not_localtime():
    fmt = JsonFormatter()
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "x", (), None)
    record.created = 0.0
    out = fmt.formatTime(record)
    assert out.startswith("1970-01-01T00:00:00.")
    assert out.endswith("Z")


def test_env_loader_handles_quoted_value_with_comment(tmp_path, monkeypatch):
    env_file = Path(tmp_path) / ".env"
    env_file.write_text('KEY="value" # note\n', encoding="utf-8")
    monkeypatch.delenv("KEY", raising=False)

    load_dotenv_safe(env_file)

    assert os.getenv("KEY") == "value"
