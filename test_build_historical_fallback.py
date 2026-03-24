from __future__ import annotations

import os

import build_historical_fallback as bhf


def test_load_env_file_fallback_sets_missing_keys_only(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROWW_API_TOKEN=token_from_file\nEXISTING_KEY=from_file\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("GROWW_API_TOKEN", raising=False)
    monkeypatch.setenv("EXISTING_KEY", "already_set")

    bhf._load_env_file_fallback(env_file)

    assert os.getenv("GROWW_API_TOKEN") == "token_from_file"
    assert os.getenv("EXISTING_KEY") == "already_set"


def test_load_env_file_fallback_strips_inline_comments_from_unquoted_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TOKEN=abc123 # prod token\nQUOTED='keep # inside quotes'\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TOKEN", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    bhf._load_env_file_fallback(env_file)

    assert os.getenv("TOKEN") == "abc123"
    assert os.getenv("QUOTED") == "keep # inside quotes"


def test_parse_args_defaults_start_to_2015():
    args = bhf._parse_args([])
    assert args.start == "2015-01-01"


def test_wbm_cdx_timestamps_merges_scheme_variants_and_month_collapses(monkeypatch):
    calls = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
        calls.append(params["url"])
        payload_by_url = {
            "https://example.com/file.csv": [
                ["timestamp", "statuscode"],
                ["20150101120000", "200"],
                ["20150125120000", "200"],
            ],
            "http://example.com/file.csv": [
                ["timestamp", "statuscode"],
                ["20150201120000", "200"],
            ],
            "*://example.com/file.csv": [
                ["timestamp", "statuscode"],
                ["20150301120000", "200"],
                ["20150305120000", "200"],
            ],
        }
        return _Resp(payload_by_url.get(params["url"], [["timestamp", "statuscode"]]))

    monkeypatch.setattr(bhf.requests, "get", _fake_get)

    out = bhf._wbm_cdx_timestamps("https://example.com/file.csv", start_year=2015)

    assert calls == [
        "https://example.com/file.csv",
        "http://example.com/file.csv",
        "*://example.com/file.csv",
    ]
    assert out == ["20150101120000", "20150201120000", "20150301120000"]
