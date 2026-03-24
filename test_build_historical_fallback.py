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
