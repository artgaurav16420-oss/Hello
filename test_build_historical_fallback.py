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
