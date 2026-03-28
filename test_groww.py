import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("GROWW_API_TOKEN"), reason="live API key required"),
]


def test_groww_live_placeholder():
    """Placeholder integration guard file for live Groww tests."""
    assert True
