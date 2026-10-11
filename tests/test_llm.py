"""LLM client: the LLM_ENABLED master switch and provider behaviour (no network: HTTP is intercepted)."""
from dataclasses import replace

import httpx
import pytest

from ssa.agents.llm import LLMClient, LLMUnavailable


@pytest.fixture()
def http_attempts(monkeypatch):
    """Record every HTTP request the SDKs try to send and fail it before it reaches the network."""
    attempts = []

    def send(self, request, *a, **k):
        attempts.append(str(request.url))
        raise httpx.ConnectError("blocked by test", request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return attempts


def _settings(ssa_home, **kw):
    from ssa.config import get_settings
    return replace(get_settings(), llm_max_retries=0, llm_cache_enabled=False, openai_api_key="fake-test-key", **kw)


def test_llm_disabled_sends_nothing_even_with_key(ssa_home, http_attempts):
    llm = LLMClient(_settings(ssa_home, llm_enabled=False), ledger=False)
    assert not llm.available
    with pytest.raises(LLMUnavailable):
        llm.complete_json("system", "user")
    with pytest.raises(LLMUnavailable):
        llm.chat([{"role": "user", "content": "hi"}])
    assert http_attempts == []
    # control: with the switch on, the same interception does see a request, so the check above is meaningful
    on = LLMClient(_settings(ssa_home, llm_enabled=True), ledger=False)
    assert on.available
    with pytest.raises(LLMUnavailable):
        on.complete_json("system", "user")
    assert http_attempts
