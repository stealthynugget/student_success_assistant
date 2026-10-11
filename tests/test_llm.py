"""LLM client: Claude backend with a mocked Anthropic SDK (no network), the LLM_ENABLED switch, routing and pricing."""
from dataclasses import replace

import anthropic
import httpx
import httpx2
import pytest
from anthropic.types import Message

from ssa.agents.llm import JSON_FIX, LLMClient, LLMRefusal, LLMUnavailable, Usage, estimate_cost
from ssa.agents.llm_schemas import FEATURE_SCHEMAS

FAKE_KEY = "fake-test-key"


def _settings(ssa_home, **kw):
    from ssa.config import get_settings
    base = {"llm_provider": "anthropic", "anthropic_api_key": FAKE_KEY, "openai_api_key": FAKE_KEY, "llm_max_retries": 2,
            "llm_cache_enabled": False}
    return replace(get_settings(), **{**base, **kw})


def _msg(content, stop="end_turn", model="claude-sonnet-5-5", usage=None):
    return Message.model_validate({"id": "msg_x", "type": "message", "role": "assistant", "model": model, "content": content,
                                   "stop_reason": stop, "stop_sequence": None,
                                   "usage": usage or {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0,
                                                      "cache_creation_input_tokens": 0}})


def _text(t):
    return {"type": "text", "text": t}


class Scripted:
    """Stands in for client.messages.create: returns (or raises) scripted items and records every request."""

    def __init__(self, *items):
        self.items, self.requests = list(items), []

    def __call__(self, **kwargs):
        self.requests.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(ssa_home, script, monkeypatch, **kw):
    llm = LLMClient(_settings(ssa_home, **kw), ledger=False)
    monkeypatch.setattr(llm._backend.client.messages, "create", script)
    monkeypatch.setattr("ssa.agents.llm.time.sleep", lambda s: None)
    return llm


def _status_error(cls, status, headers=None):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("scripted", response=httpx2.Response(status, request=req, headers=headers or {}), body=None)


# ------------------------------------------------------------------ LLM_ENABLED
@pytest.fixture()
def http_attempts(monkeypatch):
    """Record every HTTP request either SDK tries to send (openai: httpx, anthropic 1.x: httpx2) and fail it locally."""
    attempts = []

    def intercept(lib):
        def send(self, request, *a, **k):
            attempts.append(str(request.url))
            raise lib.ConnectError("blocked by test", request=request)
        return send

    monkeypatch.setattr(httpx.Client, "send", intercept(httpx))
    monkeypatch.setattr(httpx2.Client, "send", intercept(httpx2))
    return attempts


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_llm_disabled_sends_nothing_even_with_key(ssa_home, http_attempts, provider):
    off = LLMClient(_settings(ssa_home, llm_enabled=False, llm_provider=provider, llm_max_retries=0), ledger=False)
    assert not off.available
    with pytest.raises(LLMUnavailable):
        off.complete_json("system", "user")
    with pytest.raises(LLMUnavailable):
        off.chat([{"role": "user", "content": "hi"}])
    assert http_attempts == []
    # control: with the switch on, the same interception does see a request, so the check above is meaningful
    on = LLMClient(_settings(ssa_home, llm_enabled=True, llm_provider=provider, llm_max_retries=0), ledger=False)
    assert on.available
    with pytest.raises(LLMUnavailable):
        on.complete_json("system", "user")
    assert http_attempts


# ------------------------------------------------------------------ structured JSON
def test_complete_json_uses_structured_outputs_and_records_usage(ssa_home, monkeypatch):
    usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    script = Scripted(_msg([{"type": "thinking", "thinking": "", "signature": "s1"},
                            _text('{"summary": "Steady work.", "strengths": ["OBS_1"], "concerns": []}')], usage=usage))
    llm = _client(ssa_home, script, monkeypatch)
    schema = FEATURE_SCHEMAS["agent:performance"]
    out = llm.complete_json("SYSTEM PROMPT", "task", temperature=0.7, feature="agent:performance", schema=schema)
    assert out == {"summary": "Steady work.", "strengths": ["OBS_1"], "concerns": []}
    req = script.requests[0]
    assert req["model"] == "claude-sonnet-5-5" and req["system"] == "SYSTEM PROMPT"
    assert req["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": schema}}
    assert "temperature" not in req  # Claude 5.x models reject non-default sampling parameters
    rec = llm.calls[-1]
    assert (rec.model, rec.prompt_tokens, rec.completion_tokens, rec.error) == ("claude-sonnet-5-5", 1000, 500, None)
    assert rec.cost_usd == pytest.approx((1000 * 2.00 + 500 * 10.00) / 1e6)


def test_features_route_to_fast_main_and_judge_models(ssa_home, monkeypatch):
    script = Scripted(*[_msg([_text("{}")]) for _ in range(3)])
    llm = _client(ssa_home, script, monkeypatch)
    llm.complete_json("s", "u", feature="search_understanding")
    llm.complete_json("s", "u", feature="precedent_synthesis")
    judge = LLMClient(_settings(ssa_home), model=llm.settings.judge_model, ledger=False)  # how LLMJudge builds its client
    monkeypatch.setattr(judge._backend.client.messages, "create", script)
    judge.complete_json("s", "u", feature="judge:analysis")
    assert [(r["model"], r["output_config"]["effort"]) for r in script.requests] == [
        ("claude-haiku-5-5", "low"), ("claude-sonnet-5-5", "medium"), ("claude-opus-5-5", "high")]


# ------------------------------------------------------------------ malformed output fallback
def test_malformed_output_is_replayed_with_a_correction(ssa_home, monkeypatch):
    bad = _msg([{"type": "thinking", "thinking": "", "signature": "s1"}, _text("Sure! Here you go: not json")])
    script = Scripted(bad, _msg([_text('{"winner": "tie", "reason": "equal"}')]))
    llm = _client(ssa_home, script, monkeypatch)
    out = llm.complete_json("s", "compare", feature="judge:pairwise", schema=FEATURE_SCHEMAS["judge:pairwise"])
    assert out == {"winner": "tie", "reason": "equal"}
    second = script.requests[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "s1"}
    assert second[2] == {"role": "user", "content": JSON_FIX}
    assert [c.error for c in llm.calls] == ["MalformedOutput", None]


def test_rejected_schema_falls_back_to_parse_path(ssa_home, monkeypatch):
    script = Scripted(_status_error(anthropic.BadRequestError, 400), _msg([_text('```json\n{"themes": []}\n```')]))
    llm = _client(ssa_home, script, monkeypatch)
    assert llm.complete_json("s", "u", feature="feedback_themes", schema=FEATURE_SCHEMAS["feedback_themes"]) == {"themes": []}
    assert "format" in script.requests[0]["output_config"] and "format" not in script.requests[1]["output_config"]


# ------------------------------------------------------------------ errors and retries
def test_transient_errors_are_retried_with_backoff(ssa_home, monkeypatch):
    script = Scripted(_status_error(anthropic.RateLimitError, 429, {"retry-after": "1"}),
                      _status_error(anthropic.InternalServerError, 529), _msg([_text('{"ok": true}')]))
    sleeps = []
    llm = _client(ssa_home, script, monkeypatch)
    monkeypatch.setattr("ssa.agents.llm.time.sleep", sleeps.append)
    assert llm.complete_json("s", "u", feature="agent:risk") == {"ok": True}
    assert [c.error for c in llm.calls] == ["RateLimitError", "InternalServerError", None]
    assert sleeps[0] == 1.0 and len(sleeps) == 2  # honours retry-after, then exponential backoff


def test_client_errors_and_refusals_are_not_retried(ssa_home, monkeypatch):
    script = Scripted(_status_error(anthropic.AuthenticationError, 401))
    llm = _client(ssa_home, script, monkeypatch)
    with pytest.raises(LLMUnavailable):
        llm.complete_json("s", "u", feature="agent:risk")
    assert len(script.requests) == 1
    refuse = Scripted(_msg([], stop="refusal"))
    llm = _client(ssa_home, refuse, monkeypatch)
    with pytest.raises(LLMRefusal):
        llm.complete_json("s", "u", feature="agent:risk")
    assert len(refuse.requests) == 1


# ------------------------------------------------------------------ copilot tool-use loop
def test_copilot_tool_loop_over_claude(ssa_home, service, features, monkeypatch):
    from ssa.agents.copilot import TOOLS, Copilot
    from tests.conftest import ref_of
    ref = ref_of(features)
    think = {"type": "thinking", "thinking": "", "signature": "sig-round-1"}
    script = Scripted(
        _msg([think, {"type": "tool_use", "id": "toolu_a", "name": "get_student", "input": ref}], stop="tool_use"),
        _msg([{"type": "tool_use", "id": "toolu_b", "name": "cohort_overview",
               "input": {"code_module": ref["code_module"], "code_presentation": ref["code_presentation"]}}], stop="tool_use"),
        _msg([_text(f"Student {ref['id_student']}'s record and the cohort overview are summarised above [get_student].")]),
    )
    llm = _client(ssa_home, script, monkeypatch)
    out = Copilot(service, llm).ask(f"How is student {ref['id_student']} doing compared with the cohort?")
    assert out["source"] == "llm" and [t["tool"] for t in out["tool_calls"]] == ["get_student", "cohort_overview"]
    assert len(script.requests) == 3
    first, third = script.requests[0], script.requests[2]
    assert first["tools"][0] == {"name": TOOLS[0]["function"]["name"], "description": TOOLS[0]["function"]["description"],
                                 "input_schema": TOOLS[0]["function"]["parameters"]}
    assert first["tool_choice"] == {"type": "auto"} and first["system"][0]["cache_control"] == {"type": "ephemeral"}
    msgs = third["messages"]
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"][0] == think  # thinking replayed verbatim
    result = msgs[2]["content"][0]
    assert msgs[2]["role"] == "user" and len(msgs[2]["content"]) == 1
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_a" and ref["code_module"] in result["content"]
    assert msgs[4]["content"][0]["tool_use_id"] == "toolu_b"
    assert third["tools"] == first["tools"] and third["system"] == first["system"]  # stable prefix: cacheable, thinking stays valid


def test_message_conversion_groups_parallel_tool_results():
    from ssa.agents.llm import AnthropicBackend
    system, msgs = AnthropicBackend.convert_messages([
        {"role": "system", "content": "S"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                        {"id": "t2", "type": "function", "function": {"name": "b", "arguments": '{"x": 1}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "r1"}, {"role": "tool", "tool_call_id": "t2", "content": "r2"}])
    assert system[0]["text"] == "S" and len(msgs) == 3
    assert [b["input"] for b in msgs[1]["content"]] == [{}, {"x": 1}]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["t1", "t2"]  # all results of one turn in one user message


# ------------------------------------------------------------------ config and pricing
def test_judge_must_differ_from_generators(monkeypatch):
    from ssa.config import _llm_models
    monkeypatch.setenv("CLAUDE_MODEL_JUDGE", "claude-haiku-5-5")
    with pytest.raises(ValueError):
        _llm_models("anthropic")


def test_haiku_long_prompt_tier_pricing(ssa_home):
    prices = _settings(ssa_home).llm_prices
    short = estimate_cost(prices, "claude-haiku-5-5", Usage(input_tokens=90_000, output_tokens=1_000))
    long = estimate_cost(prices, "claude-haiku-5-5", Usage(input_tokens=90_000, output_tokens=1_000, cache_read_tokens=20_000))
    assert short == pytest.approx((90_000 * 0.10 + 1_000 * 0.50) / 1e6)
    assert long == pytest.approx((90_000 * 0.50 + 1_000 * 2.50 + 20_000 * 0.05) / 1e6)
    assert estimate_cost(prices, "unknown-model", Usage(1, 1)) is None


def test_budget_cap_stops_instead_of_falling_back(ssa_home, monkeypatch):
    import ssa.agents.llm as L
    monkeypatch.setitem(L._SPEND, "usd", 0.0)
    script = Scripted(_msg([_text('{"ok": 1}')]), _msg([_text('{"ok": 2}')]))
    llm = _client(ssa_home, script, monkeypatch, llm_budget_usd=0.001)
    assert llm.complete_json("s", "u", feature="agent:risk") == {"ok": 1}  # 1000 in + 200 out on sonnet = $0.004 > cap
    with pytest.raises(L.LLMBudgetExceeded):
        try:
            llm.complete_json("s", "u2", feature="agent:risk")
        except Exception:  # feature code catches Exception-derived LLM errors; the cap must get past that
            pytest.fail("budget stop was swallowed by an except Exception handler")
    assert len(script.requests) == 1
