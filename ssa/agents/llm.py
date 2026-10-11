"""LLM access layer used by every generative feature. Provider: Claude (Anthropic SDK) by default, OpenAI optional.

* ``complete_json`` - one JSON object per call (agents, judge, query understanding, synthesis, briefing). On Claude it uses
  structured outputs (``output_config.format``) with the feature's schema; the parse-and-correct retry path stays as the
  fallback (models without structured outputs, a rejected schema, or a malformed answer).
* ``chat`` - one tool-calling turn for the copilot. Messages and tool definitions stay in our provider-neutral format
  (system/user/assistant/tool roles, ``tool_calls``) and are converted at the provider boundary. Assistant turns carry an
  opaque ``provider_content`` so Claude's thinking blocks are replayed verbatim (append-only history).
* The model is chosen per feature (``Settings.model_for``) unless the client is pinned to one (the judge).
* Bounded retries with exponential backoff + jitter (honours ``retry-after``) for rate limits, overload, 5xx, timeouts,
  connection errors and malformed output; other 4xx and refusals fail fast. SDK-internal retries are off.
* SQLite response cache keyed by (provider, model, request). Usage ledger: every call (feature, actual model, tokens incl.
  cache reads/writes, cost, latency, cache hit, error) feeds the cost/latency dashboard (``/api/llm/usage``).
* ``LLM_ENABLED=false`` blocks every request, whatever keys are configured. ``FakeLLM`` serves tests and offline demos.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from ssa.config import Settings, get_settings
from ssa.utils import get_logger

log = get_logger("ssa.llm")


class LLMUnavailable(RuntimeError):
    pass


class LLMRefusal(LLMUnavailable):
    """The model declined (``stop_reason == "refusal"``). Not retried: the caller uses its deterministic fallback."""


class LLMBudgetExceeded(BaseException):
    """LLM_BUDGET_USD reached. A BaseException on purpose: features catch LLM errors and fall back, which would quietly turn
    the rest of an evaluation run into fallback results; this stops the run instead."""


_SPEND = {"usd": 0.0}  # estimated spend of this process, checked against LLM_BUDGET_USD
_SPEND_LOCK = threading.Lock()


def process_spend_usd() -> float:
    return _SPEND["usd"]


class MalformedOutput(ValueError):
    """Unparseable or truncated model output. Retried with a corrective instruction."""

    def __init__(self, msg: str, content=None):
        super().__init__(msg)
        self.content = content  # the raw answer, replayed (append-only) with the correction on the next attempt


@dataclass
class CallRecord:
    model: str
    latency_s: float
    cached: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    feature: str = "unspecified"
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class Usage:
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0  # includes thinking tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def estimate_cost(prices: dict, model: str | None, u: Usage) -> float | None:
    """USD for one call from a per-million-token price table; None when the model has no price entry."""
    p = prices.get(model or "")
    if p is None:
        return None
    prompt = u.input_tokens + u.cache_read_tokens + u.cache_write_tokens
    if p.get("long") and prompt > p.get("long_context_threshold", float("inf")):
        p = {**p, **p["long"]}
    return (u.input_tokens * p["input"] + u.output_tokens * p["output"] + u.cache_read_tokens * p.get("cache_read", p["input"])
            + u.cache_write_tokens * p.get("cache_write", p["input"])) / 1e6


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model text (tolerates code fences / surrounding prose on the fallback path)."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise MalformedOutput("no JSON object in model output") from None
        try:
            data = json.loads(t[i: j + 1])
        except json.JSONDecodeError:
            raise MalformedOutput("malformed JSON from model") from None
    if not isinstance(data, dict):
        raise MalformedOutput("model returned non-object JSON")
    return data


class _Cache:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, created REAL)")

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def get(self, k):
        with self._lock, self._conn() as c:
            row = c.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v):
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (k, json.dumps(v), time.time()))


def _is_transient(exc: Exception) -> bool:
    """Retry rate limits, overload/5xx, timeouts, connection errors and malformed output (both SDKs share these shapes)."""
    if isinstance(exc, MalformedOutput):
        return True
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in (408, 429) or status >= 500  # 529 = overloaded
    return any(b.__name__ in ("APIConnectionError", "APITimeoutError") for b in type(exc).__mro__)


def _retry_after(exc: Exception) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after"))  # type: ignore[attr-defined]
    except Exception:
        return None


def _schema_key(schema: dict) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- Anthropic (Claude)
# Models that accept output_config.format (structured outputs) and output_config.effort, per the Claude docs.
STRUCTURED_OUTPUT_MODELS = {"claude-fable-5-1", "claude-fable-5", "claude-mythos-5-1", "claude-mythos-5", "claude-opus-5-5",
                            "claude-opus-5", "claude-opus-4-8", "claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5",
                            "claude-haiku-4-5", "claude-opus-4-5", "claude-opus-4-1"}
EFFORT_MODELS = {"claude-fable-5-1", "claude-fable-5", "claude-mythos-5-1", "claude-mythos-5", "claude-opus-5-5", "claude-opus-5",
                 "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6",
                 "claude-haiku-5-5", "claude-opus-4-5"}
JSON_FIX = "That was not a valid JSON object for the task. Return only the JSON object."


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, settings: Settings):
        import anthropic

        self.s = settings
        self.sdk = anthropic
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s, max_retries=0)
        self._schema_rejected: set[str] = set()

    def _output_config(self, model: str, effort: str | None, schema: dict | None) -> dict:
        oc = {}
        if effort and model in EFFORT_MODELS:
            oc["effort"] = effort
        if schema is not None and model in STRUCTURED_OUTPUT_MODELS and _schema_key(schema) not in self._schema_rejected:
            oc["format"] = {"type": "json_schema", "schema": schema}
        return oc

    @staticmethod
    def _usage(resp) -> Usage:
        u = resp.usage
        return Usage(u.input_tokens or 0, u.output_tokens or 0, getattr(u, "cache_read_input_tokens", 0) or 0,
                     getattr(u, "cache_creation_input_tokens", 0) or 0)

    @staticmethod
    def _check_stop(resp):
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise LLMRefusal(f"model declined (category: {cat})")

    def complete_json(self, *, model, system, user, max_tokens, effort, schema, previous) -> tuple[dict, Usage, str]:
        messages = [{"role": "user", "content": user}]
        if previous is not None:  # malformed earlier answer: replay it verbatim, then correct (append-only)
            messages += [{"role": "assistant", "content": previous}, {"role": "user", "content": JSON_FIX}]
        kwargs = {"model": model, "max_tokens": max(max_tokens, self.s.claude_max_tokens), "system": system, "messages": messages}
        oc = self._output_config(model, effort, schema)
        try:
            resp = self.client.messages.create(**kwargs, **({"output_config": oc} if oc else {}))
        except self.sdk.BadRequestError as exc:
            if "format" not in oc:
                raise
            # schema not accepted (e.g. an unsupported keyword): remember it and fall back to plain JSON prompting
            self._schema_rejected.add(_schema_key(schema))
            log.warning(f"structured-output schema rejected for {model}; using parse-and-retry ({str(exc)[:160]})")
            oc = {k: v for k, v in oc.items() if k != "format"}
            resp = self.client.messages.create(**kwargs, **({"output_config": oc} if oc else {}))
        self._check_stop(resp)
        content = [b.to_dict() for b in resp.content]
        if resp.stop_reason == "max_tokens":
            raise MalformedOutput("response truncated at max_tokens", content)
        try:
            data = parse_json_object("".join(b.text for b in resp.content if b.type == "text"))
        except MalformedOutput as exc:
            raise MalformedOutput(str(exc), content) from None
        return data, self._usage(resp), resp.model

    # ---- chat (copilot) ----
    @staticmethod
    def convert_tools(tools: list[dict] | None) -> list[dict]:
        out = []
        for t in tools or []:
            f = t.get("function", t)
            out.append({"name": f["name"], "description": f.get("description", ""),
                        "input_schema": f.get("parameters") or f.get("input_schema") or {"type": "object", "properties": {}}})
        return out

    @staticmethod
    def convert_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Provider-neutral history -> (Claude system blocks, Claude messages). Tool results of one turn share one user message."""
        system_parts, out = [], []
        for m in messages:
            role = m.get("role")
            if role == "system":
                system_parts.append(m.get("content") or "")
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content") or ""}
                prev = out[-1] if out else None
                results_turn = prev and prev["role"] == "user" and isinstance(prev["content"], list) \
                    and all(b.get("type") == "tool_result" for b in prev["content"])
                if results_turn:
                    prev["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif role == "assistant":
                if m.get("provider_content"):
                    out.append({"role": "assistant", "content": m["provider_content"]})  # verbatim, thinking blocks included
                    continue
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"],
                                   "input": json.loads(args or "{}") if isinstance(args, str) else args})
                out.append({"role": "assistant", "content": blocks or (m.get("content") or "")})
            else:
                out.append({"role": "user", "content": m.get("content") or ""})
        # One breakpoint at the end of system caches tools + system (render order: tools -> system -> messages).
        system = [{"type": "text", "text": "\n\n".join(system_parts), "cache_control": {"type": "ephemeral"}}] if system_parts else []
        return system, out

    @staticmethod
    def _strip_thinking(msgs: list[dict]) -> list[dict]:
        return [{**m, "content": [b for b in m["content"] if b.get("type") not in ("thinking", "redacted_thinking")]}
                if m["role"] == "assistant" and isinstance(m["content"], list) else m for m in msgs]

    def chat(self, *, model, messages, tools, max_tokens, effort, tool_choice, _sender=None) -> tuple[dict, Usage, str]:
        system, msgs = self.convert_messages(messages)
        kwargs = {"model": model, "max_tokens": max(max_tokens, self.s.claude_max_tokens), "messages": msgs,
                  "cache_control": {"type": "ephemeral"}}  # automatic breakpoint on the growing history (tool-loop rounds)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self.convert_tools(tools)
            kwargs["tool_choice"] = {"type": "none" if tool_choice == "none" else "auto"}  # forced choice is a 400 on these models
        oc = self._output_config(model, effort, None)
        if oc:
            kwargs["output_config"] = oc
        return self._send(kwargs, msgs, _sender or self.client.messages.create)

    def chat_stream(self, *, model, messages, tools, max_tokens, effort, tool_choice, on_text) -> tuple[dict, Usage, str]:
        """Like ``chat`` but streams text deltas to ``on_text`` as they are generated (SDK ``messages.stream``)."""
        def stream(**kwargs):
            with self.client.messages.stream(**kwargs) as st:
                for delta in st.text_stream:
                    on_text(delta)
                return st.get_final_message()
        return self.chat(model=model, messages=messages, tools=tools, max_tokens=max_tokens, effort=effort,
                         tool_choice=tool_choice, _sender=stream)

    def _send(self, kwargs: dict, msgs: list[dict], sender) -> tuple[dict, Usage, str]:
        try:
            resp = sender(**kwargs)
        except self.sdk.BadRequestError as exc:
            if "signature" not in str(exc) or "thinking" not in str(exc):
                raise
            log.warning("thinking blocks rejected for this conversation; retrying once without them")
            kwargs["messages"] = self._strip_thinking(msgs)
            resp = sender(**kwargs)
        self._check_stop(resp)
        text = "".join(b.text for b in resp.content if b.type == "text").strip() or None
        calls = [{"id": b.id, "name": b.name, "arguments": dict(b.input or {})} for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason == "max_tokens" and not calls:
            raise MalformedOutput("response truncated at max_tokens")
        turn = {"content": text, "tool_calls": calls, "provider_content": [b.to_dict() for b in resp.content]}
        return turn, self._usage(resp), resp.model


# --------------------------------------------------------------------------- OpenAI (optional provider)
class OpenAIBackend:
    name = "openai"

    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.s = settings
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url, timeout=settings.llm_timeout_s,
                             max_retries=0)

    @staticmethod
    def _usage(resp) -> Usage:
        u = getattr(resp, "usage", None)
        return Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)

    def complete_json(self, *, model, system, user, max_tokens, effort, schema, previous, temperature=0.1) -> tuple[dict, Usage, str]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if previous is not None:
            messages += [{"role": "assistant", "content": previous}, {"role": "user", "content": JSON_FIX}]
        resp = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                                                   response_format={"type": "json_object"})
        content = resp.choices[0].message.content or ""
        try:
            data = parse_json_object(content)
        except MalformedOutput as exc:
            raise MalformedOutput(str(exc), content) from None
        return data, self._usage(resp), getattr(resp, "model", model)

    def chat(self, *, model, messages, tools, max_tokens, effort, tool_choice, temperature=0.1) -> tuple[dict, Usage, str]:
        msgs = [{k: v for k, v in m.items() if k != "provider_content"} for m in messages]
        kwargs = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            kwargs.update(tools=tools, tool_choice="none" if tool_choice == "none" else "auto")
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"content": msg.content, "tool_calls": calls, "provider_content": None}, self._usage(resp), getattr(resp, "model", model)


# --------------------------------------------------------------------------- client
class LLMClient:
    def __init__(self, settings: Settings | None = None, model: str | None = None, use_cache: bool | None = None,
                 ledger: bool = True):
        self.settings = settings or get_settings()
        self._pinned = model  # e.g. the judge; otherwise the model is chosen per feature
        self.calls: list[CallRecord] = []
        self._backend = None
        use_cache = self.settings.llm_cache_enabled if use_cache is None else use_cache
        self._cache = _Cache(self.settings.llm_cache_db) if use_cache else None
        self._ledger = None
        if ledger:
            try:
                from ssa.appdb import AppDB
                self._ledger = AppDB(self.settings)
            except Exception as exc:  # pragma: no cover
                log.warning(f"usage ledger unavailable: {exc}")
        if self.settings.llm_enabled and self.settings.llm_api_key:
            try:
                self._backend = (AnthropicBackend if self.settings.llm_provider == "anthropic" else OpenAIBackend)(self.settings)
            except Exception as exc:  # pragma: no cover - SDK not installed
                log.warning(f"{self.settings.llm_provider} SDK unavailable: {exc}")

    @property
    def model(self) -> str:
        return self._pinned or self.settings.llm_model

    @property
    def provider(self) -> str:
        return self.settings.llm_provider

    @property
    def available(self) -> bool:
        return self.settings.llm_enabled and self._backend is not None

    def model_for(self, feature: str) -> str:
        return self._pinned or self.settings.model_for(feature)

    def _record(self, rec: CallRecord):
        self.calls.append(rec)
        if self._ledger is not None:
            try:
                self._ledger.log_usage(rec.feature, rec.model, rec.prompt_tokens, rec.completion_tokens, rec.latency_s, rec.cached,
                                       rec.error,
                                       cache_read_tokens=rec.cache_read_tokens, cache_write_tokens=rec.cache_write_tokens,
                                       cost_usd=rec.cost_usd)
            except Exception:  # pragma: no cover
                pass

    def _run(self, key: str, feature: str, model: str, call: Callable[[int, object], tuple[dict, Usage, str]]) -> dict:
        if not self.available:
            raise LLMUnavailable("LLM not configured or disabled (set LLM_ENABLED=true and the provider's API key)")
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self._record(CallRecord(model, 0.0, True, feature=feature, cost_usd=0.0))
                return hit
        last_exc, previous = None, None
        for attempt in range(self.settings.llm_max_retries + 1):
            budget = self.settings.llm_budget_usd
            if budget is not None and _SPEND["usd"] >= budget:
                raise LLMBudgetExceeded(f"LLM_BUDGET_USD={budget} reached (estimated spend ${_SPEND['usd']:.2f})")
            t0 = time.time()
            try:
                data, usage, actual = call(attempt, previous)
            except LLMRefusal:
                self._record(CallRecord(model, round(time.time() - t0, 3), False, error="Refusal", feature=feature))
                raise
            except Exception as exc:
                last_exc = exc
                self._record(CallRecord(model, round(time.time() - t0, 3), False, error=type(exc).__name__, feature=feature))
                if not _is_transient(exc):
                    raise LLMUnavailable(f"LLM call failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
                previous = getattr(exc, "content", None)
                if attempt < self.settings.llm_max_retries:
                    wait = _retry_after(exc)
                    time.sleep(min(20.0, wait if wait is not None else (2 ** attempt) * 0.5 + random.random() * 0.3))
                continue
            cost = estimate_cost(self.settings.llm_prices, actual, usage)
            with _SPEND_LOCK:
                _SPEND["usd"] += cost or 0.0
            self._record(CallRecord(actual, round(time.time() - t0, 3), False, usage.input_tokens, usage.output_tokens, feature=feature,
                                    cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=usage.cache_write_tokens,
                                    cost_usd=cost))
            if self._cache is not None:
                self._cache.put(key, data)
            return data
        raise LLMUnavailable(f"LLM call failed after retries: {type(last_exc).__name__}")

    def complete_json(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 900,
                      feature: str = "unspecified", schema: dict | None = None) -> dict:
        """One JSON object. ``schema`` enables structured outputs where the model supports them. ``temperature`` applies to
        OpenAI only: current Claude models reject non-default sampling parameters, so effort per feature is used instead."""
        model, effort = self.model_for(feature), self.settings.effort_for(feature)
        key = hashlib.sha256(json.dumps(["json", self.provider, model, effort, system, user, temperature, max_tokens, schema],
                                        sort_keys=True).encode()).hexdigest()

        def call(attempt, previous):
            kw = {"model": model, "system": system, "user": user, "max_tokens": max_tokens * (2 if attempt else 1), "effort": effort,
                  "schema": schema, "previous": previous}
            if self.provider == "openai":
                kw["temperature"] = temperature
            return self._backend.complete_json(**kw)

        return self._run(key, feature, model, call)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.1, max_tokens: int = 900,
             feature: str = "copilot", tool_choice: str = "auto") -> dict:
        """One assistant turn with optional tool calling. Returns ``{"content": str|None, "tool_calls": [{"id","name",
        "arguments"(dict)}], "provider_content": opaque|None}``. Put ``provider_content`` on the assistant message you append
        to the history so the provider can replay its own blocks; ``tool_choice="none"`` keeps tools defined but unused."""
        model, effort = self.model_for(feature), self.settings.effort_for(feature)
        key = hashlib.sha256(json.dumps(["chat", self.provider, model, effort, messages, tools, tool_choice, temperature, max_tokens],
                                        default=str, sort_keys=True).encode()).hexdigest()

        def call(attempt, previous):
            kw = {"model": model, "messages": messages, "tools": tools, "max_tokens": max_tokens, "effort": effort,
                  "tool_choice": tool_choice}
            if self.provider == "openai":
                kw["temperature"] = temperature
            return self._backend.chat(**kw)

        return self._run(key, feature, model, call)

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.1, max_tokens: int = 900,
                    feature: str = "copilot", tool_choice: str = "auto", on_text=None, on_reset=None) -> dict:
        """``chat`` with the answer text streamed to ``on_text(delta)``. If an attempt fails after streaming some text, ``on_reset()``
        is called before the retry. Cache hits and backends without streaming deliver the whole text as one delta."""
        model, effort = self.model_for(feature), self.settings.effort_for(feature)
        key = hashlib.sha256(json.dumps(["chat", self.provider, model, effort, messages, tools, tool_choice, temperature, max_tokens],
                                        default=str, sort_keys=True).encode()).hexdigest()
        sent = {"any": False}

        def emit(delta):
            if delta:
                sent["any"] = True
                if on_text:
                    on_text(delta)

        def call(attempt, previous):
            if sent["any"]:
                sent["any"] = False
                if on_reset:
                    on_reset()
            kw = {"model": model, "messages": messages, "tools": tools, "max_tokens": max_tokens, "effort": effort,
                  "tool_choice": tool_choice}
            if hasattr(self._backend, "chat_stream"):
                return self._backend.chat_stream(**kw, on_text=emit)
            if self.provider == "openai":
                kw["temperature"] = temperature
            return self._backend.chat(**kw)

        turn = self._run(key, feature, model, call)
        if not sent["any"] and turn.get("content"):
            emit(turn["content"])
        return turn

    def usage_summary(self) -> dict:
        return {"calls": len(self.calls), "cached": sum(c.cached for c in self.calls),
                "errors": sum(c.error is not None for c in self.calls),
                "latency_s": round(sum(c.latency_s for c in self.calls), 3),
                "prompt_tokens": sum(c.prompt_tokens or 0 for c in self.calls),
                "completion_tokens": sum(c.completion_tokens or 0 for c in self.calls),
                "cache_read_tokens": sum(c.cache_read_tokens for c in self.calls),
                "cache_write_tokens": sum(c.cache_write_tokens for c in self.calls),
                "estimated_cost_usd": round(sum(c.cost_usd or 0 for c in self.calls), 6)}


@dataclass
class FakeLLM:
    """Deterministic stand-in. ``responder(system, user) -> dict`` for JSON calls and optional
    ``chat_responder(messages, tools) -> {"content", "tool_calls"}`` for tool-calling turns."""
    responder: Callable[[str, str], dict]
    chat_responder: Callable[[list, list], dict] | None = None
    model: str = "fake-llm"
    calls: list = field(default_factory=list)

    @property
    def available(self) -> bool:
        return True

    def complete_json(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 900, feature: str = "x",
                      schema: dict | None = None) -> dict:
        self.calls.append(CallRecord(self.model, 0.0, False, feature=feature))
        return self.responder(system, user)

    def chat(self, messages, tools=None, temperature=0.1, max_tokens=900, feature="copilot", tool_choice="auto") -> dict:
        self.calls.append(CallRecord(self.model, 0.0, False, feature=feature))
        if self.chat_responder is None:
            raise LLMUnavailable("fake chat not scripted")
        return self.chat_responder(messages, tools or [])

    def usage_summary(self) -> dict:
        return {"calls": len(self.calls), "cached": 0, "errors": 0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
