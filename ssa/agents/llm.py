"""LLM access layer (OpenAI-compatible) used by every generative feature.

* ``complete_json`` - JSON-mode completions (agents, judge, query understanding, synthesis, briefing).
* ``chat`` - tool/function-calling turns for the copilot.
* Timeouts, bounded retries with exponential backoff + jitter, one corrective retry on malformed JSON.
* SQLite response cache keyed by (model, messages, params).
* Usage ledger: every call (feature, model, tokens, latency, cache hit, error) is recorded for the cost/latency
  dashboard (``/api/llm/usage``).
* ``FakeLLM`` provides deterministic, scriptable responses for tests and offline demos.
"""
from __future__ import annotations

import hashlib
import json
import random
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


@dataclass
class CallRecord:
    model: str
    latency_s: float
    cached: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    feature: str = "unspecified"


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


class LLMClient:
    def __init__(self, settings: Settings | None = None, model: str | None = None, use_cache: bool | None = None,
                 ledger: bool = True):
        self.settings = settings or get_settings()
        self.model = model or self.settings.llm_model
        self.calls: list[CallRecord] = []
        self._client = None
        use_cache = self.settings.llm_cache_enabled if use_cache is None else use_cache
        self._cache = _Cache(self.settings.llm_cache_db) if use_cache else None
        self._ledger = None
        if ledger:
            try:
                from ssa.appdb import AppDB
                self._ledger = AppDB(self.settings)
            except Exception as exc:  # pragma: no cover
                log.warning(f"usage ledger unavailable: {exc}")
        if self.settings.llm_enabled and self.settings.openai_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url,
                                      timeout=self.settings.llm_timeout_s, max_retries=0)
            except Exception as exc:  # pragma: no cover - openai optional
                log.warning(f"openai SDK unavailable: {exc}")

    @property
    def available(self) -> bool:
        return self.settings.llm_enabled and self._client is not None

    def _record(self, rec: CallRecord):
        self.calls.append(rec)
        if self._ledger is not None:
            try:
                self._ledger.log_usage(rec.feature, rec.model, rec.prompt_tokens, rec.completion_tokens, rec.latency_s,
                                       rec.cached, rec.error)
            except Exception:  # pragma: no cover
                pass

    def _with_retries(self, key: str, feature: str, call: Callable[[], tuple[dict, object]]) -> dict:
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self._record(CallRecord(self.model, 0.0, True, feature=feature))
                return hit
        last_exc = None
        for attempt in range(self.settings.llm_max_retries + 1):
            t0 = time.time()
            try:
                data, usage = call()
                self._record(CallRecord(self.model, round(time.time() - t0, 3), False, getattr(usage, "prompt_tokens", None),
                                        getattr(usage, "completion_tokens", None), feature=feature))
                if self._cache is not None:
                    self._cache.put(key, data)
                return data
            except Exception as exc:
                last_exc = exc
                self._record(CallRecord(self.model, round(time.time() - t0, 3), False, error=type(exc).__name__, feature=feature))
                if attempt < self.settings.llm_max_retries:
                    time.sleep(min(8.0, (2 ** attempt) * 0.5 + random.random() * 0.3))
        raise LLMUnavailable(f"LLM call failed after retries: {type(last_exc).__name__}")

    def complete_json(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 900,
                      feature: str = "unspecified") -> dict:
        if not self.available:
            raise LLMUnavailable("LLM not configured (set OPENAI_API_KEY)")
        key = hashlib.sha256(json.dumps(["json", self.model, system, user, temperature, max_tokens]).encode()).hexdigest()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def call():
            nonlocal messages
            resp = self._client.chat.completions.create(model=self.model, messages=messages, temperature=temperature,
                                                        max_tokens=max_tokens, response_format={"type": "json_object"})
            content = resp.choices[0].message.content or ""
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                messages = messages + [{"role": "assistant", "content": content},
                                       {"role": "user", "content": "That was not valid JSON. Return only the JSON object."}]
                raise ValueError("malformed JSON from model")
            if not isinstance(data, dict):
                raise ValueError("model returned non-object JSON")
            return data, getattr(resp, "usage", None)

        return self._with_retries(key, feature, call)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.1, max_tokens: int = 900,
             feature: str = "copilot") -> dict:
        """One assistant turn with optional function calling. Returns
        ``{"content": str|None, "tool_calls": [{"id","name","arguments"(dict)}]}``."""
        if not self.available:
            raise LLMUnavailable("LLM not configured (set OPENAI_API_KEY)")
        key = hashlib.sha256(json.dumps(["chat", self.model, messages, tools, temperature, max_tokens], default=str).encode()).hexdigest()

        def call():
            kwargs = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            if tools:
                kwargs.update(tools=tools, tool_choice="auto")
            resp = self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            calls = []
            for tc in getattr(msg, "tool_calls", None) or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
            return {"content": msg.content, "tool_calls": calls}, getattr(resp, "usage", None)

        return self._with_retries(key, feature, call)

    def usage_summary(self) -> dict:
        return {"calls": len(self.calls), "cached": sum(c.cached for c in self.calls),
                "errors": sum(c.error is not None for c in self.calls),
                "latency_s": round(sum(c.latency_s for c in self.calls), 3),
                "prompt_tokens": sum(c.prompt_tokens or 0 for c in self.calls),
                "completion_tokens": sum(c.completion_tokens or 0 for c in self.calls)}


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

    def complete_json(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 900, feature: str = "x") -> dict:
        self.calls.append(CallRecord(self.model, 0.0, False, feature=feature))
        return self.responder(system, user)

    def chat(self, messages, tools=None, temperature=0.1, max_tokens=900, feature="copilot") -> dict:
        self.calls.append(CallRecord(self.model, 0.0, False, feature=feature))
        if self.chat_responder is None:
            raise LLMUnavailable("fake chat not scripted")
        return self.chat_responder(messages, tools or [])

    def usage_summary(self) -> dict:
        return {"calls": len(self.calls), "cached": 0, "errors": 0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
