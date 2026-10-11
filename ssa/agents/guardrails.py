"""Deterministic guardrails applied to every agent output and every free-text input.

These checks are code, not prompts: an LLM cannot talk its way past them.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from ssa.data import vocab

_PROTECTED = [
    r"\bgender\b", r"\bmale\b", r"\bfemale\b", r"\bman\b", r"\bwoman\b", r"\bhis\b", r"\bher\b", r"\bhe\b", r"\bshe\b",
    r"\bage(d)?\b", r"\bage band\b", r"\bolder\b", r"\byounger\b", r"\belderly\b", r"\bimd\b", r"\bdepriv\w*",
    r"\bpoverty\b", r"\bsocio-?economic\b", r"\bincome\b", r"\bdisab\w*", r"\bethnic\w*", r"\brace\b", r"\breligio\w*",
    r"\bregion\b", r"\bprior education\b", r"\bqualification(s)?\b", r"\bno formal quals\b", r"\ba level\b",
] + [rf"\b{re.escape(r.lower())}\b" for r in vocab.REGIONS]
_PUNITIVE = [r"\bpenalt\w*", r"\bpunish\w*", r"\bdisciplin\w*", r"\bsanction\w*", r"\bprobation\b", r"\bexpel\w*",
             r"\blazy\b", r"\bwarning letter\b", r"\bconsequences\b", r"\bfail(ing)? you\b", r"\bthreat\w*", r"\bremov(e|al) from\b"]
_DIAGNOSTIC = [r"\bdepress\w*", r"\banxiety disorder\b", r"\badhd\b", r"\bmental illness\b", r"\bdiagnos\w*", r"\bdisorder\b",
               r"\btherapy\b"]
_STIGMA_IN_OUTREACH = [r"\brisk\b", r"\bprobabilit\w*", r"\bpercentile\b", r"\bpredict\w*", r"\bmodel\b", r"%", r"\bwithdraw\w*",
                       r"\bdrop(ping)? out\b", r"\balgorithm\w*"]
_INJECTION = [r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)", r"disregard (the )?(system|previous)",
              r"\bsystem prompt\b", r"\byou are now\b", r"\bdeveloper mode\b", r"\bjailbreak\b", r"<\s*/?\s*(script|system|evidence)",
              r"\bact as\b", r"\bpretend to be\b", r"\breveal (your|the) (prompt|instructions)", r"\bBEGIN (SYSTEM|PROMPT)\b"]


def _hits(text: str, patterns: Iterable[str]) -> list[str]:
    t = (text or "").lower()
    out = []
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            out.append(m.group(0))
    return sorted(set(out))


def protected_mentions(text: str) -> list[str]:
    return _hits(text, _PROTECTED)


def punitive_terms(text: str) -> list[str]:
    return _hits(text, _PUNITIVE)


def diagnostic_terms(text: str) -> list[str]:
    return _hits(text, _DIAGNOSTIC)


def outreach_stigma_terms(text: str) -> list[str]:
    return _hits(text, _STIGMA_IN_OUTREACH)


def injection_signals(text: str) -> list[str]:
    return _hits(text, _INJECTION)


def sanitize_text(text: str, max_len: int = 500) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if ch.isprintable() or ch in " \t")
    return re.sub(r"\s+", " ", text).strip()[:max_len]


# "1,702" is one number (thousands separators), not 1 and 702; plain numbers as before.
_NUM = re.compile(r"(?<![\w.,])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|(?<![\w.])-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in _NUM.findall(text or "")]


def _collect_numbers(obj: Any, out: set[float]):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        if v == v:
            for x in (v, v * 100):
                out.update({round(x, 0), round(x, 1), round(x, 2)})
        return
    if isinstance(obj, str):
        for n in numbers_in(obj):
            out.add(n)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def ungrounded_numbers(text: str, evidence: Any, tolerance: float = 0.51) -> list[float]:
    """Numbers stated in ``text`` that do not appear (within tolerance) anywhere in ``evidence``."""
    pool: set[float] = set()
    _collect_numbers(evidence, pool)
    bad = []
    for n in numbers_in(text):
        if not any(abs(n - p) <= tolerance or (p != 0 and abs(n - p) / abs(p) < 0.01) for p in pool):
            bad.append(n)
    return bad
