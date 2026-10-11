"""JSON schemas for every structured LLM output, used for Claude structured outputs (``output_config.format``).

They follow the structured-outputs subset: ``additionalProperties: false`` on every object, no numeric or length
constraints (ranges, ID membership and lengths are enforced by the code validators that check every answer anyway).
"""
from __future__ import annotations

STR = {"type": "string"}
INT = {"type": "integer"}
NUM = {"type": "number"}
NULLABLE_STR = {"anyOf": [STR, {"type": "null"}]}


def arr(items: dict) -> dict:
    return {"type": "array", "items": items}


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": list(props) if required is None else required,
            "additionalProperties": False}


def scores_schema(dimensions) -> dict:
    """Judge answer: one score per rubric dimension, unsupported claims and a rationale."""
    return obj({"scores": obj({d: NUM for d in dimensions}), "unsupported_claims": arr(STR), "rationale": STR})


_RANGE = obj({"min": NUM, "max": NUM}, required=[])
_ONE_OR_MANY = {"anyOf": [STR, arr(STR)]}

FEATURE_SCHEMAS: dict[str, dict] = {
    "search_understanding": obj({
        "filters": obj({"code_module": _ONE_OR_MANY, "code_presentation": STR, "region": _ONE_OR_MANY, "highest_education": STR,
                        "age_band": STR, "studied_credits": _RANGE, "num_of_prev_attempts": _RANGE, "final_result": _ONE_OR_MANY},
                       required=[]),
        "descriptors": arr(STR), "explanation": STR}),
    "precedent_synthesis": obj({"what_happened": STR, "patterns": arr(STR), "implication": STR, "cited_cases": arr(INT)}),
    "cohort_briefing": obj({"headline": STR, "key_points": arr(STR), "focus_this_week": arr(STR), "students_to_contact": arr(INT)}),
    "outreach_rewrite": obj({"subject": NULLABLE_STR, "message": STR, "english_back_translation": NULLABLE_STR}),
    "feedback_themes": obj({"themes": arr(obj({"theme": STR, "count": INT, "note_ids": arr(INT), "suggestion": STR}))}),
    "agent:performance": obj({"summary": STR, "strengths": arr(STR), "concerns": arr(STR)}),
    "agent:risk": obj({"risk_summary": STR, "key_factors": arr(STR), "historical_context": STR, "uncertainty_note": STR}),
    "agent:intervention": obj({"action_ids": arr(STR), "outreach_message": STR, "rationale": STR}),
    "judge:pairwise": obj({"winner": {"type": "string", "enum": ["plan_1", "plan_2", "tie"]}, "reason": STR}),
}


_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
                "minItems", "maxItems", "uniqueItems", "pattern", "default", "title"}


def strict(schema: dict) -> dict:
    """Adapt a generic JSON schema (e.g. a pydantic one from DeepEval) to the structured-outputs subset."""
    if isinstance(schema, list):
        return [strict(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in ("properties", "$defs", "definitions") and isinstance(v, dict):
            out[k] = {name: strict(sub) for name, sub in v.items()}  # keys here are names, not keywords
        elif k not in _UNSUPPORTED:
            out[k] = strict(v)
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
        out.setdefault("properties", {})
    return out
