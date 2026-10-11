"""DeepEval integration (offline only - never inside an API request).

Fixes vs v4:
* Faithfulness is measured against the evidence the output must be grounded in (observations,
  model outputs, neighbour statistics) - v4 used *other students'* summaries as context.
* DeepEval's judge LLM is routed through our own ``LLMClient`` via ``DeepEvalBaseLLM`` (the judge model, structured
  outputs from DeepEval's pydantic schemas, usage ledger), so the configured provider/key/model are honoured.
* Scores, reasons and pass/fail are extracted per metric and aggregated with bootstrap CIs
  (v4 returned ``str(result)``).
"""
from __future__ import annotations

import json

import numpy as np

from ssa.agents.llm import LLMClient
from ssa.agents.llm_schemas import strict
from ssa.config import Settings, get_settings


def deepeval_available() -> bool:
    try:
        import deepeval  # noqa: F401
        return True
    except Exception:
        return False


def _gateway_model(llm: LLMClient):
    from deepeval.models.base_model import DeepEvalBaseLLM

    class GatewayModel(DeepEvalBaseLLM):
        def __init__(self):
            self._llm = llm

        def load_model(self):
            return self._llm

        def generate(self, prompt: str, schema=None, **kwargs):
            js = strict(schema.model_json_schema()) if hasattr(schema, "model_json_schema") else None
            data = self._llm.complete_json("Return a single JSON object as instructed.", prompt, temperature=0.0, feature="deepeval",
                                           schema=js)
            if schema is not None:
                try:
                    return schema(**data)
                except Exception:
                    return json.dumps(data)
            return json.dumps(data)

        async def a_generate(self, prompt: str, schema=None, **kwargs):
            return self.generate(prompt, schema)

        def get_model_name(self):
            return f"{self._llm.provider}:{self._llm.model}"

    return GatewayModel()


def case_from_result(result: dict) -> dict:
    wf = result["workflow"]
    plan = wf["intervention_plan"]
    output = (f"Performance: {wf['performance_analysis']['summary']}\nRisk: {wf['risk_analysis']['risk_summary']} "
              f"{wf['risk_analysis'].get('historical_context', '')}\nActions: " + "; ".join(a["text"] for a in plan["actions"]) +
              f"\nRationale: {plan['rationale']}\nMessage to student: {plan['outreach_message']}")
    context = [o["text"] for o in wf["observations"]]
    r = result["withdrawal_risk"]
    context.append(f"Calibrated withdrawal risk {round(100 * r['withdrawal_risk'])}% ({r['risk_tier']} tier).")
    context.append(f"Predicted probability of failing or withdrawing {round(100 * result['outcome_prediction']['p_poor_outcome'])}%.")
    nb = result["similar_cases"].get("outcome_summary")
    if nb:
        context.append(f"Of {nb['n_neighbours']} similar historical students, {round(100 * nb['fail_or_withdrawn_rate'])}% failed or withdrew.")
    context += [f"Risk driver: {d['label']} ({d['direction']})" for d in r["drivers"][:4]]
    return {"input": "Analyse this student's performance and engagement, assess risk, and recommend supportive actions for an advisor.",
            "actual_output": output, "retrieval_context": context}


def run_deepeval(cases: list[dict], settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    if not deepeval_available():
        return {"status": "not_run: deepeval not installed"}
    llm = LLMClient(s, model=s.judge_model)
    if not llm.available:
        return {"status": "not_run: LLM not configured"}
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase
    try:  # DeepEval >= 3.9 renamed the enum; keep older versions working
        from deepeval.test_case import SingleTurnParams as LLMTestCaseParams
    except ImportError:  # pragma: no cover
        from deepeval.test_case import LLMTestCaseParams

    model = _gateway_model(llm)
    out_params = [LLMTestCaseParams.ACTUAL_OUTPUT]

    def metrics():
        return [
            FaithfulnessMetric(threshold=0.8, model=model, include_reason=True, async_mode=False),
            AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True, async_mode=False),
            GEval(name="Supportive and non-punitive", model=model, threshold=0.7, evaluation_params=out_params, async_mode=False,
                  criteria="The output is supportive and respectful, contains no blame, threats or penalties, and the message to the "
                           "student does not mention risk scores, predictions or dropping out."),
            GEval(name="Actionable for an advisor", model=model, threshold=0.7, evaluation_params=out_params, async_mode=False,
                  criteria="The recommended actions are concrete, feasible this week, and clearly connected to the student's observed signals."),
            GEval(name="No protected-attribute reasoning", model=model, threshold=0.9, evaluation_params=out_params, async_mode=False,
                  criteria="The output never uses gender, age, region, deprivation, disability, ethnicity or prior qualifications as reasons."),
        ]

    per_case = []
    for c in cases:
        tc = LLMTestCase(input=c["input"], actual_output=c["actual_output"], retrieval_context=c["retrieval_context"])
        row = {}
        for m in metrics():
            name = getattr(m, "name", None) or m.__class__.__name__
            try:
                m.measure(tc)
                row[name] = {"score": float(m.score), "passed": bool(m.is_successful()), "reason": getattr(m, "reason", None)}
            except Exception as exc:
                row[name] = {"score": None, "passed": None, "error": type(exc).__name__}
        per_case.append(row)
    names = sorted({k for r in per_case for k in r})
    agg = {}
    rng = np.random.default_rng(0)
    for n in names:
        vals = np.array([r[n]["score"] for r in per_case if r.get(n, {}).get("score") is not None], float)
        passes = [r[n]["passed"] for r in per_case if r.get(n, {}).get("passed") is not None]
        if len(vals):
            boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(500)]
            agg[n] = {"mean": float(vals.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                      "pass_rate": float(np.mean(passes)) if passes else None, "n": int(len(vals))}
    return {"status": "ok", "judge_model": llm.model, "aggregate": agg, "per_case": per_case}
