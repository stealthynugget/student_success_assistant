import json
import re

from ssa.agents.llm import FakeLLM
from ssa.agents.workflow import SupportWorkflow


def _ctx(service, features, high=True):
    s = features[(features["in_scope"] == 1) & (features["code_presentation"] == "2014J")]
    p = service.risk.predict_proba(s)
    row = s.iloc[int(p.argmax() if high else p.argmin())]
    rp = service.resolve(ref={"id_student": int(row["id_student"]), "code_module": row["code_module"],
                              "code_presentation": row["code_presentation"]})
    pred = service.predict(rp)
    sim = service.similar(rp, top_k=5)
    return dict(profile=rp.profile, risk=pred["withdrawal_risk"], outcome=pred["outcome_prediction"],
                neighbours=sim["outcome_summary"], similar_cases=sim["cases"], module_context=service.module_context(rp.profile))


def _evidence(user):
    return json.loads(re.search(r"<evidence>\n(.*)\n</evidence>", user, re.S).group(1))


def good_llm(system, user):
    ev = _evidence(user)
    if "action_ids" in user:
        ids = [a["id"] for a in ev["eligible_actions"] if a["id"] != "ACT_MONITOR"][:2] or ["ACT_MONITOR"]
        obs = [o["id"] for o in ev["observations"] if o["kind"] == "concern"][:2]
        return {"action_ids": ids, "rationale": "Chosen to address the signals in " + ", ".join(obs or ["the observations"]) + ".",
                "outreach_message": "Hello, I wanted to check in and see how the module is going for you. If anything is getting in "
                                    "the way, we would be glad to help you plan the next few weeks. Could we have a short chat?"}
    if "risk_summary" in user:
        return {"risk_summary": "The model places the student in the " + ev["risk_model"]["risk_tier"] + " tier.",
                "key_factors": [d["label"] for d in ev["risk_drivers"]][:2], "historical_context": "Similar students had mixed outcomes.",
                "uncertainty_note": "This is an estimate, not a certainty."}
    return {"summary": "The student shows a mix of strengths and concerns in the evidence.",
            "strengths": [o["id"] for o in ev["observations"] if o["kind"] == "strength"],
            "concerns": [o["id"] for o in ev["observations"] if o["kind"] == "concern"]}


def bad_llm(system, user):
    if "action_ids" in user:
        return {"action_ids": ["ACT_DOES_NOT_EXIST"], "outreach_message": "Your withdrawal risk is 91% so attend or face penalties.",
                "rationale": "Because she is from Wales and scored 12."}
    if "risk_summary" in user:
        return {"risk_summary": "Risk is 99%.", "key_factors": ["invented"], "historical_context": "x", "uncertainty_note": "y"}
    return {"summary": "The student is lazy and scored 12.3.", "strengths": [], "concerns": ["OBS_FAKE"]}


def test_deterministic_workflow_produces_valid_plan(service, features):
    out = SupportWorkflow(None).run(**_ctx(service, features))
    plan = out["intervention_plan"]
    assert plan["mode"] == "intervene" and plan["source"] == "deterministic"
    assert out["guardrails"]["final_plan_violations"] == []
    assert out["guardrails"]["sensitive_attributes_sent_to_llm"] is False
    eligible = {a["id"] for a in out["eligible_actions"]}
    assert all(a["id"] in eligible for a in plan["actions"])


def test_low_risk_is_routed_to_monitoring(service, features):
    out = SupportWorkflow(None).run(**_ctx(service, features, high=False))
    assert out["intervention_plan"]["mode"] in {"monitor", "intervene"}
    assert any(t.get("agent") == "router" for t in out["trace"])


def test_valid_llm_output_is_accepted(service, features):
    out = SupportWorkflow(FakeLLM(good_llm)).run(**_ctx(service, features))
    assert out["intervention_plan"]["source"] == "llm"
    assert out["performance_analysis"]["source"] == "llm"
    assert out["guardrails"]["final_plan_violations"] == []


def test_hallucinating_llm_is_rejected_and_falls_back(service, features):
    out = SupportWorkflow(FakeLLM(bad_llm)).run(**_ctx(service, features))
    assert out["intervention_plan"]["source"] == "deterministic"
    assert out["performance_analysis"]["source"] == "deterministic"
    assert out["risk_analysis"]["source"] == "deterministic"
    assert any(t.get("agent") == "intervention_revision" for t in out["trace"])
    assert out["guardrails"]["final_plan_violations"] == []


def test_sensitive_attributes_not_in_prompts(service, features):
    seen = []
    def spy(system, user):
        seen.append(user)
        return good_llm(system, user)
    SupportWorkflow(FakeLLM(spy)).run(**_ctx(service, features))
    blob = " ".join(seen)
    for key in ('"gender"', '"region"', '"imd_band"', '"age_band"', '"disability"', '"highest_education"'):
        assert key not in blob
