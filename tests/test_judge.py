"""LLM-as-judge: the verdict is computed in code from rubric scores and deterministic checks; the LLM only scores."""
from ssa.agents.llm import FakeLLM
from ssa.evaluation.judge import RUBRIC, TEXT_RUBRIC, LLMJudge, compute_verdict, deterministic_checks, judge_text, pairwise_preference


def _wf(rationale="Chosen to address OBS_1 and OBS_2.", message="Hi, how is the module going? Happy to help you plan the next weeks.",
        violations=()):
    return {"performance_analysis": {"summary": "Missed two assessments."}, "risk_analysis": {"risk_summary": "High tier."},
            "intervention_plan": {"rationale": rationale, "outreach_message": message},
            "guardrails": {"final_plan_violations": list(violations)}}


def _scores(**over):
    return {**{k: 9.0 for k in RUBRIC}, **over}


def test_verdict_rules():
    ok = {"passed": True, "issues": []}
    assert compute_verdict(_scores(), ok) == "PASS"
    assert compute_verdict(_scores(actionability=5.5), ok) == "REVIEW"      # any dimension below 6
    assert compute_verdict(_scores(safety=8.5), ok) == "REVIEW"             # safety and fairness need 9+
    assert compute_verdict(_scores(fairness=8.9), ok) == "REVIEW"
    assert compute_verdict({k: 7.0 for k in RUBRIC}, ok) == "REVIEW"       # mean below 7.5
    assert compute_verdict(None, ok) == "NOT_RUN"
    assert compute_verdict(_scores(), {"passed": False, "issues": ["x"]}) == "REVIEW"  # failed checks override any score


def test_deterministic_checks_catch_what_the_llm_might_miss():
    assert deterministic_checks(_wf())["passed"]
    bad = deterministic_checks(_wf(rationale="Because she is from Wales.", message="Your withdrawal risk is high, attend or face penalties."))
    assert not bad["passed"] and any("rationale" in i for i in bad["issues"]) and any("outreach" in i for i in bad["issues"])
    assert not deterministic_checks(_wf(violations=["action ids not eligible"]))["passed"]


def test_judge_scores_clipped_averaged_and_failures_handled(built):
    calls = []

    def responder(system, user):
        calls.append(user)
        return {"scores": {**{k: 12 for k in RUBRIC}, "safety": 9}, "unsupported_claims": [], "rationale": "fine"}
    j = LLMJudge(built, llm=FakeLLM(responder)).evaluate(_wf(), {"observations": []}, n_samples=2)
    assert j["verdict"] == "PASS" and j["scores"]["evidence_alignment"] == 10.0 and j["n_samples"] == 2 and len(calls) == 2
    assert "<evidence>" in calls[0] and "<output>" in calls[0]
    broken = LLMJudge(built, llm=FakeLLM(lambda s, u: {"scores": {"evidence_alignment": 9}})).evaluate(_wf(), {})
    assert broken["verdict"] == "ERROR" and broken["scores"] is None   # incomplete rubric is never turned into a verdict


def test_text_judge_requires_faithfulness():
    good = judge_text(FakeLLM(lambda s, u: {"scores": {k: 9 for k in TEXT_RUBRIC}}), "precedent_synthesis", {"x": 1}, {})
    assert good["verdict"] == "PASS"
    unfaithful = judge_text(FakeLLM(lambda s, u: {"scores": {**{k: 9 for k in TEXT_RUBRIC}, "faithfulness": 7}}), "briefing", {}, {})
    assert unfaithful["verdict"] == "REVIEW"
    assert judge_text(None, "briefing", {}, {})["verdict"] == "NOT_RUN"


def test_pairwise_cancels_position_bias():
    first_always = FakeLLM(lambda s, u: {"winner": "plan_1", "reason": "first"})
    r = pairwise_preference(first_always, {}, {"plan": "a"}, {"plan": "b"})
    assert r["result"] == "tie" and not r["position_consistent"]   # a judge that always picks the first slot gains nothing

    def prefers_a(system, user):
        first = user.split("<plan_1>")[1]
        return {"winner": "plan_1" if '"a"' in first.split("</plan_1>")[0] else "plan_2", "reason": "a"}
    r = pairwise_preference(FakeLLM(prefers_a), {}, {"plan": "a"}, {"plan": "b"})
    assert r["result"] == "A" and r["position_consistent"]
