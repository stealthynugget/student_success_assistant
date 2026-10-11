import pytest

from ssa.feedback.loop import action_learner, calibration_learner, run_loop
from ssa.feedback.store import FeedbackStore, UnknownRecommendation


def _analysis(service, features):
    row = features[(features["in_scope"] == 1) & (features["code_presentation"] == "2014J")].iloc[0]
    rp = service.resolve(ref={"id_student": int(row["id_student"]), "code_module": row["code_module"],
                              "code_presentation": row["code_presentation"]})
    return service.analyze(rp, use_llm=False)


def test_feedback_requires_registered_recommendation(service, features, tmp_path, built):
    store = FeedbackStore(built, path=tmp_path / "fb.sqlite3")
    a = _analysis(service, features)
    rid = store.register(a)
    sid = a["student"]["id_student"]
    action = a["workflow"]["intervention_plan"]["actions"][0]["id"]
    case = str(a["similar_cases"]["cases"][0]["case_id"])
    assert store.add_feedback({"recommendation_id": rid, "id_student": sid, "rating": 4, "plan_accepted": True,
                               "action_feedback": {action: {"accepted": True}}, "case_feedback": {case: True}}) > 0
    with pytest.raises(UnknownRecommendation):
        store.add_feedback({"recommendation_id": "f" * 20, "id_student": sid, "rating": 3, "plan_accepted": False})
    with pytest.raises(ValueError):
        store.add_feedback({"recommendation_id": rid, "id_student": sid + 1, "rating": 3, "plan_accepted": False})
    with pytest.raises(ValueError):
        store.add_feedback({"recommendation_id": rid, "id_student": sid, "rating": 3, "plan_accepted": False,
                            "action_feedback": {"ACT_NOT_SHOWN": {"accepted": True}}})
    assert store.summary()["count"] == 1


def test_action_learner_demotes_consistently_rejected_actions():
    rows = [{"action_feedback": {"ACT_X": {"accepted": False}, "ACT_Y": {"accepted": True}}} for _ in range(15)]
    prop = action_learner(rows)
    assert prop["payload"]["actions"]["ACT_X"]["demoted"] is True
    assert prop["payload"]["actions"]["ACT_Y"]["demoted"] is False


def test_calibration_learner_needs_enough_outcomes():
    assert calibration_learner([{"risk_score": 0.2, "observed_outcome": "Pass"}] * 10)["status"] == "insufficient_data"
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):
        p = rng.uniform(0.05, 0.6)
        rows.append({"risk_score": p, "observed_outcome": "Withdrawn" if rng.random() < min(1, p * 1.5) else "Pass"})
    assert calibration_learner(rows)["status"] in {"proposed", "no_change"}


def test_loop_applies_approved_action_stats(built, service, features):
    store = FeedbackStore(built)
    a = _analysis(service, features)
    rid = store.register(a)
    action = a["workflow"]["intervention_plan"]["actions"][0]["id"]
    for _ in range(12):
        store.add_feedback({"recommendation_id": rid, "id_student": a["student"]["id_student"], "rating": 2,
                            "plan_accepted": False, "action_feedback": {action: {"accepted": False}}})
    plan = run_loop(built, apply=True, approved_by="test")
    assert "action_catalog_ranking" in plan["applied"]
    from ssa.agents.evidence import action_stats
    assert action_stats(built)[action]["demoted"] is True
