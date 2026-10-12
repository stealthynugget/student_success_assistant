"""The LangGraph engine must behave exactly like the reference orchestrator on every path."""
from dataclasses import replace

import pytest

from ssa.agents.llm import FakeLLM
from ssa.agents.workflow import SupportWorkflow
from tests.test_workflow import _ctx, bad_llm, good_llm


def _fix_then_good():
    """First plan invalid, the revision valid: exercises intervention -> critic -> revision -> critic -> end."""
    calls = {"plans": 0}

    def llm(system, user):
        if "action_ids" in user:
            calls["plans"] += 1
            if calls["plans"] == 1:
                return {"action_ids": ["ACT_DOES_NOT_EXIST"], "outreach_message": "short", "rationale": "too short"}
        return good_llm(system, user)
    return llm


def _comparable(out: dict) -> dict:
    out = {k: v for k, v in out.items() if k not in ("recommendation_id", "seconds", "llm_usage", "orchestration")}
    out["trace"] = [{k: v for k, v in t.items() if k != "seconds"} for t in out["trace"]]
    return out


def _run(engine: str, llm_fn, ctx, settings):
    wf = SupportWorkflow(FakeLLM(llm_fn) if llm_fn else None, replace(settings, workflow_engine=engine))
    return wf.run(**ctx)


@pytest.mark.parametrize("scenario", ["no_llm_high", "no_llm_low", "good", "bad", "revised"])
def test_langgraph_matches_reference(service, features, built, scenario):
    ctx = _ctx(service, features, high=scenario != "no_llm_low")
    llm = {"no_llm_high": None, "no_llm_low": None, "good": good_llm, "bad": bad_llm, "revised": _fix_then_good()}[scenario]
    a = _run("langgraph", llm, ctx, built)
    if scenario == "revised":
        llm = _fix_then_good()  # fresh counter for the reference run
    b = _run("python", llm, ctx, built)
    assert a["orchestration"] == "langgraph" and b["orchestration"] == "python"
    assert _comparable(a) == _comparable(b)
    if scenario == "revised":
        assert a["intervention_plan"]["source"] == "llm_revised"
        assert [t["agent"] for t in a["trace"] if t["agent"] in ("intervention", "critic", "intervention_revision")] == \
            ["intervention", "critic", "intervention_revision", "critic"]


def test_graph_shape():
    from ssa.agents.graph import mermaid
    m = mermaid()
    for node in ("performance", "risk", "router", "intervention", "critic", "revision", "deterministic_planner"):
        assert node in m
