"""The support workflow as a LangGraph state graph.

Same agents, same guardrails, same output as the reference orchestrator (``SupportWorkflow._run_python``); LangGraph
makes the control flow explicit and inspectable::

    performance -> risk -> router --monitor / no LLM--> deterministic_planner -> finalize
                                  \\--intervene-------> intervention -> critic --ok--> finalize
                                                            ^             |  \\--violations after revision--> deterministic_planner
                                                            |  revise 1x  |
                                                            +-- revision <+

Nodes call the workflow's own agent methods, so the LLM client (cache, ledger, budget), validation and fallbacks are
unchanged; the per-step callback still drives the console's live agent steps. ``tests/test_graph.py`` checks that both
engines give identical results on every path.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict, total=False):
    st: dict[str, Any]            # shared evidence and outputs (same dict the agents use)
    mode: str                     # router decision: intervene | monitor
    plan: dict | None             # current candidate plan
    violations: list[str]         # critic findings for the current plan
    revised: bool                 # one revision already requested


def build_graph(wf, step):
    """Compile the graph for one run. ``wf`` is the SupportWorkflow; ``step(name, fn)`` records trace + callback."""

    def performance(s: WorkflowState) -> dict:
        s["st"]["performance"] = step("performance", lambda: wf.performance_agent(s["st"]))
        return {}

    def risk(s: WorkflowState) -> dict:
        s["st"]["risk_analysis"] = step("risk", lambda: wf.risk_agent(s["st"]))
        return {}

    def router(s: WorkflowState) -> dict:
        mode = wf.route(s["st"])
        s["st"]["trace"].append({"agent": "router", "decision": mode})
        return {"mode": mode, "plan": None, "violations": [], "revised": False}

    def intervention(s: WorkflowState) -> dict:
        plan = step("intervention", lambda: wf.intervention_agent(s["st"]) or {"source": "unavailable"})
        return {"plan": None if plan.get("source") == "unavailable" else plan}

    def critic(s: WorkflowState) -> dict:
        if s["revised"]:
            plan = s["plan"]
            v = wf._validate_plan(plan, s["st"]) if plan.get("source") != "unavailable" else ["revision unavailable"]
            s["st"]["trace"].append({"agent": "critic", "violations_after_revision": v})
            return {"violations": v, "plan": plan if not v else None}
        v = wf._validate_plan(s["plan"], s["st"])
        s["st"]["trace"].append({"agent": "critic", "violations": v})
        return {"violations": v}

    def revision(s: WorkflowState) -> dict:
        plan = step("intervention_revision", lambda: wf.intervention_agent(s["st"], s["violations"]) or {"source": "unavailable"})
        return {"plan": plan, "revised": True}

    def deterministic_planner(s: WorkflowState) -> dict:
        plan = wf._deterministic_plan(s["st"], s["mode"])
        s["st"]["trace"].append({"agent": "deterministic_planner", "violations": wf._validate_plan(plan, s["st"])})
        return {"plan": plan}

    def after_router(s: WorkflowState) -> str:
        return "intervention" if (s["mode"] == "intervene" and wf.llm is not None) else "deterministic_planner"

    def after_intervention(s: WorkflowState) -> str:
        return "critic" if s["plan"] is not None else "deterministic_planner"

    def after_critic(s: WorkflowState) -> str:
        if not s["violations"]:
            return END
        return "deterministic_planner" if s["revised"] else "revision"

    g = StateGraph(WorkflowState)
    for name, fn in (("performance", performance), ("risk", risk), ("router", router), ("intervention", intervention),
                     ("critic", critic), ("revision", revision), ("deterministic_planner", deterministic_planner)):
        g.add_node(name, fn)
    g.add_edge(START, "performance")
    g.add_edge("performance", "risk")
    g.add_edge("risk", "router")
    g.add_conditional_edges("router", after_router, ["intervention", "deterministic_planner"])
    g.add_conditional_edges("intervention", after_intervention, ["critic", "deterministic_planner"])
    g.add_conditional_edges("critic", after_critic, ["revision", "deterministic_planner", END])
    g.add_edge("revision", "critic")
    g.add_edge("deterministic_planner", END)
    return g.compile()


def run_graph(wf, st: dict, step) -> tuple[str, dict]:
    """Run the graph; returns (mode, final plan)."""
    out = build_graph(wf, step).invoke({"st": st})
    return out["mode"], out["plan"]


def mermaid() -> str:
    """Mermaid source of the graph, for the docs and slides."""
    class _Stub:
        llm = object()
    return build_graph(_Stub(), lambda name, fn: fn()).get_graph().draw_mermaid()
