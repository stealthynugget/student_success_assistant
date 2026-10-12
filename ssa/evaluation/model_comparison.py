"""LLM selection: the same features run on each Claude tier, scored for quality, latency and cost.

* **Search understanding** (production tier: fast). Filter precision/recall/F1 and exact match on the labelled templated
  queries and the paraphrase set, the share of answers that pass the code checks, latency and cost per call.
* **Precedent synthesis** (production tier: main). Share of drafts that pass the code checks (cited IDs, numeric
  grounding, protected/punitive terms), the judge's scores against the same evidence, latency and cost per call. The judge
  is the judge-tier model; where the generator is that same model the row is marked ``self_judged`` (expect leniency).

Effort per feature is the production setting for every model, so only the model changes. Results:
``artifacts/metrics/model_comparison.json``; ``LLM_BUDGET_USD`` caps the run.
"""
from __future__ import annotations

import numpy as np

from ssa.config import Settings, get_settings
from ssa.utils import get_logger, write_json

log = get_logger("ssa.model_comparison")


def _per_call(client, feature: str) -> dict:
    """Cost and size per live call. When every call of this run was answered from the response cache, the figures come from
    the usage ledger's earlier live calls of the same feature and model (``from_ledger``)."""
    u = client.usage_summary()
    live = u["calls"] - u["cached"]
    out = {"calls": u["calls"], "cached": u["cached"], "cost_usd": round(u["estimated_cost_usd"], 4)}
    if live:
        lat = [r.latency_s for r in client.calls if not r.cached and r.error is None]
        return {**out, "cost_per_call_usd": round(u["estimated_cost_usd"] / live, 5),
                "output_tokens_per_call": round(u["completion_tokens"] / live), "median_latency_s": round(float(np.median(lat)), 2),
                "from_ledger": False}
    import sqlite3
    with sqlite3.connect(client.settings.app_db) as c:
        rows = c.execute("SELECT latency_s, cost_usd, completion_tokens FROM llm_usage WHERE feature = ? AND model = ? AND cached = 0 "
                         "AND error IS NULL AND cost_usd IS NOT NULL", (feature, client.model)).fetchall()
    if not rows:
        return {**out, "cost_per_call_usd": None, "output_tokens_per_call": None, "median_latency_s": None, "from_ledger": True}
    lat, cost, toks = (np.array(col, dtype=float) for col in zip(*rows, strict=True))
    return {**out, "cost_per_call_usd": round(float(cost.mean()), 5), "output_tokens_per_call": round(float(toks.mean())),
            "median_latency_s": round(float(np.median(lat)), 2), "from_ledger": True, "ledger_calls": len(rows)}


def compare_search(models: list[str], s: Settings, n: int = 30) -> dict:
    from ssa.agents.genai import understand_query
    from ssa.agents.llm import LLMClient
    from ssa.evaluation.offline_eval import _filter_scores
    from ssa.evaluation.search_queries import PARAPHRASES, labelled_queries
    data = [*labelled_queries(n), *PARAPHRASES]
    out = {}
    for m in models:
        client = LLMClient(s, model=m)
        tp = fp = fn = exact = accepted = 0
        for d in data:
            ans = understand_query(client, d["query"])
            a, b, c = _filter_scores(ans["filters"], d["expected"])
            tp, fp, fn = tp + a, fp + b, fn + c
            exact += int(b == 0 and c == 0)
            accepted += int(ans.get("source") == "llm")
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[m] = {"n": len(data), "filter_precision": round(p, 4), "filter_recall": round(r, 4),
                  "filter_f1": round(2 * p * r / max(p + r, 1e-9), 4), "exact_match": round(exact / len(data), 4),
                  "accepted_by_code_checks": round(accepted / len(data), 4), **_per_call(client, "search_understanding")}
        log.info(f"search_understanding {m}: {out[m]}")
    return out


def compare_synthesis(models: list[str], s: Settings, n_cases: int = 10) -> dict:
    from ssa.agents.genai import precedent_evidence, synthesize_precedents
    from ssa.agents.llm import LLMClient
    from ssa.evaluation.judge import LLMJudge, judge_text
    from ssa.evaluation.offline_eval import golden_set
    from ssa.service import StudentSuccessService
    svc = StudentSuccessService(s).warm()
    refs = golden_set(svc, per_tier=4)[:n_cases]
    inputs = []
    for ref in refs:
        rp = svc.resolve(ref=ref)
        res = svc.analyze(rp, use_llm=False, synthesize=False)
        inputs.append((rp, res))
    judge = LLMJudge(s).llm
    out = {}
    for m in models:
        client = LLMClient(s, model=m)
        rows = []
        for rp, res in inputs:
            ps = synthesize_precedents(client, rp.profile, res, res["similar_cases"])
            ev = precedent_evidence(res, res["similar_cases"])
            j = judge_text(judge, "precedent_synthesis", ps, ev) if ps["source"] == "llm" else None
            rows.append({"accepted": ps["source"] == "llm", "checks": ps["checks"], "judge": j})
        js = [r["judge"] for r in rows if r["judge"] and r["judge"].get("scores")]
        out[m] = {"n": len(rows), "accepted_by_code_checks": round(float(np.mean([r["accepted"] for r in rows])), 4),
                  "rejected_checks": [c for r in rows for c in r["checks"]][:6],
                  "judge_pass_rate": round(float(np.mean([j["verdict"] == "PASS" for j in js])), 4) if js else None,
                  "judge_mean_scores": {k: round(float(np.mean([j["scores"][k] for j in js])), 2) for k in js[0]["scores"]} if js else None,
                  "self_judged": m == s.judge_model, **_per_call(client, "precedent_synthesis")}
        log.info(f"precedent_synthesis {m}: {out[m]}")
    return out


def run(settings: Settings | None = None, models: list[str] | None = None) -> dict:
    s = settings or get_settings()
    models = models or list(dict.fromkeys([s.fast_model, s.llm_model, s.judge_model]))
    result = {"models": models, "production_choice": {"search_understanding": s.model_for("search_understanding"),
                                                      "precedent_synthesis": s.model_for("precedent_synthesis"),
                                                      "judge": s.judge_model},
              "prices_usd_per_million_tokens": {m: s.llm_prices.get(m) for m in models},
              "search_understanding": compare_search(models, s), "precedent_synthesis": compare_synthesis(models, s)}
    write_json(s.global_metrics_dir / "model_comparison.json", result)
    return result
