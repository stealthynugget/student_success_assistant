"""Command-line entry point: ``python -m ssa.cli <command>``.

prepare | train | index | evaluate | llm-eval | feedback-loop | quality-gate | build | demo-data | serve
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

from ssa.config import PROJECT_ROOT, get_settings
from ssa.utils import get_logger, read_json, write_json

log = get_logger("ssa.cli")
warnings.filterwarnings("ignore", category=UserWarning)


def _checkpoints(args):
    s = get_settings()
    if getattr(args, "checkpoint", None):
        return [s.at(float(args.checkpoint))]
    return [s.at(c) for c in s.checkpoints]


def cmd_prepare(args):
    from ssa.data.pipeline import build_features
    # primary last is not required: course-level artifacts are written when the primary checkpoint is built
    for sc in _checkpoints(args):
        meta = build_features(settings=sc).metadata
        log.info(f"[{sc.tag}] {meta['records']:,} rows, {meta['in_scope_records']:,} in scope")


def cmd_train(args):
    from ssa.data.pipeline import load_features
    from ssa.models.outcome import OutcomeModel
    from ssa.models.prioritization import Prioritizer
    from ssa.models.risk import WithdrawalRiskModel
    summary = []
    for sc in _checkpoints(args):
        df = load_features(sc)
        risk = WithdrawalRiskModel(sc)
        m = risk.fit(df)
        log.info(f"[{sc.tag}] withdrawal: ROC-AUC {m['test']['roc_auc']:.3f} PR-AUC {m['test']['pr_auc']:.3f} ECE {m['test']['ece_10']:.3f}")
        outcome = OutcomeModel(sc)
        o = outcome.fit(df)
        log.info(f"[{sc.tag}] outcome: macro-F1 {o['test']['macro_f1']:.3f}")
        p = Prioritizer(sc).fit(df, risk, outcome)
        summary.append({"checkpoint": sc.cutoff_fraction, "tag": sc.tag, "approx_week": round(sc.cutoff_fraction * 262 / 7, 1),
                        "in_scope": int((df["in_scope"] == 1).sum()), "withdrawal": m["test"], "thresholds": m["thresholds"],
                        "outcome_macro_f1": o["test"]["macro_f1"],
                        "p_poor_auc": o["poor_outcome_from_p_fail_plus_p_withdrawn_test"]["roc_auc"],
                        "prioritization_precision_at_10": p["test_comparison"]["learned_weights"]["at_capacity"][0]["precision"]})
    if not getattr(args, "checkpoint", None):
        write_json(get_settings().global_metrics_dir / "checkpoint_summary.json", {"checkpoints": summary})


def cmd_index(args):
    from ssa.data.pipeline import load_features
    from ssa.models.risk import WithdrawalRiskModel
    from ssa.retrieval.benchmark import tune_fusion_weights
    from ssa.retrieval.hybrid import HistoricalCaseIndex
    for sc in _checkpoints(args):
        df = load_features(sc)
        risk = WithdrawalRiskModel(sc).load()
        idx = HistoricalCaseIndex(sc)
        man = idx.build(df, importance=risk.art.importance, with_dense=not args.no_dense)
        tuned = tune_fusion_weights(df, idx, settings=sc, n_queries=200 if not sc.is_primary else 300)
        log.info(f"[{sc.tag}] index {man['n_docs']:,} docs, dense={man['dense_channel']}, fusion={tuned.get('weights')}")


def cmd_evaluate(args):
    from ssa.data.pipeline import load_features
    from ssa.evaluation.model_analysis import ablations, factor_analysis, fairness_audit, trend_analysis
    from ssa.models.risk import WithdrawalRiskModel
    from ssa.retrieval.benchmark import run_benchmark
    from ssa.retrieval.hybrid import HistoricalCaseIndex
    s = get_settings()
    df = load_features(s)
    risk = WithdrawalRiskModel(s).load()
    ablations(df, s)
    fairness_audit(df, risk, s)
    factor_analysis(df, risk, s)
    trend_analysis(s)
    b = run_benchmark(df, HistoricalCaseIndex(s).load(), n_queries=args.queries, settings=s)
    log.info(f"retrieval: hybrid vote-AUC(poor) {b['modes']['hybrid']['neighbour_vote_auc_poor_outcome']:.3f} vs random "
             f"{b['modes']['random']['neighbour_vote_auc_poor_outcome']:.3f}")


def cmd_llm_eval(args):
    from ssa.evaluation.offline_eval import run
    res = run(per_tier=args.per_tier, use_llm=not args.no_llm)
    print(json.dumps({k: res[k] for k in ("agents", "search", "copilot", "synthesis", "briefing", "outreach")}, indent=2, default=str)[:6000])


def cmd_feedback_loop(args):
    from ssa.feedback.loop import run_loop
    plan = run_loop(apply=args.apply, approved_by=args.approved_by)
    print(json.dumps({"feedback_rows": plan["feedback_rows"], "applied": plan["applied"],
                      "proposals": [{"type": p["type"], "status": p["status"]} for p in plan["proposals"]],
                      "monitors": plan["monitors"]}, indent=2))


SECRET_PATTERNS = [re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), re.compile(r"(?i)api[_-]?key\s*=\s*['\"]?[A-Za-z0-9_\-]{24,}")]
SCAN_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".cfg", ".ini", ".example", ".sh"}


def secret_scan(root: Path) -> list[str]:
    hits = []
    for p in root.rglob("*"):
        if any(part in {".git", ".venv", "node_modules", "artifacts", "data"} for part in p.parts):
            continue
        if p.is_file() and (p.suffix in SCAN_SUFFIXES or p.name.startswith(".env")) and p.name != ".env":
            text = p.read_text(errors="ignore")
            if any(rx.search(text) for rx in SECRET_PATTERNS):
                hits.append(str(p.relative_to(root)))
    return hits


def quality_gate(run_tests: bool = False) -> dict:
    s = get_settings()
    M = lambda n: read_json(s.metrics_dir / f"{n}.json", {})  # noqa: E731
    data, wd, oc, pr, rb, fa = (M("data_build"), M("withdrawal_model"), M("outcome_model"), M("prioritization"),
                                M("retrieval_benchmark"), M("fairness"))
    profile = "smoke" if data.get("synthetic_fixture") else "official"
    th = {"official": {"auc": 0.65, "ece": 0.05, "retr": 0.10, "prio": 0.20},
          "smoke": {"auc": 0.55, "ece": 0.15, "retr": 0.0, "prio": 0.0}}[profile]
    checks, warnings_ = [], []

    def check(name, ok, detail=None, warn=False):
        (warnings_ if (warn and not ok) else checks).append({"check": name, "passed": bool(ok), "detail": detail})

    check("artifacts_present", all([data, wd, oc, pr, rb]), {"data": bool(data), "withdrawal": bool(wd), "outcome": bool(oc),
                                                              "prioritization": bool(pr), "retrieval": bool(rb)})
    if data:
        check("leakage_scope_applied", data.get("in_scope_records", 0) < data.get("records", 0) or profile == "smoke",
              {"records": data.get("records"), "in_scope": data.get("in_scope_records")})
    if wd:
        t = wd["test"]
        check("withdrawal_auc", (t.get("roc_auc") or 0) >= th["auc"], t.get("roc_auc"))
        check("withdrawal_calibration_ece", (t.get("ece_10") or 1) <= th["ece"], t.get("ece_10"))
        base = wd["baseline_logistic_regression_test"].get("roc_auc") or 0
        check("withdrawal_beats_logistic_baseline", (t.get("roc_auc") or 0) >= base - 0.005, {"gbm": t.get("roc_auc"), "lr": base},
              warn=profile == "smoke")
        tiers = wd["test_tiers"]
        rates = [tiers[k]["observed_withdrawal_rate"] for k in ("High", "Medium", "Low")]
        check("risk_tiers_monotonic", None not in rates and rates[0] > rates[1] > rates[2], rates, warn=profile == "smoke")
        check("thresholds_separated", wd["thresholds"]["medium"] < wd["thresholds"]["high"], wd["thresholds"])
        check("population_audit_reported", wd.get("population_audit", {}).get("available", False) or profile == "smoke")
    if oc:
        t, b = oc["test"], oc["baselines_test"]
        check("outcome_beats_majority", t["macro_f1"] > b["majority_class"]["macro_f1"], {"model": t["macro_f1"], "majority": b["majority_class"]["macro_f1"]})
        check("outcome_vs_logistic", t["macro_f1"] >= b["multinomial_logistic"]["macro_f1"] - 0.01,
              {"model": t["macro_f1"], "lr": b["multinomial_logistic"]["macro_f1"]}, warn=True)
    if pr:
        tc = pr["test_comparison"]
        learned, v4, rnd = (tc["learned_weights"]["objective_mean_precision"], tc["v4_hand_formula"]["objective_mean_precision"],
                            tc["random"]["objective_mean_precision"])
        check("prioritization_beats_random", learned - rnd >= th["prio"], {"learned": learned, "random": rnd})
        check("prioritization_vs_v4_formula", learned >= v4 - 0.01, {"learned": learned, "v4": v4}, warn=profile == "smoke")
    if rb:
        h, r = rb["hybrid_minus_random"], rb["modes"]
        check("retrieval_beats_random_vote_auc", (h.get("neighbour_vote_auc_poor_outcome") or 0) >= th["retr"], h)
        check("retrieval_beats_random_ndcg", (h.get("reference_ndcg_at_10_global_ideal") or 0) >= th["retr"], h)
        man = read_json(s.index_dir / "manifest.json", {})
        check("index_complete", man.get("n_docs") == data.get("in_scope_records"), {"index": man.get("n_docs"), "in_scope": data.get("in_scope_records")})
    if fa:
        check("sensitive_attributes_excluded_from_model", fa.get("model_uses_sensitive_attributes") == s.include_sensitive_features)
        for attr, rep in fa.get("attributes", {}).items():
            gap = rep.get("equal_opportunity_gap")
            check(f"fairness_equal_opportunity_{attr}", gap is None or gap <= 0.15, gap, warn=True)
    llm = read_json(s.global_metrics_dir / "llm_evaluation.json", {})
    if llm and llm.get("agents"):
        det = llm["agents"]["deterministic"]
        check("agent_plans_always_valid", det["final_plan_valid_rate"] == 1.0, det["final_plan_valid_rate"])
    cps = read_json(s.global_metrics_dir / "checkpoint_summary.json", {}).get("checkpoints", [])
    if cps:
        check("all_checkpoints_trained", len(cps) == len(s.checkpoints), [c["tag"] for c in cps])
    hits = secret_scan(PROJECT_ROOT)
    check("no_secrets_in_source", not hits, hits)
    if run_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        check("tests", r.returncode == 0, r.stdout[-1500:])
    result = {"profile": profile, "passed": all(c["passed"] for c in checks), "checks": checks, "warnings": warnings_,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "official_evidence": profile == "official",
              "note": "Metrics from a synthetic fixture must never be reported as OULAD performance."}
    write_json(s.global_metrics_dir / "quality_gate.json", result)
    return result


def cmd_quality_gate(args):
    res = quality_gate(run_tests=args.with_tests)
    for c in res["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    for w in res["warnings"]:
        print("WARN " + w["check"] + f"  {w['detail']}")
    print(f"\nprofile={res['profile']} passed={res['passed']}")
    raise SystemExit(0 if res["passed"] else 1)


def cmd_build(args):
    t0 = time.time()
    s = get_settings()
    s.ensure_dirs()
    for name, fn in [("prepare", cmd_prepare), ("train", cmd_train), ("index", cmd_index), ("evaluate", cmd_evaluate)]:
        log.info(f"== {name} ==")
        fn(args)
    if args.llm_eval:
        cmd_llm_eval(args)
    res = quality_gate(run_tests=False)
    log.info(f"build finished in {time.time() - t0:.0f}s; quality gate passed={res['passed']} ({res['profile']})")
    if not res["passed"]:
        raise SystemExit(1)


def cmd_demo_data(args):
    from ssa.data.synthetic import write_synthetic_oulad
    s = get_settings()
    out = Path(args.out) if args.out else s.raw_dir
    write_synthetic_oulad(out, n_per_presentation=args.n, seed=args.seed)
    print(f"synthetic OULAD-shaped fixture written to {out} (marked .synthetic_fixture; never report its metrics)")


def cmd_report(args):
    from ssa.reporting import write_reports
    print(json.dumps(write_reports(), indent=2))


def cmd_serve(args):  # pragma: no cover
    from ssa.api.app import create_app
    create_app().run(host=args.host, port=args.port, debug=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ssa")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("prepare"); sp.add_argument("--checkpoint", type=float); sp.set_defaults(fn=cmd_prepare)  # noqa: E702
    sp = sub.add_parser("train"); sp.add_argument("--checkpoint", type=float); sp.set_defaults(fn=cmd_train)  # noqa: E702
    sp = sub.add_parser("index"); sp.add_argument("--no-dense", action="store_true"); sp.add_argument("--checkpoint", type=float)
    sp.set_defaults(fn=cmd_index)
    sp = sub.add_parser("evaluate"); sp.add_argument("--queries", type=int, default=600); sp.set_defaults(fn=cmd_evaluate)  # noqa: E702
    sp = sub.add_parser("llm-eval"); sp.add_argument("--per-tier", type=int, default=8); sp.add_argument("--no-llm", action="store_true")
    sp.set_defaults(fn=cmd_llm_eval)
    sp = sub.add_parser("feedback-loop"); sp.add_argument("--apply", action="store_true"); sp.add_argument("--approved-by")
    sp.set_defaults(fn=cmd_feedback_loop)
    sp = sub.add_parser("quality-gate"); sp.add_argument("--with-tests", action="store_true"); sp.set_defaults(fn=cmd_quality_gate)  # noqa: E702
    sp = sub.add_parser("build")
    sp.add_argument("--checkpoint", type=float, default=None); sp.add_argument("--no-dense", action="store_true")
    sp.add_argument("--queries", type=int, default=600); sp.add_argument("--llm-eval", action="store_true")
    sp.add_argument("--per-tier", type=int, default=8); sp.add_argument("--no-llm", action="store_true")
    sp.set_defaults(fn=cmd_build)
    sp = sub.add_parser("demo-data"); sp.add_argument("--out"); sp.add_argument("--n", type=int, default=150)
    sp.add_argument("--seed", type=int, default=7); sp.set_defaults(fn=cmd_demo_data)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sp = sub.add_parser("serve"); sp.add_argument("--host", default="127.0.0.1"); sp.add_argument("--port", type=int, default=5000)
    sp.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
