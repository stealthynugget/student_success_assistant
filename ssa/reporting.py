"""Generate the model card, results summary and the 10-minute panel deck *from the metric
artifacts*. Nothing numeric is hard-coded: re-run ``python -m ssa.cli report`` after every build
(v4's deck and README advertised a leaked 0.90 ROC-AUC)."""
from __future__ import annotations

import time
from pathlib import Path

from ssa.config import PROJECT_ROOT, Settings, get_settings
from ssa.utils import read_json

NAVY, TEAL, AMBER, RED, SLATE, PAPER = "1F2A44", "0F766E", "B45309", "B91C1C", "64748B", "FBFBF9"


def load_all(s: Settings) -> dict:
    s = s.at(None)
    names = ["data_build", "withdrawal_model", "outcome_model", "ablations", "fairness", "prioritization", "retrieval_benchmark",
             "text_search_benchmark"]
    m = {n: read_json(s.metrics_dir / f"{n}.json", {}) for n in names}
    for n in ("llm_evaluation", "quality_gate", "checkpoint_summary"):
        m[n] = read_json(s.global_metrics_dir / f"{n}.json", {})
    m["factors"] = read_json(s.analysis_dir / "factor_analysis.json", {})
    m["trends"] = read_json(s.analysis_dir / "trend_analysis.json", {})
    return m


def _f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def _pct(x, d=0):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _llm_lines(e: dict) -> str:
    if not e or not e.get("agents"):
        return "Not run yet: `python -m ssa.cli llm-eval`."
    ag, se, co, rb = e["agents"], e["search"], e["copilot"], e["robustness"]
    rows = [f"Generator: {e.get('generator_model') or 'not connected'}; judge: {e.get('judge_model') or 'not connected'}.", "",
            "| Check | Result |", "|---|---|",
            f"| Support plans passing every guardrail | {_pct(ag['deterministic']['final_plan_valid_rate'])} |",
            f"| AI plan drafts accepted / revised | {_pct(ag['deterministic']['llm_plan_accepted_rate'])} / {_pct(ag['deterministic']['revision_rate'])} |",
            f"| Judge pass rate | {_pct(ag['judge'].get('pass_rate')) if ag['judge'].get('pass_rate') is not None else 'not run'} |",
            f"| AI plan preferred in order-swapped A/B | {_pct(ag['pairwise_llm_vs_deterministic'].get('llm_win_rate')) if ag['pairwise_llm_vs_deterministic'].get('llm_win_rate') is not None else 'not run'} |",
            f"| Search understanding F1, templated (rules / AI) | {_f(se.get('templated_rules', {}).get('filter_f1'))} / {_f(se.get('templated_llm', {}).get('filter_f1'))} |",
            f"| Search understanding F1, paraphrased (rules / AI) | {_f(se.get('paraphrased_rules', {}).get('filter_f1'))} / {_f(se.get('paraphrased_llm', {}).get('filter_f1'))} |",
            f"| Copilot correct answers (rules / AI) | {_pct(co['rules'].get('answer_correct_rate'))} / {_pct(co.get('llm', {}).get('answer_correct_rate'))} |",
            f"| Adversarial copilot prompts handled safely ({rb['mode']}) | {_pct(rb['safe_rate'])} |"]
    return "\n".join(rows)


def _typed_search_lines(t: dict) -> list[str]:
    if not t.get("parsers"):
        return []
    tw = t.get("hybrid_text_weights", {})
    lines = ["", "### Typed search", f"{t['queries']} labelled plain-English queries (templated, paraphrased and behaviour-led); a returned "
             "case is relevant when it meets every expected filter and shows every expected behaviour. Hybrid typed search uses "
             f"its own tuned weights (lexical {tw.get('lexical')}, dense {tw.get('dense')}).", "",
             "| Query parser | Mode | Precision@10 | Filter match@10 | Behaviour match@10 |", "|---|---|---|---|---|"]
    for parser, res in t["parsers"].items():
        for mode, v in res["modes"].items():
            lines.append(f"| {parser} | {mode} | {_f(v.get('precision_at_10'))} | {_f(v.get('filter_match_at_10'))} | "
                         f"{_f(v.get('behaviour_match_at_10'))} |")
    return lines


def model_card(s: Settings | None = None) -> str:
    s = s or get_settings()
    m = load_all(s)
    d, w, o, a, fa, p, r, q = (m[k] for k in ("data_build", "withdrawal_model", "outcome_model", "ablations", "fairness",
                                               "prioritization", "retrieval_benchmark", "quality_gate"))
    if not w:
        return "# Model card\n\nNo metrics yet. Run `python -m ssa.cli build`.\n"
    t = w["test"]
    pa = w.get("population_audit", {})
    tiers = w["test_tiers"]
    cap = {c["fraction"]: c for c in w["test_capacity"]}
    lines = [
        "# Model card — Student Success Assistant v5", "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')} from `artifacts/metrics/*.json` by `python -m ssa.cli report`. "
        f"Engine: **{w.get('engine')}**. Evidence profile: **{q.get('profile', 'unknown')}** "
        f"(quality gate {'passed' if q.get('passed') else 'NOT passed'})._", "",
        "## Intended use",
        "Early-warning support for academic advisors at ~40% of a course presentation: surface students still enrolled who may "
        "withdraw or fail, explain why, show comparable historical cases and draft supportive outreach. A human advisor reviews "
        "every recommendation. Not for admissions, grading, sanctions or any automated adverse decision.", "",
        "## Data and population",
        f"- OULAD, {d.get('records', 'n/a'):,} enrolments; snapshot at {int(100 * d.get('cutoff_fraction', 0.4))}% of each presentation.",
        f"- **In scope:** {d.get('in_scope_records', 0):,} enrolments still registered at the snapshot. "
        f"{d.get('withdrawn_before_snapshot', 0):,} students who had already unregistered "
        f"({d.get('withdrawn_before_start', 0):,} before day 0) are excluded — they cannot be helped by an early warning and "
        "including them inflates every metric.",
        f"- Temporal protocol: {w['split'].get('train')} train, {w['split'].get('validation')} validation, "
        f"{w['split'].get('test')} untouched test.", "",
        "## Withdrawal model (target: withdrawal after the snapshot)",
        "| Metric (test presentation, in scope) | Value |", "|---|---|",
        f"| ROC-AUC | {_f(t['roc_auc'])} (95% CI {_f(t['roc_auc_95ci']['lower'])}–{_f(t['roc_auc_95ci']['upper'])}) |",
        f"| PR-AUC | {_f(t['pr_auc'])} (base rate {_f(t['positive_rate'])}) |",
        f"| Brier / ECE | {_f(t['brier'])} / {_f(t['ece_10'])} |",
        f"| Logistic-regression baseline ROC-AUC | {_f(w['baseline_logistic_regression_test']['roc_auc'])} |",
        f"| Precision / recall when contacting top 10% | {_pct(cap[0.1]['precision'])} / {_pct(cap[0.1]['recall'])} (lift {_f(cap[0.1]['lift'], 2)}) |",
        f"| Precision / recall when contacting top 20% | {_pct(cap[0.2]['precision'])} / {_pct(cap[0.2]['recall'])} |", "",
        f"Risk tiers (thresholds learned out-of-fold: Medium ≥ {_f(w['thresholds']['medium'])}, High ≥ {_f(w['thresholds']['high'])}):", "",
        "| Tier | Share of students | Observed withdrawal rate | Share of all withdrawals |", "|---|---|---|---|",
    ] + [f"| {k} | {_pct(v['share'])} | {_pct(v['observed_withdrawal_rate'], 1)} | {_pct(v['share_of_all_withdrawals'])} |"
         for k, v in tiers.items()] + [
        "",
        f"**Leakage audit.** Scoring the same model on *all* test enrolments, including students who had already left, gives "
        f"ROC-AUC {_f(pa.get('all_enrolments_including_already_withdrawn', {}).get('roc_auc'))} vs {_f(pa.get('in_scope_only', {}).get('roc_auc'))} "
        f"in scope; {_pct(pa.get('already_withdrawn_share_of_test_withdrawals'))} of the test presentation's withdrawals had already "
        "happened by the snapshot. v4 reported 0.90 by training and testing on that population.", "",
        "## Outcome model (Distinction / Pass / Fail / Withdrawn)",
        f"- Test macro-F1 {_f(o['test']['macro_f1'])} vs multinomial logistic {_f(o['baselines_test']['multinomial_logistic']['macro_f1'])} "
        f"and majority class {_f(o['baselines_test']['majority_class']['macro_f1'])}; class weighting `{o['chosen_weighting']}` chosen on validation.",
        f"- P(fail or withdraw): ROC-AUC {_f(o['poor_outcome_from_p_fail_plus_p_withdrawn_test']['roc_auc'])}, "
        f"ECE {_f(o['poor_outcome_from_p_fail_plus_p_withdrawn_test']['ece_10'])}.", "",
        "## Feature-group ablation (test ROC-AUC / PR-AUC)",
        "| Variant | ROC-AUC | PR-AUC |", "|---|---|---|",
    ] + [f"| {k} | {_f(v['roc_auc'])} | {_f(v['pr_auc'])} |" for k, v in a.get("variants", {}).items()] + [
        "", "Sensitive attributes add essentially nothing, so they are excluded from the model by default.", "",
        "## Prioritisation (target: fail or withdraw)",
        f"Weights learned on the validation presentation: `{p.get('weights')}`.", "",
        "| Ranking | Mean precision @10/20/30% | Precision @10% |", "|---|---|---|",
    ] + [f"| {k} | {_f(v['objective_mean_precision'])} | {_pct(v['at_capacity'][0]['precision'])} |"
         for k, v in p.get("test_comparison", {}).items() if k in ("learned_weights", "v4_hand_formula", "only_p_poor", "only_p_withdraw", "random")] + [
        "", "## Historical-case retrieval",
        f"Queries: {r.get('queries')} in-scope students of {r.get('query_presentation')}; corpus: earlier presentations, same module.", "",
        "| Mode | Neighbour-vote AUC (fail/withdraw) | Outcome agreement@10 | nDCG@10 (global ideal) |", "|---|---|---|---|",
    ] + [f"| {k} | {_f(v['neighbour_vote_auc_poor_outcome'])} | {_f(v['outcome_agreement_at_10'])} | {_f(v['reference_ndcg_at_10_global_ideal'])} |"
         for k, v in r.get("modes", {}).items()] + _typed_search_lines(m.get("text_search_benchmark", {})) + [
        "", "## Fairness audit (High tier, test presentation)",
        "| Attribute | Equal-opportunity gap (TPR max−min) | Selection-rate ratio | Max calibration gap |", "|---|---|---|---|",
    ] + [f"| {k} | {_f(v['equal_opportunity_gap'])} | {_f(v['selection_rate_ratio'])} | {_f(v['max_calibration_gap'])} |"
         for k, v in fa.get("attributes", {}).items()] + [
        "", "Gaps persist even with sensitive attributes excluded (proxies via engagement/scores and different base rates). "
        "They are surfaced as quality-gate warnings and must be reviewed with the institution's equity office before deployment. "
        "Mitigation options: per-group threshold review, outreach-capacity floors, monitoring by group via the feedback loop.", "",
        "## Early warning across the course (test presentation, students still enrolled at each point)",
        "| Course week | Still enrolled | ROC-AUC | PR-AUC | Base rate | ECE | Outcome macro-F1 |", "|---|---|---|---|---|---|---|",
    ] + [f"| {round(c['approx_week'])} | {c['in_scope']:,} | {_f(c['withdrawal']['roc_auc'])} | {_f(c['withdrawal']['pr_auc'])} | "
         f"{_f(c['withdrawal']['positive_rate'])} | {_f(c['withdrawal']['ece_10'])} | {_f(c['outcome_macro_f1'])} |"
         for c in m["checkpoint_summary"].get("checkpoints", [])] + [
        "", "## AI features (offline evaluation)", _llm_lines(m["llm_evaluation"]), "",
        "## Limitations",
        "- Associations, not causes. Drivers explain the model, not the student.",
        "- One test presentation (2014J); performance can drift across cohorts — monitored by the feedback loop's tier-drift check.",
        "- OULAD is from one institution and 2013–2014; revalidate before use elsewhere.",
        "- LLM narration is optional; all facts, scores, priorities and eligible actions are computed deterministically.",
    ]
    return "\n".join(lines) + "\n"


def build_deck(s: Settings | None = None, out: Path | None = None) -> Path:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
    from pptx.util import Inches, Pt

    s = s or get_settings()
    m = load_all(s)
    out = Path(out or PROJECT_ROOT / "docs" / "Student_Success_Assistant_10min.pptx")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    rgb = lambda h: RGBColor.from_string(h)  # noqa: E731

    def slide(title, dark=False, notes=""):
        sl = prs.slides.add_slide(blank)
        bg = sl.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(NAVY if dark else PAPER)
        tb = sl.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12), Inches(1.0)).text_frame
        tb.word_wrap = True
        r = tb.paragraphs[0].add_run()
        r.text = title
        r.font.size, r.font.bold = Pt(30 if not dark else 40), True
        r.font.color.rgb = rgb("FFFFFF" if dark else NAVY)
        if notes:
            sl.notes_slide.notes_text_frame.text = notes
        return sl

    def text(sl, x, y, w, h, items, size=16, color=NAVY, bold_first=False):
        tf = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = p.add_run()
            run.text = it
            run.font.size = Pt(size)
            run.font.color.rgb = rgb(color)
            run.font.bold = bold_first and i == 0
            p.space_after = Pt(8)
        return tf

    def stat(sl, x, y, value, label, color=TEAL):
        text(sl, x, y, 3.0, 0.9, [value], size=40, color=color)
        text(sl, x, y + 0.85, 3.0, 0.8, [label], size=13, color=SLATE)

    def bar(sl, x, y, w, h, cats, series, colors, title, fmt='0.00', maxv=None, horizontal=False):
        cd = CategoryChartData()
        cd.categories = cats
        for name, vals in series:
            cd.add_series(name, vals)
        kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
        ch = sl.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
        ch.has_legend = len(series) > 1
        va = ch.value_axis
        va.has_major_gridlines = False
        va.tick_labels.font.size = Pt(11)
        va.minimum_scale = 0
        if maxv:
            va.maximum_scale = maxv
        ch.category_axis.tick_labels.font.size = Pt(11)
        for i, ser in enumerate(ch.series):
            pl = ser
            pl.format.fill.solid()
            pl.format.fill.fore_color.rgb = rgb(colors[i % len(colors)])
        plot = ch.plots[0]
        plot.has_data_labels = True
        plot.data_labels.number_format = fmt
        plot.data_labels.number_format_is_linked = False
        plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        plot.data_labels.font.size = Pt(11)
        return ch

    d, w, o, a, fa, p, r, llm, q = (m[k] for k in ("data_build", "withdrawal_model", "outcome_model", "ablations", "fairness",
                                                    "prioritization", "retrieval_benchmark", "llm_evaluation", "quality_gate"))
    t = w["test"]
    pa = w["population_audit"]

    sl = slide("AI-powered Student Success Assistant", dark=True,
               notes="Ten-minute walkthrough: the problem, how the system works, honest evidence, and how it stays safe and improves.")
    text(sl, 0.6, 2.2, 12, 2, ["Early warning, comparable historical cases and supportive action plans for advisors",
                                f"OULAD · {d['records']:,} enrolments · evidence profile: {q.get('profile')} · quality gate "
                                f"{'passed' if q.get('passed') else 'failed'}"], size=20, color="E2E8F0")

    sl = slide("The question an advisor actually has",
               notes="Frame the problem around the advisor at week 15: who still enrolled needs help, why, and what to do.")
    text(sl, 0.6, 1.5, 6.2, 5, [
        "At ~40% of the course, which students who are still enrolled are likely to fail or withdraw?",
        "Why — in terms an advisor can act on?",
        "Have students with similar patterns struggled before?",
        "What supportive, non-punitive action should we take, and who first given limited capacity?"], size=18)
    stat(sl, 7.4, 1.6, f"{d['in_scope_records']:,}", "students still enrolled at the snapshot (in scope)")
    stat(sl, 10.2, 1.6, f"{d['withdrawn_before_snapshot']:,}", "had already unregistered — excluded", AMBER)
    stat(sl, 7.4, 3.6, _pct(t["positive_rate"], 1), "of in-scope students later withdraw (test)", SLATE)
    stat(sl, 10.2, 3.6, _pct(o["poor_outcome_from_p_fail_plus_p_withdrawn_test"]["positive_rate"]), "later fail or withdraw", SLATE)

    sl = slide("Architecture", notes="Offline build produces calibrated models, the case index and evaluation; the online API serves "
               "lookups, scores, retrieval and the agent workflow; feedback closes the loop with human approval.")
    img = PROJECT_ROOT / "docs" / "architecture.png"
    if img.exists():
        pic = sl.shapes.add_picture(str(img), Inches(0.4), Inches(1.25), width=Inches(8.3))
        if pic.height > Inches(6.0):
            r_ = Inches(6.0) / pic.height
            pic.height, pic.width = Inches(6.0), int(pic.width * r_)
    text(sl, 8.9, 1.5, 4.1, 5.5, [
        "Offline build: features → calibrated models → prioritiser → historical-case index → evaluation and quality gate.",
        "Online: API (auth, rate limit, strict validation) → service (real snapshots, what-if with imputation flags) → agent workflow.",
        "LLM is optional and narrates only; facts, scores, priorities and eligible actions are computed in code.",
        "Feedback store and loop (amber) feed approved changes back into actions, retrieval weights and calibration."], size=16)

    sl = slide("Honest evaluation: removing the leakage",
               notes="v4 trained and tested on students who had already left, so silence in the logs 'predicted' withdrawal. "
                     "We score only students still enrolled, on a later, untouched presentation.")
    bar(sl, 0.6, 1.4, 6.0, 5.4, ["All enrolments (incl. already left)", "Still enrolled (valid)"],
        [("ROC-AUC", [pa["all_enrolments_including_already_withdrawn"]["roc_auc"], pa["in_scope_only"]["roc_auc"]])],
        [AMBER], "Same model, two populations (test presentation)", maxv=1.0)
    text(sl, 7.0, 1.6, 5.8, 5, [
        f"{_pct(pa['already_withdrawn_share_of_test_withdrawals'])} of test-presentation withdrawals had already happened by the snapshot.",
        "Temporal protocol: train on earlier presentations, calibrate out-of-fold by presentation, test once on the latest.",
        "Missing scores stay missing; zero-activity weeks count; peer-relative percentiles make modules comparable.",
        "Sensitive attributes are excluded from the model — the ablation shows they add essentially nothing."], size=16)

    sl = slide("Withdrawal risk: calibrated and useful at advisor capacity",
               notes="Explain ROC-AUC with CI, the logistic baseline, calibration, and that tiers really separate risk.")
    tiers = w["test_tiers"]
    bar(sl, 0.6, 1.4, 6.0, 5.2, list(tiers), [("Observed withdrawal rate", [tiers[k]["observed_withdrawal_rate"] for k in tiers])],
        [RED], "Observed withdrawal rate by risk tier", fmt='0%')
    cap = {c["fraction"]: c for c in w["test_capacity"]}
    stat(sl, 7.2, 1.5, _f(t["roc_auc"]), f"ROC-AUC (95% CI {_f(t['roc_auc_95ci']['lower'])}–{_f(t['roc_auc_95ci']['upper'])}); "
         f"logistic baseline {_f(w['baseline_logistic_regression_test']['roc_auc'])}")
    stat(sl, 10.2, 1.5, _f(t["ece_10"]), "calibration error (ECE)")
    stat(sl, 7.2, 3.6, f"{cap[0.1]['lift']:.1f}×", "lift when contacting the top 10%")
    stat(sl, 10.2, 3.6, _pct(cap[0.2]["recall"]), "of eventual withdrawals reached at 20% capacity")

    cps = m["checkpoint_summary"].get("checkpoints", [])
    if cps:
        sl = slide("How early can we tell?", notes="Separate calibrated models at four points in the course; each is tested on the later, "
                   "untouched presentation using only what was known at that point.")
        bar(sl, 0.6, 1.4, 7.4, 5.4, [f"week {round(c['approx_week'])}" for c in cps],
            [("ROC-AUC", [c["withdrawal"]["roc_auc"] for c in cps]), ("PR-AUC", [c["withdrawal"]["pr_auc"] for c in cps]),
             ("Base rate", [c["withdrawal"]["positive_rate"] for c in cps])], [NAVY, "4C4BC2", "9AA2B1"], "Withdrawal model by course week", maxv=1.0)
        text(sl, 8.4, 1.6, 4.5, 5, ["Later points see more evidence but fewer students remain to help.",
                                    f"Week {round(cps[0]['approx_week'])}: {cps[0]['in_scope']:,} still enrolled; week {round(cps[-1]['approx_week'])}: {cps[-1]['in_scope']:,}.",
                                    "The console's course clock switches every view between these points."], size=15)

    sl = slide("What is associated with failing or withdrawing",
               notes="Factor analysis on historical in-scope students; associations not causes. TreeSHAP gives per-student drivers.")
    top = m["factors"].get("numeric_factors_ranked", [])[:7]
    if top:
        ch = bar(sl, 0.6, 1.4, 8.2, 5.6, [f["label"] for f in top][::-1], [("Spread", [f["poor_outcome_rate_spread"] for f in top][::-1])],
                 [TEAL], "Gap in fail/withdraw rate between best and worst quintile", fmt='0%', horizontal=True, maxv=1.0)
    div = {k: v["first_week_below_60pct_of_pass"].get("Withdrawn") for k, v in m["trends"].get("modules", {}).items()}
    text(sl, 9.1, 1.6, 3.8, 5, ["Engagement trends", "Week at which eventual withdrawers fall below 60% of passers' activity:"] +
         [f"{k}: week {v}" for k, v in div.items() if v is not None], size=14, bold_first=True)

    sl = slide("Similar historical students — retrieval that beats chance",
               notes="Behavioural kNN weighted by model attributions + BM25 over qualitative descriptors (+ optional dense). "
                     "Only earlier presentations are searchable. Evaluated against real outcomes and a random baseline.")
    modes = r.get("modes", {})
    bar(sl, 0.6, 1.4, 7.6, 5.4, list(modes), [("Neighbour-vote AUC", [v["neighbour_vote_auc_poor_outcome"] for v in modes.values()]),
                                              ("nDCG@10", [v["reference_ndcg_at_10_global_ideal"] for v in modes.values()])],
        [NAVY, TEAL], "Do retrieved precedents predict the real outcome?", maxv=1.0)
    text(sl, 8.6, 1.6, 4.3, 5, ["Metadata filters: module, presentation, region, education, age band, credits (range), "
                                "previous attempts, outcome.",
                                "Each answer reports the outcome mix of the 50 nearest precedents with a 95% CI.",
                                "v4's index shipped with 2 test records and scored at random level."], size=15)

    sl = slide("Who first? Prioritisation learned for limited capacity",
               notes="Weights chosen on the validation presentation to maximise precision at 10/20/30% capacity; compared on test.")
    tc = p["test_comparison"]
    names = [("learned_weights", "Learned"), ("v4_hand_formula", "v4 formula"), ("only_p_withdraw", "Risk only"), ("random", "Random")]
    bar(sl, 0.6, 1.4, 7.6, 5.4, [n for _, n in names], [("Precision @10%", [tc[k]["at_capacity"][0]["precision"] for k, _ in names]),
                                                       ("Precision @20%", [tc[k]["at_capacity"][1]["precision"] for k, _ in names])],
        [TEAL, SLATE], "Share of contacted students who went on to fail or withdraw", fmt='0%', maxv=1.0)
    text(sl, 8.6, 1.6, 4.3, 5, [f"Learned weights: {', '.join(f'{k} {v:.1f}' for k, v in p['weights'].items() if v)}",
                                "Signals are cohort-relative so a click means the same in every module.",
                                "Every ranked student carries reason codes and top risk drivers."], size=15)

    sl = slide("Agents that cannot make things up",
               notes="Agents select and narrate ID-tagged evidence; actions come from an eligibility-gated catalogue; "
                     "the critic enforces guardrails in code with one revision, else a deterministic plan.")
    det = (llm.get("agents") or {}).get("deterministic", {}) if llm else {}
    text(sl, 0.6, 1.5, 6.5, 5.5, [
        "Performance → Risk → Router → Intervention ⇄ Critic, or Monitoring for low-risk students.",
        "Facts are computed in code; the LLM only narrates and must cite observation IDs.",
        "Critic checks: catalogue eligibility, numeric grounding, protected attributes, punitive or clinical language, "
        "and no risk scores in messages to students.",
        "Sensitive attributes are never sent to the LLM; priorities come from calibrated models, not the LLM.",
        "Copilot answers through 8 tools and must back every number with a tool result; search, precedent summaries, "
        "briefings and outreach rewrites follow the same checks.",
        "Each AI feature has its own offline evaluation; LLM-as-Judge uses a separate model and verdicts are computed in code."], size=15)
    stat(sl, 7.8, 1.6, _pct(det.get("final_plan_valid_rate"), 0), f"of {det.get('n_cases', 0)} golden-set plans pass every guardrail")
    j = (llm.get("agents") or {}).get("judge", {}) if llm else {}
    stat(sl, 7.8, 3.6, _pct(j.get("pass_rate")) if j.get("pass_rate") is not None else "run with key",
         "LLM-judge pass rate (python -m ssa.cli llm-eval)", SLATE)

    shot = PROJECT_ROOT / "docs" / "screenshots" / "student.jpg"
    if shot.exists():
        sl = slide("The advisor console", notes="React console served by the FastAPI service. Walk through: course clock, why flagged, "
                   "risk across the course, live agent plan, outreach rewriting, precedents with what happened next, feedback.")
        from PIL import Image
        im = Image.open(shot)
        crop = PROJECT_ROOT / "docs" / "screenshots" / "_deck_student.jpg"
        im.crop((0, 0, im.size[0], min(im.size[1], int(im.size[0] * 0.62)))).save(crop, quality=85)
        sl.shapes.add_picture(str(crop), Inches(0.5), Inches(1.25), height=Inches(5.9))
        text(sl, 10.2, 1.5, 2.8, 5.5, ["Course clock across the top: every view is as of one point in the course.",
                                       "Computed facts in ink; AI-drafted text marked in indigo.",
                                       "Copilot, case finder, insights, model quality and the feedback loop use the same API."], size=14)

    sl = slide("Fairness and limits — stated, not hidden",
               notes="Gaps remain via proxies; they are flagged by the gate and need institutional review.")
    attrs = fa.get("attributes", {})
    bar(sl, 0.6, 1.4, 7.6, 5.4, list(attrs), [("Equal-opportunity gap", [v["equal_opportunity_gap"] or 0 for v in attrs.values()])],
        [AMBER], "Difference in true-positive rate between groups (High tier)", fmt='0.00')
    text(sl, 8.6, 1.6, 4.3, 5, ["Sensitive attributes are excluded from the model and from agent prompts, yet gaps persist through "
                                "proxies and base-rate differences.", "The quality gate warns above 0.15; deployment needs equity review.",
                                "One institution, 2013–14 data: revalidate before use elsewhere."], size=15)

    sl = slide("Closing the loop", dark=True, notes="Feedback becomes versioned, human-approved changes; then re-evaluate.")
    text(sl, 0.6, 1.8, 12, 5, [
        "Advisors rate plans, individual actions and individual precedents, and later record real outcomes.",
        "Learners re-rank actions (Beta-Binomial), re-learn retrieval fusion weights, and recalibrate risk when enough outcomes exist.",
        "Changes are proposals until approved with --apply; drift monitors flag tier precision changes and retraining needs.",
        "Everything in this deck is regenerated from artifacts: python -m ssa.cli report"], size=20, color="E2E8F0")
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


def write_reports(s: Settings | None = None) -> dict:
    s = s or get_settings()
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "MODEL_CARD.md").write_text(model_card(s), encoding="utf-8")
    out = {"model_card": str(docs / "MODEL_CARD.md")}
    try:
        out["deck"] = str(build_deck(s))
    except ImportError:
        out["deck"] = "python-pptx not installed"
    return out
