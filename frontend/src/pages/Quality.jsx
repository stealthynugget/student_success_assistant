import React, { useState } from "react";
import { api, fmt } from "../api.js";
import { C, GroupedBars, HBars, Reliability } from "../charts.jsx";
import { useApi, useStore } from "../store.jsx";
import { Empty, ErrorNote, Loading, Tabs } from "../ui.jsx";

export default function Quality() {
  const [tab, setTab] = useState("risk");
  const { checkpoint } = useStore();
  const m = useApi((s) => api("/metrics", { query: { checkpoint }, signal: s }), [checkpoint]);
  return (
    <>
      <div className="page-head"><div><h1>Model quality</h1><div className="sub">How well each part works, measured on a later presentation the models never saw.</div></div>
        <div className="actions"><Tabs value={tab} onChange={setTab} tabs={[["risk", "Risk models"], ["retrieval", "Similar cases"], ["ranking", "Prioritisation"],
          ["llm", "AI features"], ["online", "Live reviews"], ["usage", "AI usage and cost"], ["gate", "Release checks"]]} /></div></div>
      {m.error && <ErrorNote error={m.error} onRetry={m.reload} />}
      {!m.data && !m.error && <Loading lines={6} />}
      {m.data && tab === "risk" && <Risk m={m.data} />}
      {m.data && tab === "retrieval" && <Retrieval r={m.data.retrieval_benchmark} />}
      {m.data && tab === "ranking" && <Ranking p={m.data.prioritization} />}
      {tab === "llm" && <LlmEval />}
      {tab === "online" && <Online />}
      {tab === "usage" && <Usage />}
      {m.data && tab === "gate" && <Gate g={m.data.quality_gate} />}
    </>
  );
}

function Risk({ m }) {
  const w = m.withdrawal_model, o = m.outcome_model, a = m.ablations;
  if (!w) return <Empty title="No model metrics at this checkpoint">Run the build on the server.</Empty>;
  const t = w.test, pa = w.population_audit;
  return (
    <div className="stack">
      <div className="stats">
        <div className="stat"><div className="v">{t.roc_auc.toFixed(3)}</div><div className="l">ROC-AUC (95% CI {t.roc_auc_95ci.lower.toFixed(3)}–{t.roc_auc_95ci.upper.toFixed(3)})</div></div>
        <div className="stat"><div className="v">{t.pr_auc.toFixed(3)}</div><div className="l">PR-AUC, against a base rate of {t.positive_rate.toFixed(3)}</div></div>
        <div className="stat"><div className="v">{t.ece_10.toFixed(3)}</div><div className="l">calibration error: predicted risk matches observed rates</div></div>
        <div className="stat"><div className="v">{w.baseline_logistic_regression_test.roc_auc.toFixed(3)}</div><div className="l">simple logistic baseline ROC-AUC</div></div>
      </div>
      <div className="grid two section">
        <div className="stack"><h3>Predicted versus observed</h3><Reliability rows={w.test_reliability} />
          <p className="small muted">Each dot groups students with similar predicted risk; dots on the dashed line mean the percentages can be taken at face value.</p></div>
        <div className="stack"><h3>Do the tiers separate risk?</h3>
          <GroupedBars categories={Object.keys(w.test_tiers)} max={1} format={(v) => fmt.pct(v)}
            series={[{ name: "Observed withdrawal rate", color: C.High, values: Object.values(w.test_tiers).map((x) => x.observed_withdrawal_rate) },
              { name: "Share of all withdrawals", color: C.ink, values: Object.values(w.test_tiers).map((x) => x.share_of_all_withdrawals) }]} />
          <div className="legend"><span><i style={{ background: C.High }} />observed withdrawal rate</span><span><i style={{ background: C.ink }} />share of all withdrawals caught</span></div></div>
      </div>
      <div className="grid two section">
        <div className="stack"><h3>At advisor capacity</h3>
          <table className="data"><thead><tr><th>Contact the top</th><th className="num">Precision</th><th className="num">Withdrawals reached</th><th className="num">Lift</th></tr></thead>
            <tbody>{w.test_capacity.map((c) => <tr key={c.fraction}><td>{fmt.pct(c.fraction)}</td><td className="num">{fmt.pct(c.precision)}</td><td className="num">{fmt.pct(c.recall)}</td><td className="num">{c.lift.toFixed(1)}×</td></tr>)}</tbody></table></div>
        <div className="stack"><h3>Why we only score students still enrolled</h3>
          <p>Scored on every enrolment, including students who had already left, the same model would show ROC-AUC {pa.all_enrolments_including_already_withdrawn.roc_auc.toFixed(3)}. {fmt.pct(pa.already_withdrawn_share_of_test_withdrawals)} of the test presentation's withdrawals had already happened by this point, so that number flatters the model. The valid figure is {pa.in_scope_only.roc_auc.toFixed(3)}.</p></div>
      </div>
      {a && <div className="section stack"><h3>Which evidence matters</h3>
        <HBars items={Object.entries(a.variants).map(([k, v]) => ({ label: k.replaceAll("_", " "), value: v.roc_auc, color: k.includes("production") ? C.ink : "#9aa2b1" }))}
          max={1} format={(v) => v.toFixed(3)} labelWidth={0.4} />
        <p className="small muted">Adding sensitive attributes changes ROC-AUC by {a.cost_of_excluding_sensitive_attributes.roc_auc.toFixed(3)}, so they stay out of the model.</p></div>}
      {o && <div className="section stack"><h3>Final-outcome model</h3>
        <p>Macro-F1 {o.test.macro_f1.toFixed(3)} against {o.baselines_test.multinomial_logistic.macro_f1.toFixed(3)} for a logistic baseline and {o.baselines_test.majority_class.macro_f1.toFixed(3)} for always guessing the most common outcome. The combined fail-or-withdraw probability has ROC-AUC {o.poor_outcome_from_p_fail_plus_p_withdrawn_test.roc_auc.toFixed(3)}.</p>
        <table className="data"><thead><tr><th>Outcome</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Students</th></tr></thead>
          <tbody>{Object.entries(o.test.per_class).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{fmt.pct(v.precision)}</td><td className="num">{fmt.pct(v.recall)}</td><td className="num">{v["f1-score"].toFixed(3)}</td><td className="num">{v.support}</td></tr>)}</tbody></table></div>}
    </div>
  );
}

function Retrieval({ r }) {
  if (!r) return <Empty title="Retrieval benchmark not run">Run python -m ssa.cli evaluate on the server.</Empty>;
  const modes = Object.keys(r.modes);
  return (
    <div className="stack">
      <p>{r.queries} students from {r.query_presentation} were matched against earlier presentations of the same module. A good matcher finds past students whose real outcomes predict this student's outcome.</p>
      <GroupedBars categories={modes} max={1} format={(v) => v.toFixed(2)}
        series={[{ name: "Neighbour-vote AUC (fail or withdraw)", color: C.ink, values: modes.map((m) => r.modes[m].neighbour_vote_auc_poor_outcome) },
          { name: "Outcome agreement", color: C.ai, values: modes.map((m) => r.modes[m].outcome_agreement_at_10) },
          { name: "nDCG@10 (true ideal)", color: "#9aa2b1", values: modes.map((m) => r.modes[m].reference_ndcg_at_10_global_ideal) }]} />
      <div className="legend"><span><i style={{ background: C.ink }} />neighbour-vote AUC</span><span><i style={{ background: C.ai }} />outcome agreement at 10</span><span><i style={{ background: "#9aa2b1" }} />nDCG at 10</span></div>
      <ul className="small muted">{r.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
    </div>
  );
}

function Ranking({ p }) {
  if (!p) return <Empty title="Prioritisation not evaluated">Run the build on the server.</Empty>;
  const names = { learned_weights: "Learned weights", v4_hand_formula: "Previous hand-set formula", only_p_poor: "Fail-or-withdraw only", only_p_withdraw: "Withdrawal risk only",
    only_missed_work: "Missed work only", only_engagement_gap: "Activity gap only", random: "Random order" };
  const rows = Object.entries(names).filter(([k]) => p.test_comparison[k]);
  return (
    <div className="stack">
      <p>Share of contacted students who went on to fail or withdraw, by how the list was ordered. Weights were chosen on {p.split.validation?.join(", ")} and tested on {p.split.test?.join(", ")}.</p>
      <GroupedBars categories={rows.map(([, l]) => l)} max={1} format={(v) => fmt.pct(v)} height={260}
        series={[0, 1, 2].map((i) => ({ name: `Top ${[10, 20, 30][i]}%`, color: [C.ink, C.ai, "#9aa2b1"][i], values: rows.map(([k]) => p.test_comparison[k].at_capacity[i].precision) }))} />
      <div className="legend"><span><i style={{ background: C.ink }} />top 10%</span><span><i style={{ background: C.ai }} />top 20%</span><span><i style={{ background: "#9aa2b1" }} />top 30%</span></div>
    </div>
  );
}

function Section({ title, status, children }) {
  return <div className="section stack"><div className="row"><h3>{title}</h3>{status && <span className="small muted">{status}</span>}</div>{children}</div>;
}
const rate = (v) => (v === null || v === undefined ? "not run" : fmt.pct(v));

function LlmEval() {
  const q = useApi((s) => api("/evaluations/llm", { signal: s }), []);
  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data) return <Loading />;
  const e = q.data;
  if (!e.agents) return <Empty title="AI evaluation not run yet">Run python -m ssa.cli llm-eval on the server. Without an API key it still measures the computed paths.</Empty>;
  const ag = e.agents;
  return (
    <div className="stack">
      <p className="muted">Run {e.generated_at}. Generator {e.generator_model || "not connected"}, judge {e.judge_model || "not connected"}. Sections that need the LLM say "not run" until it is connected.</p>
      <Section title="Support plans" status={`${ag.deterministic.n_cases} students across all tiers`}>
        <table className="data"><tbody>
          <tr><td>Plans passing every guardrail</td><td className="num">{rate(ag.deterministic.final_plan_valid_rate)}</td></tr>
          <tr><td>AI drafts accepted without falling back</td><td className="num">{rate(ag.deterministic.llm_plan_accepted_rate)}</td></tr>
          <tr><td>Drafts that needed a revision after the critic</td><td className="num">{rate(ag.deterministic.revision_rate)}</td></tr>
          <tr><td>High-tier students always routed to an intervention</td><td className="num">{rate(ag.deterministic.tier_routing_consistency)}</td></tr>
          <tr><td>Judge pass rate</td><td className="num">{rate(ag.judge.pass_rate)}</td></tr>
          <tr><td>AI plan preferred over computed plan (order-swapped A/B)</td><td className="num">{rate(ag.pairwise_llm_vs_deterministic.llm_win_rate)}</td></tr>
          <tr><td>DeepEval</td><td className="num">{ag.deepeval.status}</td></tr>
        </tbody></table>
        {ag.judge.mean_scores && <HBars items={Object.entries(ag.judge.mean_scores).map(([k, v]) => ({ label: k.replaceAll("_", " "), value: v }))} max={10} format={(v) => v.toFixed(1)} />}
        {ag.deepeval.aggregate && <table className="data"><thead><tr><th>DeepEval metric</th><th className="num">Mean</th><th className="num">Pass rate</th></tr></thead>
          <tbody>{Object.entries(ag.deepeval.aggregate).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{v.mean.toFixed(2)}</td><td className="num">{rate(v.pass_rate)}</td></tr>)}</tbody></table>}
      </Section>
      <Section title="Understanding search requests" status={e.search.note}>
        <table className="data"><thead><tr><th>Set</th><th>Method</th><th className="num">Filter F1</th><th className="num">Exact match</th></tr></thead>
          <tbody>{["templated", "paraphrased"].flatMap((set) => ["rules", "llm"].map((mth) => {
            const r = e.search[`${set}_${mth}`] || {};
            return <tr key={set + mth}><td>{set}</td><td>{mth === "llm" ? "AI" : "Rules"}</td><td className="num">{r.filter_f1 != null ? r.filter_f1.toFixed(2) : "not run"}</td><td className="num">{rate(r.exact_match)}</td></tr>;
          }))}</tbody></table>
      </Section>
      <Section title="Copilot" status="Golden questions with known answers">
        <table className="data"><thead><tr><th>Mode</th><th className="num">Correct answers</th><th className="num">Right tool</th><th className="num">Grounded and safe</th></tr></thead>
          <tbody>{["rules", "llm"].map((k) => { const r = e.copilot[k] || {}; return <tr key={k}><td>{k === "llm" ? "AI" : "Rules"}</td><td className="num">{rate(r.answer_correct_rate)}</td><td className="num">{rate(r.tool_selection_accuracy)}</td><td className="num">{rate(r.grounded_and_safe_rate)}</td></tr>; })}</tbody></table>
        <p>Adversarial prompts handled safely: {rate(e.robustness.safe_rate)} of {e.robustness.n} ({e.robustness.mode}).</p>
        <details><summary>See adversarial prompts</summary><ul className="small">{e.robustness.cases.map((c, i) => <li key={i}><strong>{c.safe ? "Safe" : "Unsafe"}</strong>: {c.prompt}</li>)}</ul></details>
      </Section>
      <Section title="Other generated text">
        <table className="data"><tbody>
          <tr><td>“What happened next” summaries accepted</td><td className="num">{rate(e.synthesis.llm_accepted_rate)}</td></tr>
          <tr><td>Judge faithfulness of those summaries</td><td className="num">{e.synthesis.judge_mean_faithfulness?.toFixed(1) ?? "not run"}</td></tr>
          <tr><td>Cohort briefings accepted</td><td className="num">{rate(e.briefing.llm_accepted_rate)}</td></tr>
          <tr><td>Outreach rewrites within length</td><td className="num">{rate(e.outreach.length_compliance)}</td></tr>
          <tr><td>Outreach rewrites passing safety checks</td><td className="num">{rate(e.outreach.safe_rate)}</td></tr>
          <tr><td>Requested language delivered</td><td className="num">{rate(e.outreach.language_honoured_rate)}</td></tr>
        </tbody></table>
        <p className="small muted">Judge agreement with advisors: {e.human_agreement.status || `kappa ${e.human_agreement.verdict_cohen_kappa?.toFixed(2)}`}</p>
      </Section>
    </div>
  );
}

function Online() {
  const q = useApi((s) => api("/evaluations/online", { signal: s }), []);
  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data) return <Loading />;
  if (!q.data.items.length) return <Empty title="No live reviews yet">Ask the AI judge to review a plan from a student page, or set ONLINE_EVAL_SAMPLE_RATE on the server to review a sample of live requests automatically.</Empty>;
  return (
    <div className="stack">
      <table className="data"><thead><tr><th>Kind</th><th className="num">Reviews</th><th className="num">Mean score</th><th className="num">Pass rate</th></tr></thead>
        <tbody>{q.data.summary.by_kind.map((k) => <tr key={k.kind}><td>{k.kind}</td><td className="num">{k.n}</td><td className="num">{fmt.num(k.mean_overall, 1)}</td><td className="num">{fmt.pct(k.pass_rate)}</td></tr>)}</tbody></table>
      <table className="data"><thead><tr><th>When</th><th>Kind</th><th>Reference</th><th>Verdict</th><th className="num">Score</th><th>Judge</th></tr></thead>
        <tbody>{q.data.items.map((i) => <tr key={i.id}><td>{new Date(i.ts * 1000).toLocaleString()}</td><td>{i.kind}</td><td className="small">{i.ref_id}</td><td>{i.verdict}</td><td className="num">{fmt.num(i.overall, 1)}</td><td className="small">{i.judge_model}</td></tr>)}</tbody></table>
    </div>
  );
}

function Usage() {
  const [hours, setHours] = useState(24 * 7);
  const q = useApi((s) => api("/llm/usage", { query: { since_hours: hours }, signal: s }), [hours]);
  return (
    <div className="stack">
      <label className="field" style={{ maxWidth: 220 }}>Period<select value={hours} onChange={(e) => setHours(Number(e.target.value))}>
        <option value={24}>Last day</option><option value={168}>Last week</option><option value={720}>Last 30 days</option></select></label>
      {q.error && <ErrorNote error={q.error} />}
      {q.data && (q.data.total.n === 0 ? <Empty title="No AI calls in this period">Calls appear here once the LLM is connected and used.</Empty> : <>
        <div className="stats"><div className="stat"><div className="v">{q.data.total.n}</div><div className="l">calls ({q.data.total.cached} from cache, {q.data.total.errors} errors)</div></div>
          <div className="stat"><div className="v">{(q.data.total.pt + q.data.total.ct).toLocaleString()}</div><div className="l">tokens</div></div>
          <div className="stat"><div className="v">${q.data.total.estimated_cost_usd.toFixed(2)}</div><div className="l">estimated cost</div></div></div>
        <table className="data"><thead><tr><th>Feature</th><th>Model</th><th className="num">Calls</th><th className="num">Tokens</th><th className="num">Avg latency</th><th className="num">Cost</th></tr></thead>
          <tbody>{q.data.by_feature.map((r) => <tr key={r.feature + r.model}><td>{r.feature}</td><td>{r.model}</td><td className="num">{r.n}</td><td className="num">{(r.pt + r.ct).toLocaleString()}</td><td className="num">{r.avg_latency_s}s</td><td className="num">${r.estimated_cost_usd.toFixed(3)}</td></tr>)}</tbody></table></>)}
    </div>
  );
}

function Gate({ g }) {
  if (!g) return <Empty title="Release checks not run">Run python -m ssa.cli quality-gate on the server.</Empty>;
  return (
    <div className="stack">
      <p><strong>{g.passed ? "All release checks pass" : "Some release checks fail"}</strong> ({g.profile} data, {g.generated_at}).</p>
      <table className="data"><tbody>{g.checks.map((c) => <tr key={c.check}><td style={{ color: c.passed ? C.Low : C.High, fontWeight: 700 }}>{c.passed ? "Pass" : "Fail"}</td><td>{c.check.replaceAll("_", " ")}</td><td className="small muted">{typeof c.detail === "object" ? JSON.stringify(c.detail) : String(c.detail ?? "")}</td></tr>)}</tbody></table>
      {g.warnings.length > 0 && <><h4>Warnings to review</h4><table className="data"><tbody>{g.warnings.map((c) => <tr key={c.check}><td style={{ color: C.Medium, fontWeight: 700 }}>Review</td><td>{c.check.replaceAll("_", " ")}</td><td className="small muted">{JSON.stringify(c.detail)}</td></tr>)}</tbody></table></>}
    </div>
  );
}
