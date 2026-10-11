import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, CircleAlert, TriangleAlert } from "lucide-react";
import { useState, type ReactNode } from "react";
import { api, fmt, type ApiError } from "../api";
import { useStore } from "../store";
import { Empty, ErrorNote, Loading, TabSet } from "../ui";
import { COL, GroupedBars, HBars, Reliability } from "../viz";

/* eslint-disable @typescript-eslint/no-explicit-any */
const rate = (v: unknown) => (v == null ? "not run" : fmt.pct(v));

export default function Quality() {
  const [tab, setTab] = useState("risk");
  const { checkpoint } = useStore();
  const m = useQuery({ queryKey: ["metrics", checkpoint], queryFn: ({ signal }) => api("/metrics", { query: { checkpoint }, signal }) });
  const d: any = m.data;
  return (
    <>
      <div className="page-head"><div><h1>Model quality</h1><div className="sub">How well each part works, measured on a later presentation the models never saw.</div></div></div>
      <TabSet label="Model quality" value={tab} onChange={setTab} tabs={[["risk", "Risk models"], ["retrieval", "Similar cases"], ["ranking", "Prioritisation"],
        ["llm", "AI features"], ["online", "Live reviews"], ["usage", "AI usage and cost"], ["gate", "Release checks"]]}>
        {m.error && <ErrorNote error={m.error as ApiError} onRetry={() => m.refetch()} />}
        {!d && !m.error && ["risk", "retrieval", "ranking", "gate"].includes(tab) && <Loading lines={6} />}
        {d && tab === "risk" && <Risk m={d} />}
        {d && tab === "retrieval" && <Retrieval r={d.retrieval_benchmark} />}
        {d && tab === "ranking" && <Ranking p={d.prioritization} />}
        {tab === "llm" && <LlmEval />}
        {tab === "online" && <Online />}
        {tab === "usage" && <Usage />}
        {d && tab === "gate" && <Gate g={d.quality_gate} />}
      </TabSet>
    </>
  );
}

function Risk({ m }: { m: any }) {
  const w = m.withdrawal_model, o = m.outcome_model, a = m.ablations;
  if (!w) return <Empty title="No model metrics at this checkpoint">Run the build on the server.</Empty>;
  const t = w.test, pa = w.population_audit;
  return (
    <div className="stack">
      <div className="stats">
        <div className="stat"><div className="v">{t.roc_auc.toFixed(3)}</div><div className="l">ROC-AUC (95% CI {t.roc_auc_95ci.lower.toFixed(3)} to {t.roc_auc_95ci.upper.toFixed(3)})</div></div>
        <div className="stat"><div className="v">{t.pr_auc.toFixed(3)}</div><div className="l">PR-AUC, against a base rate of {t.positive_rate.toFixed(3)}</div></div>
        <div className="stat"><div className="v">{t.ece_10.toFixed(3)}</div><div className="l">calibration error: predicted risk matches observed rates</div></div>
        <div className="stat"><div className="v">{w.baseline_logistic_regression_test.roc_auc.toFixed(3)}</div><div className="l">simple logistic baseline ROC-AUC</div></div>
      </div>
      <div className="grid two section">
        <div className="stack"><h2 className="h3">Predicted versus observed</h2><Reliability rows={w.test_reliability} />
          <p className="small muted">Each dot groups students with similar predicted risk (bigger dot, more students); dots on the dashed line mean the percentages can be taken at face value.</p></div>
        <div className="stack"><h2 className="h3">Do the tiers separate risk?</h2>
          <GroupedBars label="Observed withdrawal rate and share of withdrawals per tier" categories={Object.keys(w.test_tiers)} max={1} format={(v) => fmt.pct(v)}
            series={[{ name: "Observed withdrawal rate", color: COL.ink, values: Object.values(w.test_tiers).map((x: any) => x.observed_withdrawal_rate) },
              { name: "Share of all withdrawals caught", color: COL.median, values: Object.values(w.test_tiers).map((x: any) => x.share_of_all_withdrawals) }]} /></div>
      </div>
      <div className="grid two section">
        <div className="stack"><h2 className="h3">At advisor capacity</h2>
          <table className="data"><thead><tr><th scope="col">Contact the top</th><th scope="col" className="num">Precision</th><th scope="col" className="num">Withdrawals reached</th><th scope="col" className="num">Lift</th></tr></thead>
            <tbody>{w.test_capacity.map((c: any) => <tr key={c.fraction}><td>{fmt.pct(c.fraction)}</td><td className="num">{fmt.pct(c.precision)}</td><td className="num">{fmt.pct(c.recall)}</td><td className="num">{c.lift.toFixed(1)}×</td></tr>)}</tbody></table></div>
        <div className="stack"><h2 className="h3">Why we only score students still enrolled</h2>
          <p>Scored on every enrolment, including students who had already left, the same model would show ROC-AUC {pa.all_enrolments_including_already_withdrawn.roc_auc.toFixed(3)}. {fmt.pct(pa.already_withdrawn_share_of_test_withdrawals)} of the test presentation's withdrawals had already happened by this point, so that number flatters the model. The valid figure is {pa.in_scope_only.roc_auc.toFixed(3)}.</p></div>
      </div>
      {a && <div className="section stack"><h2 className="h3">Which evidence matters</h2>
        <HBars label="ROC-AUC by feature set" max={1} format={(v) => v.toFixed(3)}
          items={Object.entries(a.variants).map(([k, v]: [string, any]) => ({ label: k.replaceAll("_", " "), value: v.roc_auc, color: k.includes("production") ? COL.ink : COL.median, strong: k.includes("production") }))} />
        <p className="small muted">Adding sensitive attributes changes ROC-AUC by {a.cost_of_excluding_sensitive_attributes.roc_auc.toFixed(3)}, so they stay out of the model.</p></div>}
      {o && <div className="section stack"><h2 className="h3">Final-outcome model</h2>
        <p>Macro-F1 {o.test.macro_f1.toFixed(3)} against {o.baselines_test.multinomial_logistic.macro_f1.toFixed(3)} for a logistic baseline and {o.baselines_test.majority_class.macro_f1.toFixed(3)} for always guessing the most common outcome. The combined fail-or-withdraw probability has ROC-AUC {o.poor_outcome_from_p_fail_plus_p_withdrawn_test.roc_auc.toFixed(3)}.</p>
        <table className="data"><thead><tr><th scope="col">Outcome</th><th scope="col" className="num">Precision</th><th scope="col" className="num">Recall</th><th scope="col" className="num">F1</th><th scope="col" className="num">Students</th></tr></thead>
          <tbody>{Object.entries(o.test.per_class).map(([k, v]: [string, any]) => <tr key={k}><td>{k}</td><td className="num">{fmt.pct(v.precision)}</td><td className="num">{fmt.pct(v.recall)}</td><td className="num">{v["f1-score"].toFixed(3)}</td><td className="num">{v.support}</td></tr>)}</tbody></table></div>}
    </div>
  );
}

function Retrieval({ r }: { r: any }) {
  if (!r) return <Empty title="Retrieval benchmark not run">Run python -m ssa.cli evaluate on the server.</Empty>;
  const modes = Object.keys(r.modes);
  return (
    <div className="stack">
      <p>{r.queries} students from {r.query_presentation} were matched against earlier presentations of the same module. A good matcher finds past students whose real outcomes predict this student's outcome.</p>
      <GroupedBars label="Retrieval quality by matching method" categories={modes} max={1} format={(v) => v.toFixed(2)}
        series={[{ name: "Neighbour-vote AUC (fail or withdraw)", color: COL.ink, values: modes.map((m) => r.modes[m].neighbour_vote_auc_poor_outcome) },
          { name: "Outcome agreement at 10", color: "var(--ink-2)", values: modes.map((m) => r.modes[m].outcome_agreement_at_10) },
          { name: "nDCG at 10 (true ideal)", color: COL.median, values: modes.map((m) => r.modes[m].reference_ndcg_at_10_global_ideal) }]} />
      <ul className="small muted">{r.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}</ul>
    </div>
  );
}

function Ranking({ p }: { p: any }) {
  if (!p) return <Empty title="Prioritisation not evaluated">Run the build on the server.</Empty>;
  const names: Record<string, string> = { learned_weights: "Learned weights", v4_hand_formula: "Previous formula", only_p_poor: "Fail-or-withdraw only",
    only_p_withdraw: "Withdrawal risk only", only_missed_work: "Missed work only", only_engagement_gap: "Activity gap only", random: "Random order" };
  const rows = Object.entries(names).filter(([k]) => p.test_comparison[k]);
  return (
    <div className="stack">
      <p>Share of contacted students who went on to fail or withdraw, by how the list was ordered. Weights were chosen on {p.split.validation?.join(", ")} and tested on {p.split.test?.join(", ")}.</p>
      <GroupedBars label="Precision of each ordering at three capacities" categories={rows.map(([, l]) => l)} max={1} format={(v) => fmt.pct(v)} height={280}
        series={[0, 1, 2].map((i) => ({ name: `Top ${[10, 20, 30][i]}%`, color: [COL.ink, "var(--ink-2)", COL.median][i], values: rows.map(([k]) => p.test_comparison[k].at_capacity[i].precision) }))} />
    </div>
  );
}

function Block({ title, status, children }: { title: string; status?: string; children: ReactNode }) {
  return <div className="section stack"><div className="row"><h2 className="h3">{title}</h2>{status && <span className="small muted">{status}</span>}</div>{children}</div>;
}

function LlmEval() {
  const q = useQuery({ queryKey: ["llm-eval"], queryFn: ({ signal }) => api("/evaluations/llm", { signal }) });
  if (q.error) return <ErrorNote error={q.error as ApiError} />;
  if (!q.data) return <Loading />;
  const e: any = q.data;
  if (!e.agents) return <Empty title="AI evaluation not run yet">Run python -m ssa.cli llm-eval on the server.</Empty>;
  const ag = e.agents;
  return (
    <div className="stack">
      <p className="muted">Run {e.generated_at}. {e.models ? `Models: ${e.models.fast} (search, outreach, themes), ${e.models.main} (agents, copilot, synthesis, briefing), judged by ${e.models.judge}.`
        : `Generator ${e.generator_model || "not connected"}, judge ${e.judge_model || "not connected"}.`}</p>
      <Block title="Support plans" status={`${ag.deterministic.n_cases} students across all tiers`}>
        <table className="data"><tbody>
          <tr><td>Plans passing every guardrail</td><td className="num">{rate(ag.deterministic.final_plan_valid_rate)}</td></tr>
          <tr><td>Plans drafted by AI (the rest were routed to monitoring and use the computed plan)</td><td className="num">{rate(ag.deterministic.llm_plan_accepted_rate)}</td></tr>
          <tr><td>Drafts that needed a revision after the critic</td><td className="num">{rate(ag.deterministic.revision_rate)}</td></tr>
          <tr><td>High-tier students always routed to an intervention</td><td className="num">{rate(ag.deterministic.tier_routing_consistency)}</td></tr>
          <tr><td>Judge pass rate</td><td className="num">{rate(ag.judge.pass_rate)}</td></tr>
          <tr><td>AI plan preferred over computed plan (order-swapped A/B)</td><td className="num">{rate(ag.pairwise_llm_vs_deterministic.llm_win_rate)}</td></tr>
        </tbody></table>
        {ag.judge.mean_scores && <HBars label="Mean judge scores out of 10" max={10} format={(v) => v.toFixed(1)}
          items={Object.entries(ag.judge.mean_scores).map(([k, v]) => ({ label: k.replaceAll("_", " "), value: v as number }))} />}
        {ag.deepeval.aggregate ? <table className="data"><thead><tr><th scope="col">DeepEval metric</th><th scope="col" className="num">Mean (95% CI)</th><th scope="col" className="num">Pass rate</th></tr></thead>
          <tbody>{Object.entries(ag.deepeval.aggregate).map(([k, v]: [string, any]) => <tr key={k}><td>{k}</td>
            <td className="num">{v.mean.toFixed(2)} ({v.ci95[0].toFixed(2)} to {v.ci95[1].toFixed(2)})</td><td className="num">{rate(v.pass_rate)}</td></tr>)}</tbody></table>
          : <p className="small muted">DeepEval: {ag.deepeval.status}</p>}
      </Block>
      <Block title="Understanding search requests" status={e.search.note}>
        <table className="data"><thead><tr><th scope="col">Set</th><th scope="col">Method</th><th scope="col" className="num">Filter F1</th><th scope="col" className="num">Exact match</th></tr></thead>
          <tbody>{["templated", "paraphrased"].flatMap((set) => ["rules", "llm"].map((mth) => {
            const r = e.search[`${set}_${mth}`] || {};
            return <tr key={set + mth}><td>{set}</td><td>{mth === "llm" ? "AI" : "Rules"}</td><td className="num">{r.filter_f1 != null ? r.filter_f1.toFixed(2) : "not run"}</td><td className="num">{rate(r.exact_match)}</td></tr>;
          }))}</tbody></table>
      </Block>
      <Block title="Copilot" status="Golden questions with known answers">
        <table className="data"><thead><tr><th scope="col">Mode</th><th scope="col" className="num">Correct answers</th><th scope="col" className="num">Right tool</th><th scope="col" className="num">Grounded and safe</th></tr></thead>
          <tbody>{["rules", "llm"].map((k) => { const r = e.copilot[k] || {}; return <tr key={k}><td>{k === "llm" ? "AI" : "Rules"}</td><td className="num">{rate(r.answer_correct_rate)}</td><td className="num">{rate(r.tool_selection_accuracy)}</td><td className="num">{rate(r.grounded_and_safe_rate)}</td></tr>; })}</tbody></table>
        <p>Adversarial prompts handled safely: {rate(e.robustness.safe_rate)} of {e.robustness.n} ({e.robustness.mode === "llm" ? "AI" : "rules"}).</p>
        <details><summary>See adversarial prompts</summary><ul className="small">{e.robustness.cases.map((c: any, i: number) => <li key={i}><strong>{c.safe ? "Handled safely" : "Flagged"}</strong>: {c.prompt}</li>)}</ul></details>
      </Block>
      <Block title="Other generated text">
        <table className="data"><tbody>
          <tr><td>“What happened next” summaries accepted by the code checks</td><td className="num">{rate(e.synthesis.llm_accepted_rate)}</td></tr>
          <tr><td>Judge faithfulness of those summaries (out of 10)</td><td className="num">{e.synthesis.judge_mean_faithfulness?.toFixed(1) ?? "not run"}</td></tr>
          <tr><td>Cohort briefings accepted by the code checks</td><td className="num">{rate(e.briefing.llm_accepted_rate)}</td></tr>
          <tr><td>Outreach rewrites within length</td><td className="num">{rate(e.outreach.length_compliance)}</td></tr>
          <tr><td>Outreach rewrites passing safety checks</td><td className="num">{rate(e.outreach.safe_rate)}</td></tr>
          <tr><td>Requested language delivered</td><td className="num">{rate(e.outreach.language_honoured_rate)}</td></tr>
        </tbody></table>
        <p className="small muted">Judge agreement with advisors: {e.human_agreement.status || `kappa ${e.human_agreement.verdict_cohen_kappa?.toFixed(2)}`}</p>
      </Block>
    </div>
  );
}

function Online() {
  const q = useQuery({ queryKey: ["online-evals"], queryFn: ({ signal }) => api("/evaluations/online", { signal }) });
  if (q.error) return <ErrorNote error={q.error as ApiError} />;
  if (!q.data) return <Loading />;
  const d: any = q.data;
  if (!d.items.length) return <Empty title="No live reviews yet">Ask the AI judge to review a plan from a student page, or set ONLINE_EVAL_SAMPLE_RATE on the server to review a sample of live requests automatically.</Empty>;
  return (
    <div className="stack">
      <table className="data"><thead><tr><th scope="col">Kind</th><th scope="col" className="num">Reviews</th><th scope="col" className="num">Mean score</th><th scope="col" className="num">Pass rate</th></tr></thead>
        <tbody>{d.summary.by_kind.map((k: any) => <tr key={k.kind}><td>{k.kind}</td><td className="num">{k.n}</td><td className="num">{fmt.num(k.mean_overall, 1)}</td><td className="num">{fmt.pct(k.pass_rate)}</td></tr>)}</tbody></table>
      <table className="data"><thead><tr><th scope="col">When</th><th scope="col">Kind</th><th scope="col">Reference</th><th scope="col">Verdict</th><th scope="col" className="num">Score</th><th scope="col">Judge</th></tr></thead>
        <tbody>{d.items.map((i: any) => <tr key={i.id}><td>{new Date(i.ts * 1000).toLocaleString("en-GB")}</td><td>{i.kind}</td><td className="small">{i.ref_id}</td><td>{i.verdict}</td><td className="num">{fmt.num(i.overall, 1)}</td><td className="small">{i.judge_model}</td></tr>)}</tbody></table>
    </div>
  );
}

function Usage() {
  const [hours, setHours] = useState(24 * 7);
  const q = useQuery({ queryKey: ["usage", hours], queryFn: ({ signal }) => api("/llm/usage", { query: { since_hours: hours }, signal }) });
  const d: any = q.data;
  return (
    <div className="stack">
      <label className="field" style={{ maxWidth: 220 }}>Period<select value={hours} onChange={(e) => setHours(Number(e.target.value))}>
        <option value={24}>Last day</option><option value={168}>Last week</option><option value={720}>Last 30 days</option></select></label>
      {q.error && <ErrorNote error={q.error as ApiError} />}
      {!d && !q.error && <Loading />}
      {d && (d.total.n === 0 ? <Empty title="No AI calls in this period">Calls appear here once the AI is connected and used.</Empty> : <>
        <div className="stats">
          <div className="stat"><div className="v">{fmt.int(d.total.n)}</div><div className="l">calls ({fmt.int(d.total.cached)} answered from cache, {fmt.int(d.total.errors)} errors)</div></div>
          <div className="stat"><div className="v">{fmt.int(d.total.pt + (d.total.crt || 0) + (d.total.cwt || 0))}</div><div className="l">input tokens ({fmt.int(d.total.crt || 0)} read from the prompt cache)</div></div>
          <div className="stat"><div className="v">{fmt.int(d.total.ct)}</div><div className="l">output tokens (thinking included)</div></div>
          <div className="stat"><div className="v">${d.total.estimated_cost_usd.toFixed(2)}</div><div className="l">estimated cost</div></div>
        </div>
        <div className="table-wrap"><table className="data"><thead><tr><th scope="col">Feature</th><th scope="col">Model</th><th scope="col" className="num">Calls</th>
          <th scope="col" className="num">Input</th><th scope="col" className="num">Cache read</th><th scope="col" className="num">Output</th><th scope="col" className="num">Avg latency</th><th scope="col" className="num">Cost</th></tr></thead>
          <tbody>{d.by_feature.map((r: any) => <tr key={r.feature + r.model}><td>{r.feature.replaceAll("_", " ")}</td><td className="small">{r.model}</td><td className="num">{fmt.int(r.n)}</td>
            <td className="num">{fmt.int(r.pt)}</td><td className="num">{fmt.int(r.crt || 0)}</td><td className="num">{fmt.int(r.ct)}</td><td className="num">{r.avg_latency_s}s</td><td className="num">${r.estimated_cost_usd.toFixed(3)}</td></tr>)}</tbody></table></div>
        {d.total.unpriced > 0 && <p className="note warn">{d.total.unpriced} calls used a model without a price entry; add it with LLM_PRICE_TABLE on the server.</p>}
        {d.prices_usd_per_million_tokens && <details><summary>Prices used (USD per million tokens)</summary>
          <table className="data small"><thead><tr><th scope="col">Model</th><th scope="col" className="num">Input</th><th scope="col" className="num">Cache write</th><th scope="col" className="num">Cache read</th><th scope="col" className="num">Output</th></tr></thead>
            <tbody>{Object.entries(d.prices_usd_per_million_tokens).map(([k, p]: [string, any]) => <tr key={k}><td>{k}</td><td className="num">{p.input}</td><td className="num">{p.cache_write ?? "–"}</td><td className="num">{p.cache_read ?? "–"}</td><td className="num">{p.output}</td></tr>)}</tbody></table></details>}
      </>)}
    </div>
  );
}

function Gate({ g }: { g: any }) {
  if (!g) return <Empty title="Release checks not run">Run python -m ssa.cli quality-gate on the server.</Empty>;
  return (
    <div className="stack">
      <p className="row"><strong>{g.passed ? "All release checks pass" : "Some release checks fail"}</strong> <span className="muted">({g.profile} data, {g.generated_at})</span></p>
      <div className="table-wrap"><table className="data"><tbody>{g.checks.map((c: any) => <tr key={c.check}>
        <td style={{ whiteSpace: "nowrap" }}>{c.passed ? <><CheckCircle2 aria-hidden="true" size={16} /> Pass</> : <strong><CircleAlert aria-hidden="true" size={16} /> Fail</strong>}</td>
        <td>{c.check.replaceAll("_", " ")}</td><td className="small muted wrap">{typeof c.detail === "object" ? JSON.stringify(c.detail) : String(c.detail ?? "")}</td></tr>)}</tbody></table></div>
      {g.warnings.length > 0 && <><h3 className="h4">Warnings to review</h3><div className="table-wrap"><table className="data"><tbody>{g.warnings.map((c: any) => <tr key={c.check}>
        <td style={{ whiteSpace: "nowrap" }}><TriangleAlert aria-hidden="true" size={16} /> Review</td><td>{c.check.replaceAll("_", " ")}</td><td className="small muted wrap">{JSON.stringify(c.detail)}</td></tr>)}</tbody></table></div></>}
    </div>
  );
}
