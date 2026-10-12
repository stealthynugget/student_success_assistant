import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, fmt, type ApiError } from "../api";
import { useStore } from "../store";
import { ErrorNote, Loading, TabSet } from "../ui";
import { COL, GroupedBars, HBars, SeriesLines } from "../viz";

/* eslint-disable @typescript-eslint/no-explicit-any */
const useGet = (key: unknown[], path: string, query?: Record<string, any>) =>
  useQuery({ queryKey: key, queryFn: ({ signal }) => api(path, { query, signal }) });

export default function Insights() {
  const [tab, setTab] = useState("signals");
  return (
    <>
      <div className="page-head"><div><h1>Insights</h1><div className="sub">What the historical records say about who struggles, when, and which assessments matter. All computed; no AI text on this page.</div></div></div>
      <TabSet label="Insights" value={tab} onChange={setTab} tabs={[["signals", "Signals"], ["engagement", "Engagement over time"], ["assessments", "Assessments"],
        ["weeks", "Early warning by week"], ["fairness", "Fairness"]]}>
        {tab === "signals" && <Signals />}
        {tab === "engagement" && <Engagement />}
        {tab === "assessments" && <Assessments />}
        {tab === "weeks" && <Weeks />}
        {tab === "fairness" && <Fairness />}
      </TabSet>
    </>
  );
}

function Signals() {
  const q = useGet(["factors"], "/insights/factors");
  const [sel, setSel] = useState(0);
  if (q.error) return <ErrorNote error={q.error as ApiError} />;
  if (!q.data) return <Loading lines={6} />;
  const d: any = q.data;
  const f = d.numeric_factors_ranked;
  const cur = f[sel];
  return (
    <div className="grid two">
      <div className="stack">
        <h2 className="h3">Biggest gaps in fail-or-withdraw rates</h2>
        <p className="small muted">Difference between the best and worst fifth of students on each signal, among {fmt.int(d.n)} historical students still enrolled at the checkpoint (base rate {fmt.pct(d.base_rates.poor_outcome)}). Select a signal to see its fifths.</p>
        <div className="table-wrap"><table className="data">
          <thead><tr><th scope="col">Signal</th><th scope="col" className="num">Rate gap</th><th scope="col" className="num">Rank correlation</th></tr></thead>
          <tbody>{f.slice(0, 14).map((x: any, i: number) => (
            <tr key={x.feature} className="click" onClick={() => setSel(i)} style={i === sel ? { background: "var(--marker-soft)" } : {}}>
              <td><button className="ghost" style={{ padding: 0, textDecoration: "none", fontWeight: i === sel ? 700 : 400 }} aria-pressed={i === sel} onClick={() => setSel(i)}>{x.label}</button></td>
              <td className="num">{fmt.pct(x.poor_outcome_rate_spread)}</td><td className="num small muted">{x.spearman_with_poor_outcome.toFixed(2)}</td></tr>))}</tbody></table></div>
      </div>
      <div className="stack" aria-live="polite">
        <h2 className="h3">{cur.label}</h2>
        <GroupedBars label={`Fail-or-withdraw and withdrawal rates by fifth of ${cur.label}`} categories={cur.quintiles.map((x: any) => `Q${x.quintile}`)} max={1}
          format={(v) => fmt.pct(v)} series={[{ name: "Fail or withdraw", color: COL.Fail, values: cur.quintiles.map((x: any) => x.poor_outcome_rate) },
            { name: "Withdraw", color: COL.Withdrawn, values: cur.quintiles.map((x: any) => x.withdrawal_rate) }]} />
        <table className="data small"><thead><tr><th scope="col">Fifth</th><th scope="col">Range</th><th scope="col" className="num">Students</th><th scope="col" className="num">Fail or withdraw (95% CI)</th></tr></thead>
          <tbody>{cur.quintiles.map((x: any) => <tr key={x.quintile}><td>Q{x.quintile}</td><td>{fmt.num(x.range[0], 2)} to {fmt.num(x.range[1], 2)}</td><td className="num">{x.n}</td>
            <td className="num">{fmt.pct(x.poor_outcome_rate)} ({fmt.pct(x.poor_outcome_ci[0])} to {fmt.pct(x.poor_outcome_ci[1])})</td></tr>)}</tbody></table>
        <p className="small muted">{d.caveat}</p>
      </div>
    </div>
  );
}

function Engagement() {
  const { module } = useStore();
  const q = useGet(["course-trends", module], "/trends/course", { code_module: module });
  if (q.error) return <ErrorNote error={q.error as ApiError} />;
  if (!q.data) return <Loading />;
  const d: any = q.data;
  if (!d.weekly_mean_clicks_by_outcome) return <p className="muted">No trend data for {module}.</p>;
  const order = ["Distinction", "Pass", "Fail", "Withdrawn"].filter((k) => d.weekly_mean_clicks_by_outcome[k]);
  return (
    <div className="stack">
      <h2 className="h3">Weekly activity by final outcome in {module}</h2>
      <SeriesLines label="Weekly activity by final outcome" yLabel="mean clicks per student"
        series={order.map((k) => ({ key: k, name: k, values: d.weekly_mean_clicks_by_outcome[k], color: COL[k] }))}
        markers={Object.entries(d.first_week_below_60pct_of_pass).filter(([, w]) => w != null).map(([k, w]) => ({ x: w as number, label: `${k} diverge`, color: COL[k] }))} />
      <p className="muted">Students who eventually withdrew fall below 60% of passing students' activity from week {d.first_week_below_60pct_of_pass.Withdrawn ?? "-"}; those who eventually failed from week {d.first_week_below_60pct_of_pass.Fail ?? "-"}. Students who had already left at the checkpoint are excluded.</p>
    </div>
  );
}

function Assessments() {
  const { module, presentation } = useStore();
  const q = useGet(["assessments", module, presentation], "/insights/assessments", { code_module: module, code_presentation: presentation });
  if (q.error) return <ErrorNote error={q.error as ApiError} />;
  if (!q.data) return <Loading />;
  const d: any = q.data;
  const rows = d.assessments.filter((a: any) => a.assessment_type !== "Exam").sort((a: any, b: any) => (a.deadline_day ?? 999) - (b.deadline_day ?? 999));
  return (
    <div className="grid two">
      <div className="stack">
        <h2 className="h3">How strongly missing each assessment marks later failure or withdrawal</h2>
        <HBars label="Non-submission AUC per assessment" max={1} format={(v) => v.toFixed(2)}
          items={rows.map((a: any) => ({ label: `${a.assessment_type} week ${a.deadline_week}`, value: a.non_submission_auc_for_poor_outcome || 0.5,
            color: a.assessment_type === "TMA" ? COL.ink : COL.median }))} />
        <p className="small muted">{d.note}</p>
      </div>
      <div className="table-wrap"><table className="data">
        <thead><tr><th scope="col">Assessment</th><th scope="col" className="num">Week</th><th scope="col" className="num">Weight</th><th scope="col" className="num">Submitted</th>
          <th scope="col" className="num">Mean score</th><th scope="col" className="num">Late</th><th scope="col" className="num">Score AUC</th></tr></thead>
        <tbody>{rows.map((a: any) => <tr key={a.id_assessment}><td>{a.assessment_type} {a.id_assessment}</td><td className="num">{a.deadline_week}</td><td className="num">{a.weight}</td>
          <td className="num">{fmt.pct(a.submission_rate)}</td><td className="num">{fmt.num(a.mean_score, 0)}</td><td className="num">{fmt.pct(a.late_rate)}</td>
          <td className="num">{a.score_auc_for_poor_outcome ? a.score_auc_for_poor_outcome.toFixed(2) : "-"}</td></tr>)}</tbody></table></div>
    </div>
  );
}

function Weeks() {
  const q = useGet(["checkpoint-metrics"], "/insights/checkpoints");
  if (q.error) return <ErrorNote error={q.error as ApiError} />;
  if (!q.data) return <Loading />;
  const cps = (q.data as any).checkpoints;
  return (
    <div className="grid two">
      <div className="stack">
        <h2 className="h3">How early can we tell?</h2>
        <GroupedBars label="Withdrawal model quality at each checkpoint" categories={cps.map((c: any) => `week ${Math.round(c.approx_week)}`)} max={1} format={(v) => v.toFixed(2)}
          series={[{ name: "ROC-AUC", color: COL.ink, values: cps.map((c: any) => c.withdrawal.roc_auc) },
            { name: "PR-AUC", color: "var(--ink-2)", values: cps.map((c: any) => c.withdrawal.pr_auc) },
            { name: "Share who later withdraw", color: COL.median, values: cps.map((c: any) => c.withdrawal.positive_rate) }]} />
      </div>
      <div className="table-wrap"><table className="data"><thead><tr><th scope="col">Week</th><th scope="col" className="num">Still enrolled</th><th scope="col" className="num">ROC-AUC</th>
        <th scope="col" className="num">Calibration error</th><th scope="col" className="num">Outcome macro-F1</th><th scope="col" className="num">Precision at 10%</th></tr></thead>
        <tbody>{cps.map((c: any) => <tr key={c.tag}><td>{Math.round(c.approx_week)}</td><td className="num">{fmt.int(c.in_scope)}</td><td className="num">{c.withdrawal.roc_auc.toFixed(3)}</td>
          <td className="num">{c.withdrawal.ece_10.toFixed(3)}</td><td className="num">{c.outcome_macro_f1.toFixed(3)}</td><td className="num">{fmt.pct(c.prioritization_precision_at_10)}</td></tr>)}</tbody></table>
        <p className="small muted">Later checkpoints see more evidence, but fewer students remain to help. Each row is scored on the untouched 2014J presentation.</p></div>
    </div>
  );
}

function ThresholdReview({ review }: { review: any }) {
  return (
    <>
      <h3 className="h4">Threshold review</h3>
      <p className="small muted">The High-tier cut-off each group would need for {fmt.pct(review.target_tpr)} of its withdrawals to be caught (the overall rate),
        against today's single cut-off of {fmt.pct(review.current_threshold, 1)}. For review with the equity office only: group-specific cut-offs would use a sensitive attribute and are not applied.</p>
      <div className="table-wrap"><table className="data small"><thead><tr><th scope="col">Group</th><th scope="col" className="num">Caught now</th>
        <th scope="col" className="num">Cut-off needed</th><th scope="col" className="num">Flagged now → then</th><th scope="col" className="num">Precision now → then</th>
        <th scope="col" className="num">Change in students flagged</th></tr></thead>
        <tbody>{Object.entries(review.groups).map(([g, r]: [string, any]) => <tr key={g}><td>{g}</td><td className="num">{fmt.pct(r.current_tpr)}</td>
          <td className="num">{fmt.pct(r.equalising_threshold, 1)}</td><td className="num">{fmt.pct(r.current_selection_rate)} → {fmt.pct(r.selection_rate_at_equalising)}</td>
          <td className="num">{fmt.pct(r.current_ppv)} → {fmt.pct(r.ppv_at_equalising)}</td>
          <td className="num">{r.change_in_flagged_students > 0 ? "+" : ""}{r.change_in_flagged_students}</td></tr>)}</tbody></table></div>
    </>
  );
}

function Fairness() {
  const q = useGet(["metrics", null], "/metrics");
  if (q.error) return <ErrorNote error={q.error as ApiError} />;
  if (!q.data) return <Loading />;
  const f = (q.data as any).fairness;
  if (!f) return <p className="muted">No fairness audit yet.</p>;
  const attrs = Object.entries(f.attributes) as [string, any][];
  return (
    <div className="stack">
      <p>A descriptive audit of the High tier on the test presentation. Each attribute's gap is set against the gap that random group labels
        would give at the same group sizes, so sampling noise is not mistaken for a group difference.</p>
      <HBars label="Equal-opportunity gap by attribute" max={Math.max(0.3, ...attrs.map(([, v]) => v.equal_opportunity_gap || 0))} format={(v) => v.toFixed(2)}
        items={attrs.map(([k, v]) => { const flag = (v.equal_opportunity_gap || 0) > 0.15; // tier colours mean tier only: flag in words
          const chance = flag && v.gap_null && v.gap_null.p_value >= 0.05;
          return { label: `${k.replaceAll("_", " ")}${flag ? (chance ? " (above 0.15, within chance)" : " (review)") : ""}`, value: v.equal_opportunity_gap || 0,
            color: flag && !chance ? COL.ink : COL.median, strong: flag && !chance }; })} />
      <p className="small muted">Gap in the share of eventual withdrawals caught by the High tier, between the best- and worst-served groups; gaps above 0.15 that chance does not explain are marked "review" and drawn darker.
        The model does not use these attributes; gaps arise through correlated signals and differing base rates. Groups shown in grey ({f.sufficient_sample_rule || "too few withdrawals"} not met) are too small to compare reliably.</p>
      {attrs.map(([k, v]) => (
        <details key={k}><summary>{k.replaceAll("_", " ")}: gap {v.equal_opportunity_gap == null ? "not computed" : v.equal_opportunity_gap.toFixed(2)}
          {v.gap_null ? `, chance alone gives about ${v.gap_null.median.toFixed(2)} (p ${v.gap_null.p_value.toFixed(3)})` : ""}</summary>
          <div className="stack" style={{ marginTop: 8 }}>
            {v.verdict && <p className="small">{v.verdict[0].toUpperCase() + v.verdict.slice(1)}.</p>}
            <div className="table-wrap"><table className="data small"><thead><tr><th scope="col">Group</th><th scope="col" className="num">Students</th><th scope="col" className="num">Withdrew</th>
              <th scope="col" className="num">Predicted</th><th scope="col" className="num">Flagged High</th><th scope="col" className="num">Caught</th>
              <th scope="col" className="num">Caught, 95% interval</th><th scope="col" className="num">Precision</th></tr></thead>
              <tbody>{Object.entries(v.groups).map(([g, x]: [string, any]) => <tr key={g} style={x.sufficient_sample ? {} : { color: "var(--muted)" }}><td>{g}</td><td className="num">{x.n}</td>
                <td className="num">{fmt.pct(x.observed_rate)}</td><td className="num">{fmt.pct(x.mean_predicted)}</td><td className="num">{fmt.pct(x.selection_rate_high)}</td>
                <td className="num">{fmt.pct(x.tpr_high)}</td><td className="num">{x.tpr_high_ci95 ? `${fmt.pct(x.tpr_high_ci95[0])} to ${fmt.pct(x.tpr_high_ci95[1])}` : "-"}</td>
                <td className="num">{fmt.pct(x.ppv_high)}</td></tr>)}</tbody></table></div>
            {v.threshold_review && v.gap_null?.p_value < 0.05 && <ThresholdReview review={v.threshold_review} />}
          </div></details>
      ))}
    </div>
  );
}
