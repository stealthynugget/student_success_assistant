import React, { useState } from "react";
import { api, fmt } from "../api";
import { C, GroupedBars, HBars, LineChart } from "../charts.jsx";
import { useApi, useStore } from "../store";
import { ErrorNote, Loading, Tabs } from "../ui";

export default function Insights() {
  const [tab, setTab] = useState("signals");
  return (
    <>
      <div className="page-head"><div><h1>Insights</h1><div className="sub">What the historical records say about who struggles, when, and which assessments matter.</div></div>
        <div className="actions"><Tabs value={tab} onChange={setTab} tabs={[["signals", "Signals"], ["engagement", "Engagement over time"], ["assessments", "Assessments"],
          ["weeks", "Early warning by week"], ["fairness", "Fairness"]]} /></div></div>
      {tab === "signals" && <Signals />}
      {tab === "engagement" && <Engagement />}
      {tab === "assessments" && <Assessments />}
      {tab === "weeks" && <Weeks />}
      {tab === "fairness" && <Fairness />}
    </>
  );
}

function Signals() {
  const q = useApi((s) => api("/insights/factors", { signal: s }), []);
  const [sel, setSel] = useState(0);
  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data) return <Loading lines={6} />;
  const f = q.data.numeric_factors_ranked;
  const cur = f[sel];
  return (
    <div className="grid two">
      <div className="stack">
        <h3>Biggest gaps in fail-or-withdraw rates</h3>
        <p className="small muted">Difference between the best and worst fifth of students on each signal, among {q.data.n.toLocaleString()} historical students still enrolled at the checkpoint (base rate {fmt.pct(q.data.base_rates.poor_outcome)}).</p>
        <div className="table-wrap"><table className="data"><tbody>{f.slice(0, 14).map((x, i) => (
          <tr key={x.feature} className="click" onClick={() => setSel(i)} style={i === sel ? { background: "#fbe9a6" } : {}}>
            <td>{x.label}</td><td className="num">{fmt.pct(x.poor_outcome_rate_spread)}</td><td className="num small muted">ρ {x.spearman_with_poor_outcome.toFixed(2)}</td></tr>))}</tbody></table></div>
      </div>
      <div className="stack">
        <h3>{cur.label}</h3>
        <GroupedBars categories={cur.quintiles.map((qq) => `Q${qq.quintile}`)} max={1} format={(v) => fmt.pct(v)}
          series={[{ name: "Fail or withdraw", color: C.Fail, values: cur.quintiles.map((qq) => qq.poor_outcome_rate) },
            { name: "Withdraw", color: C.Withdrawn, values: cur.quintiles.map((qq) => qq.withdrawal_rate) }]} />
        <div className="legend"><span><i style={{ background: C.Fail }} />fail or withdraw</span><span><i style={{ background: C.Withdrawn }} />withdraw</span></div>
        <table className="data small"><thead><tr><th>Fifth</th><th>Range</th><th className="num">Students</th><th className="num">Fail or withdraw (95% CI)</th></tr></thead>
          <tbody>{cur.quintiles.map((qq) => <tr key={qq.quintile}><td>Q{qq.quintile}</td><td>{fmt.num(qq.range[0], 2)} to {fmt.num(qq.range[1], 2)}</td><td className="num">{qq.n}</td>
            <td className="num">{fmt.pct(qq.poor_outcome_rate)} ({fmt.pct(qq.poor_outcome_ci[0])}–{fmt.pct(qq.poor_outcome_ci[1])})</td></tr>)}</tbody></table>
        <p className="small muted">{q.data.caveat}</p>
      </div>
    </div>
  );
}

function Engagement() {
  const { module } = useStore();
  const q = useApi((s) => api("/trends/course", { query: { code_module: module }, signal: s }), [module]);
  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data) return <Loading />;
  const d = q.data;
  if (!d.weekly_mean_clicks_by_outcome) return <p className="muted">No trend data for {module}.</p>;
  const order = ["Distinction", "Pass", "Fail", "Withdrawn"].filter((k) => d.weekly_mean_clicks_by_outcome[k]);
  return (
    <div className="stack">
      <h3>Weekly activity by final outcome in {module}</h3>
      <LineChart height={280} ariaLabel="Weekly activity by final outcome" xLabel="course week" yLabel="mean clicks per student"
        series={order.map((k) => ({ name: k, values: d.weekly_mean_clicks_by_outcome[k], color: C[k], width: 2.2 }))}
        vlines={Object.entries(d.first_week_below_60pct_of_pass).filter(([, w]) => w != null).map(([k, w]) => ({ x: w, label: `${k} diverge`, color: C[k] }))} />
      <div className="legend">{order.map((k) => <span key={k}><i style={{ background: C[k] }} />{k}</span>)}</div>
      <p className="muted">Students who eventually withdrew fall below 60% of passing students' activity from week {d.first_week_below_60pct_of_pass.Withdrawn ?? "–"}; those who eventually failed from week {d.first_week_below_60pct_of_pass.Fail ?? "–"}. Students who had already left at the checkpoint are excluded.</p>
    </div>
  );
}

function Assessments() {
  const { module, presentation } = useStore();
  const q = useApi((s) => api("/insights/assessments", { query: { code_module: module, code_presentation: presentation }, signal: s }), [module, presentation]);
  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data) return <Loading />;
  const rows = q.data.assessments.filter((a) => a.assessment_type !== "Exam").sort((a, b) => (a.deadline_day ?? 999) - (b.deadline_day ?? 999));
  return (
    <div className="grid two">
      <div className="stack">
        <h3>How strongly missing each assessment marks later failure or withdrawal</h3>
        <HBars items={rows.map((a) => ({ label: `${a.assessment_type} wk ${a.deadline_week}`, value: a.non_submission_auc_for_poor_outcome || 0.5, color: a.assessment_type === "TMA" ? C.ink : "#7d8798" }))}
          max={1} format={(v) => v.toFixed(2)} />
        <p className="small muted">{q.data.note}</p>
      </div>
      <div className="table-wrap"><table className="data">
        <thead><tr><th>Assessment</th><th className="num">Week</th><th className="num">Weight</th><th className="num">Submitted</th><th className="num">Mean score</th><th className="num">Late</th><th className="num">Score AUC</th></tr></thead>
        <tbody>{rows.map((a) => <tr key={a.id_assessment}><td>{a.assessment_type} {a.id_assessment}</td><td className="num">{a.deadline_week}</td><td className="num">{a.weight}</td>
          <td className="num">{fmt.pct(a.submission_rate)}</td><td className="num">{fmt.num(a.mean_score, 0)}</td><td className="num">{fmt.pct(a.late_rate)}</td>
          <td className="num">{a.score_auc_for_poor_outcome ? a.score_auc_for_poor_outcome.toFixed(2) : "–"}</td></tr>)}</tbody></table></div>
    </div>
  );
}

function Weeks() {
  const q = useApi((s) => api("/insights/checkpoints", { signal: s }), []);
  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data) return <Loading />;
  const cps = q.data.checkpoints;
  return (
    <div className="grid two">
      <div className="stack">
        <h3>How early can we tell?</h3>
        <GroupedBars categories={cps.map((c) => `wk ${Math.round(c.approx_week)}`)} max={1} format={(v) => v.toFixed(2)}
          series={[{ name: "ROC-AUC", color: C.ink, values: cps.map((c) => c.withdrawal.roc_auc) }, { name: "PR-AUC", color: C.ai, values: cps.map((c) => c.withdrawal.pr_auc) },
            { name: "Base rate", color: "#9aa2b1", values: cps.map((c) => c.withdrawal.positive_rate) }]} />
        <div className="legend"><span><i style={{ background: C.ink }} />ROC-AUC</span><span><i style={{ background: C.ai }} />PR-AUC</span><span><i style={{ background: "#9aa2b1" }} />share who later withdraw</span></div>
      </div>
      <div className="table-wrap"><table className="data"><thead><tr><th>Week</th><th className="num">Still enrolled</th><th className="num">ROC-AUC</th><th className="num">Calibration error</th><th className="num">Outcome macro-F1</th><th className="num">Precision at 10%</th></tr></thead>
        <tbody>{cps.map((c) => <tr key={c.tag}><td>{Math.round(c.approx_week)}</td><td className="num">{fmt.int(c.in_scope)}</td><td className="num">{c.withdrawal.roc_auc.toFixed(3)}</td>
          <td className="num">{c.withdrawal.ece_10.toFixed(3)}</td><td className="num">{c.outcome_macro_f1.toFixed(3)}</td><td className="num">{fmt.pct(c.prioritization_precision_at_10)}</td></tr>)}</tbody></table>
        <p className="small muted">Later checkpoints see more evidence, but fewer students remain to help. Each row is scored on the untouched 2014J presentation.</p></div>
    </div>
  );
}

function Fairness() {
  const q = useApi((s) => api("/metrics", { signal: s }), []);
  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data) return <Loading />;
  const f = q.data.fairness;
  if (!f) return <p className="muted">No fairness audit yet.</p>;
  const attrs = Object.entries(f.attributes);
  return (
    <div className="stack">
      <p>{f.interpretation}</p>
      <HBars items={attrs.map(([k, v]) => ({ label: k.replaceAll("_", " "), value: v.equal_opportunity_gap || 0, color: (v.equal_opportunity_gap || 0) > 0.15 ? C.High : C.ink }))}
        max={Math.max(0.3, ...attrs.map(([, v]) => v.equal_opportunity_gap || 0))} format={(v) => v.toFixed(2)} />
      <p className="small muted">Gap in the share of eventual withdrawals caught by the High tier, between the best- and worst-served groups. Red bars exceed 0.15. The model does not use these attributes ({String(!f.model_uses_sensitive_attributes)}); gaps arise through correlated signals and differing base rates.</p>
      {attrs.map(([k, v]) => (
        <details key={k}><summary>{k.replaceAll("_", " ")} groups</summary>
          <table className="data small"><thead><tr><th>Group</th><th className="num">Students</th><th className="num">Withdrew</th><th className="num">Flagged High</th><th className="num">Caught</th><th className="num">Precision</th></tr></thead>
            <tbody>{Object.entries(v.groups).map(([g, x]) => <tr key={g} style={x.sufficient_sample ? {} : { color: "#5c6577" }}><td>{g}</td><td className="num">{x.n}</td>
              <td className="num">{fmt.pct(x.observed_rate)}</td><td className="num">{fmt.pct(x.selection_rate_high)}</td><td className="num">{fmt.pct(x.tpr_high)}</td><td className="num">{fmt.pct(x.ppv_high)}</td></tr>)}</tbody></table></details>
      ))}
    </div>
  );
}
