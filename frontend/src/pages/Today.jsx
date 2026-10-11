import React, { useState } from "react";
import { api, fmt, runJob } from "../api.js";
import { C, Histogram, LineChart } from "../charts.jsx";
import { studentPath, useApi, useStore } from "../store.jsx";
import { Empty, ErrorNote, Generated, Loading, Section, Stat, Steps, Tier } from "../ui.jsx";

export default function Today() {
  const { module, presentation, checkpoint } = useStore();
  const cohort = useApi((s) => api("/cohort", { query: { code_module: module, code_presentation: presentation, checkpoint }, signal: s }), [module, presentation, checkpoint]);
  const overview = useApi((s) => api("/overview", { query: { checkpoint }, signal: s }), [checkpoint]);
  const top = useApi((s) => api("/prioritize", { method: "POST", body: { code_module: module, code_presentation: presentation, capacity: 0.1, limit: 8, checkpoint }, signal: s }), [module, presentation, checkpoint]);
  const [brief, setBrief] = useState({ status: "idle" });

  const makeBrief = async () => {
    setBrief({ status: "running", steps: [] });
    try {
      const r = await runJob("/jobs/briefing", { code_module: module, code_presentation: presentation, checkpoint, capacity: 0.2 },
        (j) => setBrief((b) => ({ ...b, steps: j.steps || [] })));
      setBrief({ status: "done", data: r.briefing });
    } catch (e) { setBrief({ status: "error", error: e }); }
  };

  const c = cohort.data;
  return (
    <>
      <div className="page-head">
        <div><h1>This week in {module} {presentation}</h1>
          <div className="sub">Students still enrolled, how many may not finish, and who to reach first.</div></div>
      </div>
      {cohort.error && <ErrorNote error={cohort.error} onRetry={cohort.reload} />}
      {!c && !cohort.error && <Loading lines={4} />}
      {c && (
        <>
          <div className="stats">
            <Stat value={fmt.int(c.in_scope)} label={`students still enrolled at week ${c.snapshot_week}`} />
            <Stat value={fmt.int(c.tiers.High)} label="in the High priority tier" tone="high" />
            <Stat value={fmt.num(c.expected_withdrawals, 0)} label="withdrawals expected from now on (sum of calibrated risks)" />
            <Stat value={fmt.num(c.expected_fail_or_withdraw, 0)} label="expected to fail or withdraw" />
            <Stat value={fmt.int(c.already_withdrawn)} label="already left before this week (not scored)" />
          </div>

          <div className="grid wide-left section">
            <div className="stack">
              <h3>Who to reach first</h3>
              {top.error && <ErrorNote error={top.error} />}
              {top.data && (
                <table className="data"><thead><tr><th>Student</th><th>Withdrawal tier</th><th className="num">Withdrawal risk</th><th className="num">Fail or withdraw</th><th>Main reasons</th></tr></thead>
                  <tbody>{top.data.ranking.map((r) => (
                    <tr key={r.id_student} className="click" onClick={() => { window.location.hash = studentPath(module, presentation, r.id_student).slice(1); }}>
                      <td><a href={studentPath(module, presentation, r.id_student)}>{r.id_student}</a></td><td><Tier tier={r.risk_tier} /></td>
                      <td className="num">{fmt.pct(r.withdrawal_risk)}</td><td className="num">{fmt.pct(r.p_poor_outcome)}</td>
                      <td className="small">{r.reasons.slice(0, 2).join("; ")}</td></tr>))}</tbody></table>
              )}
              <a href="#/priorities">See the full ranked list</a>
            </div>
            <div className="stack">
              <h3>Weekly briefing</h3>
              {brief.status === "idle" && <Empty title="Get a briefing for the module team"
                action={<button className="primary" onClick={makeBrief}>Write briefing</button>}>Summarises this week's risk picture, upcoming deadlines and who to contact, from the computed data.</Empty>}
              {brief.status === "running" && <Steps steps={brief.steps || []} expected={[["cohort", "Read cohort figures"], ["ranking", "Rank students"], ["briefing", "Write briefing"]]} />}
              {brief.status === "error" && <ErrorNote error={brief.error} onRetry={makeBrief} />}
              {brief.status === "done" && (
                <Generated source={brief.data.source} title="Briefing">
                  <p><strong>{brief.data.headline}</strong></p>
                  <ul>{brief.data.key_points.map((k, i) => <li key={i}>{k}</li>)}</ul>
                  <h4>Focus this week</h4>
                  <ul>{brief.data.focus_this_week.map((k, i) => <li key={i}>{k}</li>)}</ul>
                  <div className="row">{brief.data.students_to_contact.map((id) => <a className="chip" key={id} href={studentPath(module, presentation, id)}>{id}</a>)}</div>
                  {brief.data.checks?.length > 0 && <p className="small muted">The AI draft was replaced because: {brief.data.checks.join("; ")}</p>}
                  <button className="ghost" onClick={makeBrief}>Write again</button>
                </Generated>
              )}
            </div>
          </div>

          <div className="grid two section">
            <div className="stack">
              <h3>How risk is spread</h3>
              <Histogram counts={c.risk_histogram.counts} edges={c.risk_histogram.edges} thresholds={c.thresholds} />
              <p className="small muted">Calibrated probability of withdrawing after week {c.snapshot_week}. Tier cut-offs were learned on earlier presentations.</p>
            </div>
            <div className="stack">
              <h3>What drives High priority here</h3>
              {c.common_high_tier_drivers.length ? (
                <table className="data"><tbody>{c.common_high_tier_drivers.map(([label, n]) => (
                  <tr key={label}><td>{label}</td><td className="num">{n} students</td></tr>))}</tbody></table>
              ) : <p className="muted">No High-tier students at this point.</p>}
            </div>
          </div>

          <div className="grid two section">
            <div className="stack">
              <h3>Typical weekly activity so far</h3>
              <LineChart series={[{ name: "Median clicks", values: c.cohort_median_weekly_clicks.slice(0, c.snapshot_week + 1), color: C.ink }]}
                xLabel="course week" yLabel="median weekly VLE clicks" ariaLabel="Median weekly activity of enrolled students" />
            </div>
            <div className="stack">
              <h3>Assessments</h3>
              <table className="data"><thead><tr><th>Type</th><th className="num">Week</th><th className="num">Weight</th><th className="num">Usually submitted</th><th className="num">Missing it signals risk</th></tr></thead>
                <tbody>{c.assessments.filter((a) => a.assessment_type !== "Exam").map((a) => (
                  <tr key={a.id_assessment} style={a.upcoming ? {} : { color: "#5c6577" }}>
                    <td>{a.assessment_type}{a.upcoming ? " (upcoming)" : ""}</td><td className="num">{a.deadline_week ?? "–"}</td><td className="num">{a.weight}</td>
                    <td className="num">{fmt.pct(a.submission_rate)}</td><td className="num">{a.non_submission_auc_for_poor_outcome ? a.non_submission_auc_for_poor_outcome.toFixed(2) : "–"}</td></tr>))}</tbody></table>
              <p className="small muted">The last column is an AUC from this presentation's complete records: 0.5 means no signal, higher means missing it more strongly marks students who later failed or withdrew.</p>
            </div>
          </div>
        </>
      )}

      <Section title="All presentations at this point" sub="Click a row to switch the course clock.">
        {overview.error && <ErrorNote error={overview.error} />}
        {overview.data && <OverviewTable data={overview.data} />}
      </Section>
    </>
  );
}

function OverviewTable({ data }) {
  const { setModule, setPresentation } = useStore();
  return (
    <div className="table-wrap"><table className="data">
      <thead><tr><th>Module</th><th>Presentation</th><th className="num">Still enrolled</th><th>Tier mix</th><th className="num">High</th><th className="num">Expected withdrawals</th><th className="num">Already left</th></tr></thead>
      <tbody>{data.presentations.map((r) => (
        <tr key={r.code_module + r.code_presentation} className="click" onClick={() => { setModule(r.code_module); setPresentation(r.code_presentation); window.scrollTo(0, 0); }}>
          <td>{r.code_module}</td><td>{r.code_presentation}</td><td className="num">{fmt.int(r.in_scope)}</td>
          <td style={{ width: 180 }}><div style={{ display: "flex", height: 10, borderRadius: 2, overflow: "hidden" }}>
            {["high", "medium", "low"].map((k) => <span key={k} style={{ width: `${(100 * r[k]) / Math.max(1, r.in_scope)}%`, background: { high: C.High, medium: C.Medium, low: C.Low }[k] }} />)}</div></td>
          <td className="num">{r.high}</td><td className="num">{fmt.num(r.expected_withdrawals, 0)}</td><td className="num">{fmt.int(r.already_withdrawn)}</td></tr>))}</tbody>
    </table></div>
  );
}
