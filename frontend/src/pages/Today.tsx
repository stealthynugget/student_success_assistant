import { useQuery } from "@tanstack/react-query";
import { ArrowRight, PenLine } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router";
import { api, fmt, runJob, type ApiError } from "../api";
import { studentPath, useStore } from "../store";
import { CountUp, Empty, ErrorNote, Generated, Loading, Section, Stat, Steps, Tier, type StepState } from "../ui";
import { ActivityChart, COL, RiskHistogram } from "../viz";

/* eslint-disable @typescript-eslint/no-explicit-any */
const BRIEF_STEPS: [string, string][] = [["cohort", "Read cohort figures"], ["ranking", "Rank students"], ["briefing", "Write the briefing"]];

export default function Today() {
  const { module, presentation, checkpoint, config } = useStore();
  const nav = useNavigate();
  const cohort = useQuery({ queryKey: ["cohort", module, presentation, checkpoint], enabled: !!checkpoint,
    queryFn: ({ signal }) => api("/cohort", { query: { code_module: module, code_presentation: presentation, checkpoint }, signal }) });
  const overview = useQuery({ queryKey: ["overview", checkpoint], enabled: !!checkpoint,
    queryFn: ({ signal }) => api("/overview", { query: { checkpoint }, signal }) });
  const top = useQuery({ queryKey: ["top", module, presentation, checkpoint], enabled: !!checkpoint,
    queryFn: ({ signal }) => api("/prioritize", { method: "POST", body: { code_module: module, code_presentation: presentation, capacity: 0.1, limit: 8, checkpoint }, signal }) });
  const [brief, setBrief] = useState<{ status: string; steps?: StepState[]; data?: any; error?: ApiError }>({ status: "idle" });
  useEffect(() => { setBrief({ status: "idle" }); }, [module, presentation, checkpoint]);

  const makeBrief = async () => {
    setBrief({ status: "running", steps: [] });
    try {
      const r = await runJob("/jobs/briefing", { code_module: module, code_presentation: presentation, checkpoint, capacity: 0.2 },
        (j) => setBrief((b) => ({ ...b, steps: (j.steps as StepState[]) || [] })));
      setBrief({ status: "done", data: r.briefing });
    } catch (e) { setBrief({ status: "error", error: e as ApiError }); }
  };

  const c: any = cohort.data;
  return (
    <>
      <div className="page-head">
        <div><h1>This week in {module} {presentation}</h1>
          <div className="sub">Students still enrolled{c ? ` at week ${c.snapshot_week}` : ""}, how many may not finish, and who to reach first.</div></div>
      </div>
      {cohort.error && <ErrorNote error={cohort.error as ApiError} onRetry={() => cohort.refetch()} />}
      {!c && !cohort.error && <Loading lines={4} label="Loading cohort" />}
      {c && (
        <>
          <div className="stats">
            <Stat value={<CountUp value={c.in_scope} format={fmt.int} />} label={`students still enrolled at week ${c.snapshot_week}`} />
            <Stat value={<CountUp value={c.tiers.High} format={fmt.int} />} label="in the High priority tier" tone="high" />
            <Stat value={<CountUp value={c.expected_withdrawals} format={(v) => fmt.num(v, 0)} />} label="withdrawals expected from now on (sum of calibrated risks)" />
            <Stat value={<CountUp value={c.expected_fail_or_withdraw} format={(v) => fmt.num(v, 0)} />} label="expected to fail or withdraw" />
            <Stat value={<CountUp value={c.already_withdrawn} format={fmt.int} />} label="already left before this week (not scored)" />
          </div>

          <div className="grid wide-left section">
            <section className="stack" aria-labelledby="reach-h">
              <h2 className="h3" id="reach-h">Who to reach first</h2>
              {top.error && <ErrorNote error={top.error as ApiError} />}
              {!top.data && !top.error && <Loading lines={5} label="Ranking students" />}
              {top.data && (
                <div className="table-wrap"><table className="data">
                  <caption className="visually-hidden">Top-ranked students at advisor capacity</caption>
                  <thead><tr><th scope="col">Student</th><th scope="col">Withdrawal tier</th><th scope="col" className="num">Withdrawal risk</th>
                    <th scope="col" className="num">Fail or withdraw</th><th scope="col">Main reasons</th></tr></thead>
                  <tbody>{(top.data as any).ranking.map((r: any) => (
                    <tr key={r.id_student} className="click" onClick={() => nav(studentPath(module, presentation, r.id_student))}>
                      <td><Link to={studentPath(module, presentation, r.id_student)}>{r.id_student}</Link></td><td><Tier tier={r.risk_tier} /></td>
                      <td className="num">{fmt.pct(r.withdrawal_risk)}</td><td className="num">{fmt.pct(r.p_poor_outcome)}</td>
                      <td className="small">{r.reasons.slice(0, 2).join("; ")}</td></tr>))}</tbody></table></div>
              )}
              <Link to="/priorities" className="row small">See the full ranked list <ArrowRight aria-hidden="true" size={14} /></Link>
            </section>
            <section className="stack" aria-labelledby="brief-h">
              <h2 className="h3" id="brief-h">Weekly briefing</h2>
              {brief.status === "idle" && <Empty title="Brief the module team"
                action={<button className="primary" onClick={makeBrief}><PenLine aria-hidden="true" />Write briefing</button>}>
                {config?.llm.configured ? "AI drafts a short briefing from this week's computed figures; code checks every number and student ID before you see it."
                  : "Summarises this week's computed figures. AI drafting is not connected, so the briefing is assembled from templates."}</Empty>}
              {brief.status === "running" && <div className="panel"><Steps steps={brief.steps || []} expected={BRIEF_STEPS} /></div>}
              {brief.status === "error" && <ErrorNote error={brief.error} onRetry={makeBrief} />}
              {brief.status === "done" && <Briefing data={brief.data} module={module} presentation={presentation} again={makeBrief} />}
            </section>
          </div>

          <div className="grid two section">
            <section className="stack" aria-labelledby="spread-h">
              <h2 className="h3" id="spread-h">How risk is spread</h2>
              <RiskHistogram counts={c.risk_histogram.counts} edges={c.risk_histogram.edges} thresholds={c.thresholds} />
              <div className="legend">{(["Low", "Medium", "High"] as const).map((t) => <span key={t}><i style={{ background: COL[t], height: 10 }} />{t} tier</span>)}</div>
              <p className="small muted">Calibrated probability of withdrawing after week {c.snapshot_week}. Tier cut-offs ({fmt.pct(c.thresholds.medium)} and {fmt.pct(c.thresholds.high)}) were learned on earlier presentations.</p>
            </section>
            <section className="stack" aria-labelledby="drivers-h">
              <h2 className="h3" id="drivers-h">What drives High priority here</h2>
              {c.common_high_tier_drivers.length ? (
                <div className="bars-h">
                  {c.common_high_tier_drivers.map(([label, n]: [string, number]) => (
                    <div key={label} style={{ display: "contents" }}><span>{label}</span>
                      <span><span className="bar" style={{ display: "block", width: `${(100 * n) / c.common_high_tier_drivers[0][1]}%`, background: "var(--high)" }} /></span>
                      <span className="small">{n} students</span></div>))}
                </div>
              ) : <p className="muted">No High-tier students at this point.</p>}
              <p className="small muted">How often each signal is among the top reasons for High-tier students' risk.</p>
            </section>
          </div>

          <div className="grid two section">
            <section className="stack" aria-labelledby="act-h">
              <h2 className="h3" id="act-h">Typical weekly activity so far</h2>
              <ActivityChart label="Median weekly activity of enrolled students" yLabel="median clicks per week" snapshotWeek={c.snapshot_week}
                deadlines={c.assessments.filter((a: any) => a.deadline_week != null && a.deadline_week <= c.snapshot_week).map((a: any) => ({ week: a.deadline_week, type: a.assessment_type }))}
                series={[{ key: "median", name: "Median clicks", values: c.cohort_median_weekly_clicks.slice(0, c.snapshot_week + 1), color: COL.ink }]} />
              <p className="small muted">Dashed lines: assessment deadlines. Yellow: the selected point in the course.</p>
            </section>
            <section className="stack" aria-labelledby="ass-h">
              <h2 className="h3" id="ass-h">Assessments</h2>
              <div className="table-wrap"><table className="data">
                <thead><tr><th scope="col">Type</th><th scope="col" className="num">Week</th><th scope="col" className="num">Weight</th>
                  <th scope="col" className="num">Usually submitted</th><th scope="col" className="num">Missing it signals risk</th></tr></thead>
                <tbody>{c.assessments.filter((a: any) => a.assessment_type !== "Exam").map((a: any) => (
                  <tr key={a.id_assessment} style={a.upcoming ? {} : { color: "var(--muted)" }}>
                    <td>{a.assessment_type}{a.upcoming ? <strong> upcoming</strong> : ""}</td><td className="num">{a.deadline_week ?? "–"}</td><td className="num">{a.weight}</td>
                    <td className="num">{fmt.pct(a.submission_rate)}</td><td className="num">{a.non_submission_auc_for_poor_outcome ? a.non_submission_auc_for_poor_outcome.toFixed(2) : "–"}</td></tr>))}</tbody></table></div>
              <p className="small muted">The last column is an AUC from this presentation's complete records: 0.5 means no signal; higher means missing it more strongly marks students who later failed or withdrew.</p>
            </section>
          </div>
        </>
      )}

      <Section id="all" title="All presentations at this point" sub="Select a row to move the course clock there.">
        {overview.error && <ErrorNote error={overview.error as ApiError} />}
        {overview.data && <OverviewTable data={overview.data} />}
      </Section>
    </>
  );
}

function Briefing({ data, module, presentation, again }: { data: any; module: string; presentation: string; again: () => void }) {
  return (
    <Generated source={data.source} title="Briefing">
      <p><strong>{data.headline}</strong></p>
      <ul>{data.key_points.map((k: string, i: number) => <li key={i}>{k}</li>)}</ul>
      <h3 className="h4">Focus this week</h3>
      <ul>{data.focus_this_week.map((k: string, i: number) => <li key={i}>{k}</li>)}</ul>
      <div className="row small"><span className="muted">Contact first:</span>
        {data.students_to_contact.map((id: number) => <Link className="chip" key={id} to={studentPath(module, presentation, id)}>{id}</Link>)}</div>
      {data.checks?.length > 0 && <p className="small muted">The AI draft was not used because: {data.checks.join("; ")}. This is the computed briefing.</p>}
      <button className="ghost" onClick={again}>Write again</button>
    </Generated>
  );
}

function OverviewTable({ data }: { data: any }) {
  const { setModule, setPresentation } = useStore();
  const pick = (r: any) => { setModule(r.code_module); setPresentation(r.code_presentation); window.scrollTo(0, 0); };
  return (
    <div className="table-wrap"><table className="data">
      <thead><tr><th scope="col">Module</th><th scope="col">Presentation</th><th scope="col" className="num">Still enrolled</th><th scope="col">Tier mix</th>
        <th scope="col" className="num">High</th><th scope="col" className="num">Expected withdrawals</th><th scope="col" className="num">Already left</th></tr></thead>
      <tbody>{data.presentations.map((r: any) => (
        <tr key={r.code_module + r.code_presentation} className="click" onClick={() => pick(r)}>
          <td><button className="ghost" style={{ padding: 0 }} onClick={(e) => { e.stopPropagation(); pick(r); }}>{r.code_module}</button></td>
          <td>{r.code_presentation}</td><td className="num">{fmt.int(r.in_scope)}</td>
          <td style={{ width: 180 }}><div role="img" aria-label={`High ${r.high}, Medium ${r.medium}, Low ${r.low}`} style={{ display: "flex", height: 10, borderRadius: 2, overflow: "hidden" }}>
            {(["high", "medium", "low"] as const).map((k) => <span key={k} style={{ width: `${(100 * r[k]) / Math.max(1, r.in_scope)}%`, background: `var(--${k})` }} />)}</div></td>
          <td className="num">{r.high}</td><td className="num">{fmt.num(r.expected_withdrawals, 0)}</td><td className="num">{fmt.int(r.already_withdrawn)}</td></tr>))}</tbody>
    </table></div>
  );
}
