import React, { useEffect, useMemo, useState } from "react";
import { api, fmt, runJob } from "../api.js";
import { C, Diverging, LineChart, RiskScale, Stacked } from "../charts.jsx";
import { useApi, useStore } from "../store.jsx";
import { CopyButton, Empty, ErrorNote, Generated, Json, Loading, Section, Source, Steps, Tier } from "../ui.jsx";

const OUTCOME_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"];
const AGENT_STEPS = [["models", "Score risk and outcome"], ["retrieval", "Find similar past students"], ["performance", "Performance agent"],
  ["risk", "Risk agent"], ["intervention", "Intervention agent"], ["intervention_revision", "Revision after critic (if needed)"],
  ["precedent_synthesis", "Explain what happened to similar students"]];

export default function Student({ parts }) {
  const [, m, p, idStr] = parts;
  const id = Number(idStr);
  const ref = useMemo(() => ({ id_student: id, code_module: m, code_presentation: p }), [id, m, p]);
  const { checkpoint, config, setModule, setPresentation } = useStore();
  useEffect(() => { setModule(m); setPresentation(p); }, [m, p]); // eslint-disable-line react-hooks/exhaustive-deps

  const pred = useApi((s) => api("/predict-risk", { method: "POST", body: { student: ref, checkpoint }, signal: s }), [ref, checkpoint]);
  const trend = useApi((s) => api("/trends/student", { method: "POST", body: { student: ref, checkpoint }, signal: s }), [ref, checkpoint]);
  const timeline = useApi((s) => api("/students/timeline", { method: "POST", body: { student: ref }, signal: s }), [ref]);
  const similar = useApi((s) => api("/retrieve-similar", { method: "POST", body: { student: ref, top_k: 8, checkpoint, understand: false }, signal: s }), [ref, checkpoint]);
  const [analysis, setAnalysis] = useState({ status: "idle" });
  useEffect(() => setAnalysis({ status: "idle" }), [ref, checkpoint]);

  const runAnalysis = async (useLlm = true) => {
    setAnalysis({ status: "running", steps: [] });
    try {
      const r = await runJob("/jobs/analyze", { student: ref, checkpoint, use_llm: useLlm }, (j) => setAnalysis((a) => ({ ...a, steps: j.steps || [] })));
      setAnalysis({ status: "done", data: r });
    } catch (e) { setAnalysis({ status: "error", error: e }); }
  };

  if (pred.error?.status === 409) {
    return (<><h1>Student {id}</h1><p className="muted">{m} {p}</p>
      <Empty title="This student had already left at this point in the course">
        They unregistered on day {pred.error.details?.date_unregistration}, before the selected week, so an early warning no longer applies. Pick an earlier week on the course clock to see how they looked then.
      </Empty></>);
  }
  const r = pred.data?.withdrawal_risk;
  const o = pred.data?.outcome_prediction;
  return (
    <>
      <div className="page-head">
        <div><h1>Student {id}</h1><div className="sub">{m} {p}{trend.data ? `, as of course week ${trend.data.snapshot_week}` : ""}</div></div>
        {r && <div style={{ alignSelf: "center" }}><Tier tier={r.risk_tier} /></div>}
        <div className="actions">
          <button className="primary" onClick={() => runAnalysis(true)} disabled={analysis.status === "running"}>Draft a support plan</button>
          <a className="btn" href={`#/copilot?q=${encodeURIComponent(`Tell me about student ${id} in ${m} ${p}`)}`}>Ask the copilot</a>
        </div>
      </div>
      {pred.error && <ErrorNote error={pred.error} onRetry={pred.reload} />}
      {!pred.data && !pred.error && <Loading lines={5} />}
      {r && (
        <div className="grid wide-left">
          <div className="stack">
            <h3>Why this student is flagged</h3>
            <RiskScale value={r.withdrawal_risk} thresholds={r.thresholds} />
            <p className="small muted">{r.interpretation}</p>
            <h4>Likely final outcome</h4>
            <Stacked parts={OUTCOME_ORDER.map((k) => ({ label: k, value: o.probabilities[k], display: fmt.pct(o.probabilities[k]), color: C[k] }))} />
            <h4 style={{ marginTop: 16 }}>What moves the estimate</h4>
            <Diverging items={r.drivers.map((d) => ({ label: d.label, value: d.contribution_logodds,
              title: `${d.label}: ${d.value ?? "n/a"} (${d.direction.replace("_", " ")})` }))} />
            <div className="small muted">Method: {r.drivers[0]?.method?.replace("_", " ")}. Effects are on the model's log-odds scale, relative to a typical student.</div>
          </div>
          <div className="stack">
            <h3>Across the course</h3>
            {timeline.loading && <div className="small muted">Recomputing risk at earlier weeks…</div>}
            {timeline.data && <CheckpointTrail points={timeline.data.points.filter((x) => x.checkpoint <= checkpoint + 1e-9)} current={checkpoint} />}
            {trend.data?.weekly_clicks && (
              <>
                <LineChart height={200} ariaLabel="Weekly activity compared with the cohort median" xLabel="course week" yLabel="VLE clicks per week"
                  series={[{ name: "Cohort median", values: trend.data.cohort_median_weekly_clicks.slice(0, trend.data.snapshot_week + 1), color: "#9aa2b1", width: 2 },
                    { name: "This student", values: trend.data.weekly_clicks.slice(0, trend.data.snapshot_week + 1), color: C.ink, width: 2.5 }]}
                  ticks={trend.data.assessment_deadlines.filter((d) => d.week <= trend.data.snapshot_week).map((d) => ({ x: d.week, label: `${d.type} due` }))} />
                <div className="legend"><span><i style={{ background: C.ink }} />this student</span><span><i style={{ background: "#9aa2b1" }} />cohort median</span><span>tick marks: assessment deadlines</span></div>
                {trend.data.signals.length > 0 && <div>{trend.data.signals.map((s) => <span key={s} className="chip">{s.replaceAll("_", " ")}</span>)}</div>}
              </>
            )}
          </div>
        </div>
      )}

      <Section title="Support plan" sub="Facts and scores are computed; the AI only selects from eligible actions and drafts wording, which code checks before you see it.">
        {analysis.status === "idle" && <Empty title="No plan drafted yet" action={<div className="row"><button className="primary" onClick={() => runAnalysis(true)}>Draft a support plan</button>
          <button onClick={() => runAnalysis(false)}>Use computed plan only</button></div>}>
          The agents review performance, assess risk with similar past students, choose supportive actions and write an outreach draft for you to edit.</Empty>}
        {analysis.status === "running" && <div className="panel"><Steps steps={analysis.steps || []} expected={AGENT_STEPS} /></div>}
        {analysis.status === "error" && <ErrorNote error={analysis.error} onRetry={() => runAnalysis(true)} />}
        {analysis.status === "done" && <Plan res={analysis.data} refStudent={ref} />}
      </Section>

      <Section title="Similar students from earlier presentations" sub="Same module, earlier cohorts only, with what happened to them after this point.">
        {similar.error && <ErrorNote error={similar.error} />}
        {similar.data && <Precedents data={similar.data} synthesis={analysis.data?.precedent_synthesis} />}
      </Section>

      {pred.data && <WhatIf ref_={ref} checkpoint={checkpoint} base={pred.data} />}
    </>
  );
}

function CheckpointTrail({ points, current }) {
  const pts = points.filter((x) => x.in_scope);
  if (pts.length < 2) return <p className="small muted">This is the earliest checkpoint; risk at earlier weeks is not available.</p>;
  return (
    <div>
      <LineChart height={150} markers yMax={1} formatY={(v) => `${Math.round(v * 100)}%`} ariaLabel="Withdrawal risk at each checkpoint"
        xMax={Math.max(...points.map((p) => p.snapshot_week)) + 2}
        series={[{ name: "Withdrawal risk", color: C.ink, values: Array.from({ length: Math.max(...points.map((p) => p.snapshot_week)) + 3 }, (_, w) => pts.find((p) => p.snapshot_week === w)?.withdrawal_risk ?? null),
          pointColor: Array.from({ length: 60 }, (_, w) => C[pts.find((p) => p.snapshot_week === w)?.risk_tier] || C.ink) }]}
        vlines={points.filter((p) => Math.abs(p.checkpoint - current) < 1e-6).map((p) => ({ x: p.snapshot_week, label: "now" }))} />
      <p className="small muted">Risk recomputed at each earlier checkpoint using only what was known then; later weeks are hidden so nothing from the future leaks in. Dots are coloured by tier.</p>
    </div>
  );
}

function Plan({ res, refStudent }) {
  const { notify } = useStore();
  const wf = res.workflow;
  const plan = wf.intervention_plan;
  const [message, setMessage] = useState(plan.outreach_message);
  const [variant, setVariant] = useState({ channel: "email", tone: "warm", language: "English" });
  const [rewrite, setRewrite] = useState({ status: "idle" });
  const [judge, setJudge] = useState({ status: "idle" });
  const doRewrite = async () => {
    setRewrite({ status: "running" });
    try { const v = await api("/outreach/variants", { method: "POST", body: { message, ...variant } }); setRewrite({ status: "done", data: v }); setMessage(v.message); }
    catch (e) { setRewrite({ status: "error", error: e }); }
  };
  const doJudge = async () => {
    setJudge({ status: "running" });
    try { setJudge({ status: "done", data: await api("/evaluations/judge", { method: "POST", body: { recommendation_id: wf.recommendation_id } }) }); }
    catch (e) { setJudge({ status: "error", error: e }); }
  };
  return (
    <div className="stack">
      <div className="grid two">
        <div className="panel stack">
          <div className="row"><h3 style={{ marginRight: "auto" }}>{plan.priority}</h3><span className="small muted">review in {plan.review_in_days} days</span></div>
          <ul>{plan.actions.map((a) => <li key={a.id}><strong>{a.text}</strong> <span className="small muted">({a.channel})</span></li>)}</ul>
          <Generated source={plan.source} title="Why these actions"><p>{plan.rationale}</p></Generated>
          <Generated source={wf.performance_analysis.source} title="Performance"><p>{wf.performance_analysis.summary}</p></Generated>
          <Generated source={wf.risk_analysis.source} title="Risk"><p>{wf.risk_analysis.risk_summary}</p><p className="small">{wf.risk_analysis.historical_context}</p>
            <p className="small muted">{wf.risk_analysis.uncertainty_note}</p></Generated>
        </div>
        <div className="panel stack">
          <h3>Message to the student</h3>
          <textarea rows={8} value={message} onChange={(e) => setMessage(e.target.value)} aria-label="Outreach message" />
          <div className="row">
            <select aria-label="Channel" value={variant.channel} onChange={(e) => setVariant({ ...variant, channel: e.target.value })}>
              <option value="email">Email</option><option value="sms">Text message</option><option value="phone_script">Phone script</option></select>
            <select aria-label="Tone" value={variant.tone} onChange={(e) => setVariant({ ...variant, tone: e.target.value })}>
              <option value="warm">Warm</option><option value="concise">Concise</option><option value="formal">Formal</option></select>
            <select aria-label="Language" value={variant.language} onChange={(e) => setVariant({ ...variant, language: e.target.value })}>
              {["English", "French", "Spanish", "German", "Welsh"].map((l) => <option key={l}>{l}</option>)}</select>
            <button onClick={doRewrite} disabled={rewrite.status === "running"}>{rewrite.status === "running" ? "Rewriting…" : "Rewrite"}</button>
            <CopyButton text={message} />
          </div>
          {rewrite.status === "done" && <div className="small"><Source source={rewrite.data.source} /> {rewrite.data.subject && <>Subject: {rewrite.data.subject}. </>}
            {rewrite.data.english_back_translation && <>English check: “{rewrite.data.english_back_translation}”</>}
            {rewrite.data.checks?.length > 0 && <div className="muted">{rewrite.data.checks.join("; ")}</div>}</div>}
          {rewrite.status === "error" && <ErrorNote error={rewrite.error} />}
          <p className="small muted">Drafts never mention risk scores, predictions or withdrawal. Edit freely before sending.</p>
        </div>
      </div>
      <div className="grid two">
        <div className="stack">
          <h4>Checks on this plan</h4>
          <p>{wf.guardrails.final_plan_violations.length === 0 ? "Passed every guardrail: eligible actions only, numbers traceable to evidence, no protected attributes, no punitive or clinical wording." :
            `Issues: ${wf.guardrails.final_plan_violations.join("; ")}`}</p>
          <div className="row"><button onClick={doJudge} disabled={judge.status === "running"}>Ask the AI judge to review</button>
            <span className="small muted">Uses a separate judge model; the verdict is computed from its scores.</span></div>
          {judge.status === "error" && <ErrorNote error={judge.error} />}
          {judge.status === "done" && (judge.data.scores ? (
            <div><strong>{judge.data.verdict}</strong> overall {fmt.num(judge.data.overall, 1)}/10
              <div className="row">{Object.entries(judge.data.scores).map(([k, v]) => <span key={k} className="chip">{k.replace("_", " ")} {v}</span>)}</div>
              {judge.data.rationale && <p className="small">{judge.data.rationale}</p>}</div>
          ) : <p className="muted">{judge.data.status}</p>)}
        </div>
        <div className="stack">
          <h4>How the agents worked</h4>
          <ol className="small">{wf.trace.map((t, i) => <li key={i}>{t.agent}{t.source ? ` (${t.source})` : ""}{t.decision ? `: ${t.decision}` : ""}
            {t.llm_rejected ? `: AI draft rejected (${t.llm_rejected.join("; ")})` : ""}{t.violations?.length ? `: ${t.violations.join("; ")}` : ""}</li>)}</ol>
          <div className="small muted">LLM calls {wf.llm_usage.calls ?? 0}, tokens {(wf.llm_usage.prompt_tokens || 0) + (wf.llm_usage.completion_tokens || 0)}</div>
          <Json data={wf.observations} label="Evidence the agents could use" />
        </div>
      </div>
      <FeedbackForm res={res} refStudent={refStudent} onDone={() => notify("Feedback saved. Thank you.")} />
    </div>
  );
}

function Precedents({ data, synthesis }) {
  const nb = data.outcome_summary;
  const af = data.aftermath;
  if (!data.cases.length) return <Empty title="No comparable students found">Earlier presentations of this module have no close matches.</Empty>;
  const sw = data.cases[0]?.story?.snapshot_week;
  return (
    <div className="stack">
      <div className="grid two">
        <div className="stack">
          {nb && <>
            <h4>Outcomes of the {nb.n_neighbours} most similar students</h4>
            <Stacked parts={OUTCOME_ORDER.map((k) => ({ label: k, value: nb.counts[k] || 0, color: C[k] }))} />
            <p className="small">{fmt.pct(nb.fail_or_withdrawn_rate)} failed or withdrew (95% interval {fmt.pct(nb.fail_or_withdrawn_ci95[0])} to {fmt.pct(nb.fail_or_withdrawn_ci95[1])}).</p>
          </>}
          {af && <dl className="kv">
            <dt>Median weeks to withdrawal</dt><dd>{af.median_weeks_to_withdrawal ?? "–"}</dd>
            <dt>Median change in activity after this point</dt><dd>{fmt.pct(af.median_activity_change_after_checkpoint)}</dd>
            <dt>More than halved their activity</dt><dd>{fmt.pct(af.share_with_activity_drop_over_50pct)}</dd>
            <dt>Median final coursework average</dt><dd>{fmt.num(af.median_final_coursework_score, 0)}</dd></dl>}
          {synthesis && <Generated source={synthesis.source} title="What happened next">
            <p>{synthesis.what_happened}</p>{synthesis.patterns?.length > 0 && <ul>{synthesis.patterns.map((x, i) => <li key={i}>{x}</li>)}</ul>}
            <p><strong>{synthesis.implication}</strong></p>
            <div className="small muted">Based on cases {synthesis.cited_cases.join(", ")}</div></Generated>}
        </div>
        <div className="stack">
          <h4>Their full-course activity</h4>
          <LineChart height={220} ariaLabel="Weekly activity of similar students over the whole course" xLabel="course week" yLabel="clicks per week"
            series={data.cases.filter((c) => c.story?.trajectory).map((c) => ({ name: `${c.case_id}`, values: c.story.trajectory, color: C[c.final_result], width: 1.5, opacity: 0.8 }))}
            vlines={sw != null ? [{ x: sw, label: "this point" }] : []} />
          <div className="legend">{OUTCOME_ORDER.map((k) => <span key={k}><i style={{ background: C[k] }} />{k}</span>)}</div>
        </div>
      </div>
      <div className="table-wrap"><table className="data">
        <thead><tr><th>Case</th><th>Presentation</th><th>Outcome</th><th>What happened after this point</th><th>Shared signals</th></tr></thead>
        <tbody>{data.cases.map((c) => (
          <tr key={c.case_id}><td>{c.case_id}</td><td>{c.code_presentation}</td><td><span style={{ color: C[c.final_result], fontWeight: 700 }}>{c.final_result}</span></td>
            <td>{c.story?.text || c.summary}</td><td>{c.shared_descriptors.filter((x) => !x.startsWith("module_")).slice(0, 4).map((x) => <span key={x} className="chip">{x.replaceAll("_", " ")}</span>)}</td></tr>))}</tbody>
      </table></div>
    </div>
  );
}

function WhatIf({ ref_, checkpoint, base }) {
  const snap = useApi((s) => api(`/students/${ref_.id_student}`, { query: { code_module: ref_.code_module, code_presentation: ref_.code_presentation, checkpoint }, signal: s }), [ref_, checkpoint]);
  const [v, setV] = useState(null);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    if (!snap.data) return;
    const s = snap.data.snapshot;
    setV({ weighted_score: s.weighted_score ?? 60, missed_rate: s.missed_rate ?? 0, days_since_last_activity: s.days_since_last_activity ?? 0,
      clicks_pct: s.clicks_pct ?? 0.5, snap: s, moved: [] });
    setRes(null);
  }, [snap.data]);
  useEffect(() => {
    if (!v) return;
    if (!v.moved.length) { setRes(null); return; } // nothing changed yet: the real record's estimate is shown above
    const t = setTimeout(async () => {
      const s = v.snap;
      try {
        setErr(null);
        // Send only the signals the user moved; a moved score also sets the score fields it implies.
        const what_if = {};
        for (const k of v.moved) {
          if (k === "weighted_score") Object.assign(what_if, { weighted_score: v.weighted_score, mean_score: v.weighted_score,
            last_score: v.weighted_score, no_scores_yet: 0, min_score: Math.min(v.weighted_score, s.min_score ?? v.weighted_score),
            max_score: Math.max(v.weighted_score, s.max_score ?? v.weighted_score) });
          else what_if[k] = v[k];
        }
        setRes(await api("/predict-risk", { method: "POST", body: { student: ref_, checkpoint, what_if } }));
      } catch (e) { setErr(e); }
    }, 300);
    return () => clearTimeout(t);
  }, [v]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!v) return null;
  const b = base.withdrawal_risk.withdrawal_risk;
  const w = res?.withdrawal_risk.withdrawal_risk;
  const slider = (key, label, min, max, step, show) => (
    <label className="field">{label}: <strong>{show(v[key])}</strong>
      <input type="range" min={min} max={max} step={step} value={v[key]} onChange={(e) => setV({ ...v, [key]: Number(e.target.value), moved: v.moved.includes(key) ? v.moved : [...v.moved, key] })} /></label>);
  return (
    <Section title="What if" sub="Change a few signals of this real record to see how the estimate responds. Associations, not guaranteed effects.">
      <div className="grid two">
        <div className="stack">
          {slider("weighted_score", "Weighted assessment score", 0, 100, 1, (x) => x.toFixed(0))}
          {slider("missed_rate", "Share of due work missed", 0, 1, 0.05, fmt.pct)}
          {slider("days_since_last_activity", "Days since last online activity", 0, v.snap.snapshot_day || 100, 1, (x) => x)}
          {slider("clicks_pct", "Activity compared with cohort (percentile)", 0, 1, 0.05, fmt.pct)}
        </div>
        <div className="stack">
          {err && <ErrorNote error={err} />}
          {!res && !err && <p className="muted">Move a slider to see how the estimate responds.</p>}
          {res && <>
            <div className="stats"><div className="stat"><div className="v">{fmt.pct(w)}</div><div className="l">withdrawal risk with these changes</div></div>
              <div className="stat"><div className="v" style={{ color: w > b ? C.High : C.Low }}>{w > b ? "+" : ""}{fmt.pct(w - b)}</div><div className="l">compared with the real record ({fmt.pct(b)})</div></div></div>
            <Tier tier={res.withdrawal_risk.risk_tier} />
            {res.input_quality.out_of_distribution.length > 0 && <div className="note warn">Some values fall outside what the model has seen: {res.input_quality.out_of_distribution.map((o) => o.label).join(", ")}.</div>}
          </>}
        </div>
      </div>
    </Section>
  );
}

function FeedbackForm({ res, refStudent, onDone }) {
  const wf = res.workflow;
  const [f, setF] = useState({ rating: 4, plan_accepted: true, risk_assessment_agrees: true, notes: "", observed_outcome: "" });
  const [acts, setActs] = useState(Object.fromEntries(wf.intervention_plan.actions.map((a) => [a.id, true])));
  const [cases, setCases] = useState({});
  const [state, setState] = useState({ status: "idle" });
  const submit = async (e) => {
    e.preventDefault();
    setState({ status: "saving" });
    try {
      await api("/feedback", { method: "POST", body: { recommendation_id: wf.recommendation_id, id_student: refStudent.id_student, rating: f.rating,
        plan_accepted: f.plan_accepted, risk_assessment_agrees: f.risk_assessment_agrees, notes: f.notes, observed_outcome: f.observed_outcome || null,
        action_feedback: Object.fromEntries(Object.entries(acts).map(([k, v]) => [k, { accepted: v }])), case_feedback: cases } });
      setState({ status: "saved" }); onDone();
    } catch (err) { setState({ status: "error", error: err }); }
  };
  if (state.status === "saved") return <p className="note">Feedback saved for recommendation {wf.recommendation_id}. It feeds the next improvement run.</p>;
  return (
    <form className="panel stack" onSubmit={submit}>
      <h3>Was this useful?</h3>
      <div className="row">
        <label className="field">Overall usefulness: {f.rating} of 5<input type="range" min="1" max="5" value={f.rating} onChange={(e) => setF({ ...f, rating: Number(e.target.value) })} /></label>
        <label><input type="checkbox" checked={f.plan_accepted} onChange={(e) => setF({ ...f, plan_accepted: e.target.checked })} /> I will act on this plan</label>
        <label><input type="checkbox" checked={f.risk_assessment_agrees} onChange={(e) => setF({ ...f, risk_assessment_agrees: e.target.checked })} /> I agree with the risk assessment</label>
      </div>
      <fieldset style={{ border: 0, padding: 0 }}><legend className="small muted">Keep or drop each action</legend>
        {wf.intervention_plan.actions.map((a) => <label key={a.id} style={{ display: "block" }}><input type="checkbox" checked={acts[a.id]} onChange={(e) => setActs({ ...acts, [a.id]: e.target.checked })} /> {a.text}</label>)}</fieldset>
      <fieldset style={{ border: 0, padding: 0 }}><legend className="small muted">Which similar students were useful precedents?</legend>
        <div className="row">{res.similar_cases.cases.map((c) => <label key={c.case_id} className="chip"><input type="checkbox" checked={!!cases[c.case_id]}
          onChange={(e) => setCases({ ...cases, [c.case_id]: e.target.checked })} /> {c.case_id} ({c.final_result})</label>)}</div></fieldset>
      <label className="field">Outcome, if already known<select value={f.observed_outcome} onChange={(e) => setF({ ...f, observed_outcome: e.target.value })}>
        <option value="">Not known yet</option>{OUTCOME_ORDER.map((o) => <option key={o}>{o}</option>)}</select></label>
      <label className="field">Notes for the team (optional)<textarea rows={3} maxLength={2000} value={f.notes} onChange={(e) => setF({ ...f, notes: e.target.value })} /></label>
      {state.status === "error" && <ErrorNote error={state.error} />}
      <div><button className="primary" type="submit" disabled={state.status === "saving"}>Save feedback</button></div>
    </form>
  );
}
