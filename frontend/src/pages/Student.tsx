import { useQuery } from "@tanstack/react-query";
import { MessagesSquare, PenLine, Scale } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router";
import { api, fmt, runJob, type ApiError } from "../api";
import { useStore } from "../store";
import { CopyButton, Empty, ErrorNote, Generated, Json, Loading, Section, Source, Steps, Tier, type StepState } from "../ui";
import { ActivityChart, COL, Diverging, OUTCOMES, RiskDial, RiskTrail, Stacked, Trajectories } from "../viz";

/* eslint-disable @typescript-eslint/no-explicit-any */
const AGENT_STEPS: [string, string, string?][] = [
  ["models", "Score withdrawal risk and likely outcome"], ["retrieval", "Find similar students from earlier presentations"],
  ["performance", "Performance agent reviews scores and engagement"], ["risk", "Risk agent explains the estimate"],
  ["intervention", "Intervention agent chooses actions and drafts outreach", "if routed to support"],
  ["intervention_revision", "Revision after the critic's checks", "if needed"],
  ["precedent_synthesis", "Summarise what happened to similar students"],
];
const OUTLINE: [string, string][] = [["overview", "Why flagged"], ["course", "Across the course"], ["plan", "Support plan"],
  ["precedents", "Similar students"], ["whatif", "What if"]];

function useScrollSpy(ids: string[]) {
  const [active, setActive] = useState(ids[0]);
  useEffect(() => {
    const obs = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (vis[0]) setActive(vis[0].target.id);
    }, { rootMargin: "-200px 0px -55% 0px" });
    ids.forEach((id) => { const el = document.getElementById(id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, [ids]);
  return active;
}

export default function Student() {
  const { module: m = "", presentation: p = "", id: idStr = "" } = useParams();
  const id = Number(idStr);
  const ref = useMemo(() => ({ id_student: id, code_module: m, code_presentation: p }), [id, m, p]);
  const { checkpoint, config, setModule, setPresentation } = useStore();
  useEffect(() => { setModule(m); setPresentation(p); }, [m, p]); // eslint-disable-line react-hooks/exhaustive-deps

  const q = (key: string, path: string, body: any, enabled = true) => useQuery({ // eslint-disable-line react-hooks/rules-of-hooks
    queryKey: [key, ref, checkpoint], enabled: enabled && !!checkpoint, retry: false,
    queryFn: ({ signal }) => api(path, { method: "POST", body, signal }) });
  const pred = q("predict", "/predict-risk", { student: ref, checkpoint });
  const ok = !!pred.data;
  const trend = q("trend", "/trends/student", { student: ref, checkpoint }, ok);
  const timeline = q("timeline", "/students/timeline", { student: ref }, ok);
  const similar = q("similar", "/retrieve-similar", { student: ref, top_k: 8, checkpoint, understand: false }, ok);
  const [analysis, setAnalysis] = useState<{ status: string; steps?: StepState[]; data?: any; error?: ApiError }>({ status: "idle" });
  useEffect(() => { setAnalysis({ status: "idle" }); }, [ref, checkpoint]);
  const active = useScrollSpy(OUTLINE.map(([k]) => k));

  const runAnalysis = async (useLlm = true) => {
    setAnalysis({ status: "running", steps: [] });
    document.getElementById("plan")?.scrollIntoView({ behavior: "smooth", block: "start" });
    try {
      const r = await runJob("/jobs/analyze", { student: ref, checkpoint, use_llm: useLlm }, (j) => setAnalysis((a) => ({ ...a, steps: (j.steps as StepState[]) || [] })));
      setAnalysis((a) => ({ status: "done", data: r, steps: a.steps }));
    } catch (e) { setAnalysis({ status: "error", error: e as ApiError }); }
  };

  const err = pred.error as ApiError | null;
  if (err?.status === 409) {
    return (<><h1>Student {id}</h1><p className="muted">{m} {p}</p>
      <Empty title="This student had already left at this point in the course">
        They unregistered on day {(err.details as any)?.date_unregistration}, before the selected week, so an early warning no longer applies.
        Pick an earlier week on the course clock to see how they looked then.</Empty></>);
  }
  const r: any = pred.data && (pred.data as any).withdrawal_risk;
  const o: any = pred.data && (pred.data as any).outcome_prediction;
  const td: any = trend.data;
  const aiOn = config?.llm.configured;
  return (
    <>
      <header className="casefile-head">
        <div><h1>Student {id}</h1><div className="sub">{m} {p}{td ? `, as of course week ${td.snapshot_week}` : ""}</div></div>
        {r && <Tier tier={r.risk_tier} />}
        {r && <RiskDial value={r.withdrawal_risk} thresholds={r.thresholds} tier={r.risk_tier} />}
        <div className="actions">
          <button className={aiOn ? "ai primary" : "primary"} onClick={() => runAnalysis(true)} disabled={analysis.status === "running" || !r}>
            <PenLine aria-hidden="true" />{aiOn ? "Draft a support plan" : "Build a support plan"}</button>
          <Link className="btn" to={`/copilot?q=${encodeURIComponent(`Tell me about student ${id} in ${m} ${p}`)}`}><MessagesSquare aria-hidden="true" />Ask the copilot</Link>
        </div>
      </header>
      {err && <ErrorNote error={err} onRetry={() => pred.refetch()} />}
      {!r && !err && <Loading lines={5} label="Loading the student's record" />}
      {r && (
        <div className="casefile">
          <nav className="outline" aria-label="Case file sections">
            <ol>{OUTLINE.map(([k, l]) => <li key={k}><a href={`#${k}`} aria-current={active === k ? "true" : undefined}
              onClick={(e) => { e.preventDefault(); document.getElementById(k)?.scrollIntoView({ behavior: "smooth", block: "start" }); }}>{l}</a></li>)}</ol>
          </nav>
          <div className="body">
            <Section id="overview" title="Why this student is flagged" sub="Computed by the risk and outcome models; no AI text here.">
              <div className="grid two">
                <div className="stack">
                  <p className="small muted" style={{ marginBottom: 0 }}>{r.interpretation}</p>
                  <h3 className="h4">Likely final outcome</h3>
                  <Stacked label="Predicted final outcome" parts={OUTCOMES.map((k) => ({ label: k, value: o.probabilities[k], display: fmt.pct(o.probabilities[k]), color: COL[k] }))} />
                </div>
                <div className="stack">
                  <h3 className="h4">What moves the estimate</h3>
                  <Diverging items={r.drivers.map((d: any) => ({ label: d.label, value: d.contribution_logodds, title: `${d.label}: ${d.value ?? "n/a"}` }))} />
                  <div className="small muted">Method: {r.drivers[0]?.method?.replace("_", " ")}. Effects on the model's log-odds scale, relative to a typical student. Drivers explain the model, not the student.</div>
                </div>
              </div>
            </Section>

            <Section id="course" title="Across the course" sub="Only what was known at each point; later weeks stay hidden.">
              <div className="grid two">
                <div className="stack">
                  <h3 className="h4">Withdrawal risk at each checkpoint</h3>
                  {timeline.isLoading && <Loading lines={2} label="Recomputing risk at earlier weeks" />}
                  {timeline.data && <RiskTrail points={(timeline.data as any).points.filter((x: any) => x.checkpoint <= (checkpoint ?? 1) + 1e-9)} current={checkpoint ?? 0} />}
                  <p className="small muted">Dots are coloured by tier; the ringed dot is the selected week.</p>
                </div>
                <div className="stack">
                  <h3 className="h4">Weekly online activity</h3>
                  {td?.weekly_clicks && <>
                    <ActivityChart label="This student's weekly activity compared with the cohort median" snapshotWeek={td.snapshot_week}
                      deadlines={td.assessment_deadlines.filter((d: any) => d.week <= td.snapshot_week)}
                      series={[{ key: "median", name: "Cohort median", values: td.cohort_median_weekly_clicks.slice(0, td.snapshot_week + 1), color: COL.median, width: 2, dash: "5 4" },
                        { key: "student", name: "This student", values: td.weekly_clicks.slice(0, td.snapshot_week + 1), color: COL.ink, width: 2.5 }]} />
                    <div className="legend"><span><i style={{ background: COL.ink }} />this student</span><span><i style={{ background: COL.median }} />cohort median</span><span>dashed lines: deadlines</span></div>
                    {td.signals.length > 0 && <div>{td.signals.map((s: string) => <span key={s} className="chip">{s.replaceAll("_", " ")}</span>)}</div>}
                  </>}
                </div>
              </div>
            </Section>

            <Section id="plan" title="Support plan" sub="Facts and scores are computed. The AI only chooses from eligible actions and drafts wording; code checks every draft before you see it.">
              {analysis.status === "idle" && <Empty title="No plan yet" action={<div className="row">
                <button className={aiOn ? "ai primary" : "primary"} onClick={() => runAnalysis(true)}><PenLine aria-hidden="true" />{aiOn ? "Draft a support plan" : "Build a support plan"}</button>
                {aiOn && <button onClick={() => runAnalysis(false)}>Computed plan only</button>}</div>}>
                The agents review performance, explain the risk with similar past students, choose supportive actions and draft an outreach message for you to edit.</Empty>}
              {analysis.status === "running" && <div className="panel" aria-label="Agent progress"><Steps steps={analysis.steps || []} expected={AGENT_STEPS} /></div>}
              {analysis.status === "error" && <ErrorNote error={analysis.error} onRetry={() => runAnalysis(true)} />}
              {analysis.status === "done" && <Plan res={analysis.data} refStudent={ref} steps={analysis.steps || []} />}
            </Section>

            <Section id="precedents" title="Similar students from earlier presentations" sub="Same module, earlier cohorts only, and what happened to them after this point.">
              {similar.error && <ErrorNote error={similar.error as ApiError} />}
              {!similar.data && !similar.error && <Loading lines={4} label="Finding similar students" />}
              {similar.data && <Precedents data={similar.data} synthesis={analysis.data?.precedent_synthesis} />}
            </Section>

            <WhatIf ref_={ref} checkpoint={checkpoint} base={pred.data} />
          </div>
        </div>
      )}
    </>
  );
}

function Plan({ res, refStudent, steps }: { res: any; refStudent: any; steps: StepState[] }) {
  const { notify } = useStore();
  const wf = res.workflow;
  const plan = wf.intervention_plan;
  const [message, setMessage] = useState<string>(plan.outreach_message);
  const [variant, setVariant] = useState({ channel: "email", tone: "warm", language: "English" });
  const [rewrite, setRewrite] = useState<{ status: string; data?: any; error?: ApiError }>({ status: "idle" });
  const [judge, setJudge] = useState<{ status: string; data?: any; error?: ApiError }>({ status: "idle" });
  const doRewrite = async () => {
    setRewrite({ status: "running" });
    try { const v = await api("/outreach/variants", { method: "POST", body: { message, ...variant } }); setRewrite({ status: "done", data: v }); setMessage(v.message); }
    catch (e) { setRewrite({ status: "error", error: e as ApiError }); }
  };
  const doJudge = async () => {
    setJudge({ status: "running" });
    try { setJudge({ status: "done", data: await api("/evaluations/judge", { method: "POST", body: { recommendation_id: wf.recommendation_id } }) }); }
    catch (e) { setJudge({ status: "error", error: e as ApiError }); }
  };
  const msgSource = rewrite.status === "done" ? rewrite.data.source : plan.source;
  const refs: Record<string, string> = Object.fromEntries([...wf.observations.map((o: any) => [o.id, o.text]),
    ...(wf.eligible_actions || []).map((a: any) => [a.id, a.text]), ...plan.actions.map((a: any) => [a.id, a.text])]);
  return (
    <div className="stack">
      <details><summary>How the agents worked ({steps.length} steps)</summary><div className="panel tight" style={{ marginTop: 8 }}><Steps steps={steps} expected={AGENT_STEPS} running={false} /></div></details>
      <div className="grid two">
        <div className="panel stack">
          <div className="row"><h3 style={{ marginRight: "auto" }}>{plan.priority}</h3><span className="small muted">review in {plan.review_in_days} days</span></div>
          <div className="computed-block"><div className="who">Actions <Source source="computed" /> <span>chosen from the eligible catalogue</span></div>
            <ul>{plan.actions.map((a: any) => <li key={a.id}><strong>{a.text}</strong> <span className="small muted">({a.channel})</span></li>)}</ul></div>
          <Generated source={plan.source} title="Why these actions"><p><RefText text={plan.rationale} refs={refs} /></p></Generated>
          <Generated source={wf.performance_analysis.source} title="Performance"><p><RefText text={wf.performance_analysis.summary} refs={refs} /></p></Generated>
          <Generated source={wf.risk_analysis.source} title="Risk"><p><RefText text={wf.risk_analysis.risk_summary} refs={refs} /></p><p className="small">{wf.risk_analysis.historical_context}</p>
            <p className="small muted">{wf.risk_analysis.uncertainty_note}</p></Generated>
        </div>
        <div className="panel stack">
          <h3>Message to the student</h3>
          <Generated source={msgSource} title="Outreach draft" extra={<span style={{ fontWeight: 400 }}>edit freely before sending</span>}>
            <label className="visually-hidden" htmlFor="outreach">Outreach message</label>
            <textarea id="outreach" rows={9} value={message} onChange={(e) => setMessage(e.target.value)} />
          </Generated>
          <div className="row">
            <select aria-label="Channel" value={variant.channel} onChange={(e) => setVariant({ ...variant, channel: e.target.value })}>
              <option value="email">Email</option><option value="sms">Text message</option><option value="phone_script">Phone script</option></select>
            <select aria-label="Tone" value={variant.tone} onChange={(e) => setVariant({ ...variant, tone: e.target.value })}>
              <option value="warm">Warm</option><option value="concise">Concise</option><option value="formal">Formal</option></select>
            <select aria-label="Language" value={variant.language} onChange={(e) => setVariant({ ...variant, language: e.target.value })}>
              {["English", "French", "Spanish", "German", "Welsh"].map((l) => <option key={l}>{l}</option>)}</select>
            <button className="ai" onClick={doRewrite} disabled={rewrite.status === "running"}><PenLine aria-hidden="true" />{rewrite.status === "running" ? "Rewriting…" : "Rewrite"}</button>
            <CopyButton text={message} />
          </div>
          {rewrite.status === "done" && <div className="small">{rewrite.data.subject && <>Subject: {rewrite.data.subject}. </>}
            {rewrite.data.english_back_translation && <>English check: “{rewrite.data.english_back_translation}”</>}
            {rewrite.data.checks?.length > 0 && <div className="muted">The AI rewrite was not used: {rewrite.data.checks.join("; ")}</div>}</div>}
          {rewrite.status === "error" && <ErrorNote error={rewrite.error} />}
          <p className="small muted">Drafts never mention risk scores, predictions or withdrawal.</p>
        </div>
      </div>
      <div className="grid two">
        <div className="stack">
          <h3 className="h4">Checks on this plan</h3>
          <p>{wf.guardrails.final_plan_violations.length === 0 ? "Passed every check: eligible actions only, numbers traceable to the evidence, no protected attributes, no punitive or clinical wording." :
            `Issues: ${wf.guardrails.final_plan_violations.join("; ")}`}</p>
          <div className="row"><button onClick={doJudge} disabled={judge.status === "running"}><Scale aria-hidden="true" />{judge.status === "running" ? "Reviewing…" : "Ask the AI judge to review"}</button>
            <span className="small muted">A separate judge model scores the plan; the verdict is computed from its scores.</span></div>
          {judge.status === "error" && <ErrorNote error={judge.error} />}
          {judge.status === "done" && (judge.data.scores ? (
            <Generated source="llm" title={`Judge verdict ${judge.data.verdict}, ${fmt.num(judge.data.overall, 1)} of 10`}>
              <div className="row">{Object.entries(judge.data.scores).map(([k, v]) => <span key={k} className="chip">{k.replaceAll("_", " ")} {String(v)}</span>)}</div>
              {judge.data.rationale && <p className="small">{judge.data.rationale}</p>}</Generated>
          ) : <p className="muted">{judge.data.status}</p>)}
        </div>
        <div className="stack">
          <h3 className="h4">Agent trace</h3>
          <ol className="small">{wf.trace.map((t: any, i: number) => <li key={i}>{t.agent.replaceAll("_", " ")}{t.source ? ` (${t.source})` : ""}{t.decision ? `: ${t.decision}` : ""}
            {t.llm_rejected ? `: AI draft rejected (${t.llm_rejected.join("; ")})` : ""}{t.violations?.length ? `: ${t.violations.join("; ")}` : ""}</li>)}</ol>
          <div className="small muted">AI calls {wf.llm_usage.calls ?? 0}, tokens {fmt.int((wf.llm_usage.prompt_tokens || 0) + (wf.llm_usage.completion_tokens || 0))}</div>
          <Json data={wf.observations} label="Evidence the agents could use" />
        </div>
      </div>
      <FeedbackForm res={res} refStudent={refStudent} onDone={() => notify("Feedback saved. Thank you.")} />
    </div>
  );
}

/** AI text cites observation and action IDs (that is how its grounding is checked). Show them as readable chips;
 *  hovering or focusing a chip shows the full observation or action. */
function RefText({ text, refs }: { text: string; refs: Record<string, string> }) {
  const parts = (text || "").split(/\b((?:ACT|OBS)_[A-Z0-9_]+)\b/);
  return <>{parts.map((p, i) => (i % 2 === 1
    ? <abbr key={i} className="chip ref" title={refs[p] || p} tabIndex={0}>{p.replace(/^(ACT|OBS)_/, "").toLowerCase().replaceAll("_", " ")}</abbr>
    : <span key={i}>{p}</span>))}</>;
}

function Precedents({ data, synthesis }: { data: any; synthesis?: any }) {
  const [hl, setHl] = useState<number | null>(null);
  const nb = data.outcome_summary, af = data.aftermath;
  if (!data.cases.length) return <Empty title="No comparable students found">Earlier presentations of this module have no close matches.</Empty>;
  const sw = data.cases[0]?.story?.snapshot_week;
  return (
    <div className="stack">
      <div className="grid two">
        <div className="stack">
          {nb && <>
            <h3 className="h4">Outcomes of the {nb.n_neighbours} most similar students</h3>
            <Stacked label="Outcomes of similar students" parts={OUTCOMES.map((k) => ({ label: k, value: nb.counts[k] || 0, color: COL[k] }))} />
            <p className="small">{fmt.pct(nb.fail_or_withdrawn_rate)} failed or withdrew (95% interval {fmt.pct(nb.fail_or_withdrawn_ci95[0])} to {fmt.pct(nb.fail_or_withdrawn_ci95[1])}).</p>
          </>}
          {af && <dl className="kv">
            <dt>Median weeks to withdrawal</dt><dd>{af.median_weeks_to_withdrawal ?? "–"}</dd>
            <dt>Median change in activity after this point</dt><dd>{fmt.pct(af.median_activity_change_after_checkpoint)}</dd>
            <dt>More than halved their activity</dt><dd>{fmt.pct(af.share_with_activity_drop_over_50pct)}</dd>
            <dt>Median final coursework average</dt><dd>{fmt.num(af.median_final_coursework_score, 0)}</dd></dl>}
          {synthesis ? <Generated source={synthesis.source} title="What happened next">
            <p>{synthesis.what_happened}</p>{synthesis.patterns?.length > 0 && <ul>{synthesis.patterns.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul>}
            <p><strong>{synthesis.implication}</strong></p>
            <div className="small muted">Based on cases {synthesis.cited_cases.join(", ")}</div></Generated>
            : <p className="small muted">A written summary of these precedents is added when you build a support plan.</p>}
        </div>
        <div className="stack">
          <h3 className="h4">Their whole-course activity</h3>
          <Trajectories label="Weekly activity of similar students over the whole course" snapshotWeek={sw} highlight={hl} onHighlight={setHl}
            items={data.cases.filter((c: any) => c.story?.trajectory).map((c: any) => ({ id: c.case_id, result: c.final_result, values: c.story.trajectory }))} />
        </div>
      </div>
      <div className="table-wrap"><table className="data">
        <caption className="visually-hidden">Similar students; hover a row to highlight their line in the chart</caption>
        <thead><tr><th scope="col">Case</th><th scope="col">Presentation</th><th scope="col">Outcome</th><th scope="col">What happened after this point</th><th scope="col">Shared signals</th></tr></thead>
        <tbody>{data.cases.map((c: any) => (
          <tr key={c.case_id} className={hl === c.case_id ? "hl" : undefined} onMouseEnter={() => setHl(c.case_id)} onMouseLeave={() => setHl(null)}
            onFocus={() => setHl(c.case_id)} onBlur={() => setHl(null)} tabIndex={0}>
            <td>{c.case_id}</td><td>{c.code_presentation}</td><td><strong style={{ color: COL[c.final_result] }}>{c.final_result}</strong></td>
            <td>{c.story?.text || c.summary}</td>
            <td>{c.shared_descriptors.filter((x: string) => !x.startsWith("module_")).slice(0, 4).map((x: string) => <span key={x} className="chip">{x.replaceAll("_", " ")}</span>)}</td></tr>))}</tbody>
      </table></div>
    </div>
  );
}

function WhatIf({ ref_, checkpoint, base }: { ref_: any; checkpoint: number | null; base: any }) {
  const snap = useQuery({ queryKey: ["snapshot", ref_, checkpoint], enabled: !!checkpoint,
    queryFn: ({ signal }) => api(`/students/${ref_.id_student}`, { query: { code_module: ref_.code_module, code_presentation: ref_.code_presentation, checkpoint }, signal }) });
  const [v, setV] = useState<any>(null);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  useEffect(() => {
    if (!snap.data) return;
    const s = (snap.data as any).snapshot;
    setV({ weighted_score: s.weighted_score ?? 60, missed_rate: s.missed_rate ?? 0, days_since_last_activity: s.days_since_last_activity ?? 0,
      clicks_pct: s.clicks_pct ?? 0.5, snap: s, moved: [] });
    setRes(null);
  }, [snap.data]);
  useEffect(() => {
    if (!v) return;
    if (!v.moved.length) { setRes(null); return; } // nothing moved: the real record's estimate is shown above
    const t = setTimeout(async () => {
      const s = v.snap;
      try {
        setErr(null);
        const what_if: Record<string, number> = {}; // send only the signals the user moved
        for (const k of v.moved) {
          if (k === "weighted_score") Object.assign(what_if, { weighted_score: v.weighted_score, mean_score: v.weighted_score, last_score: v.weighted_score,
            no_scores_yet: 0, min_score: Math.min(v.weighted_score, s.min_score ?? v.weighted_score), max_score: Math.max(v.weighted_score, s.max_score ?? v.weighted_score) });
          else what_if[k] = v[k];
        }
        setRes(await api("/predict-risk", { method: "POST", body: { student: ref_, checkpoint, what_if } }));
      } catch (e) { setErr(e as ApiError); }
    }, 300);
    return () => clearTimeout(t);
  }, [v]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!v || !base) return null;
  const b = base.withdrawal_risk.withdrawal_risk;
  const w = res?.withdrawal_risk.withdrawal_risk;
  const slider = (key: string, label: string, min: number, max: number, step: number, show: (x: number) => string) => (
    <label className="field">{label}: <strong>{show(v[key])}</strong>
      <input type="range" min={min} max={max} step={step} value={v[key]}
        onChange={(e) => setV({ ...v, [key]: Number(e.target.value), moved: v.moved.includes(key) ? v.moved : [...v.moved, key] })} /></label>);
  return (
    <Section id="whatif" title="What if" sub="Change a few signals of this real record to see how the estimate responds. Associations, not guaranteed effects.">
      <div className="grid two">
        <div className="stack">
          {slider("weighted_score", "Weighted assessment score", 0, 100, 1, (x) => x.toFixed(0))}
          {slider("missed_rate", "Share of due work missed", 0, 1, 0.05, (x) => fmt.pct(x))}
          {slider("days_since_last_activity", "Days since last online activity", 0, v.snap.snapshot_day || 100, 1, (x) => String(x))}
          {slider("clicks_pct", "Activity compared with cohort (percentile)", 0, 1, 0.05, (x) => fmt.pct(x))}
        </div>
        <div className="stack" aria-live="polite">
          {err && <ErrorNote error={err} />}
          {!res && !err && <p className="muted">Move a slider to see how the estimate responds.</p>}
          {res && <>
            <div className="stats"><div className="stat"><div className="v">{fmt.pct(w)}</div><div className="l">withdrawal risk with these changes</div></div>
              <div className="stat"><div className="v" style={{ color: w > b ? COL.High : COL.Low }}>{w > b ? "+" : ""}{fmt.pct(w - b)}</div>
                <div className="l">compared with the real record ({fmt.pct(b)})</div></div></div>
            <Tier tier={res.withdrawal_risk.risk_tier} />
            {res.input_quality.out_of_distribution.length > 0 && <div className="note warn">Some changed values fall outside what the model has seen: {res.input_quality.out_of_distribution.map((x: any) => x.label).join(", ")}.</div>}
          </>}
        </div>
      </div>
    </Section>
  );
}

function FeedbackForm({ res, refStudent, onDone }: { res: any; refStudent: any; onDone: () => void }) {
  const wf = res.workflow;
  const [f, setF] = useState({ rating: 4, plan_accepted: true, risk_assessment_agrees: true, notes: "", observed_outcome: "" });
  const [acts, setActs] = useState<Record<string, boolean>>(Object.fromEntries(wf.intervention_plan.actions.map((a: any) => [a.id, true])));
  const [cases, setCases] = useState<Record<string, boolean>>({});
  const [state, setState] = useState<{ status: string; error?: ApiError }>({ status: "idle" });
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setState({ status: "saving" });
    try {
      await api("/feedback", { method: "POST", body: { recommendation_id: wf.recommendation_id, id_student: refStudent.id_student, rating: f.rating,
        plan_accepted: f.plan_accepted, risk_assessment_agrees: f.risk_assessment_agrees, notes: f.notes, observed_outcome: f.observed_outcome || null,
        action_feedback: Object.fromEntries(Object.entries(acts).map(([k, val]) => [k, { accepted: val }])), case_feedback: cases } });
      setState({ status: "saved" }); onDone();
    } catch (err) { setState({ status: "error", error: err as ApiError }); }
  };
  if (state.status === "saved") return <p className="note">Feedback saved for recommendation {wf.recommendation_id}. It feeds the next improvement run.</p>;
  return (
    <form className="panel stack" onSubmit={submit} aria-labelledby="fb-h">
      <h3 id="fb-h">Was this useful?</h3>
      <div className="row">
        <label className="field">Overall usefulness: {f.rating} of 5<input type="range" min="1" max="5" value={f.rating} onChange={(e) => setF({ ...f, rating: Number(e.target.value) })} /></label>
        <label><input type="checkbox" checked={f.plan_accepted} onChange={(e) => setF({ ...f, plan_accepted: e.target.checked })} /> I will act on this plan</label>
        <label><input type="checkbox" checked={f.risk_assessment_agrees} onChange={(e) => setF({ ...f, risk_assessment_agrees: e.target.checked })} /> I agree with the risk assessment</label>
      </div>
      <fieldset style={{ border: 0, padding: 0 }}><legend className="small muted">Keep or drop each action</legend>
        {wf.intervention_plan.actions.map((a: any) => <label key={a.id} style={{ display: "block" }}><input type="checkbox" checked={acts[a.id]} onChange={(e) => setActs({ ...acts, [a.id]: e.target.checked })} /> {a.text}</label>)}</fieldset>
      <fieldset style={{ border: 0, padding: 0 }}><legend className="small muted">Which similar students were useful precedents?</legend>
        <div className="row">{res.similar_cases.cases.map((c: any) => <label key={c.case_id} className="chip"><input type="checkbox" checked={!!cases[c.case_id]}
          onChange={(e) => setCases({ ...cases, [c.case_id]: e.target.checked })} /> {c.case_id} ({c.final_result})</label>)}</div></fieldset>
      <label className="field">Outcome, if already known<select value={f.observed_outcome} onChange={(e) => setF({ ...f, observed_outcome: e.target.value })}>
        <option value="">Not known yet</option>{OUTCOMES.map((x) => <option key={x}>{x}</option>)}</select></label>
      <label className="field">Notes for the team (optional)<textarea rows={3} maxLength={2000} value={f.notes} onChange={(e) => setF({ ...f, notes: e.target.value })} /></label>
      {state.status === "error" && <ErrorNote error={state.error} />}
      <div><button className="primary" type="submit" disabled={state.status === "saving"}>Save feedback</button></div>
    </form>
  );
}
