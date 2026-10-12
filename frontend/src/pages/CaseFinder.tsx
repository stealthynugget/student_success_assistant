import { Search } from "lucide-react";
import { useState } from "react";
import { api, fmt, type ApiError } from "../api";
import { useStore } from "../store";
import { Empty, ErrorNote, Generated, Loading, Seg } from "../ui";
import { COL, OUTCOMES, Stacked, Trajectories } from "../viz";

/* eslint-disable @typescript-eslint/no-explicit-any */
const EXAMPLES = ["repeat students in FFF with heavy workloads who dropped out", "students who stopped turning in their coursework",
  "first attempt students in BBB from Wales who passed", "learners in DDD whose marks kept slipping"];

export default function CaseFinder() {
  const { config, checkpoint } = useStore();
  const [query, setQuery] = useState("");
  const [interp, setInterp] = useState<any>(null);
  const [f, setF] = useState({ code_module: "", region: "", highest_education: "", age_band: "", final_result: "", credits_min: "", credits_max: "", attempts: "" });
  const [mode, setMode] = useState("hybrid");
  const [topK, setTopK] = useState(12);
  const [res, setRes] = useState<{ status: string; data?: any; error?: ApiError }>({ status: "idle" });
  if (!config) return null;

  const filters = () => {
    const out: Record<string, unknown> = {};
    (["code_module", "region", "highest_education", "age_band", "final_result"] as const).forEach((k) => { if (f[k]) out[k] = f[k]; });
    if (f.credits_min || f.credits_max) out.studied_credits = { ...(f.credits_min ? { min: Number(f.credits_min) } : {}), ...(f.credits_max ? { max: Number(f.credits_max) } : {}) };
    if (f.attempts === "first") out.num_of_prev_attempts = { max: 0 };
    if (f.attempts === "repeat") out.num_of_prev_attempts = { min: 1 };
    return Object.keys(out).length ? out : null;
  };
  const search = async (e?: React.FormEvent, text = query) => {
    e?.preventDefault();
    setRes({ status: "loading" });
    try {
      const q = text.trim();
      const data = await api("/retrieve-similar", { method: "POST", body: { query: q || "students", filters: filters(), top_k: topK, mode, checkpoint, understand: !!q } });
      setInterp(q ? data.interpretation : null);
      setRes({ status: "done", data });
    } catch (err) { setRes({ status: "error", error: err as ApiError }); }
  };
  const set = (k: keyof typeof f) => (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => setF({ ...f, [k]: e.target.value });
  const v = config.vocab;
  return (
    <>
      <div className="page-head"><div><h1>Case finder</h1>
        <div className="sub">Search every completed historical enrolment by description, filters or both, and see what happened next.</div></div></div>
      <form className="panel stack" onSubmit={search} role="search">
        <label className="field">Describe the students you're looking for
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. repeat students in FFF who stopped logging in and later withdrew" maxLength={500} /></label>
        <div className="row small"><span className="muted">Try:</span>{EXAMPLES.map((x) => <button type="button" key={x} className="ghost" onClick={() => { setQuery(x); search(undefined, x); }}>{x}</button>)}</div>
        <div className="grid three">
          <label className="field">Course<select value={f.code_module} onChange={set("code_module")}><option value="">Any</option>{config.modules.map((m) => <option key={m}>{m}</option>)}</select></label>
          <label className="field">Region<select value={f.region} onChange={set("region")}><option value="">Any</option>{v.regions.map((m) => <option key={m}>{m}</option>)}</select></label>
          <label className="field">Education level<select value={f.highest_education} onChange={set("highest_education")}><option value="">Any</option>{v.education.map((m) => <option key={m}>{m}</option>)}</select></label>
          <label className="field">Age band<select value={f.age_band} onChange={set("age_band")}><option value="">Any</option>{v.age_bands.map((m) => <option key={m}>{m}</option>)}</select></label>
          <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}><legend className="small muted" style={{ padding: 0, marginBottom: 4 }}>Studied credits</legend>
            <div className="row"><input type="number" min="0" max="700" placeholder="min" aria-label="Minimum credits" value={f.credits_min} onChange={set("credits_min")} style={{ width: 90 }} />
              <input type="number" min="0" max="700" placeholder="max" aria-label="Maximum credits" value={f.credits_max} onChange={set("credits_max")} style={{ width: 90 }} /></div></fieldset>
          <label className="field">Previous attempts<select value={f.attempts} onChange={set("attempts")}><option value="">Any</option><option value="first">First attempt</option><option value="repeat">Repeating</option></select></label>
          <label className="field">Final outcome<select value={f.final_result} onChange={set("final_result")}><option value="">Any</option>{v.outcomes.map((m) => <option key={m}>{m}</option>)}</select></label>
          <div className="field"><span>Matching method</span><Seg value={mode} onChange={setMode} label="Matching method" options={[["hybrid", "Combined"], ["lexical", "Keywords"], ["dense", "Meaning"]]} /></div>
          <label className="field">Results: {topK}<input type="range" min="5" max="50" value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></label>
        </div>
        <div><button className="primary" type="submit"><Search aria-hidden="true" />Search cases</button></div>
      </form>
      <div className="section" aria-live="polite">
        {interp && <div style={{ marginBottom: 16 }}><Generated source={interp.source} title="Your request was understood as">
          <div>{Object.entries(interp.filters).map(([k, val]) => <span key={k} className="chip">{k.replaceAll("_", " ")}: {typeof val === "object" ? JSON.stringify(val) : String(val)}</span>)}
            {interp.descriptors.map((d: string) => <span key={d} className="chip">{d.replaceAll("_", " ")}</span>)}
            {!Object.keys(interp.filters).length && !interp.descriptors.length && <span className="muted">No specific filters recognised; the text is matched as written.</span>}</div>
          <div className="small muted">Filters you set above take precedence. {interp.source === "llm" && interp.explanation ? interp.explanation : ""}</div>
        </Generated></div>}
        {res.status === "idle" && <Empty title="Start with a description or a filter">Results come from completed presentations, as of the selected point on the course clock.</Empty>}
        {res.status === "loading" && <Loading lines={6} label="Searching cases" />}
        {res.status === "error" && <ErrorNote error={res.error} onRetry={() => search()} />}
        {res.status === "done" && <Results data={res.data} />}
      </div>
    </>
  );
}

function Results({ data }: { data: any }) {
  const [hl, setHl] = useState<number | null>(null);
  if (!data.cases.length) return <Empty title="No matching students">Loosen a filter, for example widen the credits range or remove the outcome.</Empty>;
  const nb = data.outcome_summary;
  const sw = data.cases[0]?.story?.snapshot_week;
  return (
    <div className="stack">
      <div className="grid two">
        <div className="stack">
          <h2 className="h3">{data.applied_filters ? `${fmt.int(data.candidates)} students matched the filters` : `Searched all ${fmt.int(data.candidates)} historical students`};
            the closest {data.cases.length} are shown</h2>
          {nb && <><Stacked label="Outcomes among the closest matches" parts={OUTCOMES.map((k) => ({ label: k, value: nb.counts[k] || 0, color: COL[k] }))} />
            <p className="small">Among the closest {nb.n_neighbours}: {fmt.pct(nb.fail_or_withdrawn_rate)} failed or withdrew (95% interval {fmt.pct(nb.fail_or_withdrawn_ci95[0])} to {fmt.pct(nb.fail_or_withdrawn_ci95[1])}).</p></>}
          <p className="small muted">Matching used: {data.channels_used.join(" and ")}.
            {data.query_descriptors?.length ? ` Signals searched: ${data.query_descriptors.map((d: string) => d.replaceAll("_", " ")).join(", ")}.` : ""}</p>
        </div>
        <Trajectories label="Whole-course activity of the matched students" snapshotWeek={sw} highlight={hl} onHighlight={setHl} height={250}
          items={data.cases.filter((c: any) => c.story?.trajectory).map((c: any) => ({ id: c.case_id, result: c.final_result, values: c.story.trajectory }))} />
      </div>
      <div className="table-wrap"><table className="data">
        <caption className="visually-hidden">Matched historical students; hover a row to highlight their line in the chart</caption>
        <thead><tr><th scope="col">Case</th><th scope="col">Course</th><th scope="col">Outcome</th><th scope="col">At the checkpoint</th><th scope="col">After the checkpoint</th><th scope="col">Profile</th></tr></thead>
        <tbody>{data.cases.map((c: any) => (
          <tr key={c.case_id} className={hl === c.case_id ? "hl" : undefined} tabIndex={0} onMouseEnter={() => setHl(c.case_id)} onMouseLeave={() => setHl(null)}
            onFocus={() => setHl(c.case_id)} onBlur={() => setHl(null)}>
            <td>{c.case_id}</td><td>{c.code_module} {c.code_presentation}</td><td><strong style={{ color: COL[c.final_result] }}>{c.final_result}</strong></td>
            <td className="small">{c.snapshot.weighted_score == null ? "no marks yet" : `score ${fmt.num(c.snapshot.weighted_score, 0)}`}, missed {fmt.pct(c.snapshot.missed_rate)}, activity percentile {fmt.pct(c.snapshot.clicks_pct)}</td>
            <td className="small">{c.story?.text}</td>
            <td className="small">{c.metadata.region}, {c.metadata.highest_education}, {c.metadata.age_band}, {c.snapshot.studied_credits} credits, {c.snapshot.num_of_prev_attempts} previous</td></tr>))}</tbody>
      </table></div>
    </div>
  );
}
