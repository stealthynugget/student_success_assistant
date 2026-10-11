import React, { useState } from "react";
import { api, fmt } from "../api.js";
import { C, LineChart, Stacked } from "../charts.jsx";
import { useStore } from "../store.jsx";
import { Empty, ErrorNote, Generated, Loading, Seg } from "../ui.jsx";

const OUT = ["Distinction", "Pass", "Fail", "Withdrawn"];
const EXAMPLES = ["repeat students in FFF with heavy workloads who dropped out", "students with declining engagement who missed assessments",
  "first attempt students in BBB from Wales who passed", "students in DDD studying more than 90 credits who failed"];

export default function CaseFinder() {
  const { config, checkpoint } = useStore();
  const [query, setQuery] = useState("");
  const [interp, setInterp] = useState(null);
  const [f, setF] = useState({ code_module: "", region: "", highest_education: "", age_band: "", final_result: "", credits_min: "", credits_max: "", attempts: "" });
  const [mode, setMode] = useState("hybrid");
  const [topK, setTopK] = useState(12);
  const [res, setRes] = useState({ status: "idle" });

  const filters = () => {
    const out = {};
    ["code_module", "region", "highest_education", "age_band", "final_result"].forEach((k) => { if (f[k]) out[k] = f[k]; });
    if (f.credits_min || f.credits_max) out.studied_credits = { ...(f.credits_min ? { min: Number(f.credits_min) } : {}), ...(f.credits_max ? { max: Number(f.credits_max) } : {}) };
    if (f.attempts === "first") out.num_of_prev_attempts = { max: 0 };
    if (f.attempts === "repeat") out.num_of_prev_attempts = { min: 1 };
    return Object.keys(out).length ? out : null;
  };
  const interpret = async () => {
    if (query.trim().length < 3) return;
    try { setInterp(await api("/search/parse", { method: "POST", body: { query } })); } catch (e) { setInterp({ error: e }); }
  };
  const search = async (e) => {
    e?.preventDefault();
    setRes({ status: "loading" });
    try {
      const body = { query: query.trim() || "students", filters: filters(), top_k: topK, mode, checkpoint, understand: !!query.trim() };
      setRes({ status: "done", data: await api("/retrieve-similar", { method: "POST", body }) });
    } catch (err) { setRes({ status: "error", error: err }); }
  };
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  const v = config.vocab;
  return (
    <>
      <div className="page-head"><div><h1>Case finder</h1><div className="sub">Search every completed historical enrolment by description, filters or both. Results show what happened next.</div></div></div>
      <form className="panel stack" onSubmit={search}>
        <label className="field">Describe the students you're looking for
          <input value={query} onChange={(e) => { setQuery(e.target.value); setInterp(null); }} onBlur={interpret} placeholder="e.g. repeat students in FFF who stopped logging in and later withdrew" maxLength={500} /></label>
        <div className="row small">Try: {EXAMPLES.map((x) => <button type="button" key={x} className="ghost" onClick={() => { setQuery(x); setInterp(null); }}>{x}</button>)}</div>
        {interp && !interp.error && (
          <Generated source={interp.source} title="Understood as">
            <div>{Object.entries(interp.filters).map(([k, val]) => <span key={k} className="chip">{k.replaceAll("_", " ")}: {typeof val === "object" ? JSON.stringify(val) : String(val)}</span>)}
              {interp.descriptors.map((d) => <span key={d} className="chip ai">{d.replaceAll("_", " ")}</span>)}
              {!Object.keys(interp.filters).length && !interp.descriptors.length && <span className="muted">No specific filters recognised; the text will be matched as written.</span>}</div>
            <div className="small muted">Filters you set below override these.</div>
          </Generated>
        )}
        {interp?.error && <ErrorNote error={interp.error} />}
        <div className="grid three">
          <label className="field">Course<select value={f.code_module} onChange={set("code_module")}><option value="">Any</option>{config.modules.map((m) => <option key={m}>{m}</option>)}</select></label>
          <label className="field">Region<select value={f.region} onChange={set("region")}><option value="">Any</option>{v.regions.map((m) => <option key={m}>{m}</option>)}</select></label>
          <label className="field">Education level<select value={f.highest_education} onChange={set("highest_education")}><option value="">Any</option>{v.education.map((m) => <option key={m}>{m}</option>)}</select></label>
          <label className="field">Age band<select value={f.age_band} onChange={set("age_band")}><option value="">Any</option>{v.age_bands.map((m) => <option key={m}>{m}</option>)}</select></label>
          <label className="field">Studied credits<div className="row"><input type="number" min="0" max="700" placeholder="min" value={f.credits_min} onChange={set("credits_min")} style={{ width: 90 }} />
            <input type="number" min="0" max="700" placeholder="max" value={f.credits_max} onChange={set("credits_max")} style={{ width: 90 }} /></div></label>
          <label className="field">Previous attempts<select value={f.attempts} onChange={set("attempts")}><option value="">Any</option><option value="first">First attempt</option><option value="repeat">Repeating</option></select></label>
          <label className="field">Final outcome<select value={f.final_result} onChange={set("final_result")}><option value="">Any</option>{v.outcomes.map((m) => <option key={m}>{m}</option>)}</select></label>
          <label className="field">Matching method<Seg value={mode} onChange={setMode} label="Matching method" options={[["hybrid", "Combined"], ["lexical", "Keywords"], ["dense", "Meaning"]]} /></label>
          <label className="field">Results: {topK}<input type="range" min="5" max="50" value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></label>
        </div>
        <div><button className="primary" type="submit">Search cases</button></div>
      </form>
      <div className="section">
        {res.status === "idle" && <Empty title="Start with a description or a filter">Results come from completed presentations, as of the selected point on the course clock.</Empty>}
        {res.status === "loading" && <Loading lines={6} />}
        {res.status === "error" && <ErrorNote error={res.error} />}
        {res.status === "done" && <Results data={res.data} />}
      </div>
    </>
  );
}

function Results({ data }) {
  if (!data.cases.length) return <Empty title="No matching students">Loosen a filter, for example widen the credits range or remove the outcome.</Empty>;
  const nb = data.outcome_summary;
  const sw = data.cases[0]?.story?.snapshot_week;
  return (
    <div className="stack">
      <div className="grid two">
        <div className="stack">
          <h3>{data.candidates.toLocaleString()} students matched the filters; showing the closest {data.cases.length}</h3>
          {nb && <><Stacked parts={OUT.map((k) => ({ label: k, value: nb.counts[k] || 0, color: C[k] }))} />
            <p className="small">Among the closest {nb.n_neighbours}: {fmt.pct(nb.fail_or_withdrawn_rate)} failed or withdrew ({fmt.pct(nb.fail_or_withdrawn_ci95[0])} to {fmt.pct(nb.fail_or_withdrawn_ci95[1])}).</p></>}
          <p className="small muted">Matching used: {data.channels_used.join(" and ")}. {data.query_descriptors?.length ? `Signals searched: ${data.query_descriptors.map((d) => d.replaceAll("_", " ")).join(", ")}.` : ""}</p>
        </div>
        <div>
          <LineChart height={210} ariaLabel="Full-course activity of the matched students" xLabel="course week" yLabel="clicks per week"
            series={data.cases.filter((c) => c.story?.trajectory).map((c) => ({ name: String(c.case_id), values: c.story.trajectory, color: C[c.final_result], width: 1.4, opacity: 0.75 }))}
            vlines={sw != null ? [{ x: sw, label: "checkpoint" }] : []} />
          <div className="legend">{OUT.map((k) => <span key={k}><i style={{ background: C[k] }} />{k}</span>)}</div>
        </div>
      </div>
      <div className="table-wrap"><table className="data">
        <thead><tr><th>Case</th><th>Course</th><th>Outcome</th><th>Snapshot</th><th>After the checkpoint</th><th>Profile</th></tr></thead>
        <tbody>{data.cases.map((c) => (
          <tr key={c.case_id}><td>{c.case_id}</td><td>{c.code_module} {c.code_presentation}</td><td><strong style={{ color: C[c.final_result] }}>{c.final_result}</strong></td>
            <td className="small">score {fmt.num(c.snapshot.weighted_score, 0)}, missed {fmt.pct(c.snapshot.missed_rate)}, activity pct {fmt.pct(c.snapshot.clicks_pct)}</td>
            <td className="small">{c.story?.text}</td>
            <td className="small">{c.metadata.region}, {c.metadata.highest_education}, {c.metadata.age_band}, {c.snapshot.studied_credits} credits, {c.snapshot.num_of_prev_attempts} previous</td></tr>))}</tbody>
      </table></div>
    </div>
  );
}
