import React, { useState } from "react";
import { api, fmt, settings } from "../api.js";
import { useApi, useStore } from "../store.jsx";
import { Empty, ErrorNote, Generated, Loading, Section } from "../ui.jsx";

export default function Feedback() {
  const { notify } = useStore();
  const summary = useApi((s) => api("/feedback/summary", { signal: s }), []);
  const list = useApi((s) => api("/feedback", { query: { limit: 30 }, signal: s }), []);
  const plan = useApi((s) => api("/improvement-plan", { signal: s }), []);
  const [themes, setThemes] = useState({ status: "idle" });
  const [run, setRun] = useState({ status: "idle" });
  const loadThemes = async () => { setThemes({ status: "loading" }); try { setThemes({ status: "done", data: await api("/feedback/themes") }); } catch (e) { setThemes({ status: "error", error: e }); } };
  const runLoop = async (apply) => {
    setRun({ status: "running" });
    try {
      const r = await api("/feedback-loop/run", { method: "POST", body: { apply, approved_by: apply ? "console" : null }, admin: true });
      setRun({ status: "done", data: r }); plan.reload();
      notify(apply ? (r.applied.length ? `Applied: ${r.applied.join(", ")}` : "Nothing was ready to apply") : "Proposals updated");
    } catch (e) { setRun({ status: "error", error: e }); }
  };
  const s = summary.data;
  const p = run.data || (plan.data?.proposals ? plan.data : null);
  return (
    <>
      <div className="page-head"><div><h1>Feedback loop</h1><div className="sub">Advisors' ratings of plans, actions and precedents, and the changes they justify. Nothing changes until an admin applies it.</div></div></div>
      {summary.error && <ErrorNote error={summary.error} />}
      {s && (s.count === 0 ? <Empty title="No feedback yet">Draft a support plan on any student page and rate it at the bottom of the plan.</Empty> :
        <div className="stats">
          <div className="stat"><div className="v">{s.count}</div><div className="l">pieces of feedback</div></div>
          <div className="stat"><div className="v">{fmt.num(s.average_rating, 1)}</div><div className="l">average usefulness out of 5</div></div>
          <div className="stat"><div className="v">{fmt.pct(s.plan_acceptance_rate)}</div><div className="l">plans advisors chose to act on</div></div>
          <div className="stat"><div className="v">{fmt.pct(s.risk_agreement_rate)}</div><div className="l">agreed with the risk assessment</div></div>
          <div className="stat"><div className="v">{s.with_observed_outcome}</div><div className="l">with a recorded real outcome</div></div>
        </div>)}

      <Section title="Proposed improvements" sub="Each learner needs enough feedback before it proposes a change."
        actions={<><button onClick={() => runLoop(false)} disabled={run.status === "running"}>Recalculate proposals</button>
          <button className="primary" onClick={() => runLoop(true)} disabled={run.status === "running" || !settings.admin} title={settings.admin ? "" : "Add an admin key in Settings"}>Apply ready changes</button></>}>
        {run.status === "error" && <ErrorNote error={run.error} />}
        {!settings.admin && <p className="small muted">Applying changes needs the admin key; add it in Settings.</p>}
        {plan.loading && !p && <Loading />}
        {p ? (
          <div className="stack">
            <table className="data"><thead><tr><th>Change</th><th>Status</th><th>Evidence</th></tr></thead>
              <tbody>{p.proposals.map((x) => <tr key={x.type}><td>{{ action_catalog_ranking: "Re-rank support actions by advisor acceptance",
                retrieval_fusion_weights: "Re-weight how similar students are matched", risk_recalibration: "Recalibrate risk on real outcomes" }[x.type] || x.type}</td>
                <td>{x.status.replaceAll("_", " ")}</td><td className="small muted">{JSON.stringify(x.evidence)}</td></tr>)}</tbody></table>
            {p.monitors?.length > 0 && <><h4>Monitors</h4><ul>{p.monitors.map((x, i) => <li key={i}>{x.type.replaceAll("_", " ")}{x.tier ? ` (${x.tier})` : ""}: {x.status}{x.action ? `. ${x.action}` : ""}</li>)}</ul></>}
            {p.applied?.length > 0 && <p>Applied in this run: {p.applied.join(", ")}.</p>}
          </div>
        ) : !plan.loading && <p className="muted">No proposals yet. Recalculate after some feedback arrives.</p>}
      </Section>

      <Section title="What advisors are saying" actions={<button onClick={loadThemes} disabled={themes.status === "loading"}>Find themes in notes</button>}>
        {themes.status === "error" && <ErrorNote error={themes.error} />}
        {themes.status === "done" && (themes.data.themes.length ? <Generated source={themes.data.source} title="Themes">
          <ul>{themes.data.themes.map((t, i) => <li key={i}><strong>{t.theme}</strong> ({t.count} notes){t.suggestion ? `: ${t.suggestion}` : ""}</li>)}</ul></Generated>
          : <p className="muted">No written notes yet.</p>)}
      </Section>

      <Section title="Recent feedback">
        {list.error && <ErrorNote error={list.error} />}
        {list.data && (list.data.items.length === 0 ? <p className="muted">Nothing yet.</p> :
          <div className="table-wrap"><table className="data"><thead><tr><th>When</th><th>Student</th><th>Tier</th><th className="num">Rating</th><th>Acted on</th><th>Outcome</th><th>Notes</th></tr></thead>
            <tbody>{list.data.items.map((r) => <tr key={r.id}><td>{new Date(r.created_at * 1000).toLocaleDateString()}</td><td>{r.code_module ? "" : ""}{r.recommendation_id.slice(0, 8)}</td>
              <td>{r.risk_tier}</td><td className="num">{r.rating}</td><td>{r.plan_accepted ? "Yes" : "No"}</td><td>{r.observed_outcome || "–"}</td><td className="small">{r.notes}</td></tr>)}</tbody></table></div>)}
      </Section>
    </>
  );
}
