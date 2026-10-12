import { useQuery } from "@tanstack/react-query";
import { PenLine } from "lucide-react";
import { useState } from "react";
import { api, fmt, settings, type ApiError } from "../api";
import { useStore } from "../store";
import { Empty, ErrorNote, Generated, Loading, Section } from "../ui";

/* eslint-disable @typescript-eslint/no-explicit-any */
const PROPOSALS: Record<string, string> = { action_catalog_ranking: "Re-rank support actions by advisor acceptance",
  retrieval_fusion_weights: "Re-weight how similar students are matched", risk_recalibration: "Recalibrate risk on real outcomes" };

export default function Feedback() {
  const { notify } = useStore();
  const summary = useQuery({ queryKey: ["fb-summary"], queryFn: ({ signal }) => api("/feedback/summary", { signal }) });
  const list = useQuery({ queryKey: ["fb-list"], queryFn: ({ signal }) => api("/feedback", { query: { limit: 30 }, signal }) });
  const plan = useQuery({ queryKey: ["fb-plan"], queryFn: ({ signal }) => api("/improvement-plan", { signal }) });
  const [themes, setThemes] = useState<{ status: string; data?: any; error?: ApiError }>({ status: "idle" });
  const [run, setRun] = useState<{ status: string; data?: any; error?: ApiError }>({ status: "idle" });
  const loadThemes = async () => {
    setThemes({ status: "loading" });
    try { setThemes({ status: "done", data: await api("/feedback/themes") }); } catch (e) { setThemes({ status: "error", error: e as ApiError }); }
  };
  const runLoop = async (apply: boolean) => {
    setRun({ status: "running" });
    try {
      const r = await api("/feedback-loop/run", { method: "POST", body: { apply, approved_by: apply ? "console" : null }, admin: true });
      setRun({ status: "done", data: r }); plan.refetch();
      notify(apply ? (r.applied.length ? `Applied: ${r.applied.join(", ")}` : "Nothing was ready to apply") : "Proposals updated");
    } catch (e) { setRun({ status: "error", error: e as ApiError }); }
  };
  const s: any = summary.data;
  const p: any = run.data || ((plan.data as any)?.proposals ? plan.data : null);
  return (
    <>
      <div className="page-head"><div><h1>Feedback loop</h1>
        <div className="sub">How advisors rated plans, actions and precedents, and the changes that feedback justifies. Nothing changes until an admin applies it.</div></div></div>
      {summary.error && <ErrorNote error={summary.error as ApiError} />}
      {s && (s.count === 0 ? <Empty title="No feedback yet">Draft a support plan on any student page and rate it at the bottom of the plan.</Empty> :
        <div className="stats">
          <div className="stat"><div className="v">{s.count}</div><div className="l">pieces of feedback</div></div>
          <div className="stat"><div className="v">{fmt.num(s.average_rating, 1)}</div><div className="l">average usefulness out of 5</div></div>
          <div className="stat"><div className="v">{fmt.pct(s.plan_acceptance_rate)}</div><div className="l">plans advisors chose to act on</div></div>
          <div className="stat"><div className="v">{fmt.pct(s.risk_agreement_rate)}</div><div className="l">agreed with the risk assessment</div></div>
          <div className="stat"><div className="v">{s.with_observed_outcome}</div><div className="l">with a recorded real outcome</div></div>
        </div>)}

      <Section id="proposals" title="Proposed improvements" sub="Each learner needs enough feedback before it proposes a change."
        actions={<><button onClick={() => runLoop(false)} disabled={run.status === "running"}>Recalculate proposals</button>
          <button className="primary" onClick={() => runLoop(true)} disabled={run.status === "running" || !settings.admin}>Apply ready changes</button></>}>
        {run.status === "error" && <ErrorNote error={run.error} />}
        {!settings.admin && <p className="small muted">Applying changes needs the admin key; add it in Settings.</p>}
        {plan.isLoading && !p && <Loading />}
        {p ? (
          <div className="stack">
            <table className="data"><thead><tr><th scope="col">Change</th><th scope="col">Status</th><th scope="col">Evidence</th></tr></thead>
              <tbody>{p.proposals.map((x: any) => <tr key={x.type}><td>{PROPOSALS[x.type] || x.type}</td>
                <td>{x.status.replaceAll("_", " ")}</td><td className="small muted">{JSON.stringify(x.evidence)}</td></tr>)}</tbody></table>
            {p.monitors?.length > 0 && <><h3 className="h4">Monitors</h3><ul>{p.monitors.map((x: any, i: number) => (
              <li key={i}>{x.type.replaceAll("_", " ")}{x.tier ? ` (${x.tier})` : ""}: {x.status}{x.action ? `. ${x.action}` : ""}</li>))}</ul></>}
            {p.applied?.length > 0 && <p>Applied in this run: {p.applied.join(", ")}.</p>}
          </div>
        ) : !plan.isLoading && <p className="muted">No proposals yet. Recalculate after some feedback arrives.</p>}
      </Section>

      <Section id="themes" title="What advisors are saying"
        actions={<button className="ai" onClick={loadThemes} disabled={themes.status === "loading"}><PenLine aria-hidden="true" />
          {themes.status === "loading" ? "Reading notes…" : "Find themes in notes"}</button>}>
        {themes.status === "error" && <ErrorNote error={themes.error} />}
        {themes.status === "done" && (themes.data.themes.length ? <Generated source={themes.data.source} title="Themes">
          <ul>{themes.data.themes.map((t: any, i: number) => <li key={i}><strong>{t.theme}</strong> ({t.count} notes){t.suggestion ? `: ${t.suggestion}` : ""}</li>)}</ul></Generated>
          : <p className="muted">No written notes yet.</p>)}
      </Section>

      <Section id="recent" title="Recent feedback">
        {list.error && <ErrorNote error={list.error as ApiError} />}
        {list.data && ((list.data as any).items.length === 0 ? <p className="muted">Nothing yet.</p> :
          <div className="table-wrap"><table className="data"><thead><tr><th scope="col">When</th><th scope="col">Recommendation</th><th scope="col">Tier</th>
            <th scope="col" className="num">Rating</th><th scope="col">Acted on</th><th scope="col">Outcome</th><th scope="col">Notes</th></tr></thead>
            <tbody>{(list.data as any).items.map((r: any) => <tr key={r.id}><td>{new Date(r.created_at * 1000).toLocaleDateString("en-GB")}</td>
              <td className="small">{r.recommendation_id.slice(0, 8)}</td><td>{r.risk_tier}</td><td className="num">{r.rating}</td><td>{r.plan_accepted ? "Yes" : "No"}</td>
              <td>{r.observed_outcome || "-"}</td><td className="small">{r.notes}</td></tr>)}</tbody></table></div>)}
      </Section>
    </>
  );
}
