import { useState } from "react";
import { api, fmt, settings, type ApiError } from "../api";
import { useStore } from "../store";
import { ErrorNote } from "../ui";

/* eslint-disable @typescript-eslint/no-explicit-any */
export default function Settings() {
  const { reload, ready, config } = useStore();
  const [f, setF] = useState({ base: settings.base, key: settings.key, admin: settings.admin });
  const [check, setCheck] = useState<{ status: string; data?: any; error?: ApiError }>({ status: "idle" });
  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    settings.save(f);
    setCheck({ status: "checking" });
    try { const r = await api("/ready"); setCheck({ status: "ok", data: r }); reload(); }
    catch (err) { setCheck({ status: "error", error: err as ApiError }); }
  };
  const r = check.data || ready;
  return (
    <>
      <div className="page-head"><div><h1>Settings</h1><div className="sub">Where the console finds the service, and the keys it uses. Stored in this browser only.</div></div></div>
      <form className="panel stack" onSubmit={save} style={{ maxWidth: 640 }}>
        <label className="field">Service address (leave empty when the console is served by the API)<input value={f.base} onChange={(e) => setF({ ...f, base: e.target.value })} placeholder="http://localhost:8000" /></label>
        <label className="field">API key<input type="password" value={f.key} onChange={(e) => setF({ ...f, key: e.target.value })} autoComplete="off" /></label>
        <label className="field">Admin key (only needed to apply feedback-loop changes)<input type="password" value={f.admin} onChange={(e) => setF({ ...f, admin: e.target.value })} autoComplete="off" /></label>
        <div><button className="primary" type="submit">Save and test connection</button></div>
        <div aria-live="polite">
          {check.status === "checking" && <p className="muted">Checking…</p>}
          {check.status === "error" && <ErrorNote error={check.error} />}
          {check.status === "ok" && <p>Connected.</p>}
        </div>
      </form>
      {r && <section className="section stack" style={{ maxWidth: 720 }} aria-labelledby="status-h">
        <h2 id="status-h">Service status</h2>
        <table className="data"><tbody>{Object.entries(r.checks).map(([k, v]) => <tr key={k}><td>{k.replaceAll("_", " ")}</td><td>{v ? "Ready" : "Missing: run the build"}</td></tr>)}
          <tr><td>Course checkpoints</td><td>{r.checkpoints.filter((c: any) => c.available).map((c: any) => `week ${fmt.week(c.checkpoint)}`).join(", ")}</td></tr>
          <tr><td>Historical cases indexed</td><td>{r.index?.n_docs?.toLocaleString("en-GB") ?? "-"}</td></tr>
          <tr><td>AI drafting</td><td>{r.llm_configured ? `${r.llm_provider === "anthropic" ? "Claude" : r.llm_provider}: ${r.fast_model} (search, outreach), ${r.llm_model} (plans, copilot), judged by ${r.judge_model}`
            : "Not connected. Everything works with computed text; set ANTHROPIC_API_KEY on the server for AI drafting."}</td></tr>
          {config && <tr><td>Meaning-based search</td><td>{config.dense_retrieval === "built" ? "On: typed searches also match by meaning" : config.dense_retrieval || "Off: keyword and profile matching only"}</td></tr>}
        </tbody></table>
        <p className="small"><a href={`${settings.base}/api/docs`} target="_blank" rel="noreferrer">Open the API documentation</a></p>
      </section>}
    </>
  );
}
