import React, { useEffect, useRef, useState } from "react";
import { runJob } from "../api.js";
import { useStore } from "../store.jsx";
import { ErrorNote, Generated, Source } from "../ui.jsx";

export default function Copilot({ query }) {
  const { module, presentation, checkpoint, config } = useStore();
  const [msgs, setMsgs] = useState([]);
  const [q, setQ] = useState(query?.q || "");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const end = useRef(null);
  useEffect(() => end.current?.scrollIntoView({ behavior: "smooth", block: "end" }), [msgs]);
  const suggestions = [`Who should I contact first in ${module} ${presentation}?`, `Give me an overview of ${module} ${presentation}`,
    `Which assessments matter most in ${module}?`, "Which factors are most associated with withdrawal?", "How accurate is the withdrawal model?"];
  const ask = async (text) => {
    const question = (text ?? q).trim();
    if (!question || busy) return;
    setQ(""); setErr(null); setBusy(true);
    const history = msgs.map((m) => ({ role: m.role, content: m.content })).slice(-10);
    setMsgs((m) => [...m, { role: "user", content: question }]);
    try {
      const r = await runJob("/jobs/copilot", { question, history, checkpoint, use_llm: true });
      setMsgs((m) => [...m, { role: "assistant", content: r.answer, meta: r }]);
    } catch (e) { setErr(e); }
    setBusy(false);
  };
  return (
    <>
      <div className="page-head"><div><h1>Ask the copilot</h1>
        <div className="sub">Questions about students, cohorts, assessments and the models. Every answer is built from tool calls you can inspect.
          {!config.llm.configured && " The LLM isn't connected, so a rule-based assistant answers common questions."}</div></div></div>
      <div className="chat" aria-live="polite">
        {msgs.length === 0 && <div className="stack"><p className="muted">Try one of these:</p><div className="row">{suggestions.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div></div>}
        {msgs.map((m, i) => m.role === "user" ? <div key={i} className="msg user">{m.content}</div> : (
          <div key={i} className="msg assistant">
            <Generated source={m.meta.source} title="Copilot" extra={m.meta.revised ? <span className="small">revised for grounding</span> : null}>
              <p style={{ whiteSpace: "pre-wrap" }}>{m.content}</p>
              <div className="small muted">
                {m.meta.tool_calls.length ? `Used ${m.meta.tool_calls.map((t) => t.tool.replaceAll("_", " ")).join(", ")}. ` : ""}
                {m.meta.grounding.passed ? "Every number checked against tool results." : `Numbers not found in tool results: ${m.meta.grounding.ungrounded_numbers.join(", ")}. Verify before relying on them.`}
                {m.meta.flags?.length ? ` Flags: ${m.meta.flags.join("; ")}` : ""} {m.meta.seconds}s</div>
              {m.meta.tool_calls.length > 0 && <details><summary>See tool calls</summary>{m.meta.tool_calls.map((t, j) => (
                <div key={j} style={{ marginTop: 8 }}><strong>{t.tool}</strong> <span className="small muted">{JSON.stringify(t.arguments)} ({t.ms} ms){t.ok ? "" : " failed"}</span>
                  <pre className="json">{JSON.stringify(t.result, null, 2).slice(0, 4000)}</pre></div>))}</details>}
            </Generated>
          </div>
        ))}
        {busy && <div className="msg assistant muted">Working on it…</div>}
        <div ref={end} />
      </div>
      {err && <ErrorNote error={err} />}
      <form className="row section" onSubmit={(e) => { e.preventDefault(); ask(); }}>
        <input style={{ flex: 1, minWidth: 260 }} value={q} onChange={(e) => setQ(e.target.value)} maxLength={1500} aria-label="Your question"
          placeholder={`e.g. Tell me about student 59272 in ${module} ${presentation}`} />
        <button className="primary" type="submit" disabled={busy || !q.trim()}>Ask</button>
        {msgs.length > 0 && <button type="button" onClick={() => setMsgs([])}>Start over</button>}
        <Source source={config.llm.configured ? "llm" : "rules"} />
      </form>
    </>
  );
}
