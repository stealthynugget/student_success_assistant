import { ArrowUp, CheckCircle2, CircleAlert, LoaderCircle, RotateCcw, Wrench } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router";
import { ApiError, streamCopilot } from "../api";
import { studentPath, useStore } from "../store";
import { ErrorNote, Generated } from "../ui";

/* eslint-disable @typescript-eslint/no-explicit-any */
interface ToolCard { id: string; tool: string; arguments: Record<string, any>; ok?: boolean; ms?: number; result?: any }
interface Turn { role: "user" | "assistant"; content: string; status?: "streaming" | "done" | "error"; tools?: ToolCard[]; meta?: any; revised?: boolean }

const label = (t: string) => t.replaceAll("_", " ");

/** Student IDs the answer may cite, mapped to their module/presentation (from the tool calls that produced them). */
function citable(tools: ToolCard[]): Map<string, { m: string; p: string }> {
  const out = new Map<string, { m: string; p: string }>();
  for (const t of tools) {
    const a = t.arguments || {}, r = t.result || {};
    if (a.id_student && a.code_module) out.set(String(a.id_student), { m: String(a.code_module).toUpperCase(), p: String(a.code_presentation).toUpperCase() });
    if (t.tool === "rank_cohort" && Array.isArray(r.ranking))
      for (const x of r.ranking) out.set(String(x.id_student), { m: String(a.code_module).toUpperCase(), p: String(a.code_presentation).toUpperCase() });
  }
  return out;
}

/** Minimal, safe rendering of the answer: paragraphs, "-"/"1." lists, **bold**, cited student IDs as links, [tool] as card links. */
function Answer({ text, ids }: { text: string; ids: Map<string, { m: string; p: string }> }) {
  const inline = (s: string, key: string): ReactNode[] =>
    s.split(/(\*\*[^*]+\*\*|\[[a-z_]+\]|\b\d{5,8}\b)/g).map((part, i) => {
      const k = `${key}-${i}`;
      if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={k}>{inline(part.slice(2, -2), k)}</strong>;
      if (/^\[[a-z_]+\]$/.test(part)) return <a key={k} className="chip" href={`#tool-${part.slice(1, -1)}`}>{label(part.slice(1, -1))}</a>;
      const ref = ids.get(part);
      if (ref) return <Link key={k} to={studentPath(ref.m, ref.p, part)} title={`Open student ${part}`}>{part}</Link>;
      return <Fragment key={k}>{part}</Fragment>;
    });
  const blocks: ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = () => {
    if (!list) return;
    const L = list.ordered ? "ol" : "ul";
    blocks.push(<L key={`l${blocks.length}`}>{list.items.map((it, i) => <li key={i}>{inline(it, `li${blocks.length}-${i}`)}</li>)}</L>);
    list = null;
  };
  let tbl: string[][] | null = null;
  const flushTable = () => {
    if (!tbl) return;
    const [head, ...body] = tbl;
    const k = `t${blocks.length}`;
    blocks.push(<div key={k} className="table-wrap"><table className="data small">
      <thead><tr>{head.map((c, i) => <th key={i} scope="col">{inline(c, `${k}h${i}`)}</th>)}</tr></thead>
      <tbody>{body.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{inline(c, `${k}-${i}-${j}`)}</td>)}</tr>)}</tbody></table></div>);
    tbl = null;
  };
  for (const line of text.split("\n")) {
    if (/^\s*\|.*\|\s*$/.test(line)) { // markdown table row
      flush();
      const cells = line.trim().slice(1, -1).split("|").map((c) => c.trim());
      if (cells.every((c) => /^:?-{2,}:?$/.test(c))) continue; // |---|---| separator
      (tbl ??= []).push(cells);
      continue;
    }
    flushTable();
    const m = line.match(/^\s*(?:([-*])|(\d+)\.)\s+(.*)$/);
    if (m) {
      const ordered = !!m[2];
      if (!list || list.ordered !== ordered) { flush(); list = { ordered, items: [] }; }
      list.items.push(m[3]);
    } else if (line.trim()) { flush(); blocks.push(<p key={`p${blocks.length}`}>{inline(line, `p${blocks.length}`)}</p>); }
    else flush();
  }
  flush();
  flushTable();
  return <>{blocks}</>;
}

function Tools({ tools, final }: { tools: ToolCard[]; final: boolean }) {
  if (!tools.length) return null;
  return (
    <div className="stack" style={{ marginBottom: 8 }}>
      <AnimatePresence initial={false}>
        {tools.map((t) => (
          <motion.details key={t.id} id={`tool-${t.tool}`} className="toolcard" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}>
            <summary>
              {t.ok === undefined ? <LoaderCircle className="spin" aria-hidden="true" /> : t.ok ? <CheckCircle2 aria-hidden="true" /> : <CircleAlert aria-hidden="true" />}
              <Wrench aria-hidden="true" /><strong>{label(t.tool)}</strong>
              <span className="small muted">{Object.entries(t.arguments || {}).map(([k, v]) => `${label(k)} ${v}`).join(", ")}</span>
              <span className="small muted" style={{ marginLeft: "auto" }}>{t.ok === undefined ? "running" : `${t.ms} ms${t.ok ? "" : ", failed"}`}</span>
            </summary>
            {final && t.result !== undefined ? <pre className="json">{JSON.stringify(t.result, null, 2).slice(0, 6000)}</pre>
              : <p className="small muted">The result appears when the answer is complete.</p>}
          </motion.details>
        ))}
      </AnimatePresence>
    </div>
  );
}

export default function Copilot() {
  const { module, presentation, checkpoint, config } = useStore();
  const [params] = useSearchParams();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState(params.get("q") || "");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const ctl = useRef<AbortController | null>(null);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [turns.length]);  // newer browsers return a Promise
  useEffect(() => () => ctl.current?.abort(), []);
  const suggestions = [`Who should I contact first in ${module} ${presentation}?`, `Give me an overview of ${module} ${presentation}`,
    `Which assessments matter most in ${module}?`, "Which factors are most associated with withdrawal?", "How accurate is the withdrawal model?"];

  const update = (fn: (t: Turn) => Turn) => setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? fn(t) : t)));
  const ask = async (text?: string) => {
    const question = (text ?? q).trim();
    if (!question || busy) return;
    setQ(""); setErr(null); setBusy(true);
    const history = turns.filter((t) => t.status !== "error").map((t) => ({ role: t.role, content: t.content })).slice(-10);
    setTurns((ts) => [...ts, { role: "user", content: question }, { role: "assistant", content: "", status: "streaming", tools: [] }]);
    ctl.current = new AbortController();
    try {
      await streamCopilot({ question, history, checkpoint, use_llm: true }, (e) => {
        if (e.kind === "tool_call") update((t) => ({ ...t, tools: [...(t.tools || []), { id: e.data.id, tool: e.data.tool, arguments: e.data.arguments }] }));
        else if (e.kind === "tool_result") update((t) => ({ ...t, tools: (t.tools || []).map((c) => (c.id === e.data.id ? { ...c, ok: e.data.ok, ms: e.data.ms } : c)) }));
        else if (e.kind === "text") update((t) => ({ ...t, content: t.content + e.data.delta }));
        else if (e.kind === "text_reset") update((t) => ({ ...t, content: "" }));
        else if (e.kind === "revision") update((t) => ({ ...t, content: e.data.answer, revised: true }));
        else if (e.kind === "done") update((t) => ({ ...t, content: e.data.answer, status: "done", meta: e.data,
          tools: e.data.tool_calls.map((c: any, i: number) => ({ id: t.tools?.[i]?.id || `t${i}`, tool: c.tool, arguments: c.arguments, ok: c.ok, ms: c.ms, result: c.result })) }));
        else if (e.kind === "error") { update((t) => ({ ...t, status: "error" })); setErr(new ApiError(500, e.data)); }
      }, ctl.current.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") { update((t) => ({ ...t, status: "error" })); setErr(e as ApiError); }
    }
    setBusy(false);
  };

  return (
    <>
      <div className="page-head"><div><h1>Ask the copilot</h1>
        <div className="sub">Questions about students, cohorts, assessments and the models. Answers are built from tool calls you can open, and every number is checked against their results.
          {!config?.llm.configured && " AI is not connected, so a rule-based assistant answers common questions."}</div></div></div>
      <div className="chat" aria-live="polite" aria-busy={busy}>
        {turns.length === 0 && <div className="stack"><p className="muted">Try one of these:</p>
          <div className="row">{suggestions.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div></div>}
        {turns.map((t, i) => t.role === "user" ? <div key={i} className="msg user">{t.content}</div> : (
          <div key={i} className="msg assistant">
            <Tools tools={t.tools || []} final={t.status === "done"} />
            {(t.content || t.status === "done") && (
              <Generated source={t.meta?.source ?? "llm"} title="Copilot" pending={t.status === "streaming"}
                extra={t.status === "streaming" ? <span style={{ fontWeight: 400 }}>writing, checks pending</span>
                  : t.revised ? <span style={{ fontWeight: 400 }}>revised after the grounding check</span> : null}>
                <Answer text={t.content} ids={citable(t.tools || [])} />{t.status === "streaming" && <span className="caret" aria-hidden="true" />}
                {t.status === "done" && <p className="small muted" style={{ marginTop: 8 }}>
                  {t.meta.grounding.passed ? "Every number checked against the tool results." :
                    `Numbers not found in the tool results: ${t.meta.grounding.ungrounded_numbers.join(", ")}. Verify before relying on them.`}
                  {t.meta.flags?.length ? ` Flags: ${t.meta.flags.join("; ")}.` : ""} {t.meta.seconds}s</p>}
              </Generated>
            )}
            {t.status === "streaming" && !t.content && !(t.tools || []).length && <p className="muted" role="status">Thinking about which tools to use…</p>}
          </div>
        ))}
        <div ref={end} />
      </div>
      {err && <ErrorNote error={err} onRetry={() => { const last = [...turns].reverse().find((t) => t.role === "user"); if (last) ask(last.content); }} />}
      <form className="row section" onSubmit={(e) => { e.preventDefault(); ask(); }}>
        <label className="visually-hidden" htmlFor="copilot-q">Your question</label>
        <input id="copilot-q" style={{ flex: 1, minWidth: 240 }} value={q} onChange={(e) => setQ(e.target.value)} maxLength={1500}
          placeholder={`e.g. Tell me about student 59272 in ${module} ${presentation}`} />
        <button className="primary" type="submit" disabled={busy || !q.trim()}><ArrowUp aria-hidden="true" />Ask</button>
        {turns.length > 0 && <button type="button" onClick={() => { ctl.current?.abort(); setTurns([]); setBusy(false); }}><RotateCcw aria-hidden="true" />Start over</button>}
      </form>
    </>
  );
}
