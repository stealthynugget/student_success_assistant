import React, { useState } from "react";

export const Tier = ({ tier }) => (tier ? <span className={`tier ${tier}`}>{tier}</span> : <span className="muted">–</span>);

export const Source = ({ source }) => {
  const label = { llm: "Drafted by AI", rules: "Rule-based", deterministic: "Computed", computed: "Computed", guardrail: "Guardrail" }[source] || source;
  return <span className={`source ${source}`}>{label}</span>;
};

export function Generated({ source, children, title, extra }) {
  if (source !== "llm") {
    return (
      <div className="stack">
        {title && <h4>{title} <Source source={source} /></h4>}
        {children}
      </div>
    );
  }
  return (
    <div className="generated">
      <div className="who">{title ? `${title} ` : ""}<Source source="llm" /> {extra}</div>
      {children}
    </div>
  );
}

export function ErrorNote({ error, onRetry }) {
  if (!error) return null;
  const hint = error.status === 401 ? " Add a valid API key in Settings." : error.status === 503 ? " Build the artifacts on the server, then retry." :
    error.status === 409 ? "" : error.status === 0 ? "" : "";
  return (
    <div className="note error" role="alert">
      <strong>{error.message}</strong>{hint}
      {error.details && Array.isArray(error.details) && (
        <ul>{error.details.slice(0, 5).map((d, i) => <li key={i}>{d.loc ? `${d.loc}: ` : ""}{d.msg || JSON.stringify(d)}</li>)}</ul>
      )}
      {error.requestId && <div className="small">Reference {error.requestId}</div>}
      {onRetry && <div><button className="ghost" onClick={onRetry}>Try again</button></div>}
    </div>
  );
}

export const Empty = ({ title, children, action }) => (
  <div className="empty"><h3>{title}</h3><div>{children}</div>{action && <div style={{ marginTop: 12 }}>{action}</div>}</div>
);

export const Loading = ({ lines = 3 }) => (
  <div className="stack" aria-busy="true" aria-label="Loading">{Array.from({ length: lines }).map((_, i) => <div key={i} className="skeleton" style={{ width: `${90 - i * 15}%` }} />)}</div>
);

export function Seg({ value, options, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => {
        const [v, l] = Array.isArray(o) ? o : [o, o];
        return <button key={String(v)} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>{l}</button>;
      })}
    </div>
  );
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="seg" role="tablist">
      {tabs.map(([v, l]) => <button key={v} role="tab" aria-selected={value === v} aria-pressed={value === v} onClick={() => onChange(v)}>{l}</button>)}
    </div>
  );
}

export function Section({ title, sub, children, actions }) {
  return (
    <section className="section">
      <div className="section-head"><h2>{title}</h2>{sub && <p>{sub}</p>}{actions && <div style={{ marginLeft: "auto" }} className="row">{actions}</div>}</div>
      {children}
    </section>
  );
}

export function Stat({ value, label, tone }) {
  return <div className={`stat ${tone || ""}`}><div className="v">{value}</div><div className="l">{label}</div></div>;
}

export function Steps({ steps, expected }) {
  const done = new Set(steps.map((s) => s.step));
  const firstPending = expected.find((e) => !done.has(e[0]));
  return (
    <ol className="steps">
      {expected.map(([key, label]) => {
        const s = steps.find((x) => x.step === key);
        const cls = s ? "done" : firstPending && firstPending[0] === key ? "active" : "";
        return (
          <li key={key} className={cls}>
            <span className="dot" aria-hidden="true" />
            <span>{label}{s?.source && <> <Source source={s.source} /></>}</span>
            <span className="small muted">{s ? `${s.t}s` : ""}</span>
          </li>
        );
      })}
    </ol>
  );
}

export function Json({ data, label = "Show raw data" }) {
  return <details><summary>{label}</summary><pre className="json">{JSON.stringify(data, null, 2)}</pre></details>;
}

export function CopyButton({ text, label = "Copy" }) {
  const [done, setDone] = useState(false);
  return (
    <button type="button" onClick={async () => { await navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>
      {done ? "Copied" : label}
    </button>
  );
}

export const Icon = ({ name }) => {
  const p = {
    today: "M3 12h4l3-8 4 16 3-8h4", students: "M8 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm-6 10a6 6 0 0 1 12 0M17 11a3 3 0 1 0 0-6m2 16a5 5 0 0 0-3-4.6",
    priorities: "M4 6h16M4 12h11M4 18h6", cases: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm10 2-4.3-4.3", copilot: "M4 5h16v11H8l-4 4z",
    insights: "M4 20V10m6 10V4m6 16v-7m4 7H2", quality: "M5 13l4 4L19 7", feedback: "M4 4h16v12H5l-1 4zM8 9h8M8 12h5",
    settings: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm8-3 2-1-2-4-2 1a7 7 0 0 0-2-1l-.5-2h-4L11 6a7 7 0 0 0-2 1L7 6 5 10l2 1v2l-2 1 2 4 2-1a7 7 0 0 0 2 1l.5 2h4l.5-2a7 7 0 0 0 2-1l2 1 2-4-2-1z",
  }[name];
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={p} /></svg>;
};

export function downloadCsv(rows, filename) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = filename;
  a.click();
}
