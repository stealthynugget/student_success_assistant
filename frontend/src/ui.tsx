// Shared UI primitives. Generated (AI) text is always labelled and visually distinct; computed facts are plain ink.
import { Check, Copy, PenLine, RotateCcw } from "lucide-react";
import { animate, AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Component, useEffect, useRef, useState, type ErrorInfo, type ReactNode } from "react";
import * as RTabs from "@radix-ui/react-tabs";
import type { ApiError } from "./api";

export const Tier = ({ tier }: { tier?: string | null }) =>
  tier ? <span className={`tier ${tier}`}>{tier}</span> : <span className="muted">-</span>;

const SOURCE_LABEL: Record<string, string> = { llm: "Drafted by AI", llm_revised: "Drafted by AI, revised", rules: "Rule-based",
  deterministic: "Computed", computed: "Computed", guardrail: "Guardrail", none: "Not available" };
export const Source = ({ source }: { source?: string }) => {
  const s = source?.startsWith("llm") ? "llm" : source || "computed";
  return <span className={`source ${s}`}>{SOURCE_LABEL[source || ""] || source}</span>;
};

/** Text that may be AI-drafted. With source "llm" it gets the indigo rule and the "Drafted by AI" label; otherwise it is
 *  shown as computed text (the deterministic fallback) with a quiet label. */
export function Generated({ source, children, title, extra, pending }: { source?: string; children: ReactNode; title?: string;
  extra?: ReactNode; pending?: boolean }) {
  if (!source?.startsWith("llm")) {
    return (
      <div className="computed-block">
        <div className="who">{title && <strong style={{ color: "var(--ink)" }}>{title}</strong>} <Source source={source} /> {extra}</div>
        {children}
      </div>
    );
  }
  return (
    <div className={`generated${pending ? " pending" : ""}`}>
      <div className="who"><PenLine aria-hidden="true" />{title ? <span>{title}:</span> : null}<span>Drafted by AI</span>
        <span style={{ fontWeight: 400 }}>review before use</span>{extra}</div>
      {children}
    </div>
  );
}

/** Keeps one broken page from blanking the console: the rail and course clock stay, the page shows what failed. */
export class PageBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error("page crashed", error, info.componentStack); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="note error" role="alert">
        <strong>This page hit an error and could not be shown.</strong> Other pages still work.
        <div className="small">{this.state.error.message}</div>
        <div><button className="ghost" onClick={() => window.location.reload()}><RotateCcw aria-hidden="true" />Reload page</button></div>
      </div>
    );
  }
}

export function ErrorNote({ error, onRetry }: { error?: ApiError | Error | null; onRetry?: () => void }) {
  if (!error) return null;
  const e = error as ApiError;
  const hint = e.status === 401 ? " Add a valid API key in Settings."
    : e.status === 503 && /api key/i.test(e.message || "") ? " Set STUDENT_ASSISTANT_API_KEY on the server (or SSA_DEV_MODE=1 locally), then try again."
    : e.status === 503 ? " Build the artifacts on the server, then try again." : "";
  return (
    <div className="note error" role="alert">
      <strong>{e.message}</strong>{hint}
      {Array.isArray(e.details) && (
        <ul>{(e.details as { loc?: string; msg?: string }[]).slice(0, 5).map((d, i) => <li key={i}>{d.loc ? `${d.loc}: ` : ""}{d.msg || JSON.stringify(d)}</li>)}</ul>
      )}
      {e.requestId && <div className="small">Reference {e.requestId}</div>}
      {onRetry && <div><button className="ghost" onClick={onRetry}><RotateCcw aria-hidden="true" />Try again</button></div>}
    </div>
  );
}

export const Empty = ({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) => (
  <div className="empty"><p className="empty-title">{title}</p><div>{children}</div>{action && <div style={{ marginTop: 12 }}>{action}</div>}</div>
);

export const Loading = ({ lines = 3, label = "Loading" }: { lines?: number; label?: string }) => (
  <div className="stack" aria-busy="true" aria-label={label} role="status">
    {Array.from({ length: lines }).map((_, i) => <div key={i} className="skeleton" style={{ width: `${90 - i * 15}%` }} />)}
  </div>
);

type Opt = string | [string, string];
export function Seg({ value, options, onChange, label }: { value: string; options: Opt[]; onChange: (v: string) => void; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => {
        const [v, l] = Array.isArray(o) ? o : [o, o];
        return <button key={v} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>{l}</button>;
      })}
    </div>
  );
}

export function Tabs({ tabs, value, onChange }: { tabs: [string, string][]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="seg" role="tablist">
      {tabs.map(([v, l]) => <button key={v} role="tab" aria-selected={value === v} onClick={() => onChange(v)}>{l}</button>)}
    </div>
  );
}

export function Section({ id, title, sub, children, actions }: { id?: string; title: string; sub?: string; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="section" id={id} aria-labelledby={id ? `${id}-h` : undefined}>
      <div className="section-head"><h2 id={id ? `${id}-h` : undefined}>{title}</h2>{sub && <p>{sub}</p>}
        {actions && <div style={{ marginLeft: "auto" }} className="row">{actions}</div>}</div>
      {children}
    </section>
  );
}

/** Animated number: counts from the previous value when it changes (e.g. moving the course clock). */
export function CountUp({ value, format }: { value: number | null | undefined; format: (v: number) => string }) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(value ?? 0);
  const prev = useRef(value ?? 0);
  useEffect(() => {
    if (value == null) return;
    if (reduce) { setShown(value); prev.current = value; return; }
    const c = animate(prev.current, value, { duration: 0.6, ease: "easeOut", onUpdate: setShown });
    prev.current = value;
    return () => c.stop();
  }, [value, reduce]);
  return <>{value == null ? "-" : format(shown)}</>;
}

export function Stat({ value, label, tone }: { value: ReactNode; label: ReactNode; tone?: string }) {
  return <div className={`stat ${tone || ""}`}><div className="v">{value}</div><div className="l">{label}</div></div>;
}

export interface StepState { step: string; t: number; source?: string; violations?: string[] }
/** Agent pipeline progress: steps appear as the server reports them; the next expected step pulses. */
export function Steps({ steps, expected, running = true }: { steps: StepState[]; expected: [string, string, string?][]; running?: boolean }) {
  const done = new Map(steps.map((s) => [s.step, s]));
  const firstPending = expected.find((e) => !done.has(e[0]) && !e[2]);
  return (
    <ol className="steps" aria-live="polite">
      <AnimatePresence initial={false}>
        {expected.map(([key, label, optional]) => {
          const s = done.get(key);
          // optional steps (third element) only run for some students: never shown as the current step
          const cls = s ? "done" : running && firstPending?.[0] === key ? "active" : running ? "pending" : optional ? "skipped" : "pending";
          return (
            <motion.li key={key} className={cls} layout initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <span className="dot" aria-hidden="true" />
              <span>{label}{optional && !s ? <span className="small muted"> ({optional})</span> : null}{s?.source && <> <Source source={s.source} /></>}
                <span className="visually-hidden">{s ? " done" : cls === "active" ? " in progress" : ""}</span></span>
              <span className="small muted">{s ? `${s.t}s` : ""}</span>
            </motion.li>
          );
        })}
      </AnimatePresence>
    </ol>
  );
}

export function Json({ data, label = "Show raw data" }: { data: unknown; label?: string }) {
  return <details><summary>{label}</summary><pre className="json">{JSON.stringify(data, null, 2)}</pre></details>;
}

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button type="button" onClick={async () => { await navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>
      {done ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}{done ? "Copied" : label}
    </button>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function downloadCsv(rows: Record<string, any>[], filename: string) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = filename;
  a.click();
}

/** Accessible tab set (Radix): arrow keys move between tabs, Enter/Space selects; each tab controls a real panel. Only the
 *  active panel renders its content (so inactive tabs fetch nothing), but every panel exists for aria-controls. */
export function TabSet({ tabs, value, onChange, label, children }: { tabs: [string, string][]; value: string; onChange: (v: string) => void;
  label: string; children: ReactNode }) {
  return (
    <RTabs.Root value={value} onValueChange={onChange} activationMode="manual">
      <RTabs.List className="seg tabs" aria-label={label}>
        {tabs.map(([v, l]) => <RTabs.Trigger key={v} value={v} asChild><button type="button">{l}</button></RTabs.Trigger>)}
      </RTabs.List>
      {tabs.map(([v]) => (
        <RTabs.Content key={v} value={v} forceMount hidden={v !== value} className="tabpanel">{v === value ? children : null}</RTabs.Content>
      ))}
    </RTabs.Root>
  );
}

/** Uplift's mark: a weekly-activity line dips, then climbs from the yellow diamond, the point in the course where support
 *  happens (the clock's yellow and deadline diamond), over the course clock's tick baseline. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M1.5 21.5H22.5M1.5 19.75V21.5M8.5 19.75V21.5M15.5 19.75V21.5M22.5 19.75V21.5" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
      <path d="M1.5 8.5L4.5 7.5L8.5 15L12 11.5L14.5 12.5L22.5 3" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinejoin="miter" />
      <rect x="6.25" y="12.75" width="4.5" height="4.5" rx="0.3" transform="rotate(45 8.5 15)" fill="var(--marker)" />
    </svg>
  );
}
