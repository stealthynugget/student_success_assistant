import React from "react";
import { api } from "./api.js";
import { useApi, useRoute, useStore } from "./store.jsx";
import { ErrorNote, Icon } from "./ui.jsx";
import Today from "./pages/Today.jsx";
import Students from "./pages/Students.jsx";
import Student from "./pages/Student.jsx";
import Priorities from "./pages/Priorities.jsx";
import CaseFinder from "./pages/CaseFinder.jsx";
import Copilot from "./pages/Copilot.jsx";
import Insights from "./pages/Insights.jsx";
import Quality from "./pages/Quality.jsx";
import Feedback from "./pages/Feedback.jsx";
import Settings from "./pages/Settings.jsx";

const NAV = [
  ["today", "This week"], ["students", "Students"], ["priorities", "Who to contact"], ["cases", "Case finder"],
  ["copilot", "Ask the copilot"], ["insights", "Insights"], ["quality", "Model quality"], ["feedback", "Feedback loop"], ["settings", "Settings"],
];

function CourseClock() {
  const { config, module, setModule, presentation, setPresentation, checkpoint, setCheckpoint } = useStore();
  const cohort = useApi((signal) => api("/cohort", { query: { code_module: module, code_presentation: presentation, checkpoint }, signal }),
    [module, presentation, checkpoint], { enabled: !!(config && presentation && checkpoint) });
  if (!config) return null;
  const lengthDays = cohort.data?.course_length_days || 262;
  const weeks = Math.ceil(lengthDays / 7);
  const W = 1000, H = 58, l = 16, r = 16;
  const x = (w) => l + (w / weeks) * (W - l - r);
  const stops = config.checkpoints.filter((c) => c.available).map((c) => ({ ...c, week: Math.round((c.checkpoint * lengthDays) / 7) }));
  const deadlines = (cohort.data?.assessments || []).filter((a) => a.deadline_week !== null);
  const current = stops.find((s) => Math.abs(s.checkpoint - checkpoint) < 1e-6);
  const onKey = (e, i) => {
    if (e.key === "ArrowRight" && stops[i + 1]) setCheckpoint(stops[i + 1].checkpoint);
    if (e.key === "ArrowLeft" && stops[i - 1]) setCheckpoint(stops[i - 1].checkpoint);
  };
  return (
    <div className="clock">
      <div className="clock-row">
        <div className="clock-title">{module} {presentation}{current ? `, as of week ${current.week}` : ""}</div>
        <label>Module <select value={module} onChange={(e) => setModule(e.target.value)}>{config.modules.map((m) => <option key={m}>{m}</option>)}</select></label>
        <label>Presentation <select value={presentation} onChange={(e) => setPresentation(e.target.value)}>
          {(config.presentations_by_module[module] || []).map((p) => <option key={p}>{p}</option>)}</select></label>
      </div>
      <div className="ribbon">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="group" aria-label="Course timeline: choose the point in the course to look at">
          <line x1={x(0)} x2={x(weeks)} y1="30" y2="30" stroke="#172036" strokeWidth="2" />
          {Array.from({ length: weeks + 1 }).map((_, w) => (
            <g key={w}><line x1={x(w)} x2={x(w)} y1="26" y2="34" stroke="#9aa2b1" />
              {w % 5 === 0 && <text x={x(w)} y="52" fontSize="11" textAnchor="middle" fill="#5c6577">wk {w}</text>}</g>
          ))}
          {deadlines.map((a) => (
            <g key={a.id_assessment}><rect x={x(a.deadline_week) - 3} y="20" width="6" height="6" transform={`rotate(45 ${x(a.deadline_week)} 23)`}
              fill={a.assessment_type === "TMA" ? "#172036" : "#ffffff"} stroke="#172036" />
              <title>{`${a.assessment_type} due week ${a.deadline_week} (weight ${a.weight})`}</title></g>
          ))}
          {stops.map((s, i) => {
            const sel = Math.abs(s.checkpoint - checkpoint) < 1e-6;
            return (
              <g key={s.tag} className="stop" tabIndex={0} role="button" aria-pressed={sel} aria-label={`Week ${s.week}`}
                 onClick={() => setCheckpoint(s.checkpoint)} onKeyDown={(e) => onKey(e, i)}>
                <rect className="hit" x={x(s.week) - 26} y="2" width="52" height="40" fill="transparent" rx="4" />
                {sel && <rect x={x(s.week) - 24} y="3" width="48" height="16" fill="#f4c53d" rx="2" />}
                <text x={x(s.week)} y="15" fontSize="12" textAnchor="middle" fill="#172036" fontWeight={sel ? 700 : 400}>wk {s.week}</text>
                <circle cx={x(s.week)} cy="30" r={sel ? 7 : 5} fill={sel ? "#f4c53d" : "#ffffff"} stroke="#172036" strokeWidth="2" />
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export default function App() {
  const { error, config, reload, toast } = useStore();
  const { parts, query } = useRoute();
  const page = parts[0] || "today";
  let body;
  if (page === "settings") body = <Settings />;
  else if (error) body = <div className="content"><h1>Can't load the console</h1><p className="muted">The service didn't respond as expected.</p>
    <ErrorNote error={error} onRetry={reload} /><p style={{ marginTop: 16 }}><a href="#/settings">Open Settings</a> to check the service address and API key.</p></div>;
  else if (!config) body = <div className="content"><p className="muted">Loading…</p></div>;
  else {
    const P = { today: Today, students: Students, student: Student, priorities: Priorities, cases: CaseFinder, copilot: Copilot,
      insights: Insights, quality: Quality, feedback: Feedback }[page] || Today;
    body = <><CourseClock /><div className="content"><P parts={parts} query={query} /></div></>;
  }
  return (
    <div className="shell">
      <nav className="rail" aria-label="Main">
        <div className="brand"><strong>Student support</strong><span>OULAD early warning</span></div>
        {NAV.map(([k, l]) => (
          <a key={k} href={`#/${k}`} aria-current={page === k || (k === "students" && page === "student") ? "page" : undefined}>
            <Icon name={k} />{l}</a>
        ))}
        <div className="foot">{config?.llm?.configured ? `LLM: ${config.llm.model}` : "LLM not connected: computed mode"}</div>
      </nav>
      <main className="main" id="main">{body}</main>
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}
