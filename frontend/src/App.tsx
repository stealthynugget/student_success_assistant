import { Command } from "cmdk";
import {
  BarChart3, CalendarClock, CheckCircle2, Clock, Contrast, MessageSquareText, MessagesSquare, Search, Settings as SettingsIcon, SlidersHorizontal,
  User, Users, type LucideIcon,
} from "lucide-react";
import { motion } from "motion/react";
import { lazy, Suspense, useEffect, useMemo, useState, type MouseEvent as ReactMouseEvent } from "react";
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate, useParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { studentPath, useStore } from "./store";
import { ErrorNote, Loading, PageBoundary } from "./ui";
// Pages load on demand (route-level code splitting).
const CaseFinder = lazy(() => import("./pages/CaseFinder"));
const Student = lazy(() => import("./pages/Student"));
const Today = lazy(() => import("./pages/Today"));
const CopilotPage = lazy(() => import("./pages/Copilot"));
const Feedback = lazy(() => import("./pages/Feedback"));
const Insights = lazy(() => import("./pages/Insights"));
const Priorities = lazy(() => import("./pages/Priorities"));
const Quality = lazy(() => import("./pages/Quality"));
const Settings = lazy(() => import("./pages/Settings"));
const Students = lazy(() => import("./pages/Students"));

const NAV: [string, string, LucideIcon][] = [
  ["/today", "This week", CalendarClock], ["/students", "Students", Users], ["/priorities", "Who to contact", SlidersHorizontal],
  ["/cases", "Case finder", Search], ["/copilot", "Ask the copilot", MessagesSquare], ["/insights", "Insights", BarChart3],
  ["/quality", "Model quality", CheckCircle2], ["/feedback", "Feedback loop", MessageSquareText], ["/settings", "Settings", SettingsIcon],
];

// ---------------------------------------------------------------- theme
type Theme = "system" | "light" | "dark";
function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => { try { return (localStorage.getItem("ssa.theme") as Theme) || "system"; } catch { return "system"; } });
  useEffect(() => {
    const el = document.documentElement;
    if (theme === "system") delete el.dataset.theme; else el.dataset.theme = theme;
    try { localStorage.setItem("ssa.theme", theme); } catch { /* ignore */ }
  }, [theme]);
  const next = () => setTheme((t) => (t === "system" ? "dark" : t === "dark" ? "light" : "system"));
  return [theme, next];
}

// ---------------------------------------------------------------- course clock
function CourseClock() {
  const { config, module, setModule, presentation, setPresentation, checkpoint, setCheckpoint } = useStore();
  const cohort = useQuery({
    queryKey: ["cohort", module, presentation, checkpoint], enabled: !!(config && presentation && checkpoint),
    queryFn: ({ signal }) => api("/cohort", { query: { code_module: module, code_presentation: presentation, checkpoint }, signal }),
  });
  if (!config) return null;
  const lengthDays: number = cohort.data?.course_length_days || 262;
  const weeks = Math.ceil(lengthDays / 7);
  const W = 1000, l = 16, r = 16;
  const x = (w: number) => l + (w / weeks) * (W - l - r);
  const stops = config.checkpoints.filter((c) => c.available).map((c) => ({ ...c, week: Math.round((c.checkpoint * lengthDays) / 7) }));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const deadlines: any[] = (cohort.data?.assessments || []).filter((a: any) => a.deadline_week !== null);
  const current = stops.find((s) => checkpoint != null && Math.abs(s.checkpoint - checkpoint) < 1e-6);
  const onKey = (e: React.KeyboardEvent, i: number) => {
    if (e.key === "ArrowRight" && stops[i + 1]) { setCheckpoint(stops[i + 1].checkpoint); e.preventDefault(); }
    if (e.key === "ArrowLeft" && stops[i - 1]) { setCheckpoint(stops[i - 1].checkpoint); e.preventDefault(); }
  };
  return (
    <div className="clock">
      <div className="clock-row">
        <div className="clock-title" aria-live="polite">{module} {presentation}{current ? `, as of week ${current.week}` : ""}</div>
        <label>Module <select value={module} onChange={(e) => setModule(e.target.value)}>{config.modules.map((m) => <option key={m}>{m}</option>)}</select></label>
        <label>Presentation <select value={presentation} onChange={(e) => setPresentation(e.target.value)}>
          {(config.presentations_by_module[module] || []).map((p) => <option key={p}>{p}</option>)}</select></label>
      </div>
      <div className="ribbon">
        <svg viewBox={`0 0 ${W} 56`} preserveAspectRatio="none" role="group" aria-label="Course timeline: choose the point in the course to look at">
          <line x1={x(0)} x2={x(weeks)} y1="30" y2="30" stroke="var(--ink)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          {Array.from({ length: weeks + 1 }).map((_, w) => (
            <line key={w} x1={x(w)} x2={x(w)} y1="26" y2="34" stroke="var(--muted)" vectorEffect="non-scaling-stroke" />
          ))}
          {deadlines.map((a) => (
            <rect key={a.id_assessment} x={x(a.deadline_week) - 3} y="20" width="6" height="6" transform={`rotate(45 ${x(a.deadline_week)} 23)`}
              fill={a.assessment_type === "TMA" ? "var(--ink)" : "var(--surface)"} stroke="var(--ink)" vectorEffect="non-scaling-stroke">
              <title>{`${a.assessment_type} due week ${a.deadline_week} (weight ${a.weight})`}</title></rect>
          ))}
          {current && (
            <motion.rect initial={false} animate={{ x: x(current.week) - 24 }} transition={{ type: "spring", stiffness: 260, damping: 28 }}
              y="3" width="48" height="16" rx="2" fill="var(--marker)" />
          )}
          {stops.map((s, i) => {
            const sel = s === current;
            return (
              <g key={s.tag} className="stop" tabIndex={sel || (!current && i === 0) ? 0 : -1} role="radio" aria-checked={sel} aria-label={`Week ${s.week}`}
                onClick={() => setCheckpoint(s.checkpoint)} onKeyDown={(e) => onKey(e, i)}>
                <rect className="hit" x={x(s.week) - 26} y="2" width="52" height="40" fill="transparent" rx="4" />
                <circle cx={x(s.week)} cy="30" r={sel ? 7 : 5} fill={sel ? "var(--marker)" : "var(--surface)"} stroke="var(--ink)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
              </g>
            );
          })}
        </svg>
        {/* Labels as HTML so text stays crisp and readable at any width */}
        <div aria-hidden="true" style={{ position: "absolute", inset: 0, pointerEvents: "none", fontSize: 12 }}>
          {stops.map((s) => (
            <span key={s.tag} style={{ position: "absolute", top: 3, left: `${(x(s.week) / W) * 100}%`, transform: "translateX(-50%)",
              fontWeight: s === current ? 700 : 400, color: s === current ? "var(--on-marker)" : "var(--ink)" }}>wk {s.week}</span>
          ))}
          {Array.from({ length: Math.floor(weeks / 5) + 1 }).map((_, i) => (
            <span key={i} style={{ position: "absolute", top: 38, left: `${(x(i * 5) / W) * 100}%`, transform: "translateX(-50%)", color: "var(--muted)" }}>wk {i * 5}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- command palette
function Palette({ open, setOpen, toggleTheme }: { open: boolean; setOpen: (v: boolean) => void; toggleTheme: () => void }) {
  const { config, module, presentation, setModule, setPresentation, setCheckpoint } = useStore();
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const run = (fn: () => void) => { fn(); setOpen(false); setQ(""); };
  const id = q.trim().match(/^\d{3,8}$/)?.[0];
  return (
    <Command.Dialog open={open} onOpenChange={setOpen} label="Command palette" overlayClassName="cmdk-overlay" contentClassName="cmdk-content">
      <Command.Input value={q} onValueChange={setQ} placeholder="Type a student ID, a page, a module or a week…" />
      <Command.List>
        <Command.Empty>No matches. Type a student ID to open their case file.</Command.Empty>
        {id && <Command.Group heading="Student">
          <Command.Item value={`student ${id}`} onSelect={() => run(() => nav(studentPath(module, presentation, id)))}>
            <User aria-hidden="true" />Open student {id} in {module} {presentation}</Command.Item>
        </Command.Group>}
        <Command.Group heading="Go to">
          {NAV.map(([to, label, Icon]) => <Command.Item key={to} value={`go ${label}`} onSelect={() => run(() => nav(to))}><Icon aria-hidden="true" />{label}</Command.Item>)}
        </Command.Group>
        {config && <Command.Group heading="Course clock">
          {config.modules.map((m) => <Command.Item key={m} value={`module ${m}`} onSelect={() => run(() => setModule(m))}><Clock aria-hidden="true" />Switch to module {m}</Command.Item>)}
          {(config.presentations_by_module[module] || []).map((p) => (
            <Command.Item key={p} value={`presentation ${p}`} onSelect={() => run(() => setPresentation(p))}><Clock aria-hidden="true" />Switch to presentation {p}</Command.Item>))}
          {config.checkpoints.filter((c) => c.available).map((c) => (
            <Command.Item key={c.tag} value={`week ${c.approx_week} checkpoint`} onSelect={() => run(() => setCheckpoint(c.checkpoint))}>
              <CalendarClock aria-hidden="true" />Look at week {Math.round(c.approx_week)}</Command.Item>))}
        </Command.Group>}
        <Command.Group heading="Actions">
          <Command.Item value="toggle theme dark light" onSelect={() => run(toggleTheme)}><Contrast aria-hidden="true" />Switch theme</Command.Item>
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}

// ---------------------------------------------------------------- old #/ links
const legacyPath = (p: string) => p.replace(/^\/student\//, "/students/");
function OldStudentRedirect() {
  const { m, p, id } = useParams();
  return <Navigate replace to={studentPath(m!, p!, id!)} />;
}

export default function App() {
  const { error, config, reload } = useStore();
  const [theme, toggleTheme] = useTheme();
  const [palette, setPalette] = useState(false);
  const nav = useNavigate();
  const loc = useLocation();

  useEffect(() => { // old #/path links and bookmarks
    const fix = () => { if (window.location.hash.startsWith("#/")) nav(legacyPath(window.location.hash.slice(1)), { replace: true }); };
    fix();
    window.addEventListener("hashchange", fix);
    return () => window.removeEventListener("hashchange", fix);
  }, [nav]);
  useEffect(() => {
    const on = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPalette((o) => !o); } };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);
  useEffect(() => { window.scrollTo(0, 0); }, [loc.pathname]);

  // In-app links rendered as plain <a> (e.g. inside generated text) navigate without a full reload.
  const onClick = (e: ReactMouseEvent) => {
    const a = (e.target as HTMLElement).closest("a");
    const href = a?.getAttribute("href");
    if (!a || !href || a.target || e.ctrlKey || e.metaKey || e.shiftKey || e.button !== 0) return;
    if (href.startsWith("#/")) { e.preventDefault(); nav(legacyPath(href.slice(1))); }
    else if (href.startsWith("/") && !href.startsWith("/api")) { e.preventDefault(); nav(href); }
  };

  const page = loc.pathname.split("/")[1] || "today";
  const llm = config?.llm;
  const routes = useMemo(() => (
    <Routes>
      <Route path="/" element={<Navigate replace to="/today" />} />
      <Route path="/today" element={<Today />} />
      <Route path="/students" element={<Students />} />
      <Route path="/students/:module/:presentation/:id" element={<Student />} />
      <Route path="/student/:m/:p/:id" element={<OldStudentRedirect />} />
      <Route path="/priorities" element={<Priorities />} />
      <Route path="/cases" element={<CaseFinder />} />
      <Route path="/copilot" element={<CopilotPage />} />
      <Route path="/insights" element={<Insights />} />
      <Route path="/quality" element={<Quality />} />
      <Route path="/feedback" element={<Feedback />} />
      <Route path="*" element={<Navigate replace to="/today" />} />
    </Routes>
  ), []);

  let body;
  if (page === "settings") body = <div className="content"><PageBoundary key={loc.pathname}><Suspense fallback={<Loading />}><Settings /></Suspense></PageBoundary></div>;
  else if (error) body = (
    <div className="content"><h1>Can't load the console</h1><p className="muted">The service didn't respond as expected.</p>
      <ErrorNote error={error} onRetry={reload} /><p style={{ marginTop: 16 }}><NavLink to="/settings">Open Settings</NavLink> to check the service address and API key.</p></div>);
  else if (!config) body = <div className="content"><p className="muted" role="status">Loading…</p></div>;
  else body = <><CourseClock /><div className="content"><PageBoundary key={loc.pathname}><Suspense fallback={<Loading lines={4} />}>{routes}</Suspense></PageBoundary></div></>;

  return (
    <div className="shell" onClick={onClick}>
      <a className="skip" href="#main">Skip to content</a>
      <nav className="rail" aria-label="Main">
        <div className="brand"><strong>Student support</strong><span>OULAD early warning</span></div>
        {NAV.map(([to, label, Icon]) => (
          <NavLink key={to} to={to} aria-current={(page === to.slice(1) || (to === "/students" && page === "students")) ? "page" : undefined}>
            <Icon aria-hidden="true" />{label}</NavLink>
        ))}
        <div className="foot">
          <span className="llm">{llm?.configured ? `AI drafting on (${llm.model})` : "AI not connected: computed text only"}</span>
          <button type="button" onClick={() => setPalette(true)} title="Open the command palette">Search <kbd>Ctrl K</kbd></button>
          <button type="button" onClick={toggleTheme}><Contrast aria-hidden="true" />Theme: {theme}<span className="visually-hidden"> (switch theme)</span></button>
        </div>
      </nav>
      <main className="main" id="main" tabIndex={-1}>{body}</main>
      <Palette open={palette} setOpen={setPalette} toggleTheme={toggleTheme} />
    </div>
  );
}
