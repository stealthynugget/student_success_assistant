import "../landing/landing.css";
import { useQuery } from "@tanstack/react-query";
import {
  Activity, ArrowRight, ArrowUpRight, ClipboardCheck, Gauge, PenLine, ShieldCheck, Split,
} from "lucide-react";
import { animate, motion, useInView, useMotionValueEvent, useReducedMotion, useScroll, useTransform } from "motion/react";
import { useEffect, useMemo, useRef, useState, type FormEvent, type PointerEvent as RPointerEvent, type ReactNode } from "react";
import { Link, useNavigate } from "react-router";
import { siClaude, siFastapi, siHuggingface, siLangchain, siPython, siReact } from "simple-icons";
import { api } from "../api";
import { DitherField } from "../landing/DitherField";
import { useStore } from "../store";
import { BrandMark } from "../ui";

/* eslint-disable @typescript-eslint/no-explicit-any */
const REPO = "https://github.com/stealthynugget/student_success_assistant";
const EASE = [0.16, 1, 0.3, 1] as const;
const COURSE_WEEKS = 38;

// ---------------------------------------------------------------- motion helpers
function Reveal({ children, delay = 0, className, as = "div" }: { children: ReactNode; delay?: number; className?: string; as?: "div" | "li" }) {
  const reduce = useReducedMotion();
  const M = as === "li" ? motion.li : motion.div;
  return (
    <M className={className} initial={reduce ? false : { opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.25 }} transition={{ duration: 0.7, delay, ease: EASE }}>{children}</M>
  );
}

/** Counts up once when scrolled into view (instant under reduced motion). */
function Count({ value, format }: { value: number | null | undefined; format: (v: number) => string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduce = useReducedMotion();
  useEffect(() => {
    if (value == null || !ref.current) return;
    if (reduce) { ref.current.textContent = format(value); return; }
    if (!inView) return;
    const c = animate(0, value, { duration: 1.4, ease: EASE, onUpdate: (v) => { if (ref.current) ref.current.textContent = format(v); } });
    return () => c.stop();
  }, [value, inView, reduce, format]);
  // always render the element, so a value that arrives after the section is already in view still counts up
  return <span ref={ref}>{value == null ? "-" : format(reduce ? value : 0)}</span>;
}
const int = (v: number) => Math.round(v).toLocaleString("en-GB");
const dec2 = (v: number) => v.toFixed(2);
const pct0 = (v: number) => `${Math.round(v * 100)}%`;

// ---------------------------------------------------------------- sections
function Nav() {
  return (
    <header className="lp-nav">
      <div className="lp-frame">
        <Link to="/" className="lp-brand"><BrandMark />Student support</Link>
        <nav className="lp-links" aria-label="Page">
          <a href="#product">Product</a><a href="#clock">Course clock</a><a href="#evaluation">Evaluation</a>
          <a href="/api/docs" target="_blank" rel="noreferrer">API</a>
        </nav>
        <Link to="/today" className="lp-btn solid lp-cta">Open console<ArrowRight aria-hidden="true" /></Link>
      </div>
    </header>
  );
}

const ASK = [["Who to contact first", "Who should I contact first in BBB 2014J?"],
  ["What drives withdrawal", "Which factors are most associated with withdrawal?"], ["How accurate is it", "How accurate is the withdrawal model?"]];

function Hero({ trends }: { trends: any }) {
  const nav = useNavigate();
  const reduce = useReducedMotion();
  const [q, setQ] = useState("");
  const series = useMemo(() => {
    const by = trends?.weekly_mean_clicks_by_outcome || {};
    return [{ values: (by.Pass || []) as number[], strength: 0.95 }, { values: (by.Withdrawn || []) as number[], strength: 0.5 }];
  }, [trends]);
  const ask = (text: string) => nav(`/copilot?q=${encodeURIComponent(text)}`);
  const submit = (e: FormEvent) => { e.preventDefault(); if (q.trim()) ask(q.trim()); };
  const item = (i: number) => (reduce ? {} : { initial: { opacity: 0, y: 18 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.8, delay: 0.08 * i, ease: EASE } });
  return (
    <section className="lp-sec lp-hero" aria-labelledby="hero-h">
      <div className="lp-hero-copy">
        <div>
          <motion.h1 id="hero-h" {...item(0)}>Reach the right students before they withdraw.</motion.h1>
          <motion.div className="ctas" {...item(2)}>
            <Link to="/today" className="lp-btn solid">Open console<ArrowRight aria-hidden="true" /></Link>
            <a href="#evaluation" className="lp-btn">See the evidence</a>
          </motion.div>
        </div>
        <motion.p className="lead" {...item(1)}>
          Early warning, similar past students and evidence-checked support plans for Open University module teams.
        </motion.p>
      </div>
      <div className="lp-dome">
        <DitherField series={series}
          label="Dot pattern of weekly online activity in module BBB: the bright wave is students who passed, the faint wave students who withdrew." />
        <div className="lp-ask-wrap"><motion.div className="lp-ask" initial={reduce ? false : { opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.35, ease: EASE }}>
          <form onSubmit={submit} role="search" aria-label="Ask the copilot">
            <label className="visually-hidden" htmlFor="lp-q">Ask the copilot</label>
            <input id="lp-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about a module, a week or a student" maxLength={500} />
            <button type="submit" className="primary send" aria-label="Ask in the console"><ArrowRight aria-hidden="true" /></button>
          </form>
          <div className="chips">{ASK.map(([label, question]) => <button key={label} type="button" onClick={() => ask(question)}>{label}</button>)}</div>
          <p className="note-line">Behind this box: real weekly activity of students who passed and who withdrew.</p>
        </motion.div></div>
      </div>
    </section>
  );
}

const LOGOS = [siClaude, siPython, siFastapi, siReact, siLangchain, siHuggingface];
function Logos() {
  return (
    <section className="lp-logos" aria-label="Built with">
      <ul>{LOGOS.map((l, i) => (
        <Reveal as="li" key={l.title} delay={i * 0.05}>
          <svg viewBox="0 0 24 24" role="img" aria-label={l.title}><title>{l.title}</title><path d={l.path} /></svg>
        </Reveal>))}</ul>
    </section>
  );
}

function Numbers({ metrics, config }: { metrics: any; config: any }) {
  const cps = metrics?.checkpoint_summary?.checkpoints || [];
  const primary = config?.checkpoints?.find((c: any) => c.primary);
  const atPrimary = cps.find((c: any) => primary && Math.abs(c.checkpoint - primary.checkpoint) < 1e-6);
  return (
    <section className="lp-sec" aria-labelledby="num-h">
      <Reveal><h2 id="num-h">One honest dataset, read at four points in the course.</h2></Reveal>
      <div className="lp-numbers">
        <Reveal><div className="v"><Count value={atPrimary?.in_scope} format={int} /></div>
          <div className="l">students still enrolled at week {primary ? Math.round(primary.approx_week) : 15}, all modules</div></Reveal>
        <Reveal delay={0.08}><div className="v"><Count value={metrics?.data_build?.records} format={int} /></div>
          <div className="l">enrolments in the dataset, each with its complete record</div></Reveal>
        <Reveal delay={0.16}><div className="v"><Count value={config?.modules?.length} format={int} /></div>
          <div className="l">modules across four presentations</div></Reveal>
        <Reveal delay={0.24}><div className="v"><Count value={cps.length || null} format={int} /></div>
          <div className="l">points in the course, each with its own calibrated model</div></Reveal>
      </div>
    </section>
  );
}

function CourseClock({ metrics }: { metrics: any }) {
  const cps: any[] = metrics?.checkpoint_summary?.checkpoints || [];
  const weeksRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: weeksRef, offset: ["start 0.6", "end 0.6"] });
  const at = (w: number) => (w / COURSE_WEEKS) * 100;
  const first = cps.length ? at(cps[0].approx_week) : 15, last = cps.length ? at(cps[cps.length - 1].approx_week) : 60;
  const left = useTransform(scrollYProgress, [0, 1], [`${first}%`, `${last}%`]);
  useMotionValueEvent(scrollYProgress, "change", (p) => {
    if (!cps.length || !labelRef.current) return;
    const i = Math.min(cps.length - 1, Math.round(p * (cps.length - 1)));
    labelRef.current.textContent = `week ${Math.round(cps[i].approx_week)}`;
  });
  return (
    <section className="lp-sec" id="clock" aria-labelledby="clock-h">
      <div className="lp-clock">
        <div className="pin">
          <Reveal><h2 id="clock-h">Every view is as of one week in the course.</h2>
            <p className="lead">Pick a point on the course clock and every list, score and plan re-scopes to what was known that week.</p></Reveal>
          <div className="lp-ribbon" aria-hidden="true">
            <div className="line" />
            {Array.from({ length: COURSE_WEEKS + 1 }).filter((_, w) => w % 2 === 0).map((_, i) => <div key={i} className="tick" style={{ left: `${at(i * 2)}%` }} />)}
            {cps.map((c) => <div key={c.tag} className="stop" style={{ left: `${at(c.approx_week)}%` }} />)}
            <motion.div className="marker" style={{ left: reduce ? `${first}%` : left }}><span ref={labelRef}>{cps.length ? `week ${Math.round(cps[0].approx_week)}` : ""}</span></motion.div>
          </div>
        </div>
        <div className="lp-weeks" ref={weeksRef}>
          {cps.length === 0 && <div className="lp-week"><h3>Course checkpoints</h3><p className="lead">Connect the console to the service to see each checkpoint.</p></div>}
          {cps.map((c, i) => (
            <Reveal key={c.tag} className="lp-week" delay={i * 0.05}>
              <h3>Week {Math.round(c.approx_week)} of the course</h3>
              <div className="big">{int(c.in_scope)}</div>
              <div className="muted">students still enrolled and scored</div>
              <dl>
                <div><dt>Withdrawal ROC-AUC</dt><dd>{dec2(c.withdrawal.roc_auc)}</dd></div>
                <div><dt>Go on to withdraw</dt><dd>{pct0(c.withdrawal.positive_rate)}</dd></div>
                <div><dt>Precision, top 10%</dt><dd>{pct0(c.prioritization_precision_at_10)}</dd></div>
              </dl>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

const TOUR = [
  { k: "a", to: "/today", img: "/landing/today.jpg", t: "This week", d: "Who to reach first, the tier mix and a briefing whose numbers are checked against the data." },
  { k: "b", to: "/students", img: "/landing/student.jpg", t: "Student case file", d: "Why a student is flagged, what happened to students like them, and a plan to review." },
  { k: "c", to: "/cases", img: "/landing/cases.jpg", t: "Case finder", d: "Search every past enrolment in plain English, with filters." },
  { k: "d", to: "/copilot", img: "/landing/copilot.jpg", t: "Copilot", d: "Answers built from tool calls you can open and check." },
  { k: "e", to: "/quality", img: "/landing/quality.jpg", t: "Model quality", d: "Accuracy, calibration, fairness and AI evaluation, each measured on a presentation the models never saw." },
];
function Tour() {
  const spot = (e: RPointerEvent<HTMLAnchorElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
    e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
  };
  return (
    <section className="lp-sec" id="product" aria-labelledby="tour-h">
      <Reveal><h2 id="tour-h">Everything a module team needs, in one console.</h2></Reveal>
      <div className="lp-bento">
        {TOUR.map((x, i) => (
          <Reveal key={x.k} delay={i * 0.06} className={`lp-tile-wrap ${x.k}`}>
            <Link to={x.to} className="lp-tile" onPointerMove={spot}>
              <div className="shot"><img src={x.img} alt={`${x.t} page of the console`} loading="lazy" decoding="async" /></div>
              <div className="cap"><strong>{x.t}<ArrowUpRight aria-hidden="true" /></strong><span>{x.d}</span></div>
            </Link>
          </Reveal>
        ))}
      </div>
    </section>
  );
}

const FLOW = [
  { icon: Activity, t: "Performance", d: "Reads the student's work and activity as numbered observations.", ai: true },
  { icon: Gauge, t: "Risk", d: "Explains the calibrated risk and its drivers. It cannot change them.", ai: true },
  { icon: Split, t: "Router", d: "Sends low-risk students to monitoring. A rule, not a model.", ai: false },
  { icon: PenLine, t: "Intervention", d: "Picks actions the student is eligible for and drafts a message.", ai: true },
  { icon: ShieldCheck, t: "Critic", d: "Checks every ID, number and word in code. One revision allowed.", ai: false },
  { icon: ClipboardCheck, t: "Plan", d: "A plan an advisor reviews before anything reaches a student.", ai: false },
];
function Flow() {
  const reduce = useReducedMotion();
  return (
    <section className="lp-sec" aria-labelledby="flow-h">
      <Reveal><h2 id="flow-h">How a support plan gets made.</h2>
        <p className="lead">Agents narrate evidence that the code computed. If the critic still finds a problem after one revision, a rule-based plan is used instead.</p></Reveal>
      <div className="lp-flow">
        <svg className="wire" viewBox="0 0 100 2" preserveAspectRatio="none" aria-hidden="true">
          <motion.line x1="2" y1="1" x2="92" y2="1" stroke="var(--field)" strokeWidth="1" vectorEffect="non-scaling-stroke"
            initial={reduce ? false : { pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true, amount: 0.6 }}
            transition={{ duration: 1.6, ease: EASE }} />
        </svg>
        <ol>{FLOW.map((f, i) => (
          <Reveal as="li" key={f.t} delay={0.12 + i * 0.12} className={f.ai ? "ai" : undefined}>
            <div className="node"><f.icon aria-hidden="true" /></div>
            <h3>{f.t}</h3><p>{f.d}</p>
          </Reveal>))}</ol>
        <div className="lp-legend"><span><i className="ai" />Drafted by AI</span><span><i />Computed in code</span></div>
      </div>
    </section>
  );
}

function Evaluation({ metrics }: { metrics: any }) {
  const w = metrics?.withdrawal_model, r = metrics?.retrieval_benchmark?.modes, ag = metrics?.llm_evaluation?.agents;
  const rows = [
    { v: w?.test?.roc_auc, f: dec2, base: w?.baseline_logistic_regression_test?.roc_auc, bl: "logistic baseline",
      t: "Withdrawal ranking, week 15", s: "ROC-AUC for students still enrolled, on a later presentation the model never saw." },
    { v: r?.hybrid?.neighbour_vote_auc_poor_outcome, f: dec2, base: r?.random?.neighbour_vote_auc_poor_outcome, bl: "random cases",
      t: "Past cases predict outcomes", s: "How well the outcomes of the retrieved similar students predict what really happened." },
    { v: ag?.judge?.pass_rate, f: pct0, base: null, bl: "",
      t: "AI plans judged sound", s: "Rated by a separate, stronger model against the same evidence the agents saw." },
    { v: ag?.deterministic?.final_plan_valid_rate, f: pct0, base: null, bl: "",
      t: "Plans that pass every guardrail", s: "Eligible actions, grounded numbers, no protected attributes, no punitive or clinical language." },
  ];
  return (
    <section className="lp-sec" id="evaluation" aria-labelledby="eval-h">
      <Reveal><h2 id="eval-h">Measured, not assumed.</h2></Reveal>
      <div className="lp-eval">
        <div className="lp-metrics">
          {rows.map((x, i) => (
            <Reveal key={x.t} delay={i * 0.06} className="lp-metric">
              <div className="v"><Count value={x.v} format={x.f} />{x.base != null && <small>vs {x.f(x.base)} {x.bl}</small>}</div>
              <div className="l"><strong>{x.t}</strong><span>{x.s}</span></div>
            </Reveal>
          ))}
        </div>
        <Reveal><aside>
          <p>Every figure here is read live from the build artifacts. Students who had already left are never scored, the test presentation
            is used once, and sensitive attributes never enter the models or the prompts.</p>
          <a className="more" href={`${REPO}/blob/master/docs/MODEL_CARD.md`} target="_blank" rel="noreferrer">Read the model card<ArrowUpRight aria-hidden="true" /></a>
        </aside></Reveal>
      </div>
    </section>
  );
}

function Close() {
  return (
    <>
      <section className="lp-sec" aria-labelledby="close-h">
        <div className="lp-close">
          <Reveal><h2 id="close-h">Pick a week. See who needs you.</h2></Reveal>
          <Reveal delay={0.1}><Link to="/today" className="lp-btn solid">Open console<ArrowRight aria-hidden="true" /></Link></Reveal>
        </div>
      </section>
      <footer className="lp-foot">
        <span className="lp-brand"><BrandMark />Student support</span>
        <span>Built on the Open University Learning Analytics Dataset.</span>
        <nav aria-label="Footer">
          <Link to="/today">Console</Link><a href="/api/docs" target="_blank" rel="noreferrer">API docs</a>
          <a href={REPO} target="_blank" rel="noreferrer">Source</a>
        </nav>
      </footer>
    </>
  );
}

export default function Landing() {
  const { config } = useStore();
  const metrics = useQuery({ queryKey: ["metrics", null], queryFn: ({ signal }) => api("/metrics", { signal }), retry: 0 });
  const trends = useQuery({ queryKey: ["trends-course", "BBB"], queryFn: ({ signal }) => api("/trends/course", { query: { code_module: "BBB" }, signal }), retry: 0 });
  useEffect(() => { document.title = "Student support: early warning for module teams"; return () => { document.title = "Student support console"; }; }, []);
  return (
    <div className="lp">
      <a className="skip" href="#lp-main">Skip to content</a>
      <Nav />
      <main id="lp-main" className="lp-frame" tabIndex={-1}>
        <Hero trends={trends.data} />
        <Logos />
        <Numbers metrics={metrics.data} config={config} />
        <CourseClock metrics={metrics.data} />
        <Tour />
        <Flow />
        <Evaluation metrics={metrics.data} />
        <Close />
      </main>
    </div>
  );
}
