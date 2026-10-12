// Charts. Rendered at real pixel size (labels stay 12px at any width) and themed with CSS variables, so the same colour
// roles hold in light and dark: tier colours = tier, outcome colours = final result, yellow = the point in the course.
import { useReducedMotion } from "motion/react";
import { useMemo, useState } from "react";
import {
  Bar, BarChart, Brush, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { fmt } from "./api";

export const OUTCOMES = ["Distinction", "Pass", "Fail", "Withdrawn"] as const;
export const COL: Record<string, string> = {
  ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", marker: "var(--marker)", median: "var(--median)", ai: "var(--ai)",
  High: "var(--high)", Medium: "var(--medium)", Low: "var(--low)",
  Distinction: "var(--o-distinction)", Pass: "var(--o-pass)", Fail: "var(--o-fail)", Withdrawn: "var(--o-withdrawn)",
};
const AXIS = { stroke: "var(--rule)", tick: { fill: "var(--muted)", fontSize: 12 }, tickLine: false } as const;

function Tip({ active, payload, label, unit = "week", fmtV = (v: number) => fmt.int(v) }: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  active?: boolean; payload?: any[]; label?: number | string; unit?: string; fmtV?: (v: number) => string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="t">{unit} {label}</div>
      {payload.filter((p) => p.value != null).slice(0, 8).map((p) => (
        <div key={p.dataKey}><span style={{ color: p.color }}>■</span> {p.name}: {fmtV(p.value)}</div>
      ))}
    </div>
  );
}

export interface Series { key: string; name: string; values: (number | null)[]; color: string; width?: number; dash?: string }

/** Weekly activity with a hover crosshair, assessment deadlines and the checkpoint marker. */
export function ActivityChart({ series, snapshotWeek, deadlines = [], height = 220, yLabel = "clicks per week", label }: {
  series: Series[]; snapshotWeek?: number; deadlines?: { week: number; type: string }[]; height?: number; yLabel?: string; label: string }) {
  const reduce = useReducedMotion();
  const n = Math.max(...series.map((s) => s.values.length));
  const data = useMemo(() => Array.from({ length: n }, (_, w) => Object.fromEntries([["week", w], ...series.map((s) => [s.key, s.values[w] ?? null])])),
    [series, n]);
  return (
    <figure className="chart" style={{ margin: 0 }} aria-label={label}>
      <div className="small muted" style={{ marginBottom: 2 }}>{yLabel} by course week</div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 20, right: 40, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="week" {...AXIS} interval="preserveStartEnd" minTickGap={24} />
          <YAxis {...AXIS} width={44} />
          <Tooltip content={<Tip />} cursor={{ stroke: "var(--ink)", strokeDasharray: "3 3" }} />
          {deadlines.map((d, i) => <ReferenceLine key={i} x={d.week} stroke="var(--muted)" strokeDasharray="2 4" ifOverflow="extendDomain" />)}
          {snapshotWeek != null && <ReferenceLine x={snapshotWeek} stroke={COL.marker} strokeWidth={4}
            label={{ value: "this point", position: "top", fill: "var(--ink)", fontSize: 12 }} />}
          {series.map((s) => <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash}
            dot={false} activeDot={{ r: 4 }} connectNulls={false} isAnimationActive={!reduce} />)}
        </LineChart>
      </ResponsiveContainer>
    </figure>
  );
}

export interface Trajectory { id: number; result: string; values: number[] }
/** Full-course activity of similar students. Hovering a line (or a table row via `highlight`) brings it forward; the legend
 *  toggles outcomes; the brush zooms into weeks. */
export function Trajectories({ items, snapshotWeek, highlight, onHighlight, height = 240, label }: {
  items: Trajectory[]; snapshotWeek?: number; highlight?: number | null; onHighlight?: (id: number | null) => void; height?: number; label: string }) {
  const reduce = useReducedMotion();
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const n = Math.max(0, ...items.map((t) => t.values.length));
  const data = useMemo(() => Array.from({ length: n }, (_, w) => Object.fromEntries([["week", w], ...items.map((t) => [`c${t.id}`, t.values[w] ?? null])])),
    [items, n]);
  const present = OUTCOMES.filter((o) => items.some((t) => t.result === o));
  return (
    <figure className="chart" style={{ margin: 0 }} aria-label={label}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 20, right: 40, bottom: 4, left: 0 }} onMouseLeave={() => onHighlight?.(null)}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="week" {...AXIS} interval="preserveStartEnd" minTickGap={24} />
          <YAxis {...AXIS} width={44} />
          <Tooltip content={<Tip />} cursor={{ stroke: "var(--ink)", strokeDasharray: "3 3" }} />
          {snapshotWeek != null && <ReferenceLine x={snapshotWeek} stroke={COL.marker} strokeWidth={4}
            label={{ value: "this point", position: "top", fill: "var(--ink)", fontSize: 12 }} />}
          {items.filter((t) => !hidden[t.result]).map((t) => {
            const on = highlight == null || highlight === t.id;
            return <Line key={t.id} dataKey={`c${t.id}`} name={`Case ${t.id} (${t.result})`} stroke={COL[t.result]} dot={false}
              strokeWidth={highlight === t.id ? 3.5 : 1.6} strokeOpacity={on ? 0.95 : 0.15} isAnimationActive={!reduce}
              activeDot={{ r: 3, onMouseEnter: () => onHighlight?.(t.id) }} onMouseEnter={() => onHighlight?.(t.id)} />;
          })}
          {n > 12 && <Brush dataKey="week" height={20} stroke="var(--muted)" fill="var(--surface)" travellerWidth={8} />}
        </LineChart>
      </ResponsiveContainer>
      <figcaption className="legend">
        {present.map((o) => (
          <button key={o} type="button" aria-pressed={!hidden[o]} onClick={() => setHidden({ ...hidden, [o]: !hidden[o] })}>
            <i style={{ background: COL[o] }} />{o}
          </button>
        ))}
        <span>drag the lower bar to zoom into weeks</span>
      </figcaption>
    </figure>
  );
}

/** Calibrated withdrawal risk distribution, bars coloured by the tier they fall into. */
export function RiskHistogram({ counts, edges, thresholds, height = 180 }: {
  counts: number[]; edges: number[]; thresholds: { medium: number; high: number }; height?: number }) {
  const reduce = useReducedMotion();
  const last = counts.reduce((a, c, i) => (c > 0 ? i : a), 0);
  const xmax = Math.min(1, Math.max(0.3, (edges[last + 1] ?? 1) * 1.1, thresholds.high * 1.4));
  const data = counts.map((c, i) => ({ lo: edges[i], hi: edges[i + 1], n: c })).filter((d) => d.lo < xmax);
  const tier = (lo: number) => (lo >= thresholds.high - 1e-9 ? "High" : lo >= thresholds.medium - 1e-9 ? "Medium" : "Low");
  return (
    <figure className="chart" style={{ margin: 0 }} aria-label="Distribution of calibrated withdrawal risk, coloured by tier">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }} barCategoryGap={1}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="lo" {...AXIS} tickFormatter={(v) => fmt.pct(v)} minTickGap={28} />
          <YAxis {...AXIS} width={44} allowDecimals={false} />
          <Tooltip cursor={{ fill: "var(--rule-2)" }} content={({ active, payload }) => active && payload?.length ? (
            <div className="chart-tip"><div className="t">{fmt.pct(payload[0].payload.lo)} to {fmt.pct(payload[0].payload.hi)}</div>
              {fmt.int(payload[0].payload.n)} students, {tier(payload[0].payload.lo)} tier</div>) : null} />
          <Bar dataKey="n" isAnimationActive={!reduce} animationDuration={500}>
            {data.map((d, i) => <Cell key={i} fill={COL[tier(d.lo)]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </figure>
  );
}

export interface TrailPoint { checkpoint: number; snapshot_week: number; withdrawal_risk?: number | null; risk_tier?: string; in_scope?: boolean }
/** Withdrawal risk recomputed at each checkpoint (only what was known then); dots coloured by tier. */
export function RiskTrail({ points, current, height = 150 }: { points: TrailPoint[]; current: number; height?: number }) {
  const reduce = useReducedMotion();
  const data = points.filter((p) => p.in_scope && p.withdrawal_risk != null).map((p) => ({ week: p.snapshot_week, risk: p.withdrawal_risk, tier: p.risk_tier,
    now: Math.abs(p.checkpoint - current) < 1e-6 }));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const dot = (props: any) => <circle key={props.index} cx={props.cx} cy={props.cy} r={props.payload.now ? 7 : 5} fill={COL[props.payload.tier] || COL.ink}
    stroke={props.payload.now ? "var(--marker)" : "var(--surface)"} strokeWidth={props.payload.now ? 3 : 1.5} />;
  return (
    <figure className="chart" style={{ margin: 0 }} aria-label="Withdrawal risk at each course checkpoint">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="week" type="number" domain={["dataMin - 1", "dataMax + 1"]} {...AXIS} tickFormatter={(w) => `wk ${w}`} allowDecimals={false} />
          <YAxis {...AXIS} width={44} domain={[0, (max: number) => Math.min(1, Math.max(0.3, max * 1.3))]} tickFormatter={(v) => fmt.pct(v)} />
          <Tooltip content={<Tip fmtV={(v) => fmt.pct(v)} />} />
          <Line dataKey="risk" name="Withdrawal risk" stroke="var(--ink)" strokeWidth={2} dot={dot} activeDot={false} isAnimationActive={!reduce} />
        </LineChart>
      </ResponsiveContainer>
    </figure>
  );
}

/** Small semicircular dial for the case-file header: calibrated withdrawal risk against the tier cut-offs. */
export function RiskDial({ value, thresholds, tier }: { value: number; thresholds: { medium: number; high: number }; tier: string }) {
  const max = Math.max(0.5, thresholds.high * 2, value * 1.1);
  const pt = (v: number, r = 26) => { const a = Math.PI * (1 - Math.min(v, max) / max); return [32 + r * Math.cos(a), 34 - r * Math.sin(a)]; };
  const arc = (a: number, b: number) => { const [x1, y1] = pt(a); const [x2, y2] = pt(b); return `M${x1},${y1} A26,26 0 0 1 ${x2},${y2}`; };
  const [nx, ny] = pt(value, 20);
  return (
    <div className="dial" role="img" aria-label={`Withdrawal risk ${fmt.pct(value)}, ${tier} tier`}>
      <svg viewBox="0 0 64 40" aria-hidden="true">
        <path d={arc(0, thresholds.medium)} stroke={COL.Low} strokeWidth="7" fill="none" />
        <path d={arc(thresholds.medium, thresholds.high)} stroke={COL.Medium} strokeWidth="7" fill="none" />
        <path d={arc(thresholds.high, max)} stroke={COL.High} strokeWidth="7" fill="none" />
        <line x1="32" y1="34" x2={nx} y2={ny} stroke="var(--ink)" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="32" cy="34" r="3" fill="var(--ink)" />
      </svg>
      <div><div style={{ fontSize: 24, fontWeight: 700, lineHeight: 1 }}>{fmt.pct(value)}</div>
        <div className="small muted">withdrawal risk</div></div>
    </div>
  );
}

/** One horizontal bar split into parts (e.g. outcome probabilities). */
export function Stacked({ parts, label }: { parts: { label: string; value: number; color: string; display?: string }[]; label: string }) {
  const total = parts.reduce((a, p) => a + (p.value || 0), 0) || 1;
  return (
    <div>
      <div role="img" aria-label={`${label}: ${parts.map((p) => `${p.label} ${p.display ?? p.value}`).join(", ")}`}
        style={{ display: "flex", height: 16, borderRadius: 3, overflow: "hidden", background: "var(--rule-2)" }}>
        {parts.map((p) => <span key={p.label} title={`${p.label}: ${p.display ?? p.value}`} style={{ width: `${(100 * p.value) / total}%`, background: p.color }} />)}
      </div>
      <div className="legend" style={{ marginTop: 6 }}>
        {parts.map((p) => <span key={p.label}><i style={{ background: p.color, height: 10 }} />{p.label} {p.display ?? p.value}</span>)}
      </div>
    </div>
  );
}

/** Signed contributions (e.g. SHAP drivers): bars left lower risk, right raise it. */
export function Diverging({ items }: { items: { label: string; value: number; title?: string }[] }) {
  const max = Math.max(1e-9, ...items.map((d) => Math.abs(d.value)));
  return (
    <div className="bars-h" role="list" aria-label="What moves the estimate">
      {items.map((d) => (
        <div key={d.label} role="listitem" title={d.title} style={{ display: "contents" }}>
          <span>{d.label}</span>
          <span style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
            <span style={{ display: "flex", justifyContent: "flex-end" }}>{d.value < 0 && <span className="bar" style={{ width: `${(100 * -d.value) / max}%`, background: COL.Low }} />}</span>
            <span style={{ borderLeft: "1px solid var(--muted)" }}>{d.value > 0 && <span className="bar" style={{ display: "block", width: `${(100 * d.value) / max}%`, background: COL.High }} />}</span>
          </span>
          <span className="small muted">{d.value > 0 ? "raises" : "lowers"} risk</span>
        </div>
      ))}
    </div>
  );
}

/** Labelled horizontal bars in HTML (crisp text at any width). */
export function HBars({ items, max, format = (v) => String(v), label }: {
  items: { label: string; value: number; color?: string; strong?: boolean }[]; max?: number; format?: (v: number) => string; label: string }) {
  const m = max ?? Math.max(1e-9, ...items.map((d) => d.value || 0));
  return (
    <div className="bars-h" role="list" aria-label={label}>
      {items.map((d) => (
        <div key={d.label} role="listitem" style={{ display: "contents" }}>
          <span style={{ fontWeight: d.strong ? 700 : 400 }}>{d.label}</span>
          <span><span className="bar" style={{ display: "block", width: `${Math.max(0, (100 * (d.value || 0)) / m)}%`, background: d.color || COL.ink }} /></span>
          <span className="small">{format(d.value)}</span>
        </div>
      ))}
    </div>
  );
}

/** Grouped bars (categories x series) with a tooltip. */
export function GroupedBars({ categories, series, max = 1, format = (v) => String(v), height = 230, label }: {
  categories: string[]; series: { name: string; color: string; values: (number | null)[] }[]; max?: number; format?: (v: number) => string;
  height?: number; label: string }) {
  const reduce = useReducedMotion();
  const data = categories.map((c, i) => Object.fromEntries([["cat", c], ...series.map((s, j) => [`s${j}`, s.values[i] ?? null])]));
  return (
    <figure className="chart" style={{ margin: 0 }} aria-label={label}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="cat" {...AXIS} interval={0} />
          <YAxis {...AXIS} width={48} domain={[0, max]} tickFormatter={(v) => format(v)} />
          <Tooltip cursor={{ fill: "var(--rule-2)" }} content={<Tip unit="" fmtV={format} />} />
          {series.map((s, j) => <Bar key={j} dataKey={`s${j}`} name={s.name} fill={s.color} isAnimationActive={!reduce} radius={[2, 2, 0, 0]} />)}
        </BarChart>
      </ResponsiveContainer>
      <figcaption className="legend">{series.map((s) => <span key={s.name}><i style={{ background: s.color, height: 10 }} />{s.name}</span>)}</figcaption>
    </figure>
  );
}

/** Calibration: mean predicted vs observed rate per bin (dot size = students); the diagonal is perfect calibration. */
export function Reliability({ rows, height = 260 }: { rows: { mean_predicted: number; observed_rate: number; n: number }[]; height?: number }) {
  const max = Math.min(1, Math.ceil(Math.max(0.1, ...rows.map((r) => Math.max(r.mean_predicted, r.observed_rate))) * 10.5) / 10);
  const data = rows.filter((r) => r.n > 0).map((r) => ({ x: r.mean_predicted, y: r.observed_rate, n: r.n }));
  const biggest = Math.max(1, ...data.map((d) => d.n));
  return (
    <figure className="chart" style={{ margin: 0 }} aria-label="Predicted versus observed withdrawal rate">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
          <CartesianGrid />
          <XAxis dataKey="x" type="number" domain={[0, max]} {...AXIS} tickFormatter={(v) => fmt.pct(v)}
            label={{ value: "predicted", position: "insideBottomRight", offset: -12, fill: "var(--muted)", fontSize: 12 }} />
          <YAxis dataKey="y" type="number" domain={[0, max]} {...AXIS} width={48} tickFormatter={(v) => fmt.pct(v)} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke="var(--muted)" strokeDasharray="4 4" />
          <Tooltip content={({ active, payload }) => active && payload?.length ? <div className="chart-tip">
            {fmt.int(payload[0].payload.n)} students: predicted {fmt.pct(payload[0].payload.x, 1)}, observed {fmt.pct(payload[0].payload.y, 1)}</div> : null} />
          <Line dataKey="y" stroke="none" isAnimationActive={false}
            dot={(p: any) => <circle key={p.index} cx={p.cx} cy={p.cy} r={4 + 8 * Math.sqrt(p.payload.n / biggest)} fill="var(--ink)" fillOpacity={0.75} />} />
        </LineChart>
      </ResponsiveContainer>
    </figure>
  );
}

/** Several weekly series with optional labelled week markers (e.g. "Withdrawn diverge"). */
export function SeriesLines({ series, markers = [], height = 280, label, yLabel }: {
  series: Series[]; markers?: { x: number; label: string; color: string }[]; height?: number; label: string; yLabel: string }) {
  const reduce = useReducedMotion();
  const n = Math.max(...series.map((s) => s.values.length));
  const data = Array.from({ length: n }, (_, w) => Object.fromEntries([["week", w], ...series.map((s) => [s.key, s.values[w] ?? null])]));
  return (
    <figure className="chart" style={{ margin: 0 }} aria-label={label}>
      <div className="small muted" style={{ marginBottom: 2 }}>{yLabel} by course week</div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 20, right: 40, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="week" {...AXIS} interval="preserveStartEnd" minTickGap={24} />
          <YAxis {...AXIS} width={44} />
          <Tooltip content={<Tip />} cursor={{ stroke: "var(--ink)", strokeDasharray: "3 3" }} />
          {markers.map((m, i) => <ReferenceLine key={i} x={m.x} stroke={m.color} strokeDasharray="4 3"
            label={{ value: m.label, position: i % 2 || m.x < 3 ? "insideTopLeft" : "insideTopRight", dy: i % 2 ? 0 : 14, fill: "var(--ink)", fontSize: 12 }} />)}
          {series.map((s) => <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={s.width ?? 2.2} dot={false} isAnimationActive={!reduce} />)}
        </LineChart>
      </ResponsiveContainer>
      <figcaption className="legend">{series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.name}</span>)}</figcaption>
    </figure>
  );
}
