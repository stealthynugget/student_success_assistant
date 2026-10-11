// Small, dependency-free SVG charts tuned for this product (tier colours, checkpoint marker, deadline ticks).
import React from "react";

export const C = { ink: "#172036", muted: "#5c6577", rule: "#dce0e4", marker: "#f4c53d", High: "#b23a2b", Medium: "#c9821a", Low: "#2e7466",
  ai: "#4c4bc2", Distinction: "#2e7466", Pass: "#7d8798", Fail: "#c9821a", Withdrawn: "#b23a2b" };

const W = 640;
const nice = (v) => { if (!v || v <= 0) return 1; const p = 10 ** Math.floor(Math.log10(v)); return Math.ceil(v / p) * p; };

export function LineChart({ series, height = 220, xMax, yMax, xLabel, yLabel, vlines = [], ticks = [], formatY = (v) => v, markers = false, ariaLabel }) {
  const pad = { l: 44, r: 12, t: 12, b: 30 };
  const allX = Math.max(1, xMax ?? Math.max(...series.map((s) => s.values.length - 1)));
  const allY = yMax ?? nice(Math.max(1e-9, ...series.flatMap((s) => s.values.filter((v) => v !== null && v !== undefined))));
  const x = (i) => pad.l + (i / allX) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - v / allY) * (height - pad.t - pad.b);
  const yt = [0, 0.25, 0.5, 0.75, 1].map((f) => f * allY);
  const step = Math.max(1, Math.ceil(allX / 10));
  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel || "line chart"} style={{ width: "100%", height: "auto" }}>
      {yt.map((v) => (
        <g key={v}><line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke={C.rule} strokeWidth="1" />
          <text x={pad.l - 6} y={y(v) + 4} fontSize="11" textAnchor="end" fill={C.muted}>{formatY(v)}</text></g>
      ))}
      {Array.from({ length: Math.floor(allX / step) + 1 }).map((_, i) => (
        <text key={i} x={x(i * step)} y={height - 10} fontSize="11" textAnchor="middle" fill={C.muted}>{i * step}</text>
      ))}
      {ticks.map((t, i) => (
        <g key={`t${i}`}><line x1={x(t.x)} x2={x(t.x)} y1={height - pad.b} y2={height - pad.b + 6} stroke={C.ink} strokeWidth="2" />
          {t.label && <title>{t.label}</title>}</g>
      ))}
      {vlines.map((v, i) => (
        <g key={`v${i}`}><rect x={x(v.x) - 2} y={pad.t} width="4" height={height - pad.t - pad.b} fill={v.color || C.marker} opacity="0.85" />
          {v.label && <text x={x(v.x) + 6} y={pad.t + 12 + i * 14} fontSize="11" fill={C.ink}>{v.label}</text>}</g>
      ))}
      {series.map((s, si) => {
        let d = "";
        s.values.forEach((v, i) => { if (v === null || v === undefined) return; d += `${d && s.values[i - 1] != null ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`; });
        return (
          <g key={si}>
            <path d={d} fill="none" stroke={s.color || C.ink} strokeWidth={s.width || 2} strokeDasharray={s.dash} opacity={s.opacity ?? 1} />
            {markers && s.values.map((v, i) => v != null && <circle key={i} cx={x(i)} cy={y(v)} r="4" fill={s.pointColor?.[i] || s.color || C.ink}><title>{`${s.name}: ${formatY(v)}`}</title></circle>)}
          </g>
        );
      })}
      {xLabel && <text x={W - pad.r} y={height - pad.b - 6} fontSize="11" textAnchor="end" fill={C.muted}>{xLabel}</text>}
      {yLabel && <text x={pad.l} y={pad.t - 2} fontSize="11" fill={C.muted}>{yLabel}</text>}
    </svg>
  );
}

export function Histogram({ counts, edges, thresholds, height = 170 }) {
  const pad = { l: 10, r: 10, t: 34, b: 24 };
  const max = Math.max(1, ...counts);
  const last = counts.reduce((a, c, i) => (c > 0 ? i : a), 0);
  const xmax = Math.min(1, Math.max(0.3, Math.ceil(((edges[last + 1] ?? 1) * 1.15) * 10) / 10, thresholds ? thresholds.high * 1.4 : 0));
  const xs = (v) => pad.l + (v / xmax) * (W - pad.l - pad.r);
  const colour = (lo) => (thresholds && lo >= thresholds.high - 1e-9 ? C.High : thresholds && lo >= thresholds.medium - 1e-9 ? C.Medium : C.Low);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * xmax);
  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="Distribution of withdrawal risk" style={{ width: "100%", height: "auto" }}>
      {counts.map((c, i) => {
        if (edges[i] >= xmax) return null;
        const h = (c / max) * (height - pad.t - pad.b);
        return <rect key={i} x={xs(edges[i]) + 1} y={height - pad.b - h} width={Math.max(1, xs(Math.min(edges[i + 1], xmax)) - xs(edges[i]) - 2)} height={h} fill={colour(edges[i])}><title>{`${(edges[i] * 100).toFixed(0)}–${(edges[i + 1] * 100).toFixed(0)}%: ${c} students`}</title></rect>;
      })}
      {thresholds && [["medium", 12], ["high", 26]].map(([k, ty]) => (
        <g key={k}><line x1={xs(thresholds[k])} x2={xs(thresholds[k])} y1={ty + 2} y2={height - pad.b} stroke={C.ink} strokeDasharray="3 3" />
          <text x={xs(thresholds[k]) + 4} y={ty} fontSize="11" fill={C.ink}>{k === "high" ? "High" : "Medium"} from {(thresholds[k] * 100).toFixed(0)}%</text></g>
      ))}
      {ticks.map((v) => <text key={v} x={xs(v)} y={height - 6} fontSize="11" textAnchor="middle" fill={C.muted}>{Math.round(v * 100)}%</text>)}
    </svg>
  );
}

export function Diverging({ items }) {
  const rowH = 26;
  const h = items.length * rowH + 10;
  const max = Math.max(1e-6, ...items.map((d) => Math.abs(d.value)));
  const labelW = W * 0.42;
  const mid = labelW + (W - labelW) / 2;
  const span = (W - labelW) / 2 - 8;
  return (
    <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-label="Factors pushing risk up or down" style={{ width: "100%", height: "auto" }}>
      <line x1={mid} x2={mid} y1="0" y2={h} stroke={C.ink} strokeWidth="1" />
      {items.map((d, i) => {
        const w = (Math.abs(d.value) / max) * span;
        const up = d.value > 0;
        return (
          <g key={i} transform={`translate(0,${i * rowH + 4})`}>
            <text x={labelW - 10} y="15" fontSize="13" textAnchor="end" fill={C.ink}>{d.label}</text>
            <rect x={up ? mid : mid - w} y="4" width={w} height="14" fill={up ? C.High : C.Low} rx="2"><title>{d.title || d.label}</title></rect>
          </g>
        );
      })}
      <text x={mid - 6} y={h - 1} fontSize="11" textAnchor="end" fill={C.muted}>lowers risk</text>
      <text x={mid + 6} y={h - 1} fontSize="11" fill={C.muted}>raises risk</text>
    </svg>
  );
}

export function HBars({ items, max, format = (v) => v, color = C.ink, labelWidth = 0.45 }) {
  const rowH = 26;
  const h = items.length * rowH + 6;
  const m = max ?? Math.max(1e-9, ...items.map((d) => d.value || 0));
  const x0 = W * labelWidth;
  return (
    <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-label="bar chart" style={{ width: "100%", height: "auto" }}>
      {items.map((d, i) => {
        const w = ((d.value || 0) / m) * (W - x0 - 70);
        return (
          <g key={i} transform={`translate(0,${i * rowH + 3})`}>
            <text x={x0 - 10} y="15" fontSize="13" textAnchor="end" fill={C.ink}>{d.label}</text>
            <rect x={x0} y="4" width={Math.max(0, w)} height="14" fill={d.color || color} rx="2" />
            <text x={x0 + w + 6} y="15" fontSize="12" fill={C.muted}>{format(d.value)}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function GroupedBars({ categories, series, max = 1, format = (v) => v, height = 220 }) {
  const pad = { l: 36, r: 8, t: 10, b: 36 };
  const gw = (W - pad.l - pad.r) / categories.length;
  const bw = (gw * 0.75) / series.length;
  const y = (v) => pad.t + (1 - v / max) * (height - pad.t - pad.b);
  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="grouped bar chart" style={{ width: "100%", height: "auto" }}>
      {[0, 0.5, 1].map((f) => <g key={f}><line x1={pad.l} x2={W - pad.r} y1={y(f * max)} y2={y(f * max)} stroke={C.rule} />
        <text x={pad.l - 4} y={y(f * max) + 4} fontSize="11" textAnchor="end" fill={C.muted}>{format(f * max)}</text></g>)}
      {categories.map((c, ci) => (
        <g key={c}>
          {series.map((s, si) => {
            const v = s.values[ci] ?? 0;
            const x = pad.l + ci * gw + gw * 0.125 + si * bw;
            return <g key={si}><rect x={x} y={y(v)} width={bw - 2} height={y(0) - y(v)} fill={s.color} rx="1"><title>{`${s.name} – ${c}: ${format(v)}`}</title></rect>
              <text x={x + bw / 2 - 1} y={y(v) - 3} fontSize="10" textAnchor="middle" fill={C.muted}>{format(v)}</text></g>;
          })}
          <text x={pad.l + ci * gw + gw / 2} y={height - 14} fontSize="12" textAnchor="middle" fill={C.ink}>{c}</text>
        </g>
      ))}
    </svg>
  );
}

export function Stacked({ parts, height = 18, showLabels = true }) {
  const total = parts.reduce((a, p) => a + (p.value || 0), 0) || 1;
  let acc = 0;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="composition" style={{ width: "100%", height: "auto", display: "block" }} preserveAspectRatio="none">
        {parts.map((p) => { const w = ((p.value || 0) / total) * W; const x = acc; acc += w; return <rect key={p.label} x={x} y="0" width={w} height={height} fill={p.color}><title>{`${p.label}: ${p.display ?? p.value}`}</title></rect>; })}
      </svg>
      {showLabels && <div className="legend" style={{ marginTop: 6 }}>{parts.map((p) => <span key={p.label}><i style={{ background: p.color, height: 10 }} />{p.label} {p.display ?? p.value}</span>)}</div>}
    </div>
  );
}

export function RiskScale({ value, thresholds }) {
  const h = 56;
  const x = (v) => 12 + v * (W - 24);
  return (
    <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-label={`Withdrawal risk ${(value * 100).toFixed(0)}%`} style={{ width: "100%", height: "auto" }}>
      <rect x={x(0)} y="20" width={x(thresholds.medium) - x(0)} height="12" fill={C.Low} opacity="0.35" />
      <rect x={x(thresholds.medium)} y="20" width={x(thresholds.high) - x(thresholds.medium)} height="12" fill={C.Medium} opacity="0.45" />
      <rect x={x(thresholds.high)} y="20" width={x(1) - x(thresholds.high)} height="12" fill={C.High} opacity="0.35" />
      <rect x={x(value) - 3} y="10" width="6" height="32" fill={C.ink} />
      <text x={x(value)} y="8" fontSize="12" textAnchor="middle" fill={C.ink} fontWeight="700">{(value * 100).toFixed(0)}%</text>
      {[0, 0.25, 0.5, 0.75, 1].map((v) => <text key={v} x={x(v)} y={h - 2} fontSize="11" textAnchor="middle" fill={C.muted}>{v * 100}%</text>)}
    </svg>
  );
}

export function Reliability({ rows, height = 240 }) {
  const pad = { l: 40, r: 12, t: 10, b: 30 };
  const m = Math.max(0.05, ...rows.flatMap((r) => [r.mean_predicted, r.observed_rate]));
  const s = (v) => v / m;
  const x = (v) => pad.l + s(v) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - s(v)) * (height - pad.t - pad.b);
  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="Calibration: predicted vs observed" style={{ width: "100%", height: "auto" }}>
      <line x1={x(0)} y1={y(0)} x2={x(m)} y2={y(m)} stroke={C.rule} strokeDasharray="4 4" />
      {rows.map((r, i) => <circle key={i} cx={x(r.mean_predicted)} cy={y(r.observed_rate)} r={Math.max(3, Math.min(14, Math.sqrt(r.n) / 3))} fill={C.ink} opacity="0.75"><title>{`predicted ${(r.mean_predicted * 100).toFixed(1)}%, observed ${(r.observed_rate * 100).toFixed(1)}% (n=${r.n})`}</title></circle>)}
      {[0, 0.5, 1].map((f) => <g key={f}><text x={x(f * m)} y={height - 10} fontSize="11" textAnchor="middle" fill={C.muted}>{(f * m * 100).toFixed(0)}%</text>
        <text x={pad.l - 6} y={y(f * m) + 4} fontSize="11" textAnchor="end" fill={C.muted}>{(f * m * 100).toFixed(0)}%</text></g>)}
      <text x={W - pad.r} y={height - 22} fontSize="11" textAnchor="end" fill={C.muted}>predicted</text>
      <text x={pad.l + 4} y={pad.t + 10} fontSize="11" fill={C.muted}>observed</text>
    </svg>
  );
}

export function Spark({ values, width = 90, height = 22, color = C.ink }) {
  const v = values.filter((x) => x != null);
  if (!v.length) return null;
  const max = Math.max(...v, 1e-9);
  const pts = values.map((x, i) => (x == null ? null : `${(i / Math.max(1, values.length - 1)) * width},${height - (x / max) * (height - 2) - 1}`)).filter(Boolean).join(" ");
  return <svg className="spark" width={width} height={height} aria-hidden="true"><polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" /></svg>;
}
