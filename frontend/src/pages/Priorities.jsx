import React, { useState } from "react";
import { api, fmt } from "../api";
import { studentPath, useApi, useStore } from "../store";
import { downloadCsv, ErrorNote, Loading, Tier } from "../ui";

const SIGNALS = { p_withdraw: "Withdrawal risk", p_poor: "Fail or withdraw", academic_gap: "Score gap", engagement_gap: "Activity gap",
  missed_work: "Missed work", recent_decline: "Recent decline" };

export default function Priorities() {
  const { module, presentation, checkpoint } = useStore();
  const [capacity, setCapacity] = useState(0.15);
  const SHOW = 300;
  const q = useApi((s) => api("/prioritize", { method: "POST", body: { code_module: module, code_presentation: presentation, capacity, limit: SHOW, checkpoint }, signal: s }),
    [module, presentation, checkpoint, capacity]);
  const [exporting, setExporting] = useState(false);
  const exportAll = async () => {
    setExporting(true);
    try {
      const all = await api("/prioritize", { method: "POST", body: { code_module: module, code_presentation: presentation, capacity, limit: 2000, checkpoint } });
      downloadCsv(all.ranking.map((r) => ({ rank: r.priority_rank, id_student: r.id_student, tier: r.risk_tier, withdrawal_risk: r.withdrawal_risk,
        fail_or_withdraw: r.p_poor_outcome, within_capacity: r.within_capacity, reasons: r.reasons.join("; ") })), `priorities_${module}_${presentation}.csv`);
    } finally { setExporting(false); }
  };
  const d = q.data;
  const k = d ? d.ranking.filter((r) => r.within_capacity).length : 0;
  return (
    <>
      <div className="page-head">
        <div><h1>Who to contact first</h1><div className="sub">Ranked for advisor capacity. Weights were learned on an earlier presentation to put the most students who later failed or withdrew at the top.</div></div>
        <div className="actions">{d && <button onClick={exportAll} disabled={exporting}>{exporting ? "Preparing file…" : `Download all ${d.cohort_size} as CSV`}</button>}</div>
      </div>
      <div className="row panel tight" style={{ marginBottom: 16 }}>
        <label className="field" style={{ minWidth: 260 }}>Advisor capacity: {fmt.pct(capacity)} of the cohort{d ? ` (${k} students)` : ""}
          <input type="range" min="0.05" max="0.5" step="0.05" value={capacity} onChange={(e) => setCapacity(Number(e.target.value))} /></label>
        {d && <div className="small muted" style={{ maxWidth: 520 }}>Ranking weights: {Object.entries(d.weights).filter(([, v]) => v > 0).map(([s, v]) => `${SIGNALS[s]} ${v}`).join(", ")}.
          The dashed line marks where capacity runs out.</div>}
      </div>
      {q.error && <ErrorNote error={q.error} onRetry={q.reload} />}
      {q.loading && !d && <Loading lines={8} />}
      {d && d.cohort_size > SHOW && <p className="small muted">Showing the top {SHOW} of {d.cohort_size}; the download includes everyone.</p>}
      {d && (
        <div className="table-wrap"><table className="data">
          <thead><tr><th className="num">Rank</th><th>Student</th><th>Withdrawal tier</th><th className="num">Withdrawal risk</th><th className="num">Fail or withdraw</th><th>Why this student</th><th>Top model drivers</th></tr></thead>
          <tbody>{d.ranking.map((r) => (
            <tr key={r.id_student} className={`click ${r.priority_rank === k ? "capacity-line" : ""}`} onClick={() => { window.location.hash = studentPath(module, presentation, r.id_student).slice(1); }}
                style={r.within_capacity ? {} : { color: "#5c6577" }}>
              <td className="num">{r.priority_rank}</td><td><a href={studentPath(module, presentation, r.id_student)}>{r.id_student}</a></td><td><Tier tier={r.risk_tier} /></td>
              <td className="num">{fmt.pct(r.withdrawal_risk)}</td><td className="num">{fmt.pct(r.p_poor_outcome)}</td>
              <td>{r.reasons.map((x) => <span key={x} className="chip">{x}</span>)}</td>
              <td className="small">{r.top_drivers.filter((x) => x.direction === "increases_risk").map((x) => x.label).join(", ")}</td></tr>))}</tbody>
        </table></div>
      )}
    </>
  );
}
