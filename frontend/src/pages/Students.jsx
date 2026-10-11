import React, { useState } from "react";
import { api, fmt } from "../api";
import { studentPath, useApi, useStore } from "../store";
import { Empty, ErrorNote, Loading, Seg, Tier } from "../ui";

const COLS = [["id_student", "Student"], ["risk_tier", "Withdrawal tier"], ["withdrawal_risk", "Withdrawal risk"], ["p_poor_outcome", "Fail or withdraw"],
  ["weighted_score", "Score"], ["score_pct", "Score vs cohort"], ["missed_rate", "Missed work"], ["clicks_pct", "Activity vs cohort"],
  ["days_since_last_activity", "Days inactive"]];
const PAGE = 50;

export default function Students() {
  const { module, presentation, checkpoint } = useStore();
  const [tier, setTier] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState(["withdrawal_risk", true]);
  const [page, setPage] = useState(0);
  const q = useApi((s) => api("/students", { query: { code_module: module, code_presentation: presentation, checkpoint, tier, search,
    sort: sort[0] === "risk_tier" ? "withdrawal_risk" : sort[0], descending: sort[1], limit: PAGE, offset: page * PAGE }, signal: s }),
  [module, presentation, checkpoint, tier, search, sort, page]);
  const toggle = (k) => { setPage(0); setSort(([s, d]) => [k, s === k ? !d : true]); };
  return (
    <>
      <div className="page-head"><div><h1>Students</h1><div className="sub">Everyone still enrolled in {module} {presentation} at this point, with their computed signals.</div></div></div>
      <div className="row" style={{ marginBottom: 16 }}>
        <Seg label="Tier" value={tier} onChange={(v) => { setTier(v); setPage(0); }} options={[["", "All"], ["High", "High"], ["Medium", "Medium"], ["Low", "Low"]]} />
        <label className="field" style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>Find student
          <input inputMode="numeric" value={search} onChange={(e) => { setSearch(e.target.value.replace(/\D/g, "")); setPage(0); }} placeholder="ID" style={{ width: 120 }} /></label>
        {q.data && <span className="muted small">{q.data.total.toLocaleString()} students</span>}
      </div>
      {q.error && <ErrorNote error={q.error} onRetry={q.reload} />}
      {q.loading && !q.data && <Loading lines={6} />}
      {q.data && q.data.total === 0 && <Empty title="No students match">Clear the tier filter or the ID search.</Empty>}
      {q.data && q.data.total > 0 && (
        <>
          <div className="table-wrap"><table className="data">
            <thead><tr>{COLS.map(([k, l]) => <th key={k} className={k === "id_student" || k === "risk_tier" ? "" : "num"} aria-sort={sort[0] === k ? (sort[1] ? "descending" : "ascending") : "none"}>
              <button onClick={() => toggle(k)}>{l}{sort[0] === k ? (sort[1] ? " ▾" : " ▴") : ""}</button></th>)}</tr></thead>
            <tbody>{q.data.students.map((s) => (
              <tr key={s.id_student} className="click" onClick={() => { window.location.hash = studentPath(module, presentation, s.id_student).slice(1); }}>
                <td><a href={studentPath(module, presentation, s.id_student)}>{s.id_student}</a></td><td><Tier tier={s.risk_tier} /></td>
                <td className="num">{fmt.pct(s.withdrawal_risk)}</td><td className="num">{fmt.pct(s.p_poor_outcome)}</td>
                <td className="num">{s.weighted_score == null ? "none yet" : fmt.num(s.weighted_score, 0)}</td><td className="num">{fmt.pct(s.score_pct)}</td>
                <td className="num">{fmt.pct(s.missed_rate)}</td><td className="num">{fmt.pct(s.clicks_pct)}</td><td className="num">{fmt.int(s.days_since_last_activity)}</td></tr>))}</tbody>
          </table></div>
          <div className="row" style={{ marginTop: 12 }}>
            <button disabled={page === 0} onClick={() => setPage((p) => p - 1)}>Previous</button>
            <span className="small muted">Page {page + 1} of {Math.max(1, Math.ceil(q.data.total / PAGE))}</span>
            <button disabled={(page + 1) * PAGE >= q.data.total} onClick={() => setPage((p) => p + 1)}>Next</button>
          </div>
        </>
      )}
    </>
  );
}
