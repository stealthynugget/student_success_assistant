import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { api, fmt, type ApiError } from "../api";
import { studentPath, useStore } from "../store";
import { colHelper, DataTable } from "../table";
import { downloadCsv, ErrorNote, Loading, Tier } from "../ui";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Row = Record<string, any>;
const h = colHelper<Row>();
const SIGNALS: Record<string, string> = { p_withdraw: "Withdrawal risk", p_poor: "Fail or withdraw", academic_gap: "Score gap",
  engagement_gap: "Activity gap", missed_work: "Missed work", recent_decline: "Recent decline" };
const num = { meta: { num: true } } as any;

export default function Priorities() {
  const { module, presentation, checkpoint } = useStore();
  const nav = useNavigate();
  const [capacity, setCapacity] = useState(0.15);
  const q = useQuery({ queryKey: ["priorities", module, presentation, checkpoint, capacity], enabled: !!checkpoint, placeholderData: (prev) => prev,
    queryFn: ({ signal }) => api("/prioritize", { method: "POST", body: { code_module: module, code_presentation: presentation, capacity, limit: 2000, checkpoint }, signal }) });
  const d: any = q.data;
  const k = d ? d.ranking.filter((r: Row) => r.within_capacity).length : 0;
  const columns = useMemo(() => [
    h.accessor("priority_rank", { header: "Rank", sortFn: "basic", ...num }),
    h.accessor("id_student", { header: "Student", sortFn: "basic", cell: (c: any) => <Link to={studentPath(module, presentation, c.getValue())}>{c.getValue()}</Link> }),
    h.accessor("withdrawal_risk", { id: "tier", header: "Withdrawal tier", sortFn: "basic", cell: (c: any) => <Tier tier={c.row.original.risk_tier} /> }),
    h.accessor("withdrawal_risk", { header: "Withdrawal risk", sortFn: "basic", cell: (c: any) => fmt.pct(c.getValue()), ...num }),
    h.accessor("p_poor_outcome", { header: "Fail or withdraw", sortFn: "basic", cell: (c: any) => fmt.pct(c.getValue()), ...num }),
    h.accessor("reasons", { header: "Why this student", enableSorting: false, cell: (c: any) => c.getValue().map((x: string) => <span key={x} className="chip">{x}</span>) }),
    h.accessor("top_drivers", { header: "Top model drivers", enableSorting: false,
      cell: (c: any) => <span className="small">{c.getValue().filter((x: any) => x.direction === "increases_risk").map((x: any) => x.label).join(", ")}</span> }),
  ], [module, presentation]);
  const exportAll = () => downloadCsv(d.ranking.map((r: Row) => ({ rank: r.priority_rank, id_student: r.id_student, tier: r.risk_tier,
    withdrawal_risk: r.withdrawal_risk, fail_or_withdraw: r.p_poor_outcome, within_capacity: r.within_capacity, reasons: r.reasons.join("; ") })),
    `priorities_${module}_${presentation}.csv`);
  return (
    <>
      <div className="page-head">
        <div><h1>Who to contact first</h1>
          <div className="sub">Ranked for advisor capacity. Ranking weights were learned on an earlier presentation to put students who later failed or withdrew at the top.</div></div>
        <div className="actions">{d && <button onClick={exportAll}><Download aria-hidden="true" />Download all {d.cohort_size} as CSV</button>}</div>
      </div>
      <div className="row panel tight" style={{ marginBottom: 16 }}>
        <label className="field" style={{ minWidth: 280 }}>Advisor capacity: {fmt.pct(capacity)} of the cohort{d ? ` (${k} students)` : ""}
          <input type="range" min="0.05" max="0.5" step="0.05" value={capacity} onChange={(e) => setCapacity(Number(e.target.value))} /></label>
        {d && <div className="small muted" style={{ maxWidth: 560 }}>Ranking weights: {Object.entries(d.weights).filter(([, v]) => (v as number) > 0)
          .map(([s, v]) => `${SIGNALS[s] || s} ${v}`).join(", ")}. Rows below the dashed line are beyond capacity.</div>}
      </div>
      {q.error && <ErrorNote error={q.error as ApiError} onRetry={() => q.refetch()} />}
      {q.isLoading && <Loading lines={8} label="Ranking students" />}
      {d && <DataTable data={d.ranking} columns={columns} caption={`Students ranked for contact in ${module} ${presentation}`} rowHeight={48}
        initialSorting={[{ id: "priority_rank", desc: false }]} onRowActivate={(r) => nav(studentPath(module, presentation, r.id_student))}
        rowClass={(r) => [r.priority_rank === k ? "capacity-line" : "", r.within_capacity ? "" : "beyond"].join(" ").trim() || undefined} />}
    </>
  );
}
