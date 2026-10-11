import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { api, fmt, type ApiError } from "../api";
import { studentPath, useStore } from "../store";
import { colHelper, DataTable } from "../table";
import { Empty, ErrorNote, Loading, Seg, Tier } from "../ui";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Row = Record<string, any>;
const h = colHelper<Row>();
const pct = (v: unknown) => fmt.pct(v);
const num = { meta: { num: true } } as any;

export default function Students() {
  const { module, presentation, checkpoint } = useStore();
  const nav = useNavigate();
  const [tier, setTier] = useState("");
  const [search, setSearch] = useState("");
  const q = useQuery({ queryKey: ["students", module, presentation, checkpoint], enabled: !!checkpoint,
    queryFn: ({ signal }) => api("/students", { query: { code_module: module, code_presentation: presentation, checkpoint, sort: "withdrawal_risk",
      descending: true, limit: 5000 }, signal }) });
  const columns = useMemo(() => [
    h.accessor("id_student", { header: "Student", sortFn: "basic", cell: (c: any) => <Link to={studentPath(module, presentation, c.getValue())}>{c.getValue()}</Link> }),
    h.accessor("withdrawal_risk", { id: "tier", header: "Withdrawal tier", sortFn: "basic", cell: (c: any) => <Tier tier={c.row.original.risk_tier} /> }),
    h.accessor("withdrawal_risk", { header: "Withdrawal risk", sortFn: "basic", cell: (c: any) => pct(c.getValue()), ...num }),
    h.accessor("p_poor_outcome", { header: "Fail or withdraw", sortFn: "basic", cell: (c: any) => pct(c.getValue()), ...num }),
    h.accessor("weighted_score", { header: "Score", sortFn: "basic", sortUndefined: "last", cell: (c: any) => (c.getValue() == null ? "none yet" : fmt.num(c.getValue(), 0)), ...num }),
    h.accessor("score_pct", { header: "Score vs cohort", sortFn: "basic", cell: (c: any) => pct(c.getValue()), ...num }),
    h.accessor("missed_rate", { header: "Missed work", sortFn: "basic", cell: (c: any) => pct(c.getValue()), ...num }),
    h.accessor("clicks_pct", { header: "Activity vs cohort", sortFn: "basic", cell: (c: any) => pct(c.getValue()), ...num }),
    h.accessor("days_since_last_activity", { header: "Days inactive", sortFn: "basic", cell: (c: any) => fmt.int(c.getValue()), ...num }),
  ], [module, presentation]);
  const all: Row[] = (q.data as any)?.students ?? EMPTY;
  const rows = useMemo(() => all.filter((s) => (!tier || s.risk_tier === tier) && (!search || String(s.id_student).includes(search))), [all, tier, search]);
  return (
    <>
      <div className="page-head"><div><h1>Students</h1>
        <div className="sub">Everyone still enrolled in {module} {presentation} at this point, with their computed signals. Select a row to open the case file.</div></div></div>
      {q.error && <ErrorNote error={q.error as ApiError} onRetry={() => q.refetch()} />}
      {q.isLoading && <Loading lines={8} label="Loading students" />}
      {q.data && (
        <DataTable data={rows} columns={columns} caption={`Students still enrolled in ${module} ${presentation}`}
          initialSorting={[{ id: "withdrawal_risk", desc: true }]} onRowActivate={(r) => nav(studentPath(module, presentation, r.id_student))}
          toolbar={<>
            <Seg label="Tier" value={tier} onChange={setTier} options={[["", "All"], ["High", "High"], ["Medium", "Medium"], ["Low", "Low"]]} />
            <label className="row small">Find student
              <input inputMode="numeric" value={search} onChange={(e) => setSearch(e.target.value.replace(/\D/g, ""))} placeholder="ID" style={{ width: 120 }} /></label>
            <span className="muted small" aria-live="polite">{fmt.int(rows.length)} of {fmt.int(all.length)} students</span>
          </>} />
      )}
      {q.data && rows.length === 0 && <Empty title="No students match">Clear the tier filter or the ID search.</Empty>}
    </>
  );
}
const EMPTY: Row[] = [];
