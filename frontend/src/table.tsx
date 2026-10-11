// Sortable, column-configurable, virtualised data table (TanStack Table v9 + TanStack Virtual). The markup stays a real
// <table> (spacer rows around the rendered window) so screen readers keep row/column semantics.
import {
  columnVisibilityFeature, createColumnHelper, createSortedRowModel, rowSortingFeature, sortFn_alphanumeric, sortFn_basic, tableFeatures, useTable,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Columns3 } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";

export const features = tableFeatures({
  rowSortingFeature, columnVisibilityFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: { basic: sortFn_basic, alphanumeric: sortFn_alphanumeric },
});
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const colHelper = <T extends Record<string, any>>() => createColumnHelper<typeof features, T>();

/* eslint-disable @typescript-eslint/no-explicit-any */
export function DataTable<T extends Record<string, any>>({ data, columns, initialSorting = [], caption, onRowActivate, rowClass, height = 600, rowHeight = 41, toolbar }: {
  data: T[]; columns: any[]; initialSorting?: { id: string; desc: boolean }[]; caption: string; onRowActivate?: (row: T) => void;
  rowClass?: (row: T, index: number) => string | undefined; height?: number; rowHeight?: number; toolbar?: ReactNode }) {
  const table = useTable({ features, columns, data, initialState: { sorting: initialSorting } as any });
  const scroll = useRef<HTMLDivElement>(null);
  const rows = table.getRowModel().rows;
  const v = useVirtualizer({ count: rows.length, getScrollElement: () => scroll.current, estimateSize: () => rowHeight, overscan: 8,
    getItemKey: (i) => rows[i].id });
  const items = v.getVirtualItems();
  const top = items.length ? items[0].start : 0;
  const bottom = items.length ? v.getTotalSize() - items[items.length - 1].end : 0;
  const [menu, setMenu] = useState(false);
  const leaf = table.getAllLeafColumns() as any[];
  const visible = leaf.filter((c) => c.getIsVisible()).length;
  return (
    <div className="stack">
      <div className="row">
        {toolbar}
        <div style={{ marginLeft: "auto", position: "relative" }}>
          <button type="button" aria-expanded={menu} aria-controls="col-menu" onClick={() => setMenu((m) => !m)}><Columns3 aria-hidden="true" />Columns</button>
          {menu && (
            <fieldset id="col-menu" className="panel tight" style={{ position: "absolute", right: 0, top: "110%", zIndex: 8, minWidth: 220, margin: 0 }}>
              <legend className="visually-hidden">Visible columns</legend>
              {leaf.filter((c) => c.getCanHide?.() !== false).map((c) => (
                <label key={c.id} style={{ display: "flex", gap: 8, padding: "3px 0" }}>
                  <input type="checkbox" checked={c.getIsVisible()} disabled={c.getIsVisible() && visible <= 2} onChange={c.getToggleVisibilityHandler()} />
                  {typeof c.columnDef.header === "string" ? c.columnDef.header : c.id}</label>))}
            </fieldset>)}
        </div>
      </div>
      <div ref={scroll} className="table-wrap vt" style={{ maxHeight: height, overflow: "auto" }} tabIndex={0} aria-label={`${caption}, scrollable`}>
        <table className="data">
          <caption className="visually-hidden">{caption}. {rows.length} rows; column headers sort.</caption>
          <thead>{table.getHeaderGroups().map((g: any) => (
            <tr key={g.id}>{g.headers.map((h: any) => {
              const sorted = h.column.getIsSorted();
              const num = h.column.columnDef.meta?.num;
              return (
                <th key={h.id} scope="col" className={num ? "num" : undefined} aria-sort={sorted === "asc" ? "ascending" : sorted === "desc" ? "descending" : "none"}>
                  {h.column.getCanSort() ? (
                    <button type="button" onClick={h.column.getToggleSortingHandler()}>
                      <table.FlexRender header={h} />{sorted === "asc" ? " ▴" : sorted === "desc" ? " ▾" : ""}</button>
                  ) : <table.FlexRender header={h} />}
                </th>);
            })}</tr>))}</thead>
          <tbody>
            {top > 0 && <tr aria-hidden="true"><td colSpan={visible} style={{ height: top, padding: 0, border: 0 }} /></tr>}
            {items.map((it) => {
              const row: any = rows[it.index];
              return (
                <tr key={row.id} data-index={it.index} className={[onRowActivate ? "click" : "", rowClass?.(row.original, it.index) || ""].join(" ").trim() || undefined}
                  onClick={onRowActivate ? () => onRowActivate(row.original) : undefined}
                  onKeyDown={onRowActivate ? (e) => { if (e.key === "Enter") onRowActivate(row.original); } : undefined}>
                  {row.getVisibleCells().map((cell: any) => (
                    <td key={cell.id} className={cell.column.columnDef.meta?.num ? "num" : undefined}><table.FlexRender cell={cell} /></td>))}
                </tr>);
            })}
            {bottom > 0 && <tr aria-hidden="true"><td colSpan={visible} style={{ height: bottom, padding: 0, border: 0 }} /></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
