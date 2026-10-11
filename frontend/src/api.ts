// API client: auth headers, normalised errors, background-job polling and the copilot event stream.
const LS = window.localStorage;
const read = (k: string) => { try { return LS.getItem(k) || ""; } catch { return ""; } };

export const settings = {
  get base() { return read("ssa.apiBase"); },
  get key() { return read("ssa.apiKey"); },
  get admin() { return read("ssa.adminKey"); },
  save({ base, key, admin }: { base?: string; key?: string; admin?: string }) {
    try {
      if (base !== undefined) LS.setItem("ssa.apiBase", base.replace(/\/$/, ""));
      if (key !== undefined) LS.setItem("ssa.apiKey", key);
      if (admin !== undefined) LS.setItem("ssa.adminKey", admin);
    } catch { /* storage unavailable: settings last for this page only */ }
  },
};

export class ApiError extends Error {
  status: number;
  details?: unknown;
  requestId?: string;
  constructor(status: number, body: { error?: string; details?: unknown; request_id?: string } | null) {
    super(body?.error || `Request failed (${status})`);
    this.status = status;
    this.details = body?.details;
    this.requestId = body?.request_id;
  }
}

type Query = Record<string, string | number | boolean | null | undefined>;
export interface ApiOptions { method?: string; body?: unknown; query?: Query; admin?: boolean; signal?: AbortSignal }

function headers(admin = false): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (settings.key) h["X-API-Key"] = settings.key;
  if (admin && settings.admin) h["X-Admin-Key"] = settings.admin;
  return h;
}

function url(path: string, query?: Query) {
  const u = new URL(`${settings.base}/api${path}`, window.location.origin);
  Object.entries(query || {}).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, String(v)); });
  return u;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function api<T = any>(path: string, { method = "GET", body, query, admin = false, signal }: ApiOptions = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url(path, query), { method, headers: headers(admin), body: body ? JSON.stringify(body) : undefined, signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, { error: `Can't reach the service at ${settings.base || window.location.origin}. Check the address in Settings.` });
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { error: text.slice(0, 200) }; }
  if (!res.ok) throw new ApiError(res.status, data);
  return data as T;
}

export interface JobStep { step: string; t: number; source?: string; [k: string]: unknown }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface Job { id: string; status: "queued" | "running" | "done" | "error"; steps?: JobStep[]; result?: any; error?: { message?: string } }

// Start a background job and poll until it finishes; onProgress receives the job (with its steps) after each poll.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function runJob<T = any>(path: string, body: unknown, onProgress?: (j: Job) => void,
  { interval = 350, signal }: { interval?: number; signal?: AbortSignal } = {}): Promise<T> {
  const job = await api<Job>(path, { method: "POST", body, signal });
  onProgress?.(job);
  for (;;) {
    await new Promise((r) => setTimeout(r, interval));
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const j = await api<Job>(`/jobs/${job.id}`, { signal });
    onProgress?.(j);
    if (j.status === "done") return j.result as T;
    if (j.status === "error") throw new ApiError(500, { error: j.error?.message || "The job failed." });
  }
}

export type CopilotEvent =
  | { kind: "tool_call"; data: { id: string; tool: string; arguments: Record<string, unknown> } }
  | { kind: "tool_result"; data: { id: string; tool: string; ok: boolean; ms: number } }
  | { kind: "text"; data: { delta: string } }
  | { kind: "text_reset"; data: Record<string, never> }
  | { kind: "revision"; data: { answer: string } }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | { kind: "done"; data: any }
  | { kind: "error"; data: { error: string } };

// POST /api/copilot/stream and dispatch server-sent events as they arrive (fetch + reader: EventSource cannot POST).
export async function streamCopilot(body: unknown, onEvent: (e: CopilotEvent) => void, signal?: AbortSignal): Promise<void> {
  let res: Response;
  try {
    res = await fetch(url("/copilot/stream"), { method: "POST", headers: { ...headers(), Accept: "text/event-stream" }, body: JSON.stringify(body), signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, { error: "Can't reach the service. Check the address in Settings." });
  }
  if (!res.ok || !res.body) {
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    throw new ApiError(res.status, data);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let kind = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) kind = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) onEvent({ kind, data: JSON.parse(data) } as CopilotEvent);
    }
  }
}

const isNum = (v: unknown): v is number => typeof v === "number" && !Number.isNaN(v);
export const fmt = {
  pct: (v: unknown, d = 0) => (isNum(v) ? `${(100 * v).toFixed(d)}%` : "–"),
  num: (v: unknown, d = 1) => (isNum(v) ? Number(v).toFixed(d) : "–"),
  int: (v: unknown) => (isNum(v) ? Math.round(v).toLocaleString("en-GB") : "–"),
  week: (cp: number, len = 262) => Math.round((cp * len) / 7),
};
