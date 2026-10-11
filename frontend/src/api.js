// Thin API client: auth headers, normalised errors, background-job polling.
const LS = window.localStorage;

export const settings = {
  get base() { return LS.getItem("ssa.apiBase") || ""; },
  get key() { return LS.getItem("ssa.apiKey") || ""; },
  get admin() { return LS.getItem("ssa.adminKey") || ""; },
  save({ base, key, admin }) {
    if (base !== undefined) LS.setItem("ssa.apiBase", base.replace(/\/$/, ""));
    if (key !== undefined) LS.setItem("ssa.apiKey", key);
    if (admin !== undefined) LS.setItem("ssa.adminKey", admin);
  },
};

export class ApiError extends Error {
  constructor(status, body) {
    super(body?.error || `Request failed (${status})`);
    this.status = status;
    this.details = body?.details;
    this.requestId = body?.request_id;
  }
}

export async function api(path, { method = "GET", body, query, admin = false, signal } = {}) {
  const url = new URL(`${settings.base}/api${path}`, window.location.origin);
  Object.entries(query || {}).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v); });
  const headers = { "Content-Type": "application/json" };
  if (settings.key) headers["X-API-Key"] = settings.key;
  if (admin && settings.admin) headers["X-Admin-Key"] = settings.admin;
  let res;
  try {
    res = await fetch(url, { method, headers, body: body ? JSON.stringify(body) : undefined, signal });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError(0, { error: `Can't reach the service at ${settings.base || window.location.origin}. Check the address in Settings.` });
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { error: text.slice(0, 200) }; }
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

// Start a background job and poll until it finishes. onProgress receives the job (with steps) after each poll.
export async function runJob(path, body, onProgress, { interval = 600, signal } = {}) {
  const job = await api(path, { method: "POST", body, signal });
  onProgress?.(job);
  for (;;) {
    await new Promise((r) => setTimeout(r, interval));
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const j = await api(`/jobs/${job.id}`, { signal });
    onProgress?.(j);
    if (j.status === "done") return j.result;
    if (j.status === "error") throw new ApiError(500, { error: j.error?.message || "The job failed." });
  }
}

export const fmt = {
  pct: (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `${(100 * v).toFixed(d)}%`),
  num: (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d)),
  int: (v) => (v === null || v === undefined ? "–" : Math.round(v).toLocaleString()),
  week: (cp, len = 262) => Math.round((cp * len) / 7),
};
