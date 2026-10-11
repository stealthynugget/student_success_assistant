// App state: service config/readiness (TanStack Query), the course clock (module, presentation, checkpoint) and toasts.
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";
import { api, ApiError } from "./api";

export interface Checkpoint { checkpoint: number; tag: string; approx_week: number; primary: boolean; available: boolean }
export interface Config {
  checkpoints: Checkpoint[]; modules: string[]; presentations_by_module: Record<string, string[]>; latest_presentation: string;
  vocab: { regions: string[]; education: string[]; age_bands: string[]; outcomes: string[] };
  features: Record<string, string>;
  llm: { configured: boolean; enabled: boolean; provider: string; model: string; fast_model: string; judge_model: string; online_eval_sample_rate: number };
  dense_retrieval: string | null;
}

interface Store {
  config: Config | null; ready: any; error: ApiError | null; // eslint-disable-line @typescript-eslint/no-explicit-any
  module: string; setModule: (m: string) => void;
  presentation: string; setPresentation: (p: string) => void;
  checkpoint: number | null; setCheckpoint: (c: number) => void;
  reload: () => void; notify: (msg: string) => void;
}
const Ctx = createContext<Store | null>(null);
const LS = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};

export function StoreProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const cfg = useQuery({ queryKey: ["config"], queryFn: ({ signal }) => api<Config>("/config", { signal }), staleTime: 5 * 60_000, retry: 1 });
  const ready = useQuery({ queryKey: ["ready"], queryFn: ({ signal }) => api("/ready", { signal }), staleTime: 60_000, retry: 1 });
  const [module, setModule] = useState(LS.get("ssa.module") || "BBB");
  const [presentation, setPresentation] = useState(LS.get("ssa.presentation") || "");
  const [checkpoint, setCheckpoint] = useState<number | null>(Number(LS.get("ssa.checkpoint")) || null);
  const config = cfg.data ?? null;

  useEffect(() => {
    if (!config) return;
    setPresentation((p) => p || config.latest_presentation);
    setCheckpoint((cp) => (cp && config.checkpoints.some((c) => Math.abs(c.checkpoint - cp) < 1e-9) ? cp
      : config.checkpoints.find((x) => x.primary)?.checkpoint ?? config.checkpoints[0]?.checkpoint ?? null));
  }, [config]);
  useEffect(() => { LS.set("ssa.module", module); }, [module]);
  useEffect(() => { if (presentation) LS.set("ssa.presentation", presentation); }, [presentation]);
  useEffect(() => { if (checkpoint) LS.set("ssa.checkpoint", String(checkpoint)); }, [checkpoint]);
  useEffect(() => {
    if (!config) return;
    const list = config.presentations_by_module[module] || [];
    if (list.length && !list.includes(presentation)) setPresentation(list[list.length - 1]);
  }, [module, config]); // eslint-disable-line react-hooks/exhaustive-deps

  const notify = useCallback((msg: string) => { toast(msg); }, []);
  const reload = useCallback(() => { qc.invalidateQueries(); }, [qc]);
  const error = (cfg.error || ready.error) as ApiError | null;
  const value = useMemo<Store>(() => ({ config, ready: ready.data ?? null, error, module, setModule, presentation, setPresentation, checkpoint,
    setCheckpoint, reload, notify }), [config, ready.data, error, module, presentation, checkpoint, reload, notify]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useStore = () => {
  const s = useContext(Ctx);
  if (!s) throw new Error("useStore outside StoreProvider");
  return s;
};

// Compatibility hook for pages not yet moved to TanStack Query: re-runs when deps change and cancels stale requests.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function useApi<T = any>(fn: (signal: AbortSignal) => Promise<T>, deps: unknown[], { enabled = true } = {}) {
  const [state, setState] = useState<{ data: T | null; error: ApiError | null; loading: boolean }>({ data: null, error: null, loading: enabled });
  const ctl = useRef<AbortController | null>(null);
  const run = useCallback(() => {
    if (!enabled) return;
    ctl.current?.abort();
    const c = new AbortController();
    ctl.current = c;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn(c.signal).then((data) => { if (!c.signal.aborted) setState({ data, error: null, loading: false }); })
      .catch((error) => { if (error.name !== "AbortError") setState({ data: null, error, loading: false }); });
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { run(); return () => ctl.current?.abort(); }, [run]);
  return { ...state, reload: run };
}

export const studentPath = (m: string, p: string, id: number | string) => `/students/${m}/${p}/${id}`;
