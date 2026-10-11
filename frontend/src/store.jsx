import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";

const Ctx = createContext(null);
const LS = window.localStorage;

export function StoreProvider({ children }) {
  const [config, setConfig] = useState(null);
  const [ready, setReady] = useState(null);
  const [error, setError] = useState(null);
  const [module, setModule] = useState(LS.getItem("ssa.module") || "BBB");
  const [presentation, setPresentation] = useState(LS.getItem("ssa.presentation") || "");
  const [checkpoint, setCheckpoint] = useState(Number(LS.getItem("ssa.checkpoint")) || null);
  const [toast, setToast] = useState(null);
  const [nonce, setNonce] = useState(0);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [r, c] = await Promise.all([api("/ready"), api("/config")]);
      setReady(r); setConfig(c);
      setPresentation((p) => p || c.latest_presentation);
      setCheckpoint((cp) => cp || c.checkpoints.find((x) => x.primary)?.checkpoint || c.checkpoints[0]?.checkpoint);
    } catch (e) { setError(e); }
  }, []);
  useEffect(() => { load(); }, [load, nonce]);
  useEffect(() => { LS.setItem("ssa.module", module); }, [module]);
  useEffect(() => { if (presentation) LS.setItem("ssa.presentation", presentation); }, [presentation]);
  useEffect(() => { if (checkpoint) LS.setItem("ssa.checkpoint", String(checkpoint)); }, [checkpoint]);
  useEffect(() => {
    if (!config) return;
    const list = config.presentations_by_module[module] || [];
    if (!list.includes(presentation)) setPresentation(list[list.length - 1]);
  }, [module, config]); // eslint-disable-line react-hooks/exhaustive-deps

  const notify = useCallback((msg) => { setToast(msg); setTimeout(() => setToast(null), 3200); }, []);
  const value = useMemo(() => ({ config, ready, error, module, setModule, presentation, setPresentation, checkpoint, setCheckpoint,
    reload: () => setNonce((n) => n + 1), notify, toast }), [config, ready, error, module, presentation, checkpoint, notify, toast]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useStore = () => useContext(Ctx);

// Fetch helper: re-runs when deps change, cancels stale requests.
export function useApi(fn, deps, { enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: enabled });
  const ctl = useRef(null);
  const run = useCallback(() => {
    if (!enabled) return;
    ctl.current?.abort();
    const c = new AbortController();
    ctl.current = c;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn(c.signal).then((data) => !c.signal.aborted && setState({ data, error: null, loading: false }))
      .catch((error) => { if (error.name !== "AbortError") setState({ data: null, error, loading: false }); });
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { run(); return () => ctl.current?.abort(); }, [run]);
  return { ...state, reload: run };
}

// Minimal hash router: #/path/a/b?x=1
export function useRoute() {
  const parse = () => {
    const [path, qs] = (window.location.hash.slice(1) || "/today").split("?");
    return { parts: path.split("/").filter(Boolean), query: Object.fromEntries(new URLSearchParams(qs || "")) };
  };
  const [route, setRoute] = useState(parse);
  useEffect(() => {
    const on = () => { setRoute(parse()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
export const go = (path) => { window.location.hash = path; };
export const studentPath = (m, p, id) => `#/student/${m}/${p}/${id}`;
