import { useEffect, useRef } from "react";

/**
 * Ordered-dither dot field shaped as a dome. The waves inside it are real data: mean weekly VLE activity of students who
 * went on to pass and of students who went on to withdraw (from /api/trends/course). The pointer acts as a lens.
 * Drawing happens on a canvas in a requestAnimationFrame loop that never touches React state; it pauses off-screen and
 * in background tabs, and renders a single still frame when reduced motion is requested.
 */
export interface DitherSeries { values: number[]; strength: number }

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + 0.5) / 16);

export function DitherField({ series, className, label }: { series: DitherSeries[]; className?: string; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const pointer = useRef({ x: -9999, y: -9999, active: false });
  const data = useRef(series);
  data.current = series;
  const redraw = useRef<() => void>(() => {});

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0, visible = true, last = 0, W = 0, H = 0, dpr = 1;
    const CELL = 5;

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const curveAt = (vals: number[], t: number) => { // t in [0, 1] -> interpolated value
      if (!vals.length) return 0;
      const f = t * (vals.length - 1), i = Math.floor(f), k = f - i;
      return (vals[i] ?? 0) * (1 - k) + (vals[Math.min(i + 1, vals.length - 1)] ?? 0) * k;
    };

    const draw = (time: number) => {
      const ink = getComputedStyle(canvas).getPropertyValue("--ink").trim() || "#ececea";
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = ink;
      const cx = W / 2, cy = H * 1.02, R = Math.min(Math.max(W * 0.48, H * 0.78), H * 0.98); // portrait: wider than the screen
      const sr = data.current;
      const max = Math.max(1e-6, ...sr.flatMap((s) => s.values));
      const breathe = reduce ? 0 : Math.sin(time / 2400) * 0.035;
      const px = pointer.current.x, py = pointer.current.y, lens = pointer.current.active ? 70 : 0;
      for (let y = 0; y < H; y += CELL) {
        for (let x = 0; x < W; x += CELL) {
          const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy);
          if (d > R) continue;
          const rim = d / R;                                    // 0 centre, 1 rim
          let b = 0.02 + 0.62 * Math.pow(rim, 9);               // bright rim, dark interior (the dome)
          const t = (x - (cx - R)) / (2 * R);                   // course progress across the dome
          if (t >= 0 && t <= 1) {
            for (const s of sr) {
              const v = curveAt(s.values, t) / max;
              const yc = cy - R * (0.06 + (0.4 + breathe) * v);  // the curve's height in the lower dome, below the ask box
              const g = Math.exp(-((y - yc) ** 2) / 55);
              const fill = y > yc ? 0.16 * Math.exp(-(y - yc) / (R * 0.12)) : 0;
              b += s.strength * (0.9 * g + fill);
            }
          }
          if (lens) { const ld = Math.hypot(x - px, y - py); if (ld < lens) b += 0.45 * (1 - ld / lens); }
          const th = BAYER[((y / CELL) & 3) * 4 + ((x / CELL) & 3)];
          if (b > th) { const s = b > th + 0.35 ? CELL - 1.4 : CELL - 2.6; ctx.fillRect(x, y, s, s); }
        }
      }
    };

    const loop = (time: number) => {
      raf = requestAnimationFrame(loop);
      if (!visible || document.hidden || time - last < 33) return; // ~30 fps is plenty for a backdrop
      last = time;
      draw(time);
    };

    resize();
    redraw.current = () => draw(performance.now());
    const ro = new ResizeObserver(() => { resize(); draw(performance.now()); });
    ro.observe(canvas);
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; }, { threshold: 0 });
    io.observe(canvas);
    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      pointer.current = { x: e.clientX - r.left, y: e.clientY - r.top, active: true };
      if (reduce) draw(performance.now());
    };
    const onLeave = () => { pointer.current.active = false; if (reduce) draw(performance.now()); };
    const host = canvas.parentElement ?? canvas;
    host.addEventListener("pointermove", onMove);
    host.addEventListener("pointerleave", onLeave);
    if (reduce) draw(0); else raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf); ro.disconnect(); io.disconnect();
      host.removeEventListener("pointermove", onMove); host.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  useEffect(() => { redraw.current(); }, [series]); // new data: draw now (matters for the still, reduced-motion frame)

  return <canvas ref={ref} className={className} role="img" aria-label={label} />;
}
