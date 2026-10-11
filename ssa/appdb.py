"""Small operational store (SQLite, WAL): LLM usage ledger and online evaluation records."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ssa.config import Settings, get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, feature TEXT, model TEXT, prompt_tokens INTEGER,
  completion_tokens INTEGER, latency_s REAL, cached INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS online_evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, ref_id TEXT, verdict TEXT, overall REAL,
  scores TEXT, judge_model TEXT, details TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_feature ON llm_usage(feature);
CREATE INDEX IF NOT EXISTS idx_evals_kind ON online_evals(kind);
"""
_lock = threading.Lock()


class AppDB:
    def __init__(self, settings: Settings | None = None, path: Path | None = None):
        self.settings = settings or get_settings()
        self.path = Path(path or self.settings.app_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        c.row_factory = sqlite3.Row
        return c

    def log_usage(self, feature: str, model: str, prompt_tokens, completion_tokens, latency_s, cached, error=None):
        with _lock, self._conn() as c:
            c.execute("INSERT INTO llm_usage (ts, feature, model, prompt_tokens, completion_tokens, latency_s, cached, error) "
                      "VALUES (?,?,?,?,?,?,?,?)", (time.time(), feature, model, prompt_tokens or 0, completion_tokens or 0,
                                                   latency_s, int(bool(cached)), error))

    def usage_summary(self, since_s: float | None = None) -> dict:
        q = "SELECT feature, model, COUNT(*) n, SUM(prompt_tokens) pt, SUM(completion_tokens) ct, AVG(latency_s) lat, " \
            "SUM(cached) cached, SUM(CASE WHEN error IS NULL THEN 0 ELSE 1 END) errors FROM llm_usage"
        args = ()
        if since_s:
            q += " WHERE ts >= ?"
            args = (time.time() - since_s,)
        q += " GROUP BY feature, model ORDER BY n DESC"
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q, args).fetchall()]
        pp, pc = self.settings.llm_price_per_1k
        for r in rows:
            r["estimated_cost_usd"] = round(((r["pt"] or 0) * pp + (r["ct"] or 0) * pc) / 1000, 4)
            r["avg_latency_s"] = round(r.pop("lat") or 0, 3)
        total = {k: sum((r.get(k) or 0) for r in rows) for k in ("n", "pt", "ct", "cached", "errors", "estimated_cost_usd")}
        total["estimated_cost_usd"] = round(total["estimated_cost_usd"], 4)
        return {"by_feature": rows, "total": total, "price_per_1k_tokens_usd": {"prompt": pp, "completion": pc}}

    def add_eval(self, kind: str, ref_id: str, verdict: str, overall, scores: dict | None, judge_model: str | None, details: dict):
        with _lock, self._conn() as c:
            cur = c.execute("INSERT INTO online_evals (ts, kind, ref_id, verdict, overall, scores, judge_model, details) VALUES (?,?,?,?,?,?,?,?)",
                            (time.time(), kind, ref_id, verdict, overall, json.dumps(scores or {}), judge_model, json.dumps(details, default=str)))
            return int(cur.lastrowid)

    def list_evals(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        q, args = "SELECT * FROM online_evals", []
        if kind:
            q += " WHERE kind=?"
            args.append(kind)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q, args).fetchall()]
        for r in rows:
            r["scores"] = json.loads(r["scores"] or "{}")
            r["details"] = json.loads(r["details"] or "{}")
        return rows

    def eval_summary(self) -> dict:
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT kind, COUNT(*) n, AVG(overall) mean_overall, "
                                               "AVG(CASE WHEN verdict='PASS' THEN 1.0 ELSE 0.0 END) pass_rate FROM online_evals "
                                               "GROUP BY kind").fetchall()]
        return {"by_kind": rows}
