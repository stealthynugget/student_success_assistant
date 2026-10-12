"""Centralised, environment-driven settings.

Design rules (fixes from the v4 audit):
* No import-time side effects: directories are created only by ``ensure_dirs()``.
* Every path is derived from ``SSA_HOME`` / ``SSA_DATA_DIR`` / ``SSA_ARTIFACT_DIR`` so tests
  and CI run against an isolated temporary tree and can never overwrite production
  artifacts (v4's test-suite wiped the production vector index).
* Secrets are read from the environment only. ``.env`` is loaded for local development
  but is excluded from git, Docker build context and release archives.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if os.getenv("SSA_DISABLE_DOTENV", "").lower() not in {"1", "true", "yes"}:
    try:  # optional convenience for local development only (tests disable it)
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:  # pragma: no cover - dotenv is optional
        pass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    home: Path
    data_dir: Path
    artifact_dir: Path

    # Snapshot / modelling. ``cutoff_fraction`` is the checkpoint this Settings view refers to;
    # ``checkpoints`` are all checkpoints built (share of the planned course length).
    cutoff_fraction: float = 0.40
    primary_checkpoint: float = 0.40
    checkpoints: tuple[float, ...] = (0.15, 0.25, 0.40, 0.60)
    include_sensitive_features: bool = False
    model_engine: str = "auto"  # auto | xgboost | sklearn
    random_state: int = 42
    priority_selection: str = "one_se_blend"  # one_se_blend | argmax (see ssa.models.prioritization)

    # Retrieval
    embedding_model_name: str = "all-MiniLM-L6-v2"
    enable_text_embeddings: bool = True
    workflow_engine: str = "langgraph"  # langgraph | python (reference orchestrator, same results)
    vector_store: str = "lancedb"  # lancedb (embedded vector database) | numpy; see ssa/retrieval/vector_store.py

    # LLM
    llm_enabled: bool = True  # master switch: when false no LLM request is ever sent, whatever keys are set
    llm_provider: str = "anthropic"  # anthropic | openai
    anthropic_api_key: str | None = None
    claude_model_fast: str = "claude-haiku-5-5"  # search understanding, outreach rewrites, note themes
    claude_model_main: str = "claude-sonnet-5-5"  # agents, copilot, precedent synthesis, briefing
    claude_model_judge: str = "claude-opus-5-5"  # judge, pairwise A/B, DeepEval wrapper
    claude_effort: tuple[str, str, str] = ("low", "medium", "high")  # (fast, main, judge) output_config.effort
    claude_max_tokens: int = 8000  # per response, thinking included (adaptive thinking is on for these models)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "claude-sonnet-5-5"  # generator for the active provider (main tier)
    judge_model: str = "claude-opus-5-5"  # judge for the active provider; always differs from every generator
    llm_timeout_s: float = 90.0
    llm_budget_usd: float | None = None  # hard cap on estimated spend per process (e.g. an evaluation run); None = no cap
    llm_max_retries: int = 2
    llm_cache_enabled: bool = True

    # API security
    api_key: str | None = None
    dev_mode: bool = False
    max_request_bytes: int = 512 * 1024
    rate_limit_per_minute: int = 120
    shared_state: str = "memory"  # memory | sqlite (several workers on one host); see ssa/api/shared.py
    allowed_origins: tuple[str, ...] = field(default_factory=tuple)
    trusted_proxy_count: int = 0
    admin_key: str | None = None
    online_eval_sample_rate: float = 0.0
    llm_prices: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_LLM_PRICES.items()})  # USD per MTok

    # ---- checkpoint views ----------------------------------------------
    def at(self, checkpoint: float | None) -> "Settings":
        if checkpoint is None:
            return self.at(self.primary_checkpoint) if self.cutoff_fraction != self.primary_checkpoint else self
        c = min(self.checkpoints, key=lambda x: abs(x - float(checkpoint)))
        return replace(self, cutoff_fraction=c)

    @property
    def tag(self) -> str:
        return f"c{int(round(self.cutoff_fraction * 100)):02d}"

    @property
    def is_primary(self) -> bool:
        return abs(self.cutoff_fraction - self.primary_checkpoint) < 1e-9

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifact_dir / "checkpoints" / self.tag

    @property
    def global_metrics_dir(self) -> Path:
        return self.artifact_dir / "metrics"

    @property
    def outcome_details_path(self) -> Path:
        return self.processed_dir / "outcome_details.csv.gz"

    @property
    def course_weekly_path(self) -> Path:
        return self.processed_dir / "course_weekly.npz"

    @property
    def app_db(self) -> Path:
        return self.artifact_dir / "app" / "app.sqlite3"

    @property
    def shared_state_db(self) -> Path:
        return self.artifact_dir / "app" / "shared_state.sqlite3"

    # ---- derived paths -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / f"features_{self.tag}.csv.gz"

    @property
    def weekly_path(self) -> Path:
        return self.processed_dir / f"weekly_{self.tag}.npz"

    @property
    def course_trends_path(self) -> Path:
        return self.artifact_dir / "analysis" / "engagement_trends_by_outcome.csv"

    @property
    def model_dir(self) -> Path:
        return self.checkpoint_dir / "models"

    @property
    def index_dir(self) -> Path:
        return self.checkpoint_dir / "index"

    @property
    def metrics_dir(self) -> Path:
        return self.checkpoint_dir / "metrics"

    @property
    def analysis_dir(self) -> Path:
        return self.artifact_dir / "analysis"

    @property
    def feedback_db(self) -> Path:
        return self.artifact_dir / "feedback" / "feedback.sqlite3"

    @property
    def llm_cache_db(self) -> Path:
        return self.artifact_dir / "cache" / "llm_cache.sqlite3"

    @property
    def action_stats_path(self) -> Path:
        return self.artifact_dir / "feedback" / "action_stats.json"

    @property
    def retrieval_weights_path(self) -> Path:
        return self.artifact_dir / "feedback" / "retrieval_weights.json"

    @property
    def improvement_plan_path(self) -> Path:
        return self.artifact_dir / "feedback" / "improvement_plan.json"

    @property
    def synthetic_marker(self) -> Path:
        return self.raw_dir / ".synthetic_fixture"

    @property
    def llm_api_key(self) -> str | None:
        return self.anthropic_api_key if self.llm_provider == "anthropic" else self.openai_api_key

    @property
    def llm_configured(self) -> bool:
        return self.llm_enabled and bool(self.llm_api_key)

    @property
    def fast_model(self) -> str:
        return self.claude_model_fast if self.llm_provider == "anthropic" else self.llm_model

    def model_for(self, feature: str) -> str:
        """Model per feature: judging on the judge model, light parsing/rewriting on the fast tier, the rest on main."""
        if feature.startswith(("judge", "deepeval")):
            return self.judge_model
        if feature in FAST_FEATURES:
            return self.fast_model
        return self.llm_model

    def effort_for(self, feature: str) -> str:
        fast, main, judge = self.claude_effort
        if feature.startswith("ragas"):  # short claim-verification prompts: the judge model at low effort
            return fast
        return judge if feature.startswith(("judge", "deepeval")) else fast if feature in FAST_FEATURES else main

    def ensure_dirs(self) -> None:
        for d in [
            self.raw_dir, self.processed_dir, self.model_dir, self.index_dir, self.metrics_dir,
            self.analysis_dir, self.feedback_db.parent, self.llm_cache_db.parent, self.global_metrics_dir, self.app_db.parent,
        ]:
            d.mkdir(parents=True, exist_ok=True)


FAST_FEATURES = frozenset({"search_understanding", "outreach_rewrite", "feedback_themes"})

# USD per million tokens, from https://platform.claude.com/docs/en/about-claude/pricing (fetched 2026-10-11).
# cache_write = 5-minute TTL writes. Haiku 5.5 is priced by prompt length: prompts over 100K tokens use the "long" row.
# Other models (e.g. for LLM_PROVIDER=openai) are priced only when added through LLM_PRICE_TABLE.
DEFAULT_LLM_PRICES: dict = {
    "claude-opus-5-5": {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50,
                         "long_context_threshold": 100_000,
                         "long": {"input": 0.50, "cache_write": 0.625, "cache_read": 0.05, "output": 2.50}},
}


def _price_table() -> dict:
    """Defaults plus LLM_PRICE_TABLE (a JSON object of the same shape) for other models or price changes."""
    table = {k: dict(v) for k, v in DEFAULT_LLM_PRICES.items()}
    raw = os.getenv("LLM_PRICE_TABLE", "").strip()
    if raw:
        import json
        table.update(json.loads(raw))
    return table


def _llm_models(provider: str) -> dict:
    if provider == "anthropic":
        fast = os.getenv("CLAUDE_MODEL_FAST") or "claude-haiku-5-5"
        main = os.getenv("CLAUDE_MODEL_MAIN") or "claude-sonnet-5-5"
        judge = os.getenv("CLAUDE_MODEL_JUDGE") or "claude-opus-5-5"
        generators = {fast, main}
        out = {"claude_model_fast": fast, "claude_model_main": main, "claude_model_judge": judge, "llm_model": main, "judge_model": judge}
    elif provider == "openai":
        main, judge = os.getenv("OPENAI_MODEL") or "gpt-4o-mini", os.getenv("JUDGE_MODEL") or "gpt-4o"
        generators = {main}
        out = {"llm_model": main, "judge_model": judge}
    else:
        raise ValueError(f"LLM_PROVIDER must be 'anthropic' or 'openai', not {provider!r}")
    if judge in generators:  # self-preference bias: the judge must never grade output from its own model
        raise ValueError(f"the judge model ({judge}) must differ from every generator model ({sorted(generators)})")
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    home = Path(os.getenv("SSA_HOME", str(PROJECT_ROOT))).resolve()
    cps = tuple(sorted(float(x) for x in os.getenv("CHECKPOINTS", "0.15,0.25,0.40,0.60").split(",") if x.strip()))
    primary = _float("CUTOFF_FRACTION", 0.40)
    if primary not in cps:
        cps = tuple(sorted(set(cps) | {primary}))
    origins = tuple(o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip() and o.strip() != "*")
    provider = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
    return Settings(
        home=home,
        data_dir=Path(os.getenv("SSA_DATA_DIR", str(home / "data"))).resolve(),
        artifact_dir=Path(os.getenv("SSA_ARTIFACT_DIR", str(home / "artifacts"))).resolve(),
        cutoff_fraction=primary,
        primary_checkpoint=primary,
        checkpoints=cps,
        include_sensitive_features=_bool("INCLUDE_SENSITIVE_FEATURES", False),
        model_engine=os.getenv("MODEL_ENGINE", "auto").lower(),
        priority_selection=(os.getenv("PRIORITY_SELECTION") or "one_se_blend").lower(),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        enable_text_embeddings=_bool("ENABLE_TEXT_EMBEDDINGS", True),
        vector_store=(os.getenv("VECTOR_STORE") or "lancedb").strip().lower(),
        workflow_engine=(os.getenv("WORKFLOW_ENGINE") or "langgraph").strip().lower(),
        llm_enabled=_bool("LLM_ENABLED", True),
        llm_provider=provider,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        claude_effort=(os.getenv("CLAUDE_EFFORT_FAST") or "low", os.getenv("CLAUDE_EFFORT_MAIN") or "medium",
                       os.getenv("CLAUDE_EFFORT_JUDGE") or "high"),
        claude_max_tokens=_int("CLAUDE_MAX_TOKENS", 8000),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        **_llm_models(provider),
        llm_timeout_s=_float("LLM_TIMEOUT_S", 90.0),
        llm_budget_usd=_float("LLM_BUDGET_USD", 0.0) or None,
        llm_max_retries=_int("LLM_MAX_RETRIES", 2),
        llm_cache_enabled=_bool("LLM_CACHE_ENABLED", True),
        api_key=os.getenv("STUDENT_ASSISTANT_API_KEY") or None,
        dev_mode=_bool("SSA_DEV_MODE", False),
        max_request_bytes=_int("MAX_REQUEST_BYTES", 512 * 1024),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 120),
        shared_state=(os.getenv("SHARED_STATE") or "memory").strip().lower(),
        allowed_origins=origins,
        trusted_proxy_count=_int("TRUSTED_PROXY_COUNT", 0),
        admin_key=os.getenv("STUDENT_ASSISTANT_ADMIN_KEY") or None,
        online_eval_sample_rate=_float("ONLINE_EVAL_SAMPLE_RATE", 0.0),
        llm_prices=_price_table(),
    )


def reset_settings() -> None:
    """Clear the cached settings (used by tests after changing environment variables)."""
    get_settings.cache_clear()
