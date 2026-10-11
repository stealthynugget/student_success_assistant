# Student Success Assistant (v6)

OULAD early-warning and student-support microservice: FastAPI backend (`ssa/`) + React console (`frontend/`).
Built in a claude.ai conversation; work now continues here. Read `docs/HANDOFF.md` first for history, what was
verified, what was not, and the task list.

## Current phase: no LLM API access
LLM API access is blocked until later today. Until the user says access is available:
- Keep `OPENAI_API_KEY` unset. Every feature must keep working through its deterministic fallback.
- Do Phase 1 tasks in `docs/HANDOFF.md` only. Do not run `python -m ssa.cli llm-eval` with a key, call any LLM
  endpoint, or try to work around the restriction.
- New or changed LLM code paths may be written, but test them only with `FakeLLM` (see `tests/test_genai.py`) and
  record them under Phase 2 for real verification.
- Never invent or hand-edit LLM evaluation results; "not run" is the correct state for now.

@docs/HANDOFF.md
@frontend/DESIGN.md

## Commands
- Backend deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.lock` (exact pins; Linux adds `--extra-index-url https://download.pytorch.org/whl/cpu`; re-pin with `make lock`)
- Data: the 7 OULAD CSVs go in `data/raw/` (never commit them)
- Full build without the LLM eval (all checkpoints, ~12 min): `python -m ssa.cli build && python -m ssa.cli report`
  (`make build` also runs `llm-eval`; without a key it only measures computed paths, which is fine)  | single step: `python -m ssa.cli {prepare,train,index,evaluate,llm-eval,quality-gate,report}`
- Tests: `python -m pytest` (isolated temp SSA_HOME + synthetic fixture; no network, no LLM)
- API: `make api` (uvicorn on :8000, docs at /api/docs, serves frontend/dist at /) | dev: `make api-dev`
- Frontend: `cd frontend && npm install && npm run dev` (Vite :5173, proxies /api to :8000) | `npm run build`

## Architecture in one breath
`ssa/data/pipeline.py` builds per-checkpoint snapshot features (CHECKPOINTS 0.15/0.25/0.40/0.60) plus complete
historical records -> `ssa/models/` (withdrawal risk, outcome, prioritiser) -> `ssa/retrieval/hybrid.py` (case index)
-> `ssa/service.py` (one Bundle per checkpoint) -> `ssa/api/handlers.py` (framework-independent) -> `ssa/api/app.py`
(FastAPI). LLM features live in `ssa/agents/` (workflow, copilot, genai) and always go through `LLMClient`.

## Rules that must not be broken
- Only students still enrolled at a checkpoint (`in_scope == 1`) are modelled, evaluated, retrieved or ranked.
  `date_unregistration` defines scope only; it is never a feature. Never evaluate on the full population.
- Sensitive attributes (gender, region, imd_band, age_band, highest_education, disability) are never model inputs,
  never shown as drivers, never sent to the LLM. They are used only for the fairness audit and retrieval filters.
- Historical retrieval only searches presentations earlier than the query's presentation; never the student's own rows.
- Every LLM output is validated in code (IDs, numeric grounding, protected/punitive/clinical/stigma terms) and has a
  deterministic fallback. Do not add an LLM feature without both, plus an offline evaluation in `ssa/evaluation/offline_eval.py`.
- Numbers in docs and the deck come from artifacts via `python -m ssa.cli report`. Never hand-type metrics.
- Tests and builds must use `SSA_HOME`-scoped paths; nothing may write to production artifacts from tests.
- Never commit `.env`, data, or `artifacts/`. Rotate any key that was ever shared.

## Style
Python: ruff (line length 140), type hints, no import-time side effects. Frontend: follow `frontend/DESIGN.md`
(course-clock concept, tokens, functional colour: tier colours = tier, indigo = AI-generated, yellow = current point).
