# Uplift: Student Success Assistant (v6)

OULAD early-warning and student-support microservice: FastAPI backend (`ssa/`) + React console (`frontend/`).
Built in a claude.ai conversation; work now continues here. Read `docs/HANDOFF.md` first for history, what was
verified, what was not, and the task list.

## LLM provider: Claude
The LLM provider is Claude through the Anthropic API (`LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY` in `.env`), and LLM
calls are allowed. Models per tier: `CLAUDE_MODEL_FAST` (claude-haiku-5-5), `CLAUDE_MODEL_MAIN` (claude-sonnet-5-5),
`CLAUDE_MODEL_JUDGE` (claude-opus-5-5); the judge must differ from the generators. `LLM_ENABLED=false` blocks all LLM
traffic. Tests never call the API: use `FakeLLM` or the mocked SDK in `tests/test_llm.py`. Never invent or hand-edit LLM
evaluation results.

@docs/HANDOFF.md
@frontend/DESIGN.md

## Commands
- Backend deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.lock` (exact pins; Linux adds `--extra-index-url https://download.pytorch.org/whl/cpu`; re-pin with `make lock`)
- Data: the 7 OULAD CSVs go in `data/raw/` (never commit them)
- Full build without the LLM eval (all checkpoints, ~12 min): `python -m ssa.cli build && python -m ssa.cli report`
  (`make build` also runs `llm-eval`, which calls Claude when a key is set; `llm-eval --no-llm` measures computed paths only)  | single step: `python -m ssa.cli {prepare,train,index,evaluate,llm-eval,quality-gate,report}`
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
(course-clock concept, dark monochrome tokens, 2px radius, Geist/Geist Mono, functional colour: tier colours = tier,
indigo = AI-generated, yellow = current point). The landing page lives at `/`, the console at `/today` and below.
