# Handoff from the claude.ai conversation

## History (short)
1. **Audit of v4** (rated 4.5/10): live API key in the archive, shipped vector index held 2 test docs (tests wiped it),
   0.90 AUC came from scoring students who had already left, retrieval at random level, front-end fabricated features.
2. **v5**: rebuild fixing everything in `docs/AUDIT_FIXES.md`; honest withdrawal ROC-AUC ~0.73 at week 15.
3. **v6** (this code): FastAPI, React console, complete-dataset features, 4 checkpoints, 8 LLM features with per-feature
   offline evaluation, usage/cost ledger, sampled online judging.

## Verified in the sandbox
- Full build on official OULAD at all checkpoints; quality gate passed (fairness warnings for region/IMD/disability).
- 55 pytest tests pass. The console was driven with Playwright against the real handlers; every page rendered with no
  runtime errors after two fixes (index-loading race, slow ranking).

## Work plan
LLM API access is blocked until later today, so work is split into two phases. Phase 1 needs no LLM API (package
downloads from PyPI, npm and Hugging Face are fine). Phase 2 starts only when the user confirms access.

### Phase 1: now, no LLM API
The sandbox had no network, so FastAPI, XGBoost, sentence-transformers and Vite were never installed or run.
- [ ] 1. `pip install -r requirements.txt`, then `python -m pytest`. `tests/test_api.py::test_fastapi_contract` has never
      run: fix whatever it finds in `ssa/api/app.py` (FastAPI + pydantic v2 were only checked against a stand-in).
      Also start `make api` and smoke-test a few routes with curl, including `/api/docs`.
- [ ] 2. `python -m ssa.cli build && python -m ssa.cli report` with XGBoost installed (MODEL_ENGINE=auto picks it).
      Compare `docs/MODEL_CARD.md` with `docs/reference_run/` (sklearn engine): expect small shifts; explanations switch
      to exact TreeSHAP. Investigate anything that moves by more than ~0.02 ROC-AUC or breaks the quality gate.
- [ ] 3. Dense retrieval channel: `ENABLE_TEXT_EMBEDDINGS=true python -m ssa.cli index`, then
      `python -m ssa.cli evaluate`; check the benchmark's dense and hybrid rows and that fusion weights were re-tuned.
- [ ] 4. `cd frontend && npm install && npm run build` (shipped dist/ was built with esbuild), then `docker compose build`
      and `docker compose up app`; open the console and check every page.
- [ ] 5. **Console upgrade** (brief below). Works fully in computed mode; AI-labelled sections show their computed
      fallbacks until Phase 2.
- [ ] 6. Copilot streaming: add `/api/copilot/stream` (SSE: tool-call events, then answer tokens) and wire it into the
      console. Test the LLM branch with `FakeLLM` only; the rule-based branch can stream its answer in one event.
- [ ] 7. Prioritisation: at week 15 the learned weights put 1.0 on P(fail or withdraw). Try a finer/regularised
      simplex search if a blended ranking is wanted; keep the test comparison honest.
- [ ] 8. Fairness: region, IMD and disability equal-opportunity gaps 0.16-0.26. Investigate proxies and add a per-group
      threshold review to the fairness report (do not add sensitive attributes to the model).
- [ ] 9. Deployment notes: jobs and rate limits are in-process; document or implement a queue + gateway limits for
      multi-worker scaling.
After each item: run `python -m pytest`, commit, and summarise what changed.

### Phase 2: when LLM API access is available
- [ ] 10. Set `OPENAI_API_KEY`, `OPENAI_MODEL` and a different `JUDGE_MODEL` in `.env`; run `python -m ssa.cli llm-eval`.
- [ ] 11. Check end to end: copilot function calling (including the new streaming route), JSON-mode prompts for every
      generative feature, DeepEval's custom model wrapper (`ssa/evaluation/deepeval_suite.py`, `GatewayModel`) against
      the installed DeepEval version, and that the judge, A/B, search and copilot sections of the evaluation populate.
- [ ] 12. Turn on `ONLINE_EVAL_SAMPLE_RATE`, use the console for a while, and check Model quality > Live reviews and
      AI usage and cost.
- [ ] 13. `python -m ssa.cli report` to regenerate the model card and deck with real AI results; review prompt wording
      for any feature whose LLM output is often rejected by the code checks.

## Console upgrade brief
Keep: the course-clock concept, the functional colour rules and every capability in `frontend/DESIGN.md`.
Network is available now, so these are allowed (pick what serves the product, not all of them):
- TypeScript + Vite; React Router (replace the hash router); TanStack Query for fetching/caching/polling jobs.
- Radix UI primitives (or shadcn/ui on Tailwind) for accessible tabs, dialogs, popovers, tooltips, sliders, selects.
- Motion (framer-motion) for meaningful motion only: the agent pipeline progressing step by step, the course clock
  marker sliding between checkpoints, numbers counting when the checkpoint changes, page transitions. Respect reduced motion.
- Charts: visx or Recharts (or ECharts) to replace the hand-rolled SVG where interaction helps: hover crosshairs on
  weekly activity, brushing the full-course trajectories of precedents, linked highlighting between the precedent list
  and its trajectory line, animated tier histogram on checkpoint change.
- TanStack Table with virtualisation for the students and ranking tables (sort, filter, column visibility).
- cmdk command palette (Ctrl/Cmd+K): jump to a student by ID, switch module/presentation/week, run actions.
- sonner toasts, lucide-react icons, a dark theme built from the same tokens.
- Student page as a "case file": sticky header with tier and risk dial, sections as a scroll-spy outline.
- Copilot: streamed answers, collapsible tool-call cards, cited IDs as links to students/cases.
Quality bar: no generic dashboard look (no card grid with identical icon tiles, no purple gradients, no uppercase
eyebrows). Lighthouse accessibility >= 95, keyboard reachable, works at 820px width. Verify with Playwright screenshots
of every page in light and dark mode, and keep `npm run build` passing.

## Prompts for Claude Code
Phase 1 (now):
> Read CLAUDE.md and docs/HANDOFF.md. We have no LLM API access until later today, so work only on Phase 1. Set up the
> Python venv and frontend dependencies, then do Phase 1 items 1-4 in order, fixing whatever breaks. Run the tests and
> commit after each item. Stop and report results before starting item 5, the console upgrade.

Phase 2 (once access is back):
> LLM API access is now available and the keys are in .env. Read docs/HANDOFF.md and do Phase 2 items 10-13 in order.
> Report the evaluation results and any prompt or code changes before regenerating the report.
