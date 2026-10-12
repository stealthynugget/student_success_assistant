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
Work was split into two phases: Phase 1 without LLM access, Phase 2 once access was available (it is now, via Claude).

### Phase 1: now, no LLM API
The sandbox had no network, so FastAPI, XGBoost, sentence-transformers and Vite were never installed or run.
- [x] 1. `pip install -r requirements.txt`, then `python -m pytest`. `tests/test_api.py::test_fastapi_contract` has never
      run: fix whatever it finds in `ssa/api/app.py` (FastAPI + pydantic v2 were only checked against a stand-in).
      Also start `make api` and smoke-test a few routes with curl, including `/api/docs`.
      *Done 2026-10-11 (Windows, Python 3.12, FastAPI 0.143, Starlette 1.7, pydantic 2.14, pandas 3.0, numpy 2.5,
      scikit-learn 1.9, XGBoost 3.4): all 56 tests pass, including `test_fastapi_contract`; no code changes needed.
      uvicorn smoke test on a synthetic SSA_HOME: health, ready, docs, openapi (3.1, 35 paths), config, overview,
      students, metrics, copilot (rules) all 200; missing key 401, bad input 400, admin route 403; unbuilt home gives
      503 with a "build first" message. Notes: Starlette warns that its httpx-based TestClient is deprecated;
      `ruff check` reports 120 pre-existing style findings (96 are E501), left for a separate cleanup.*
- [x] 2. `python -m ssa.cli build && python -m ssa.cli report` with XGBoost installed (MODEL_ENGINE=auto picks it).
      Compare `docs/MODEL_CARD.md` with `docs/reference_run/` (sklearn engine): expect small shifts; explanations switch
      to exact TreeSHAP. Investigate anything that moves by more than ~0.02 ROC-AUC or breaks the quality gate.
      *Done 2026-10-11 (`build --no-dense`, XGBoost 3.4.1, 278 s; `llm-eval --no-llm` as in the reference; `report`).
      Official quality gate passes with the same three fairness warnings. Withdrawal ROC-AUC vs sklearn: c15 -0.007,
      c25 -0.002, c40 -0.005, c60 +0.015 (all within 0.02); in-scope counts identical. Drivers now report
      `method: tree_shap`. Other moves > 0.02: High-tier threshold 0.158 -> 0.141 at c40 (High tier now 24% of students,
      tiers still monotonic); outcome P(fail or withdraw) ECE 0.080 -> 0.056; behavioural/hybrid retrieval vote-AUC up
      0.03-0.05 (TreeSHAP-based feature weights; benchmark now uses 600 queries vs 500); prioritiser weights now
      p_withdraw 0.1 / p_poor 0.9. Fairness: highest_education EO gap 0.03 -> 0.14 is one student in the
      "Post Graduate Qualification" group (~13 positives; TPR 6/13 -> 5/13); the large groups stay within 0.04. The
      audit's `sufficient_sample` rule (>= 10 positives) is too loose for gap metrics (see item 8).
      Fixes: `report` crashed on Windows (cp1252) writing MODEL_CARD.md, now UTF-8; `llm-eval --no-llm` still ran the
      LLM judge whenever a key was configured, now disabled (regression test added); `copilot_golden` no longer fails
      on datasets without BBB/FFF in the latest presentation. `docs/reference_run/` is kept as the sklearn baseline.*
- [x] 3. Dense retrieval channel: `ENABLE_TEXT_EMBEDDINGS=true python -m ssa.cli index`, then
      `python -m ssa.cli evaluate`; check the benchmark's dense and hybrid rows and that fusion weights were re-tuned.
      *Done 2026-10-11 (all-MiniLM-L6-v2 on CPU, about 7 min per checkpoint). `dense_channel: built` at all 4 checkpoints;
      fusion re-tuned with dense in the grid, and the tuner picked dense = 0.0 everywhere: the best grid point with
      dense scored 0.007-0.028 lower on the validation objective. Hybrid is therefore unchanged (c40 vote-AUC poor
      0.835). Dense alone at c40: vote-AUC withdrawal 0.543 (random 0.554), poor outcome 0.775, outcome agreement@10
      0.725 (best of all modes), nDCG 0.724. Likely reasons: narratives are templated text from discretised
      descriptors (dense ends up a softer lexical channel), and corpus narratives end with "Final outcome: ..." while
      queries never do. The benchmark uses profile queries, so it does not measure the channel's real job
      (free-text case search); judge it in Phase 2 with the labelled search queries before changing the narratives.*
- [x] 4. `cd frontend && npm install && npm run build` (shipped dist/ was built with esbuild), then `docker compose build`
      and `docker compose up app`; open the console and check every page.
      *Done 2026-10-11. Vite 5.4 build passes (250 kB JS, 75 kB gzip); package-lock.json committed and the Dockerfile
      now uses `npm ci`. The image was 10.3 GB because the default Linux torch wheel ships CUDA; the Dockerfile now
      installs CPU torch first (3.57 GB). The container ran on the host-built artifacts with an override blanking
      OPENAI_API_KEY. Playwright drove all 9 pages plus a checkpoint switch at 1366 px and 820 px: no console errors,
      page errors, HTTP errors or horizontal overflow; briefing, computed support plan, case search, copilot and
      feedback-loop dry run all complete. Known issues: (a) the container resolves Python 3.11 / XGBoost 3.2 / numpy 2.4
      while the host build used 3.12 / 3.4.1 / 2.5; XGBoost warns when unpickling the host models, so either build
      artifacts in the container (`docker compose run --rm build`) or pin versions; (b) the MiniLM model is downloaded
      from the HF Hub on the first dense query (7 s cold start, fails offline) and could be baked into the image (item 9);
      (c) `npm audit`: esbuild/Vite dev-server advisory GHSA-67mh-4wv8-2f99 (dev only; fixed by the Vite upgrade in
      item 5); (d) console follow-ups for item 5: the What-if panel says "values fall outside what the model has seen"
      on an unmodified real record, and the ablation chart labels on Model quality render oversized.*
- [x] 5. **Console upgrade** (brief below). Works fully in computed mode; AI-labelled sections show their computed
      fallbacks until Phase 2.
      *Done 2026-10-12: Vite 8, React 19, TypeScript (all pages), React Router (real URLs; the API serves the console with an
      SPA fallback), TanStack Query, TanStack Table v9 + Virtual (Students, Who to contact: sort, filter, column choice,
      virtualised rows), Radix tabs, Motion (agent steps, clock marker, counting stats; reduced motion respected), cmdk
      palette (Ctrl/Cmd+K), Recharts at pixel size (fixes the oversized Model quality labels), sonner, lucide, self-hosted
      Atkinson Hyperlegible, light and dark themes from the DESIGN.md tokens. Student page is a case file (sticky header,
      tier, risk dial, scroll-spy outline). AI text is always labelled "Drafted by AI, review before use"; computed fallbacks
      are labelled "Computed". Lighthouse accessibility 100 on all 10 pages, light and dark, desktop and 820 px; Playwright
      sweep of every page in both themes at 1366 and 820 px: no console errors, failed requests or overflow. npm audit: 0.
      Docker web stage on Node 22. Note: a cold container's first Student page takes ~50 s (embedding model download and
      checkpoint recompute); see item 9.*
- [x] 6. Copilot streaming: add `/api/copilot/stream` (SSE: tool-call events, then answer tokens) and wire it into the
      console. Test the LLM branch with `FakeLLM` only; the rule-based branch can stream its answer in one event.
      *Done 2026-10-11/12: events tool_call, tool_result, text (unverified until done), text_reset, revision, done, error;
      Claude streaming via the SDK's messages.stream; incremental delivery verified under uvicorn and with a real answer.
      The console shows tool-call cards as tools run, streams the answer marked "checks pending", renders Markdown lists and
      tables safely, and links cited student IDs to their case files.*
- [x] 7. Prioritisation: at week 15 the learned weights put 1.0 on P(fail or withdraw). Try a finer/regularised
      simplex search if a blended ranking is wanted; keep the test comparison honest.
      Done: 0.05-step grid (53,130 weightings) plus a one-standard-error rule (`PRIORITY_SELECTION=one_se_blend`, default):
      bootstrap the validation objective and, among weightings within one SE of the best, take the most blended. The test
      comparison reports both the chosen blend (`learned_weights`) and the plain best-on-validation (`validation_argmax`).
      Test mean precision@10/20/30%, blend vs argmax: c15 0.756 vs 0.738, c25 0.837 vs 0.824, c40 0.860 vs 0.861,
      c60 0.889 vs 0.895; every checkpoint keeps about 0.6-0.75 on P(fail or withdraw) plus small weights on the
      actionable signals. Refit alone with `python -m ssa.cli train --only prioritization`. Quality gate passes.
- [x] 8. Fairness: region, IMD and disability equal-opportunity gaps 0.16-0.26. Investigate proxies and add a per-group
      threshold review to the fairness report (do not add sensitive attributes to the model).
      Done (`ssa/evaluation/model_analysis.py`, model card, Insights > Fairness): a group counts towards a gap only with >= 30
      withdrawn and >= 30 not withdrawn (was 10 withdrawn); per-group TPR with Wilson intervals; a permutation null (group
      labels shuffled among withdrawn students, overall and within module) with a verdict; proxy analysis (how well the
      model's inputs identify the group, and which inputs differ most); and a per-group cut-off review (shown, never applied).
      Findings at week 15: region (0.26) and IMD (0.19) are within chance at these group sizes (p 0.14 and 0.35). Gender
      (0.08) is beyond chance but explained by module mix (BBB). Disability (0.23, also 0.18 at week 22) is real and holds
      within modules: disabled students withdraw more (15.6% vs 10.8%), are under-predicted (12.3%), and have more of
      their withdrawals caught (70% vs 47%). Equalising TPR would flag 90 fewer disabled students; this is for the equity
      office to decide. The gate check is unchanged (warn above 0.15) and now carries the p-value and verdict.
- [x] 9. Deployment notes: jobs and rate limits are in-process; document or implement a queue + gateway limits for
      multi-worker scaling.
      Done: `SHARED_STATE=sqlite` (`ssa/api/shared.py`) shares job records, rate-limit buckets and recent analyses
      between the uvicorn workers on one host (`WEB_CONCURRENCY=N`); jobs whose worker died are reported as lost.
      Checked with 3 workers: 30/30 polls of a job found it (19/30 were 404 with `memory`), and one rate budget was
      enforced across workers. `RATE_LIMIT_PER_MINUTE=0` hands limits to the gateway; DEPLOYMENT.md has an nginx
      example, the multi-host options (sticky polls, Redis/Postgres job store, RQ/Celery) and the measured ~1 GB per
      worker. The image bakes the embedding model (`HF_HOME=/opt/hf`, `HF_HUB_OFFLINE=1`): cold first search at a
      checkpoint 11-15 s instead of ~49 s.
After each item: run `python -m pytest`, commit, and summarise what changed.

### Phase 2: LLM provider is Claude (access available 2026-10-11)
Step 2 of the 2026-10-11 plan replaced the OpenAI-compatible gateway with Claude: `LLMClient` has an Anthropic backend
(structured outputs per feature schema in `ssa/agents/llm_schemas.py`, tool use with tool_use/tool_result conversion at the
boundary and verbatim replay of thinking blocks, prompt caching for the copilot prefix, retries on 429/529/5xx/timeouts,
per-call usage and cost in the ledger). Models: haiku-5-5 (fast), sonnet-5-5 (main), opus-5-5 (judge). Mocked-SDK tests in
`tests/test_llm.py`; no real call verified yet when this was written.
- [x] 10. Smoke test (2026-10-11): one call per tier (haiku search understanding, sonnet briefing, opus judge) and a copilot
      question with two parallel tool calls; all OK, prompt cache hits on the copilot prefix, $0.09. Fixed: the number
      checker read "1,702" as 1 and 702. `llm-eval` estimated at ~$9.50 (dry run with a recording stand-in, calibrated on the
      smoke test) and run under `LLM_BUDGET_USD=10`: 592 calls, 0 errors, $6.09.
- [x] 11. Every section populated with real results; DeepEval 3.9.9 works through `GatewayModel` (enum renamed to
      `SingleTurnParams`; schemas pass through as structured outputs). Above 20% code-check rejection: precedent synthesis
      87.5% (numbers phrased unlike the evidence: "95% CI" labels, signs dropped, medians computed by the model), outreach
      33% (SMS over 320 characters), copilot robustness 25% (refusals repeating "depression", "age or gender"). Fixed in the
      prompts only (no guardrail loosened): synthesis 92% accepted, outreach 100%, robustness 88%. The judge was also given
      less evidence than the generators (no module context / action catalogue / briefing week); with the same evidence the
      analysis judge pass rate is 96% (was 46%). LLM search parser: off-by-one bounds fixed (F1 templated 0.84, paraphrased
      0.88 vs rules 0.97 / 0.74). Typed search with the LLM parser: precision@10 lexical 0.86, hybrid 0.77, dense 0.71 (rules
      parser: 0.52 / 0.62 / 0.63), so the text fusion weights (tuned with the rules parser) should be tuned per parser.
      Follow-up (2026-10-12): text weights are now tuned per parser (`search-benchmark --tune --llm`, stored as
      `text_weights_by_parser`; the API uses the weights of whichever parser interpreted the query). Tuning picks
      lexical-only for the LLM parser at every checkpoint and keeps dense 2.0 (0.25 at week 6) for the rules parser; LLM-parsed
      hybrid precision@10 on the reported queries 0.769 -> 0.862. Cost $0.01 (161 of 200 parser calls cached).
      Follow-up (2026-10-12): text weights are now tuned per parser (`search-benchmark --tune --llm`, stored as
      `text_weights_by_parser`; the API uses the weights of whichever parser interpreted the query). Tuning picks
      lexical-only for the LLM parser at every checkpoint and keeps dense 2.0 (0.25 at week 6) for the rules parser; LLM-parsed
      hybrid precision@10 on the reported queries 0.769 -> 0.862. Cost $0.01 (161 of 200 parser calls cached).
      Follow-up (2026-10-12): text weights are now tuned per parser (`search-benchmark --tune --llm`, stored as
      `text_weights_by_parser`; the API uses the weights of whichever parser interpreted the query). Tuning picks
      lexical-only for the LLM parser at every checkpoint and keeps dense 2.0 (0.25 at week 6) for the rules parser; LLM-parsed
      hybrid precision@10 on the reported queries 0.769 -> 0.862. Cost $0.01 (161 of 200 parser calls cached).
      Follow-up (2026-10-12): text weights are now tuned per parser (`search-benchmark --tune --llm`, stored as
      `text_weights_by_parser`; the API uses the weights of whichever parser interpreted the query). Tuning picks
      lexical-only for the LLM parser at every checkpoint and keeps dense 2.0 (0.25 at week 6) for the rules parser; LLM-parsed
      hybrid precision@10 on the reported queries 0.769 -> 0.862. Cost $0.01 (161 of 200 parser calls cached).
- [x] 12. `ONLINE_EVAL_SAMPLE_RATE=0.1` through the API (29 analyses, 8 copilot questions): live reviews hold sampled copilot
      (2, both PASS) and analysis (1, PASS 9.3) judgements; usage and cost view populated per feature and serving model.
- [x] 13. `python -m ssa.cli report` regenerated the model card (AI section now lists synthesis, briefing, outreach, DeepEval,
      models and spend) and the deck. Total LLM spend for steps 3-4: $10.26.
      Re-run (2026-10-12) after the risk-agent and briefing prompt fixes: $5.94, 0 errors. Analysis judge pass 96% (as
      before); all 19 intervene plans accepted without revision (the 5 monitor cases are deterministic by design); A/B AI
      preference 95% -> 89% (ties 5% -> 11%); DeepEval actionable pass 71% -> 75%. Briefings 4/5 accepted: BBB said "the
      top 12 priority students", a true count missing from the evidence, so `top_priorities_count` was added to the
      briefing evidence (the judge sees it too; no check changed) and the eval now records briefing rejections. Second
      run, $0.20 (603 of 613 calls cached): briefings 5/5 accepted, judge pass 100%, usefulness 8.2. Total for the
      re-runs: $6.14.

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

## Later additions (2026-10-12)
- **Vector database.** The case-index vectors (behavioural features and MiniLM narrative embeddings) moved from `.npy`
  arrays to LanceDB (`ssa/retrieval/vector_store.py`; one embedded table per checkpoint with the filterable metadata and
  bitmap/B-tree scalar indexes). Filters, earlier-presentations-only and self-exclusion run as a pre-filter in the
  database; kNN is exact. Checked against the NumPy scan on 1,800 sampled searches over the four checkpoints: identical
  top 10 and outcome mix in all of them (ties to 1e-5 now break by index position in both backends; the stored c40
  benchmark moved by at most 5e-4). Search latency about 40 ms vs 31 ms. `VECTOR_STORE=numpy` keeps arrays;
  `python -m ssa.cli index --migrate-vectors` converts an existing index without re-encoding.
- **Rubric gap work (2026-10-12).** All results are in the model card (regenerated by `report`):
  - *RAGAS* (`scripts/ragas_eval.py`, own environment: RAGAS's LangChain stack forced upgrades that broke the pinned
    environment) on precedent synthesis and copilot answers, evaluator = judge model through `LLMClient`. First run scored
    synthesis low because the student's own tier and drivers were missing from the context and the advice section was
    scored as factual claims; the case framing was corrected (context = exactly what the generator saw; advice scored
    by the judge and DeepEval), no guardrail changed. Copilot "unsupported" statements are explanations of metrics,
    not invented data. Spend $7.14 including two pilots.
  - *LLM tier comparison* (`llm-compare`, $1.37): Haiku parses search queries as well as Sonnet and Opus at a fraction of
    the cost; Sonnet gives the best precedent syntheses; Opus rows are self-judged and marked so.
  - *Retrieval sensitivity* (`retrieval-study`): top-K, pool and RRF constant; 6 stories / 50-neighbour outcome mix kept.
  - *LangGraph* workflow (default; `WORKFLOW_ENGINE=python` keeps the reference orchestrator, parity-tested).
  - *Human review* tab (blind ratings → `human_labels.jsonl` → Cohen's kappa against the judge). No labels yet: advisors
    must rate; labels are never generated.
  - Coverage measured with pytest-cov (`python -m pytest --cov`); judge tests added.
  - Docs: `SECURITY.md`, `DECISIONS.md`, `AGENT_GRAPH.md`, README, architecture diagram (rendered with Graphviz in a
    container), PRD traceability, presentation script; deck gains model-choice and evaluation slides.
  - *Embedding evaluation* (`embedding-eval`, c40, CPU): e5-small-v2 and gte-small beat the production MiniLM on the dense
    channel (typed search and precedents) at the same query latency; e5-small-v2 has the best precedent quality. With the
    LLM parser, hybrid typed search is lexical-only either way. Not switched yet: it means re-encoding all four indexes
    (~10 min each) and adding e5's "query: "/"passage: " prefixes. gte-small's encoding speed was measured while other jobs
    shared the CPU, so its docs/s figure is pessimistic.
  - *Embedding switch (2026-10-12).* Production dense channel is now intfloat/e5-small-v2 (`EMBEDDING_MODEL_NAME`, with
    "query: "/"passage: " prefixes). The index manifest records model and prefixes and queries always use them, so an older
    index keeps working with its own model. All four indexes re-encoded; fusion and per-parser text weights re-tuned.
    Typed search (rules parser) hybrid precision@10 0.622 -> 0.737; LLM parser unchanged (lexical-only); profile hybrid at
    week 15 unchanged (dense weight still 0), so the AI evaluation stays valid; week 6 now gives dense weight 0.5. Docker
    bakes e5 (`EMBEDDING_MODEL` build arg); verified offline in the container. `evaluate` now includes the LLM parser in
    the typed-search benchmark when an LLM is configured (it used to overwrite those rows with rules-only results).
  - *Hugging Face cache moved* from E: (removable drive) to `D:\huggingface\hub` (user-level `HF_HOME`; the machine-level
    variable still says E: and needs an administrator to change).
