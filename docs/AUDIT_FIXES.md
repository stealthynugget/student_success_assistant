# v4 audit → v5 fixes

Each row is a finding from the v4 review and what v5 does instead. "Verified by" points to the test, metric artifact or
quality-gate check that proves it.

## Severity 1 — blockers

| # | v4 problem | v5 fix | Verified by |
|---|---|---|---|
| 1 | Live `OPENAI_API_KEY` in `.env` inside the archive; `COPY . .` with no `.dockerignore` baked it into images | `.env` never shipped; `.gitignore` + `.dockerignore` exclude it; Dockerfile copies only code; secret scan in quality gate and tests; `.env.example` only. **Rotate the old key.** | `test_release.py`, gate `no_secrets_in_source` |
| 2 | Shipped Chroma index held 2 test documents; `test_retriever_fallback` deleted the production collection on every `pytest`/quality-gate run | All paths derive from `SSA_HOME`; tests run in a temp home; index built in a temp dir and swapped atomically with a versioned manifest + checksum; gate checks `n_docs == in-scope rows` | `test_index_is_isolated_complete_and_versioned`, gate `index_complete` |
| 3 | 72.7 % of withdrawn students had already unregistered by the snapshot; AUC 0.90 was mostly "no activity after leaving" | `in_scope` scoping (unregistered on/before snapshot excluded from training, calibration, test, retrieval, ranking); API returns 409 for them; population audit reports both numbers | `test_scope_flags...`, `withdrawal_model.json → population_audit`, gate `leakage_scope_applied` |
| 4 | Retrieval ≈ random: MiniLM/BM25 over numeric strings; nDCG normalised by the list itself (random scored 0.854) | Supervised behavioural kNN + BM25 over cohort-relative descriptors + optional dense narratives; weighted RRF tuned on validation; benchmark on future cohort vs historical corpus with neighbour-vote AUC, outcome agreement, global-ideal nDCG and a random baseline | `retrieval_benchmark.json`, gate `retrieval_beats_random_*` |
| 5 | Front-end fabricated 15+ features (`last_activity_day=0`, `weeks_active=1`, `min=max=avg` …) | Console analyses real snapshots fetched by ID; what-if edits a real record; supplied partial profiles are imputed from module medians with `imputed_fields`, OOD flags and a confidence label | `test_partial_profile_is_imputed_and_reported`, `frontend/app.py` |

## Severity 2 — major

| # | v4 problem | v5 fix | Verified by |
|---|---|---|---|
| 6 | Medium threshold static 0.35 vs learned High 0.37 → tiers collapsed | Both thresholds learned on out-of-fold historical predictions (Medium = 80 % recall watch list, High = F1-optimal above it); tier withdrawal rates reported on test | `test_learned_thresholds...`, gate `risk_tiers_monotonic`, `thresholds_separated` |
| 7 | API rejected 739 real rows (negative `last_activity_day`); silent defaults (score 50, clicks 0); no cross-field checks; free-text categoricals into prompts | `days_since_last_activity` (bounded, real-data ranges); all features optional with reported imputation; OULAD enums; cross-field validators; injection screen on text queries; sensitive fields never sent to LLM | `test_validation_rejects_bad_input`, `test_injection_like_query_rejected` |
| 8 | 25–40 synchronous LLM calls per analysis (agents + 4 judges + DeepEval) | ≤ 3 agent calls (+1 revision) per request, response cache, timeouts/backoff; judge only on request (`judge=true`, 1 call); DeepEval offline only | `ssa/agents/llm.py`, `ssa/evaluation/offline_eval.py` |
| 9 | DeepEval faithfulness vs *other students'* summaries; `str(result)`; judge = generator; LLM decided its own verdict; n=1 | Faithfulness vs the actual evidence; per-metric scores/reasons aggregated with CIs; gateway-routed judge model distinct from generator; verdict computed in code with deterministic checks; 24-case stratified golden set; judge–human kappa when labels exist | `llm_evaluation.json`, `ssa/evaluation/judge.py` |
| 10 | Sensitive attributes were model features while agents were told not to use them; misleading `protected_attributes_are_not_model_features` flag | Excluded from models by default; never surfaced as drivers; stripped from prompts; ablation quantifies cost; fairness audit (TPR gap, selection ratio, calibration gap) with gate warnings | `test_sensitive_attributes_never_model_inputs_or_drivers`, `test_sensitive_attributes_not_in_prompts`, `fairness.json` |
| 11 | Class-weighted outcome model worse than its baseline; capped `top_k_recall` | Weighting chosen on validation by macro-F1; prior-corrected calibrated probabilities; majority + multinomial-logistic baselines; capacity metrics report precision, recall, lift and max possible recall | `outcome_model.json`, `test_outcome_weighting_selected_on_validation` |
| 12 | "Feedback loop" appended JSONL and printed generic advice | SQLite registry; feedback must reference a real recommendation and its actions/cases; learners for action ranking, fusion weights, recalibration; drift/retrain/prompt monitors; human `--apply` with approver stamp; service reads applied artifacts | `test_feedback.py` |
| 13 | Hand-typed prioritisation weights, double counting, absolute clicks, client had to send 30 engineered features | Cohort-relative rank-normalised signals; weights learned for precision at capacity on validation; compared with v4 formula, single signals and random on test; ranks whole presentations from the feature store | `prioritization.json`, gate `prioritization_*`, `test_rank_presentation_with_capacity` |

## Severity 3 — hygiene

| v4 problem | v5 fix |
|---|---|
| Flask dev server as root, no healthcheck | FastAPI on uvicorn, non-root user, healthcheck, multi-stage Docker build including the console |
| Unbounded per-process rate limiter, `remote_addr` behind proxy, auth off, CORS `*`, `str(e)` to clients | token bucket with LRU bound, trusted-proxy aware IP, auth required, explicit CORS allow-list, generic errors + request id, security headers |
| Tests overwrote production artifacts; tautological tests; readiness test accepted 500 | isolated temp `SSA_HOME`; 47 behavioural tests (leakage injection, guardrails, critic, API contract, feedback integrity) |
| Quality gate checked file existence only | thresholds on AUC, calibration, baselines, tier monotonicity, retrieval vs random, prioritisation vs v4, plan validity, secrets |
| 750 MB archive with raw data, caches, Windows paths, orphaned index segments | source-only release; data/artifacts built locally |
| "Multi-agent" = 3 sequential calls; priority from substring "high" | stateful workflow with router, catalogue-gated intervention, code critic with one revision, deterministic fallback, trace; priority from calibrated policy |
| ASCII architecture diagram; deck hard-coded 0.90 | Graphviz diagram (dot/svg/png); model card and deck generated from metrics |
| Dead code (`fillna(50)`), circular unused `poor_outcome` label | missing scores kept as NaN + flag; labels derived only from `final_result` |
