# PRD traceability

| PRD requirement | Implementation | Evidence |
|---|---|---|
| **Q1** Similar students who struggled/withdrew? | Hybrid historical-case retrieval with outcome mix (Wilson CI) of the 50 nearest earlier-cohort students | `ssa/retrieval/hybrid.py`, `/api/retrieve-similar`, `retrieval_benchmark.json` |
| **Q2** Factors associated with poor performance/withdrawal | Factor analysis (quintile rates with CIs, rank correlation), global and per-student TreeSHAP drivers, ablations | `ssa/evaluation/model_analysis.py`, `/api/insights/factors`, `factor_analysis.json` |
| **Q3** Who to prioritise | Learned capacity-aware prioritiser with reason codes | `ssa/models/prioritization.py`, `/api/prioritize` |
| Basic RAG for historical similarity | Index of in-scope historical enrolments stored in a LanceDB vector store; retrieved cases (6 stories, outcome mix of 50) ground the risk agent, plan and precedent synthesis; top-K and fusion sensitivity measured | `ssa/retrieval`, `ssa/retrieval/vector_store.py`, `retrieval_sensitivity.json` |
| Hybrid keyword + semantic search | BM25 over descriptors + behavioural kNN + dense narrative embeddings (model chosen by comparing five), weighted RRF tuned on validation and per query parser | `ssa/retrieval/hybrid.py`, `benchmark.py`, `embedding_eval.json` |
| Performance & risk classification agent | Performance and risk agents over calibrated models; four-class outcome model; tiers | `ssa/agents/workflow.py`, `ssa/models/` |
| Metadata filtering (course, region, education, age band, credits, previous attempts) | Exact / list / min-max / ne filters validated against an allow-list, pushed down to the vector database as a pre-filter (scalar indexes); plus presentation and outcome | `RetrievalFilters`, `filter_sql()`, `tests/test_vector_store.py` |
| Basic support recommendations | Eligibility-gated action catalogue, deterministic planner | `ssa/agents/evidence.py` |
| Input validation guardrails | Enums, real-data ranges, cross-field checks, injection screen, size limit | `ssa/api/schemas.py`, `ssa/agents/guardrails.py` |
| Prioritisation by assessment, activity, risk | Signals: P(withdraw), P(fail/withdraw), academic gap, engagement gap, missed work, decline | `ssa/models/prioritization.py` |
| API endpoints | FastAPI service with ~35 routes, OpenAPI docs, API/admin keys, rate limit, background jobs | `ssa/api/app.py`, `docs/API.md` |
| DeepEval | Faithfulness vs evidence, answer relevancy, three GEval criteria; aggregated offline alongside per-feature evaluations (search, copilot, synthesis, briefing, outreach, adversarial prompts, A/B); RAGAS faithfulness, groundedness and context relevance for the RAG features | `ssa/evaluation/`, `python -m ssa.cli llm-eval`, `scripts/ragas_eval.py` |
| Performance & withdrawal prediction models | Temporal, calibrated GBMs with baselines and population audit | `ssa/models/risk.py`, `outcome.py`, `MODEL_CARD.md` |
| LLM-as-Judge | Anchored rubrics for plans and generated text, separate judge model (Claude Opus), code-computed verdict, order-swapped pairwise A/B, on-demand and sampled online reviews, blind human review in the console with Cohen's kappa | `ssa/evaluation/judge.py`, `tests/test_judge.py`, `/api/evaluations/*` |
| Multi-agent workflow | LangGraph state graph: Performance → Risk → Router → Intervention ⇄ Critic / deterministic planner, with trace and live steps | `ssa/agents/graph.py`, `docs/AGENT_GRAPH.md` |
| Engagement & performance trend analysis | Student weekly series vs cohort median; course trajectories by outcome with divergence weeks; score slope | `/api/trends/student`, `/api/trends/course`, `trend_analysis.json` |
| Explainable analysis | TreeSHAP drivers, factor analysis, ablations, observations with IDs | `risk.explain()`, `factor_analysis.json` |
| Feedback loop | Registry + feedback + learners + approval gate + monitors | `ssa/feedback/` |
| Front-end | React console covering every API capability: course clock, this week, students, student workspace (plan, outreach, precedents, what-if, feedback), who to contact, case finder, copilot, insights, model quality, feedback loop, settings | `frontend/src`, `frontend/DESIGN.md`, `docs/screenshots` |
| Deliverable: architecture diagram | Graphviz source + SVG/PNG | `docs/architecture.*`, `docs/ARCHITECTURE.md` |
| Deliverable: design | `docs/DESIGN.md`, `docs/DECISIONS.md`, `docs/SECURITY.md`, `docs/MODEL_CARD.md` | |
| Deliverable: executable microservice | Package, CLI, FastAPI service serving the React console, Docker/compose, tests | repository |
| Deliverable: 10-minute presentation | Generated deck + script | `docs/Student_Success_Assistant_10min.pptx`, `docs/PRESENTATION_SCRIPT.md` |

## Beyond the PRD

| Addition | Where |
|---|---|
| Early warning at four points in the course, with each student's risk trajectory | checkpoints, `/api/students/timeline`, course clock |
| Complete historical records: what happened to similar students after this point | `precedent_stories`, case finder, student page |
| Assessment analytics: which deadlines signal later failure | `/api/insights/assessments` |
| Copilot over the whole system, with tool traces and grounding checks | `ssa/agents/copilot.py`, Copilot page |
| Outreach in several channels, tones and languages with safety checks | `/api/outreach/variants` |
| LLM usage, latency and cost per feature | `/api/llm/usage`, Model quality page |
| LLM tier comparison (cost vs capability) and embedding-model comparison | `python -m ssa.cli llm-compare`, `embedding-eval`; model card |
| Fairness with permutation tests, proxy analysis and a per-group threshold review | `ssa/evaluation/model_analysis.py`; model card |
| Multi-worker shared state, streamed copilot, vector database | `ssa/api/shared.py`, `/api/copilot/stream`, LanceDB |
