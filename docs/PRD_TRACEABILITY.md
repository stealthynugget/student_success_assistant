# PRD traceability

| PRD requirement | Implementation | Evidence |
|---|---|---|
| **Q1** Similar students who struggled/withdrew? | Hybrid historical-case retrieval with outcome mix (Wilson CI) of the 50 nearest earlier-cohort students | `ssa/retrieval/hybrid.py`, `/api/retrieve-similar`, `retrieval_benchmark.json` |
| **Q2** Factors associated with poor performance/withdrawal | Factor analysis (quintile rates with CIs, rank correlation), global and per-student TreeSHAP drivers, ablations | `ssa/evaluation/model_analysis.py`, `/api/insights/factors`, `factor_analysis.json` |
| **Q3** Who to prioritise | Learned capacity-aware prioritiser with reason codes | `ssa/models/prioritization.py`, `/api/prioritize` |
| Basic RAG for historical similarity | Index of in-scope historical enrolments; retrieved cases ground the risk agent and plan | `ssa/retrieval`, `ssa/agents/workflow.py` |
| Hybrid keyword + semantic search | BM25 over descriptors + behavioural kNN + optional dense narratives, weighted RRF tuned on validation | `ssa/retrieval/hybrid.py`, `benchmark.py` |
| Performance & risk classification agent | Performance and risk agents over calibrated models; four-class outcome model; tiers | `ssa/agents/workflow.py`, `ssa/models/` |
| Metadata filtering (course, region, education, age band, credits, previous attempts) | Exact / list / min-max / ne filters validated against an allow-list; plus presentation and outcome | `RetrievalFilters`, `_allowed()` |
| Basic support recommendations | Eligibility-gated action catalogue, deterministic planner | `ssa/agents/evidence.py` |
| Input validation guardrails | Enums, real-data ranges, cross-field checks, injection screen, size limit | `ssa/api/schemas.py`, `ssa/agents/guardrails.py` |
| Prioritisation by assessment, activity, risk | Signals: P(withdraw), P(fail/withdraw), academic gap, engagement gap, missed work, decline | `ssa/models/prioritization.py` |
| API endpoints | FastAPI service with ~35 routes, OpenAPI docs, API/admin keys, rate limit, background jobs | `ssa/api/app.py`, `docs/API.md` |
| DeepEval | Faithfulness vs evidence, answer relevancy, three GEval criteria; aggregated offline alongside per-feature evaluations (search, copilot, synthesis, briefing, outreach, adversarial prompts, A/B) | `ssa/evaluation/`, `python -m ssa.cli llm-eval` |
| Performance & withdrawal prediction models | Temporal, calibrated GBMs with baselines and population audit | `ssa/models/risk.py`, `outcome.py`, `MODEL_CARD.md` |
| LLM-as-Judge | Anchored rubrics for plans and generated text, separate judge model, code-computed verdict, order-swapped pairwise A/B, on-demand and sampled online reviews, human agreement | `ssa/evaluation/judge.py`, `/api/evaluations/*` |
| Multi-agent workflow | Performance → Risk → Router → Intervention ⇄ Critic / Monitoring with trace | `ssa/agents/workflow.py` |
| Engagement & performance trend analysis | Student weekly series vs cohort median; course trajectories by outcome with divergence weeks; score slope | `/api/trends/student`, `/api/trends/course`, `trend_analysis.json` |
| Explainable analysis | TreeSHAP drivers, factor analysis, ablations, observations with IDs | `risk.explain()`, `factor_analysis.json` |
| Feedback loop | Registry + feedback + learners + approval gate + monitors | `ssa/feedback/` |
| Front-end | React console covering every API capability: course clock, this week, students, student workspace (plan, outreach, precedents, what-if, feedback), who to contact, case finder, copilot, insights, model quality, feedback loop, settings | `frontend/src`, `frontend/DESIGN.md`, `docs/screenshots` |
| Deliverable: architecture diagram | Graphviz source + SVG/PNG | `docs/architecture.*`, `docs/ARCHITECTURE.md` |
| Deliverable: design | `docs/DESIGN.md`, `docs/MODEL_CARD.md` | |
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
