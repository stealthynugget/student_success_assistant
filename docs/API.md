# API (FastAPI)

Interactive documentation: `GET /api/docs` (OpenAPI JSON at `/api/openapi.json`). All routes except `/api/health`
need `X-API-Key`; applying feedback-loop changes also needs `X-Admin-Key`. Every route that reads model output accepts
`checkpoint` (0.15, 0.25, 0.40 or 0.60 of the course; nearest is used; default from `CUTOFF_FRACTION`).

Errors: `{"error", "request_id", "details"?}` with 400 validation, 401 key, 403 admin key, 404 unknown student / job /
recommendation, 409 student already left before the checkpoint, 413 body too large, 429 rate limit, 503 not built.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/ready`, `/api/config` | liveness; artifact readiness; checkpoints, modules, vocabularies, LLM status |
| GET | `/api/overview` | every presentation: enrolled, still enrolled, tier counts, expected withdrawals |
| GET | `/api/cohort` | one presentation: tiers, risk histogram, common High-tier drivers, median weekly activity, assessments |
| GET | `/api/students` | still-enrolled students with scores; `tier`, `search`, `sort`, `descending`, `limit`, `offset` |
| GET | `/api/students/{id}` | real snapshot (no labels, no sensitive attributes) |
| POST | `/api/predict-risk` | risk, tier, drivers, outcome probabilities; by reference, reference + `what_if`, or partial profile |
| POST | `/api/students/timeline` | risk at every checkpoint for one enrolment |
| POST | `/api/trends/student` | weekly activity vs cohort median, deadlines, trend signals |
| POST | `/api/analyze` · `/api/jobs/analyze` | full agent workflow + precedent synthesis (optionally judged); job version streams steps |
| GET | `/api/jobs/{id}` | job status, completed steps, result |
| POST | `/api/copilot` · `/api/jobs/copilot` | tool-using copilot; returns answer, tool calls, grounding check |
| POST | `/api/cohort/briefing` · `/api/jobs/briefing` | weekly module-team briefing |
| POST | `/api/outreach/variants` | rewrite a message for email / text / phone script, tone and language |
| POST | `/api/search/parse` | how a plain-English case search will be interpreted |
| POST | `/api/retrieve-similar` | hybrid case search (student, profile and/or text) with filters, stories and aftermath |
| POST | `/api/prioritize` | ranking for a presentation, a list of students or supplied profiles, with capacity |
| GET | `/api/trends/course`, `/api/insights/factors`, `/api/insights/assessments`, `/api/insights/checkpoints` | historical insight |
| GET | `/api/metrics` | model, retrieval, prioritisation, fairness, LLM evaluation, release checks |
| GET | `/api/evaluations/llm` · `/api/evaluations/online` | offline AI evaluation; judged live samples |
| POST | `/api/evaluations/judge` | judge a recent recommendation now |
| GET | `/api/llm/usage` | calls, tokens, latency, estimated cost by feature |
| POST/GET | `/api/feedback`, `/api/feedback/summary`, `/api/feedback`, `/api/feedback/themes` | advisor feedback |
| GET/POST | `/api/improvement-plan`, `/api/feedback-loop/run` | proposals; recompute or apply (admin) |

```bash
H='-H X-API-Key:'$KEY' -H Content-Type:application/json'
curl -s $H localhost:8000/api/jobs/analyze -d '{"student":{"id_student":59272,"code_module":"BBB","code_presentation":"2014J"},"checkpoint":0.4}'
curl -s $H localhost:8000/api/copilot -d '{"question":"Who should I contact first in FFF 2014J and why?"}'
curl -s $H localhost:8000/api/retrieve-similar -d '{"query":"repeat students in DDD with heavy workloads who dropped out","top_k":10}'
```
