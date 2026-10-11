# Deployment and security

- **One service.** `uvicorn ssa.api.app:app_factory --factory` serves the API and the built console. Docker builds the
  console in a Node stage and runs as a non-root user with a healthcheck.
- **Workers.** Background jobs, the recent-recommendation cache and the rate limiter live in the process. Run one worker
  per container and scale horizontally behind a load balancer with sticky sessions, or move jobs to a queue (RQ/Celery)
  and rate limiting to the gateway for multi-worker deployments.
- **Secrets** only in the environment. `.env` is excluded from git, Docker and releases; the quality gate scans for keys.
- **Auth.** `STUDENT_ASSISTANT_API_KEY` for every route except health; `STUDENT_ASSISTANT_ADMIN_KEY` to apply
  feedback-loop changes. Without a key (and without `SSA_DEV_MODE=1`) protected routes return 503. Put TLS in front.
- **CORS** only for `ALLOWED_ORIGINS`; not needed when the console is served by the API.
- **LLM data handling.** Only evidence statements, model outputs, catalogue entries and advisor questions are sent; never
  sensitive attributes. Copilot questions are screened for injection patterns. Use a provider agreement that fits your
  institution's policy. `ONLINE_EVAL_SAMPLE_RATE` controls how much live traffic the judge reviews; usage and estimated
  cost are visible at `/api/llm/usage` and in the console.
- **Data stores.** SQLite (WAL) for feedback, LLM usage, online evaluations and the LLM cache. For several hosts move them
  to Postgres (same schema).
