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
- **LLM provider.** Claude through the Anthropic API (`LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`). Three tiers:
  `CLAUDE_MODEL_FAST` (claude-haiku-5-5: search understanding, outreach rewrites, note themes), `CLAUDE_MODEL_MAIN`
  (claude-sonnet-5-5: agents, copilot, precedent synthesis, briefing) and `CLAUDE_MODEL_JUDGE` (claude-opus-5-5: judge,
  pairwise A/B, DeepEval). The service refuses to start if the judge equals a generator. JSON features use structured
  outputs with a per-feature schema; every answer is still validated in code and has a deterministic fallback.
  `LLM_ENABLED=false` stops all LLM traffic without removing keys (incident switch). `LLM_PROVIDER=openai` with
  `OPENAI_*` remains available as an alternative.
- **Timeouts, retries, limits.** `LLM_TIMEOUT_S` (default 90 s; thinking is always on for these models) and
  `LLM_MAX_RETRIES` (default 2). Rate limits (429, honouring `retry-after`), overload (529), 5xx, timeouts and malformed
  output are retried with backoff; other 4xx and refusals fall back immediately. Per-request latency is dominated by the
  judge (Opus, effort high); keep judging in background jobs (`ONLINE_EVAL_SAMPLE_RATE`), never on the request path.
- **LLM data handling.** Only evidence statements, model outputs, catalogue entries and advisor questions are sent; never
  sensitive attributes. Copilot questions are screened for injection patterns. Use a provider agreement that fits your
  institution's policy. `ONLINE_EVAL_SAMPLE_RATE` controls how much live traffic the judge reviews.
- **Cost.** Every call is logged with the model that served it, uncached input, cache read/write and output tokens, and a
  cost from the per-model price table (USD per million tokens, defaults from Anthropic's pricing page; override or extend
  with `LLM_PRICE_TABLE`, see `.env.example`). Totals by feature and model: `/api/llm/usage` and the console. The copilot's
  system prompt and tool definitions are prompt-cached.
- **Data stores.** SQLite (WAL) for feedback, LLM usage, online evaluations and the LLM cache. For several hosts move them
  to Postgres (same schema).
