# Deployment and security

- **One service.** `uvicorn ssa.api.app:app_factory --factory` serves the API and the built console. Docker builds the
  console in a Node stage and runs as a non-root user with a healthcheck.
- **Workers and shared state.** Three things are shared between requests: background-job records (the console polls
  `/api/jobs/{id}`), per-client rate-limit buckets and recent analyses (for on-demand judging). `SHARED_STATE` picks
  where they live (`ssa/api/shared.py`):
  - `memory` (default): in the process. Right for one worker per container (`WEB_CONCURRENCY=1`).
  - `sqlite`: `artifacts/app/shared_state.sqlite3` (WAL), shared by every worker that mounts the same `artifacts/`.
    Use it with `WEB_CONCURRENCY=N` on one host. A job runs in the worker that accepted it; its record is shared, so a
    poll can land on any worker. A record that stops updating for 15 minutes is reported as lost (its worker died), and
    the console offers to run it again. Measured on this machine with 3 workers: 30 of 30 polls of a finished job found
    it with `sqlite`; 19 of 30 returned 404 with `memory`. The service logs a warning when `WEB_CONCURRENCY` > 1 with
    `memory`.
  Each worker loads its own models and case indexes: about 1 GB once all four checkpoints are loaded (measured in the
  container with the dense channel), so size memory per worker.
- **Several hosts.** SQLite does not span machines. Keep `SHARED_STATE=sqlite` per host for jobs and either route a
  client's polls to the host that accepted the job (sticky sessions on the `X-API-Key` or a cookie) or replace
  `SQLiteJobStore` with a Redis/Postgres store; the interface is two methods (`put`, `get`). For a full queue, the job
  functions are plain callables and can be moved to RQ or Celery workers unchanged. Enforce rate limits in the gateway
  and set `RATE_LIMIT_PER_MINUTE=0` in the app, for example with nginx:

  ```nginx
  limit_req_zone $binary_remote_addr zone=ssa_api:10m rate=2r/s;           # 120 per minute per client
  limit_req_zone $binary_remote_addr zone=ssa_ai:10m rate=10r/m;           # LLM-backed routes
  server {
    location /api/jobs/       { limit_req zone=ssa_ai burst=5 nodelay;  proxy_pass http://ssa; }
    location /api/copilot/    { limit_req zone=ssa_ai burst=5 nodelay;  proxy_pass http://ssa;
                                proxy_buffering off; proxy_read_timeout 300s; }   # server-sent events
    location /api/            { limit_req zone=ssa_api burst=60 nodelay; proxy_pass http://ssa; }
    location /                { proxy_pass http://ssa; }
  }
  ```
  Set `TRUSTED_PROXY_COUNT=1` behind one proxy so the app sees the real client address. Polling `GET /api/jobs/{id}`
  matches `/api/jobs/` too; give it its own `location ~ ^/api/jobs/[0-9a-f]+$` under the general zone if the console
  polls quickly.
- **Cold start.** The image bakes the sentence-embedding model (`EMBEDDING_MODEL` build arg, cached in `HF_HOME=/opt/hf`),
  so the first dense search in a new container does not download it, and `HF_HUB_OFFLINE=1` stops run-time Hub calls.
  Models and indexes still load on first use per checkpoint: the first search at each checkpoint took 11–15 s in a
  fresh container (about 49 s before, when the model was downloaded), later ones under 0.1 s. `GET /api/ready` reports
  whether every artifact is present.
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
