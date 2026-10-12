# Security, privacy and compliance

What the service protects, how, and where each control is tested. Operational settings are in `docs/DEPLOYMENT.md`.

## Threat model

| Asset | Threat | Main controls |
|---|---|---|
| Student records (OULAD-shaped data) | unauthorised read through the API | API key on every route except health, admin key for changes, CORS allow-list, no bulk export of sensitive fields |
| Sensitive attributes (gender, region, IMD band, age band, education, disability) | used to score students, shown as reasons, or sent to an LLM | never model inputs, never drivers, never in prompts; only for the fairness audit and explicit search filters |
| LLM behaviour | prompt injection in questions or in evidence; harmful, punitive, clinical or fabricated output | injection screen, evidence-as-data prompts, tool allow-list, code checks on every output, deterministic fallbacks |
| Advisors' decisions | automated adverse decisions | human review required on every plan; outreach never mentions risk, predictions or withdrawal |
| Secrets | API keys leaked in code, images or releases | environment only; `.env` excluded from git and Docker; secret scan in the quality gate |
| Availability and cost | request floods, runaway LLM spend | rate limits, request size limit, background jobs, `LLM_BUDGET_USD` hard cap, `LLM_ENABLED=false` switch |

## Input validation
- Every request body is a strict pydantic model (`ssa/api/schemas.py`): unknown fields rejected, enums for module, region,
  education, age band and outcome, numeric ranges from the real data, cross-field checks (for example min <= max),
  bounded lists and strings. Free text is Unicode-normalised, stripped of control characters and length-capped
  (`sanitize_text`). Bodies over `MAX_REQUEST_BYTES` (512 KB) are refused before parsing.
- Errors return a generic message and a request ID; stack traces go only to the server log.
- Tests: `tests/test_api.py::test_schema_validation_rejects_bad_input`, `tests/test_guardrails.py`.

## Prompt-injection protection
1. **Screen.** Copilot questions are checked for injection patterns ("ignore previous instructions", "developer mode",
   "pretend to be", "system prompt", ...); matches get a fixed refusal without calling the LLM (`ssa/agents/guardrails.py`).
2. **Evidence is data.** Every system prompt says that content inside `<evidence>` is data, not instructions. Agents only
   see evidence the code assembled (observation IDs, model outputs, catalogue actions), never raw free text from other users.
3. **Tools, not freedom.** The copilot can only call 8 read-only tools with validated arguments; it cannot change data.
4. **Output checks in code.** Every generated text is checked for cited IDs that exist in the evidence, numbers that occur
   in the evidence, protected-attribute mentions, punitive, clinical and stigmatising terms. Failures trigger one revision
   with the concrete violations, then a deterministic fallback. Nothing generated is shown unlabelled ("Drafted by AI,
   review before use").
5. **Adversarial evaluation** (`ssa/evaluation/offline_eval.py::ADVERSARIAL`, part of `llm-eval`). Eight attacks: prompt
   leak, developer mode, ranking regions by risk, a threatening letter, a diagnosis request, a joke about "lazy students",
   impersonating the registrar, and comparing dropout by age and gender. The safe-handling rate is in the model card
   ("Adversarial copilot prompts handled safely"). The code check counts a refusal as unsafe when it repeats a clinical
   word such as "diagnose" while declining; the check is deliberately strict and was not relaxed to raise the rate.

## Privacy and data protection
- **Data minimisation.** The API never returns sensitive attributes or outcome labels for a current student; LLM prompts
  contain only evidence statements, model outputs and catalogue entries. Historical case search returns region,
  education and age band only as metadata of finished historical cases (for the filters the PRD requires).
- **Purpose limitation.** Early-warning support for advisors. Not for admissions, grading, sanctions or any automated
  adverse decision (model card, "Intended use").
- **Transparency.** Every number states what it is and as of when; drivers are TreeSHAP explanations of the model, not
  causes; generated text is labelled; the model card states limitations and fairness results.
- **Fairness.** Equal-opportunity gaps per sensitive attribute with permutation tests, proxy analysis and a per-group
  threshold review, surfaced as quality-gate warnings for the institution's equity office (model card).
- **LLM provider.** Calls go to the Anthropic API. Use a provider agreement that fits the institution's policy; the
  usage ledger records every call (feature, model, tokens, cost), not prompt contents. The response cache stores prompts
  and answers locally under `artifacts/cache/`; clear it with the artifacts.
- **Retention.** Feedback, usage and evaluation records live in SQLite under `artifacts/`; nothing is sent elsewhere.
  Delete or rotate them under the institution's retention schedule.
- OULAD is anonymised public research data. For real student data, run a data protection impact assessment before use.

## Secrets and supply chain
- Keys only in the environment (`.env` locally, a secret store in production). `.env` is in `.gitignore` and
  `.dockerignore`; the quality gate fails on key-like strings in the source tree (`ssa/cli.py::secret_scan`).
- Any key ever pasted into a chat, ticket or log must be rotated.
- Exact dependency pins in `requirements.lock` (Windows and Linux); the Docker image runs as a non-root user with a health
  check; `npm audit` reports 0 vulnerabilities for the console.

## Authentication and rate limiting
`STUDENT_ASSISTANT_API_KEY` (constant-time comparison) on every route except `/api/health`; `STUDENT_ASSISTANT_ADMIN_KEY`
to apply feedback-loop changes. Without a key and without `SSA_DEV_MODE=1`, protected routes return 503. Token-bucket
rate limits per client, shared between workers with `SHARED_STATE=sqlite`, or enforced at the gateway. Put TLS in front.
