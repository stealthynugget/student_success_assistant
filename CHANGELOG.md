# Changelog

## 6.0.0
- FastAPI service (routers over framework-independent handlers, OpenAPI docs, admin key, background jobs with live steps).
- React console replacing Streamlit: course clock, this week, students, student workspace, who to contact, case finder,
  copilot, insights, model quality, feedback loop, settings.
- Complete-dataset use: planned schedule coverage, TMA/CMA split, whole-course historical records and assessment analytics.
- Four course checkpoints with separate calibrated models, indexes and rankings; per-student risk trajectory.
- LLM features: tool-using copilot, search understanding, precedent synthesis, cohort briefing, outreach rewriting,
  note themes, on-demand and sampled online judging; usage and cost ledger.
- Offline evaluation per AI feature, including labelled search queries, copilot golden questions, adversarial prompts
  and order-swapped A/B of AI vs computed plans.
- Fixes found through UI testing: race in index loading, slow cohort ranking (explanations only for shown students).

## 5.0.0
Complete rebuild after the v4 audit. See `docs/AUDIT_FIXES.md` for the issue-by-issue mapping. Highlights: leakage-free
population scoping, LOPO calibration and learned tiers, rebuilt hybrid retrieval with a non-circular benchmark, learned
prioritisation, evidence-gated multi-agent workflow with code guardrails, offline judge/DeepEval, closed feedback loop,
secure API, real-data console, isolated tests, generated reports.
