# Design

## Problem framing
- **Decision point:** 40 % of the planned presentation length (configurable `CUTOFF_FRACTION`). Everything used as a
  predictor must be known on that day.
- **Population:** students still registered at the snapshot (`in_scope`). Students who already unregistered are shown as
  "already withdrawn" and excluded from training, evaluation, retrieval and ranking.
- **Targets:** withdrawal after the snapshot (risk model); final outcome in four classes (outcome model);
  fail-or-withdraw (prioritisation, "needs academic support").

## Features (`ssa.models.features`)
- *Enrolment*: previous attempts, credits, registration timing, snapshot day.
- *Academic*: assessments due, submitted, missed (count/rate), weighted/mean/min/max/last score, score slope, late rate,
  days early, banked results, cohort score percentile. Missing scores stay `NaN` with a `no_scores_yet` flag.
- *Engagement*: clicks by activity group and shares, active days, resources, days since last activity, dense weekly grid
  statistics (slope, relative slope, early/recent means, momentum, irregularity, longest inactive streak, inactive weeks in
  the last three), cohort percentiles for clicks, active days and momentum.
- *Sensitive* (gender, region, IMD, age, education, disability): audit only; excluded from models and LLM prompts by
  default (`INCLUDE_SENSITIVE_FEATURES=false`). The ablation measures the cost (≈ 0).

## Modelling choices
- **Gradient boosting** with native missing-value handling; XGBoost preferred, scikit-learn HistGradientBoosting fallback.
- **Temporal protocol:** hyper-parameters selected on the validation presentation; calibration and tier thresholds learned
  from leave-one-presentation-out predictions; final refit on all historical presentations; one score on the test presentation.
- **Calibration:** Platt scaling of margins (out-of-fold). Outcome model: inverse-frequency weighting selected on
  validation for macro-F1, prior correction (`p ∝ p_w / w`) plus temperature scaling for calibrated probabilities.
- **Tiers:** Medium = highest threshold keeping 80 % recall on historical out-of-fold data (watch list); High = F1-optimal
  above Medium. Evaluated on test as observed withdrawal rate per tier.
- **Explanations:** exact TreeSHAP via XGBoost `pred_contribs`; baseline substitution for the fallback engine. Grouped
  per feature; sensitive groups never surfaced.

## Retrieval
- **Behavioural kNN:** robust-scaled snapshot features, weighted by the risk model's mean |attribution| (supervised
  similarity); exact search (≤ 25k vectors).
- **Lexical:** BM25 over discretised, cohort-relative descriptors (e.g. `missed_most`, `engagement_declining`), so keyword
  matches are meaningful; natural-language queries map onto descriptors.
- **Dense (optional):** sentence embeddings of qualitative narratives (no raw numbers) for free-text queries.
- **Fusion:** weighted RRF; weights tuned on validation queries, overridable by feedback-learned weights.
- **Guards:** earlier presentations only, self excluded, filter vocabulary validated, atomic index build with manifest.

## Prioritisation
Signals: P(withdraw), P(fail or withdraw), cohort-relative academic gap, engagement gap, missed work, recent decline —
each rank-normalised within the cohort. Weights: exhaustive simplex search (step 0.1) on the validation presentation
maximising mean precision at 10/20/30 % capacity. Output includes reason codes and top drivers.

## Responsible use
Human review is required for every plan. Outreach drafts never mention risk scores, predictions or withdrawal. No
diagnosis or speculation about causes. Fairness audit per sensitive attribute with gate warnings; the model card states
limitations. Data minimisation: the API never returns labels or sensitive attributes for a student.

## Non-functional
- Build ≈ 3 min, < 2 GB RAM on full OULAD; analysis request ≈ 50 ms without LLM.
- Deterministic path works without any LLM; LLM calls have timeouts, retries, caching and usage accounting.
- Reproducibility: fixed seeds, versioned feature/index formats, manifests and generated reports.

## v6 additions

**Checkpoints.** The same pipeline, models, calibration, tiers, prioritiser and index are built at 15, 25, 40 and 60 %
of the planned course length. Students who left before a checkpoint are out of scope at that checkpoint only, so a
student's risk can be followed over time (`/api/students/timeline`); the console hides checkpoints later than the one
selected so no future information leaks into a review.

**Complete records.** Besides snapshot features, the pipeline keeps each enrolment's whole-course weekly activity,
withdrawal week, final coursework average and exam score. Retrieval returns these as "what happened next" stories for
historical precedents only (earlier presentations), and assessment analytics measure how strongly missing or failing
each assessment separated students who later failed or withdrew.

**LLM use.** The LLM is used where language matters: planning narration, a tool-using copilot, understanding search
requests, summarising precedents, briefings, rewriting outreach and grouping advisor notes. Facts, scores, priorities,
eligible actions and IDs always come from code, and every generated output is validated before it is shown; failures
fall back to computed text and are labelled as such. Each feature has an offline evaluation and can be judged live.
