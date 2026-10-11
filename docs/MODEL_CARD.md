# Model card — Student Success Assistant v5

_Generated 2026-10-10 16:02 from `artifacts/metrics/*.json` by `python -m ssa.cli report`. Engine: **sklearn**. Evidence profile: **official** (quality gate passed)._

## Intended use
Early-warning support for academic advisors at ~40% of a course presentation: surface students still enrolled who may withdraw or fail, explain why, show comparable historical cases and draft supportive outreach. A human advisor reviews every recommendation. Not for admissions, grading, sanctions or any automated adverse decision.

## Data and population
- OULAD, 32,593 enrolments; snapshot at 40% of each presentation.
- **In scope:** 25,204 enrolments still registered at the snapshot. 7,389 students who had already unregistered (3,097 before day 0) are excluded — they cannot be helped by an early warning and including them inflates every metric.
- Temporal protocol: ['2013B', '2013J'] train, ['2014B'] validation, ['2014J'] untouched test.

## Withdrawal model (target: withdrawal after the snapshot)
| Metric (test presentation, in scope) | Value |
|---|---|
| ROC-AUC | 0.730 (95% CI 0.713–0.745) |
| PR-AUC | 0.266 (base rate 0.113) |
| Brier / ECE | 0.092 / 0.006 |
| Logistic-regression baseline ROC-AUC | 0.703 |
| Precision / recall when contacting top 10% | 30% / 27% (lift 2.68) |
| Precision / recall when contacting top 20% | 26% / 45% |

Risk tiers (thresholds learned out-of-fold: Medium ≥ 0.080, High ≥ 0.158):

| Tier | Share of students | Observed withdrawal rate | Share of all withdrawals |
|---|---|---|---|
| High | 20% | 25.5% | 46% |
| Medium | 34% | 11.8% | 36% |
| Low | 46% | 4.6% | 19% |

**Leakage audit.** Scoring the same model on *all* test enrolments, including students who had already left, gives ROC-AUC 0.783 vs 0.730 in scope; 75% of the test presentation's withdrawals had already happened by the snapshot. v4 reported 0.90 by training and testing on that population.

## Outcome model (Distinction / Pass / Fail / Withdrawn)
- Test macro-F1 0.479 vs multinomial logistic 0.460 and majority class 0.169; class weighting `inverse` chosen on validation.
- P(fail or withdraw): ROC-AUC 0.853, ECE 0.080.

## Feature-group ablation (test ROC-AUC / PR-AUC)
| Variant | ROC-AUC | PR-AUC |
|---|---|---|
| enrolment_only | 0.597 | 0.154 |
| academic_only | 0.717 | 0.261 |
| engagement_only | 0.676 | 0.203 |
| full_without_sensitive (production) | 0.731 | 0.270 |
| full_with_sensitive | 0.729 | 0.275 |

Sensitive attributes add essentially nothing, so they are excluded from the model by default.

## Prioritisation (target: fail or withdraw)
Weights learned on the validation presentation: `{'p_withdraw': 0.0, 'p_poor': 1.0, 'academic_gap': 0.0, 'engagement_gap': 0.0, 'missed_work': 0.0, 'recent_decline': 0.0}`.

| Ranking | Mean precision @10/20/30% | Precision @10% |
|---|---|---|
| learned_weights | 0.861 | 99% |
| v4_hand_formula | 0.761 | 84% |
| random | 0.349 | 34% |
| only_p_withdraw | 0.709 | 78% |
| only_p_poor | 0.861 | 99% |

## Historical-case retrieval
Queries: 500 in-scope students of 2014J; corpus: earlier presentations, same module.

| Mode | Neighbour-vote AUC (fail/withdraw) | Outcome agreement@10 | nDCG@10 (global ideal) |
|---|---|---|---|
| hybrid | 0.802 | 0.694 | 0.880 |
| behavioural | 0.785 | 0.684 | 0.874 |
| lexical | 0.832 | 0.701 | 0.755 |
| random | 0.505 | 0.529 | 0.546 |

## Fairness audit (High tier, test presentation)
| Attribute | Equal-opportunity gap (TPR max−min) | Selection-rate ratio | Max calibration gap |
|---|---|---|---|
| gender | 0.093 | 0.713 | 0.006 |
| region | 0.261 | 0.605 | 0.027 |
| highest_education | 0.029 | 0.673 | 0.030 |
| imd_band | 0.162 | 0.741 | 0.024 |
| age_band | 0.065 | 0.736 | 0.014 |
| disability | 0.211 | 0.700 | 0.029 |

Gaps persist even with sensitive attributes excluded (proxies via engagement/scores and different base rates). They are surfaced as quality-gate warnings and must be reviewed with the institution's equity office before deployment. Mitigation options: per-group threshold review, outreach-capacity floors, monitoring by group via the feedback loop.

## Early warning across the course (test presentation, students still enrolled at each point)
| Course week | Still enrolled | ROC-AUC | PR-AUC | Base rate | ECE | Outcome macro-F1 |
|---|---|---|---|---|---|---|
| 6 | 27,157 | 0.693 | 0.347 | 0.179 | 0.011 | 0.393 |
| 9 | 26,241 | 0.703 | 0.281 | 0.150 | 0.014 | 0.448 |
| 15 | 25,204 | 0.730 | 0.266 | 0.113 | 0.006 | 0.479 |
| 22 | 23,926 | 0.758 | 0.173 | 0.058 | 0.011 | 0.528 |

## AI features (offline evaluation)
Generator: not connected; judge: not connected.

| Check | Result |
|---|---|
| Support plans passing every guardrail | 100% |
| AI plan drafts accepted / revised | 0% / 0% |
| Judge pass rate | not run |
| AI plan preferred in order-swapped A/B | not run |
| Search understanding F1, templated (rules / AI) | 0.975 / n/a |
| Search understanding F1, paraphrased (rules / AI) | 0.741 / n/a |
| Copilot correct answers (rules / AI) | 100% / n/a |
| Adversarial copilot prompts handled safely (rules) | 100% |

## Limitations
- Associations, not causes. Drivers explain the model, not the student.
- One test presentation (2014J); performance can drift across cohorts — monitored by the feedback loop's tier-drift check.
- OULAD is from one institution and 2013–2014; revalidate before use elsewhere.
- LLM narration is optional; all facts, scores, priorities and eligible actions are computed deterministically.
