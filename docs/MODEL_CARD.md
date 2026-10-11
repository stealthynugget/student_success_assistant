# Model card — Student Success Assistant v5

_Generated 2026-10-11 13:14 from `artifacts/metrics/*.json` by `python -m ssa.cli report`. Engine: **xgboost**. Evidence profile: **official** (quality gate passed)._

## Intended use
Early-warning support for academic advisors at ~40% of a course presentation: surface students still enrolled who may withdraw or fail, explain why, show comparable historical cases and draft supportive outreach. A human advisor reviews every recommendation. Not for admissions, grading, sanctions or any automated adverse decision.

## Data and population
- OULAD, 32,593 enrolments; snapshot at 40% of each presentation.
- **In scope:** 25,204 enrolments still registered at the snapshot. 7,389 students who had already unregistered (3,097 before day 0) are excluded — they cannot be helped by an early warning and including them inflates every metric.
- Temporal protocol: ['2013B', '2013J'] train, ['2014B'] validation, ['2014J'] untouched test.

## Withdrawal model (target: withdrawal after the snapshot)
| Metric (test presentation, in scope) | Value |
|---|---|
| ROC-AUC | 0.726 (95% CI 0.708–0.740) |
| PR-AUC | 0.263 (base rate 0.113) |
| Brier / ECE | 0.093 / 0.009 |
| Logistic-regression baseline ROC-AUC | 0.703 |
| Precision / recall when contacting top 10% | 29% / 26% (lift 2.61) |
| Precision / recall when contacting top 20% | 25% / 44% |

Risk tiers (thresholds learned out-of-fold: Medium ≥ 0.077, High ≥ 0.141):

| Tier | Share of students | Observed withdrawal rate | Share of all withdrawals |
|---|---|---|---|
| High | 24% | 23.6% | 50% |
| Medium | 28% | 11.7% | 29% |
| Low | 48% | 4.9% | 21% |

**Leakage audit.** Scoring the same model on *all* test enrolments, including students who had already left, gives ROC-AUC 0.749 vs 0.726 in scope; 75% of the test presentation's withdrawals had already happened by the snapshot. v4 reported 0.90 by training and testing on that population.

## Outcome model (Distinction / Pass / Fail / Withdrawn)
- Test macro-F1 0.485 vs multinomial logistic 0.460 and majority class 0.169; class weighting `inverse` chosen on validation.
- P(fail or withdraw): ROC-AUC 0.852, ECE 0.056.

## Feature-group ablation (test ROC-AUC / PR-AUC)
| Variant | ROC-AUC | PR-AUC |
|---|---|---|
| enrolment_only | 0.606 | 0.163 |
| academic_only | 0.723 | 0.262 |
| engagement_only | 0.679 | 0.191 |
| full_without_sensitive (production) | 0.726 | 0.261 |
| full_with_sensitive | 0.728 | 0.270 |

Sensitive attributes add essentially nothing, so they are excluded from the model by default.

## Prioritisation (target: fail or withdraw)
Weights learned on the validation presentation: `{'p_withdraw': 0.1, 'p_poor': 0.9, 'academic_gap': 0.0, 'engagement_gap': 0.0, 'missed_work': 0.0, 'recent_decline': 0.0}`.

| Ranking | Mean precision @10/20/30% | Precision @10% |
|---|---|---|
| learned_weights | 0.857 | 98% |
| v4_hand_formula | 0.762 | 84% |
| random | 0.349 | 34% |
| only_p_withdraw | 0.707 | 78% |
| only_p_poor | 0.860 | 99% |

## Historical-case retrieval
Queries: 600 in-scope students of 2014J; corpus: earlier presentations, same module.

| Mode | Neighbour-vote AUC (fail/withdraw) | Outcome agreement@10 | nDCG@10 (global ideal) |
|---|---|---|---|
| hybrid | 0.835 | 0.715 | 0.866 |
| behavioural | 0.829 | 0.718 | 0.870 |
| lexical | 0.836 | 0.708 | 0.754 |
| dense | 0.775 | 0.725 | 0.724 |
| random | 0.512 | 0.528 | 0.544 |

## Fairness audit (High tier, test presentation)
| Attribute | Equal-opportunity gap (TPR max−min) | Selection-rate ratio | Max calibration gap |
|---|---|---|---|
| gender | 0.082 | 0.877 | 0.009 |
| region | 0.265 | 0.582 | 0.030 |
| highest_education | 0.140 | 0.593 | 0.038 |
| imd_band | 0.195 | 0.653 | 0.030 |
| age_band | 0.048 | 0.804 | 0.018 |
| disability | 0.232 | 0.708 | 0.034 |

Gaps persist even with sensitive attributes excluded (proxies via engagement/scores and different base rates). They are surfaced as quality-gate warnings and must be reviewed with the institution's equity office before deployment. Mitigation options: per-group threshold review, outreach-capacity floors, monitoring by group via the feedback loop.

## Early warning across the course (test presentation, students still enrolled at each point)
| Course week | Still enrolled | ROC-AUC | PR-AUC | Base rate | ECE | Outcome macro-F1 |
|---|---|---|---|---|---|---|
| 6 | 27,157 | 0.686 | 0.316 | 0.179 | 0.017 | 0.397 |
| 9 | 26,241 | 0.700 | 0.289 | 0.150 | 0.011 | 0.454 |
| 15 | 25,204 | 0.726 | 0.263 | 0.113 | 0.009 | 0.485 |
| 22 | 23,926 | 0.773 | 0.176 | 0.058 | 0.008 | 0.531 |

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
