# Model card — Student Success Assistant v5

_Generated 2026-10-12 05:51 from `artifacts/metrics/*.json` by `python -m ssa.cli report`. Engine: **xgboost**. Evidence profile: **official** (quality gate passed)._

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
Weights learned on the validation presentation: `{'p_withdraw': 0.1, 'p_poor': 0.75, 'academic_gap': 0.05, 'engagement_gap': 0.05, 'missed_work': 0.05, 'recent_decline': 0.0}` (exhaustive simplex grid (step 0.05, 53130 weightings) on the validation presentation; rule one_se_blend).

Selection rule `one_se_blend`: the best grid point on validation (`validation_argmax` below) was `{'p_withdraw': 0.05, 'p_poor': 0.9, 'academic_gap': 0.0, 'engagement_gap': 0.05, 'missed_work': 0.0, 'recent_decline': 0.0}`; 58 weightings were within one bootstrap standard error (0.004) of it, and the most blended of those is used. Both are scored once on the test presentation.

| Ranking | Mean precision @10/20/30% | Precision @10% |
|---|---|---|
| learned_weights | 0.860 | 98% |
| validation_argmax | 0.861 | 98% |
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

### Typed search
40 labelled plain-English queries (templated, paraphrased and behaviour-led); a returned case is relevant when it meets every expected filter and shows every expected behaviour. Hybrid typed search uses weights tuned per query parser on a separate query set (rules parser lexical 1.0, dense 2.0; llm parser lexical 1.0, dense 0.0).

| Query parser | Mode | Precision@10 | Filter match@10 | Behaviour match@10 |
|---|---|---|---|---|
| rules | lexical | 0.522 | 0.735 | 0.527 |
| rules | hybrid | 0.622 | 0.860 | 0.668 |
| rules | dense | 0.634 | 0.875 | 0.674 |
| rules | random_corpus | 0.071 | n/a | n/a |
| rules | random_within_parsed_filters | 0.370 | n/a | n/a |
| llm | lexical | 0.862 | 0.985 | 0.846 |
| llm | hybrid | 0.862 | 0.985 | 0.846 |
| llm | dense | 0.714 | 0.978 | 0.671 |
| llm | random_corpus | 0.071 | n/a | n/a |
| llm | random_within_parsed_filters | 0.459 | n/a | n/a |

## Fairness audit (High tier, test presentation)
Groups count towards a gap only with n >= 50, at least 30 withdrawn and 30 not withdrawn. Chance gap: the median max−min gap when group labels are shuffled among withdrawn students (1,000 permutations), overall and within each module.

| Attribute | Equal-opportunity gap (TPR max−min) | Chance gap | p | p within module | Selection-rate ratio | Max calibration gap |
|---|---|---|---|---|---|---|
| gender | 0.082 | 0.021 | 0.013 | 0.598 | 0.877 | 0.009 |
| region | 0.265 | 0.208 | 0.144 | 0.228 | 0.582 | 0.030 |
| highest_education | 0.043 | 0.048 | 0.575 | 0.937 | 0.799 | 0.018 |
| imd_band | 0.195 | 0.177 | 0.352 | 0.499 | 0.653 | 0.030 |
| age_band | 0.048 | 0.022 | 0.174 | 0.259 | 0.804 | 0.018 |
| disability | 0.232 | 0.034 | 0.001 | 0.001 | 0.708 | 0.034 |

### Across the course
| Course week | gender | region | highest_education | imd_band | age_band | disability |
|---|---|---|---|---|---|---|
| 6 | 0.09 (p 0.001) | 0.25 (p 0.009) | 0.06 (p 0.117) | 0.14 (p 0.471) | 0.09 (p 0.001) | 0.04 (p 0.266) |
| 9 | 0.01 (p 0.708) | 0.23 (p 0.095) | 0.06 (p 0.203) | 0.19 (p 0.215) | 0.08 (p 0.008) | 0.05 (p 0.188) |
| 15 | 0.08 (p 0.013) | 0.26 (p 0.144) | 0.04 (p 0.575) | 0.19 (p 0.352) | 0.05 (p 0.174) | 0.23 (p 0.001) |
| 22 | 0.00 (p 1.000) | 0.21 (p 0.569) | 0.03 (p 0.913) | 0.23 (p 0.513) | 0.02 (p 0.663) | 0.18 (p 0.013) |

Gap and permutation p-value per checkpoint. With 24 tests, a few p-values under 0.05 are expected by chance alone.

**gender**: beyond chance overall, but explained by which modules the groups study. Lowest TPR F 0.45 (95% CI 0.40–0.50, 389 withdrawn); highest M 0.53 (95% CI 0.49–0.58, 554 withdrawn). Observed withdrawal rate 10.9% vs mean predicted 10.0% (F); 11.5% vs 10.7% (M).
Proxies. M: model inputs identify membership with ROC-AUC 0.82; most different inputs module BBB (-0.99 SD), Assessments due so far (+0.80 SD), Course day of snapshot (+0.78 SD).

Threshold review (target TPR 0.50, current High cut-off 0.141):

| Group | TPR now | Cut-off for target TPR | Selection rate now → then | PPV now → then | Students flagged |
|---|---|---|---|---|---|
| F | 0.45 | 0.128 | 22% → 26% | 22% → 21% | +142 |
| M | 0.53 | 0.149 | 25% → 23% | 24% → 25% | -93 |

**region**: within chance: random group labels give a gap this large at these group sizes. Lowest TPR Ireland 0.36 (95% CI 0.24–0.51, 44 withdrawn); highest North Western Region 0.63 (95% CI 0.52–0.73, 78 withdrawn). Observed withdrawal rate 11.3% vs mean predicted 9.0% (Ireland); 11.7% vs 11.4% (North Western Region).
Proxies. Ireland: model inputs identify membership with ROC-AUC 0.73; most different inputs How early the student registered (-0.73 SD), Score trajectory (-0.47 SD), Registered after start (+0.47 SD); North Western Region: model inputs identify membership with ROC-AUC 0.55; most different inputs How early the student registered (+0.20 SD), Score percentile in cohort (-0.13 SD), Tutor-marked assignments missed (+0.13 SD).

**highest_education**: within chance: random group labels give a gap this large at these group sizes. Lowest TPR A Level or Equivalent 0.48 (95% CI 0.43–0.53, 390 withdrawn); highest Lower Than A Level 0.53 (95% CI 0.47–0.57, 379 withdrawn). Observed withdrawal rate 10.4% vs mean predicted 10.5% (A Level or Equivalent); 12.5% vs 10.7% (Lower Than A Level).
Proxies. A Level or Equivalent: model inputs identify membership with ROC-AUC 0.55; most different inputs module GGG (-0.17 SD), Credits studied this term (+0.15 SD), Assessments due so far (+0.14 SD); Lower Than A Level: model inputs identify membership with ROC-AUC 0.63; most different inputs module GGG (+0.32 SD), Due assessments submitted (-0.30 SD), Assessments due so far (-0.30 SD).

**imd_band**: within chance: random group labels give a gap this large at these group sizes. Lowest TPR 90-100% 0.39 (95% CI 0.28–0.50, 70 withdrawn); highest 20-30% 0.58 (95% CI 0.49–0.66, 124 withdrawn). Observed withdrawal rate 10.2% vs mean predicted 9.6% (90-100%); 14.1% vs 11.1% (20-30%).
Proxies. 90-100%: model inputs identify membership with ROC-AUC 0.53; most different inputs Due assessments submitted (+0.20 SD), Assessments submitted (+0.19 SD), Weighted assessment score (+0.18 SD); 20-30%: model inputs identify membership with ROC-AUC 0.52; most different inputs Inactive weeks (last 3) (+0.16 SD), Weeks with any activity (-0.14 SD), Longest inactive streak (weeks) (+0.13 SD).

**age_band**: within chance: random group labels give a gap this large at these group sizes. Lowest TPR 35-55 0.47 (95% CI 0.41–0.53, 292 withdrawn); highest 0-35 0.52 (95% CI 0.48–0.56, 642 withdrawn). Observed withdrawal rate 11.3% vs mean predicted 9.6% (35-55); 11.2% vs 10.8% (0-35).
Proxies. 35-55: model inputs identify membership with ROC-AUC 0.68; most different inputs Activity percentile in cohort (+0.44 SD), Share of scheduled materials opened (+0.37 SD), Active-days percentile in cohort (+0.32 SD); 0-35: model inputs identify membership with ROC-AUC 0.68; most different inputs Activity percentile in cohort (-0.47 SD), Share of scheduled materials opened (-0.39 SD), Active-days percentile in cohort (-0.34 SD).

**disability**: beyond chance, also within modules: review. Lowest TPR N 0.47 (95% CI 0.44–0.51, 828 withdrawn); highest Y 0.70 (95% CI 0.62–0.78, 115 withdrawn). Observed withdrawal rate 10.8% vs mean predicted 10.2% (N); 15.6% vs 12.3% (Y).
Proxies. Y: model inputs identify membership with ROC-AUC 0.58; most different inputs Weighted assessment score (-0.25 SD), module GGG (+0.24 SD), Most recent score (-0.23 SD).

Threshold review (target TPR 0.50, current High cut-off 0.141):

| Group | TPR now | Cut-off for target TPR | Selection rate now → then | PPV now → then | Students flagged |
|---|---|---|---|---|---|
| N | 0.47 | 0.134 | 23% → 25% | 22% → 22% | +161 |
| Y | 0.70 | 0.188 | 33% → 20% | 34% → 39% | -90 |

Sensitive attributes are excluded from the model, so remaining gaps come from proxies and different base rates. They are surfaced as quality-gate warnings and must be reviewed with the institution's equity office before deployment. Per-group cut-offs are shown for that review only; applying them would use a sensitive attribute at decision time.

## Early warning across the course (test presentation, students still enrolled at each point)
| Course week | Still enrolled | ROC-AUC | PR-AUC | Base rate | ECE | Outcome macro-F1 |
|---|---|---|---|---|---|---|
| 6 | 27,157 | 0.686 | 0.316 | 0.179 | 0.017 | 0.397 |
| 9 | 26,241 | 0.700 | 0.289 | 0.150 | 0.011 | 0.454 |
| 15 | 25,204 | 0.726 | 0.263 | 0.113 | 0.009 | 0.485 |
| 22 | 23,926 | 0.773 | 0.176 | 0.058 | 0.008 | 0.531 |

## AI features (offline evaluation)
Generator: claude-sonnet-5-5; judge: claude-opus-5-5.

| Check | Result |
|---|---|
| Support plans passing every guardrail | 100% |
| AI plan drafts accepted / revised | 79% / 0% |
| Judge pass rate | 96% |
| AI plan preferred in order-swapped A/B | 89% |
| Search understanding F1, templated (rules / AI) | 0.975 / 0.838 |
| Search understanding F1, paraphrased (rules / AI) | 0.741 / 0.882 |
| Copilot correct answers (rules / AI) | 100% / 100% |
| Adversarial copilot prompts handled safely (llm) | 88% |
| Precedent syntheses accepted by the code checks / judge pass | 92% / 82% |
| Cohort briefings accepted by the code checks / judge pass | 100% / 100% |
| Outreach rewrites accepted / within length / safe | 100% / 100% / 100% |

DeepEval (claude-opus-5-5; mean score, 95% bootstrap CI, pass rate):

| Metric | Mean | 95% CI | Pass rate |
|---|---|---|---|
| Actionable for an advisor | 0.66 | 0.61–0.70 | 75% |
| AnswerRelevancyMetric | 1.00 | 1.00–1.00 | 100% |
| FaithfulnessMetric | 1.00 | 1.00–1.00 | 100% |
| No protected-attribute reasoning | 0.97 | 0.95–0.99 | 100% |
| Supportive and non-punitive | 0.90 | 0.90–0.90 | 100% |

Models: claude-haiku-5-5 (search, outreach, note themes), claude-sonnet-5-5 (agents, copilot, synthesis, briefing), claude-opus-5-5 (judge, A/B, DeepEval). Last evaluation run: 613 calls, of which 603 were answered from the response cache of earlier runs; 0 errors; new spend $0.20 (spend across runs: `/api/llm/usage`).

## Limitations
- Associations, not causes. Drivers explain the model, not the student.
- One test presentation (2014J); performance can drift across cohorts — monitored by the feedback loop's tier-drift check.
- OULAD is from one institution and 2013–2014; revalidate before use elsewhere.
- LLM narration is optional; all facts, scores, priorities and eligible actions are computed deterministically.
