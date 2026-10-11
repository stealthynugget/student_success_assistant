"""Feature contract shared by training, serving, retrieval and explanation.

Sensitive / protected attributes (gender, region, IMD, age, education, disability) are
**excluded from model inputs by default** (``INCLUDE_SENSITIVE_FEATURES=false``). They are kept
in the data only for fairness auditing. v4 used them as predictors while instructing the
LLM agents never to use them, so SHAP drivers such as "Imd Band 0-10%" could reach the agents.
The ablation report quantifies the (small) accuracy cost of excluding them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssa.data import vocab

ACADEMIC = [
    "assessments_due", "submitted_count", "submitted_due_count", "missed_count", "missed_rate", "no_scores_yet",
    "mean_score", "weighted_score", "score_std", "min_score", "max_score", "last_score", "score_slope",
    "late_rate", "mean_days_early", "banked_count", "score_pct", "missed_rate_pct",
    "tma_mean_score", "cma_mean_score", "tma_due", "cma_due", "tma_missed", "cma_missed",
]
ENGAGEMENT = [
    "total_clicks", "log_total_clicks", "active_days", "resources_used", "activity_types_used", "pre_start_clicks",
    "clicks_content", "clicks_resource", "clicks_forum", "clicks_quiz", "clicks_homepage", "clicks_other",
    "share_content", "share_resource", "share_forum", "share_quiz", "share_homepage", "no_vle_activity",
    "days_since_last_activity", "clicks_per_active_day", "weeks_active", "engagement_weekly_mean",
    "engagement_slope", "engagement_slope_rel", "engagement_early_mean", "engagement_recent_mean",
    "engagement_momentum", "engagement_cv", "longest_inactive_streak", "inactive_weeks_last3",
    "engagement_drop_ratio", "clicks_pct", "active_days_pct", "planned_materials_due", "planned_coverage",
]
ENROLMENT = [
    "num_of_prev_attempts", "studied_credits", "registration_days_before_start", "registration_date_missing",
    "late_registration", "snapshot_day", "weeks_observed",
]
NUMERIC = ENROLMENT + ACADEMIC + ENGAGEMENT
CONTEXT_CATEGORICAL = {"code_module": vocab.MODULES}
SENSITIVE_CATEGORICAL = {
    "gender": vocab.GENDERS, "region": vocab.REGIONS, "highest_education": vocab.EDUCATION,
    "imd_band": vocab.IMD_BANDS, "age_band": vocab.AGE_BANDS, "disability": vocab.DISABILITY,
}
SENSITIVE = tuple(SENSITIVE_CATEGORICAL)

FRIENDLY = {
    "num_of_prev_attempts": "Previous attempts at this module", "studied_credits": "Credits studied this term",
    "registration_days_before_start": "How early the student registered", "late_registration": "Registered after start",
    "registration_date_missing": "Registration date missing", "snapshot_day": "Course day of snapshot",
    "weeks_observed": "Weeks observed", "assessments_due": "Assessments due so far",
    "submitted_count": "Assessments submitted", "submitted_due_count": "Due assessments submitted",
    "missed_count": "Missed assessments", "missed_rate": "Share of due assessments missed",
    "no_scores_yet": "No scored assessment yet", "mean_score": "Average assessment score",
    "weighted_score": "Weighted assessment score", "score_std": "Score variability", "min_score": "Lowest score",
    "max_score": "Highest score", "last_score": "Most recent score", "score_slope": "Score trajectory",
    "late_rate": "Late submission rate", "mean_days_early": "Average days submitted before deadline",
    "banked_count": "Banked results", "score_pct": "Score percentile in cohort",
    "missed_rate_pct": "Missed-assessment percentile in cohort", "total_clicks": "VLE activity (clicks)",
    "log_total_clicks": "VLE activity (log clicks)", "active_days": "Active learning days",
    "resources_used": "Distinct resources used", "activity_types_used": "Distinct activity types used",
    "pre_start_clicks": "Pre-course activity", "clicks_content": "Course-content activity",
    "clicks_resource": "Resource activity", "clicks_forum": "Forum/collaboration activity",
    "clicks_quiz": "Quiz activity", "clicks_homepage": "Homepage visits", "clicks_other": "Other activity",
    "share_content": "Share of activity on content", "share_resource": "Share of activity on resources",
    "share_forum": "Share of activity on forums", "share_quiz": "Share of activity on quizzes",
    "share_homepage": "Share of activity on homepage", "no_vle_activity": "No VLE activity",
    "days_since_last_activity": "Days since last VLE activity", "clicks_per_active_day": "Activity per active day",
    "weeks_active": "Weeks with any activity", "engagement_weekly_mean": "Average weekly activity",
    "engagement_slope": "Weekly engagement trend", "engagement_slope_rel": "Relative weekly engagement trend",
    "engagement_early_mean": "Early-weeks activity", "engagement_recent_mean": "Recent-weeks activity",
    "engagement_momentum": "Recent engagement momentum", "engagement_cv": "Engagement irregularity",
    "longest_inactive_streak": "Longest inactive streak (weeks)", "inactive_weeks_last3": "Inactive weeks (last 3)",
    "engagement_drop_ratio": "Engagement decline ratio", "clicks_pct": "Activity percentile in cohort",
    "active_days_pct": "Active-days percentile in cohort", "code_module": "Module",
    "tma_mean_score": "Tutor-marked assignment average", "cma_mean_score": "Computer-marked assessment average",
    "tma_due": "Tutor-marked assignments due", "cma_due": "Computer-marked assessments due",
    "tma_missed": "Tutor-marked assignments missed", "cma_missed": "Computer-marked assessments missed",
    "planned_materials_due": "Scheduled materials so far", "planned_coverage": "Share of scheduled materials opened",
    "gender": "Gender", "region": "Region", "highest_education": "Prior education", "imd_band": "Deprivation band",
    "age_band": "Age band", "disability": "Disability",
}


class FeatureEncoder:
    """Deterministic encoder: numeric passthrough (NaN preserved for native GBM missing handling)
    and one-hot categoricals over *fixed* OULAD vocabularies (never learned from data)."""

    def __init__(self, include_sensitive: bool = False, numeric: list[str] | None = None,
                 include_context: bool = True):
        self.include_sensitive = include_sensitive
        self.numeric = list(numeric or NUMERIC)
        self.categorical = dict(CONTEXT_CATEGORICAL) if include_context else {}
        if include_sensitive:
            self.categorical.update(SENSITIVE_CATEGORICAL)
        self.feature_names: list[str] = list(self.numeric)
        self.groups: list[str] = list(self.numeric)
        for col, values in self.categorical.items():
            for v in values:
                self.feature_names.append(f"{col}={v}")
                self.groups.append(col)
        self.group_names = list(dict.fromkeys(self.groups))
        self.group_index = {g: [i for i, x in enumerate(self.groups) if x == g] for g in self.group_names}

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        X = np.full((n, len(self.feature_names)), np.nan, dtype=np.float32)
        for j, col in enumerate(self.numeric):
            if col in df:
                X[:, j] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
        j = len(self.numeric)
        for col, values in self.categorical.items():
            series = df[col].astype(str) if col in df else pd.Series(["Unknown"] * n)
            if col == "imd_band":
                series = series.map(vocab.normalise_imd)
            for v in values:
                X[:, j] = (series.to_numpy() == v).astype(np.float32)
                j += 1
        return X

    def to_dict(self) -> dict:
        return {"include_sensitive": self.include_sensitive, "numeric": self.numeric,
                "categorical": {k: list(v) for k, v in self.categorical.items()}}
