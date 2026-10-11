"""Evidence layer shared by all agents.

* ``observations()`` turns a snapshot into a list of factual, ID-tagged statements. Agents (LLM
  or deterministic) may only *select and narrate* these; they never compute facts themselves.
* ``ACTION_CATALOG`` is the closed set of supportive actions. Each action has an eligibility
  predicate on the facts, so the intervention agent cannot recommend, e.g., a catch-up plan for
  a student who has missed nothing. Acceptance/effectiveness statistics learned by the
  feedback loop re-order eligible actions (``artifacts/feedback/action_stats.json``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ssa.config import Settings, get_settings
from ssa.utils import read_json


def _num(p, k, default=None):
    v = p.get(k, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


@dataclass
class Facts:
    tier: str
    no_scores: bool
    score: float | None
    score_pct: float | None
    low_scores: bool
    declining_scores: bool
    missed: int
    due: int
    missed_any: bool
    late_rate: float | None
    late_often: bool
    clicks_pct: float | None
    low_engagement: bool
    momentum: float
    engagement_drop: bool
    days_inactive: float
    recently_inactive: bool
    long_streak: bool
    repeat_attempt: bool
    heavy_load: bool
    low_forum: bool
    irregular: bool


def derive_facts(p: dict, tier: str) -> Facts:
    score = _num(p, "weighted_score")
    if score is None:
        score = _num(p, "mean_score")
    score_pct = _num(p, "score_pct")
    missed = int(_num(p, "missed_count", 0) or 0)
    due = int(_num(p, "assessments_due", 0) or 0)
    late = _num(p, "late_rate")
    clicks_pct = _num(p, "clicks_pct")
    mom = _num(p, "engagement_momentum", 0.0) or 0.0
    mom_pct = _num(p, "momentum_pct")
    days_inactive = _num(p, "days_since_last_activity", 0.0) or 0.0
    no_scores = bool(int(_num(p, "no_scores_yet", 0) or 0)) or score is None
    return Facts(
        tier=tier, no_scores=no_scores, score=score, score_pct=score_pct,
        low_scores=(not no_scores) and ((score is not None and score < 55) or (score_pct is not None and score_pct < 0.25)),
        declining_scores=(_num(p, "score_slope", 0) or 0) < -3,
        missed=missed, due=due, missed_any=missed > 0, late_rate=late, late_often=(late or 0) >= 0.5,
        clicks_pct=clicks_pct, low_engagement=(clicks_pct is not None and clicks_pct < 0.25) or bool(int(_num(p, "no_vle_activity", 0) or 0)),
        # Activity naturally tapers during a course, so a "drop" is judged relative to the cohort.
        momentum=mom, engagement_drop=(mom < -0.3 and mom_pct is not None and mom_pct < 0.25) or (mom_pct is None and mom < -0.6),
        days_inactive=days_inactive, recently_inactive=days_inactive >= 14 or (_num(p, "inactive_weeks_last3", 0) or 0) >= 2,
        long_streak=(_num(p, "longest_inactive_streak", 0) or 0) >= 3,
        repeat_attempt=(_num(p, "num_of_prev_attempts", 0) or 0) > 0,
        heavy_load=(_num(p, "studied_credits", 0) or 0) >= 120,
        low_forum=(_num(p, "share_forum", 0) or 0) < 0.02 and (_num(p, "total_clicks", 0) or 0) > 0,
        irregular=(_num(p, "engagement_cv", 0) or 0) > 1.2,
    )


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pct(x):
    return f"{_ordinal(int(round(100 * x)))} percentile" if x is not None else "an unknown percentile"


def observations(p: dict, f: Facts) -> list[dict]:
    obs = []

    def add(oid, kind, text, field, value):
        obs.append({"id": oid, "kind": kind, "text": text, "field": field, "value": value})

    if f.no_scores:
        add("OBS_NO_SCORES", "concern" if f.due else "neutral",
            "No assessment has been scored yet" + (f" although {f.due} were due." if f.due else "."), "no_scores_yet", 1)
    else:
        kind = "concern" if f.low_scores else "strength" if (f.score_pct or 0) >= 0.6 else "neutral"
        add("OBS_SCORE", kind, f"Weighted assessment score is {f.score:.1f}/100 ({_pct(f.score_pct)} of the cohort).",
            "weighted_score", round(f.score, 1))
        if f.declining_scores:
            add("OBS_SCORE_TREND", "concern", "Assessment scores have been falling across recent submissions.", "score_slope",
                round(_num(p, "score_slope", 0), 1))
    if f.due:
        if f.missed_any:
            add("OBS_MISSED", "concern", f"{f.missed} of {f.due} assessments due so far have not been submitted.", "missed_count", f.missed)
        else:
            add("OBS_ALL_SUBMITTED", "strength", f"All {f.due} assessments due so far have been submitted.", "missed_count", 0)
    if f.late_rate is not None and f.late_rate > 0:
        add("OBS_LATE", "concern" if f.late_often else "neutral",
            f"{round(100 * f.late_rate):d}% of submissions were after the deadline.", "late_rate", round(f.late_rate, 2))
    if f.clicks_pct is not None:
        kind = "concern" if f.low_engagement else "strength" if f.clicks_pct >= 0.6 else "neutral"
        add("OBS_ENGAGEMENT", kind, f"Online learning activity is at the {_pct(f.clicks_pct)} of the cohort.", "clicks_pct",
            round(f.clicks_pct, 2))
    if f.engagement_drop:
        add("OBS_DECLINE", "concern", "Online activity has fallen more steeply over recent weeks than for most of the cohort.",
            "engagement_momentum", round(f.momentum, 2))
    if f.recently_inactive:
        add("OBS_INACTIVE", "concern", f"No online learning activity for {int(f.days_inactive)} days.", "days_since_last_activity",
            int(f.days_inactive))
    if f.long_streak:
        add("OBS_STREAK", "concern", f"Longest run without any activity: {int(_num(p, 'longest_inactive_streak', 0))} weeks.",
            "longest_inactive_streak", int(_num(p, "longest_inactive_streak", 0)))
    if f.repeat_attempt:
        add("OBS_REPEAT", "neutral", f"This is attempt number {int(_num(p, 'num_of_prev_attempts', 0)) + 1} at this module.",
            "num_of_prev_attempts", int(_num(p, "num_of_prev_attempts", 0)))
    if f.heavy_load:
        add("OBS_LOAD", "neutral", f"Studying {int(_num(p, 'studied_credits', 0))} credits this term.", "studied_credits",
            int(_num(p, "studied_credits", 0)))
    return obs


@dataclass
class Action:
    id: str
    text: str
    channel: str
    eligible: Callable[[Facts], bool]
    addresses: str


ACTION_CATALOG: list[Action] = [
    Action("ACT_TUTOR_REVIEW", "Offer a one-to-one tutor session to go through recent assessment feedback", "Course tutor",
           lambda f: f.low_scores or f.declining_scores, "assessment performance"),
    Action("ACT_ASSESSMENT_PLAN", "Agree a catch-up plan for missed work, including extension options where policy allows",
           "Course tutor", lambda f: f.missed_any or (f.no_scores and f.due > 0), "missed assessments"),
    Action("ACT_CHECKIN", "Make a friendly, low-pressure check-in to ask how the module is going", "Student support team",
           lambda f: f.engagement_drop or f.recently_inactive or f.tier == "High", "engagement decline"),
    Action("ACT_STUDY_SCHEDULE", "Co-create a realistic weekly study schedule around the module's key activities", "Academic advisor",
           lambda f: f.low_engagement or f.irregular or f.long_streak, "low or irregular engagement"),
    Action("ACT_WORKLOAD_REVIEW", "Talk through the current study load and the options available if it feels unmanageable",
           "Academic advisor", lambda f: f.heavy_load and (f.tier != "Low" or f.missed_any), "study load"),
    Action("ACT_REPEAT_SUPPORT", "Offer targeted support on the parts of the module that were hardest last time", "Course tutor",
           lambda f: f.repeat_attempt, "previous attempt"),
    Action("ACT_PEER_SUPPORT", "Signpost study groups and the module forum for peer support", "Learning community",
           lambda f: f.low_forum and f.tier != "Low", "isolation from peers"),
    Action("ACT_STUDY_SKILLS", "Share study-skills resources on planning assignments and managing time", "Study skills service",
           lambda f: f.late_often or f.missed_any, "deadline management"),
    Action("ACT_WELLBEING_SIGNPOST", "Mention the confidential student support services that are available, without assuming any cause",
           "Student support team", lambda f: f.tier == "High", "general wellbeing (signposting only)"),
    Action("ACT_ENCOURAGE", "Send encouraging feedback that acknowledges the progress made so far", "Course tutor",
           lambda f: f.tier == "Low" or (f.score_pct or 0) >= 0.6, "motivation"),
    Action("ACT_MONITOR", "Keep under routine review and re-check after the next assessment", "Course team", lambda f: True,
           "monitoring"),
]
ACTIONS_BY_ID = {a.id: a for a in ACTION_CATALOG}


def action_stats(settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    return read_json(s.action_stats_path, {}).get("actions", {})


def eligible_actions(f: Facts, settings: Settings | None = None) -> list[dict]:
    stats = action_stats(settings)
    out = []
    for a in ACTION_CATALOG:
        if not a.eligible(f):
            continue
        st = stats.get(a.id, {})
        out.append({"id": a.id, "text": a.text, "channel": a.channel, "addresses": a.addresses,
                    "advisor_acceptance": st.get("acceptance_posterior_mean"), "feedback_n": st.get("n", 0),
                    "demoted": bool(st.get("demoted", False))})
    # Feedback loop: demoted actions sink; then by posterior acceptance (unknown = prior 0.5).
    out.sort(key=lambda a: (a["demoted"], -(a["advisor_acceptance"] if a["advisor_acceptance"] is not None else 0.5)))
    return out
