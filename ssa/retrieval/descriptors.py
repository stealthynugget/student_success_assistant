"""Turn numeric learning records into *meaningful* text for lexical and semantic retrieval.

v4 embedded strings such as "Average score 45.2; VLE clicks 312" with MiniLM and matched them
with BM25. Neither understands numeric proximity ("45" vs "46" share no token), so retrieval
scored at the level of random selection. Here numbers never reach the text channels:
features are discretised into cohort-relative bands ("score_low", "engagement_declining") that
BM25 can match exactly and that a sentence encoder can relate semantically, while numeric
similarity is handled by the behavioural-vector channel.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _band(v, cuts, labels, missing="unknown"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return missing
    for c, lab in zip(cuts, labels):
        if v < c:
            return lab
    return labels[-1]


def descriptor_tokens(r: dict, include_outcome: bool = True) -> list[str]:
    t: list[str] = []
    t.append(f"module_{str(r.get('code_module', '')).lower()}")
    prev = int(r.get("num_of_prev_attempts") or 0)
    t.append("first_attempt" if prev == 0 else "repeat_attempt")
    credits = float(r.get("studied_credits") or 0)
    t.append(_band(credits, [60, 90, 120], ["credits_light", "credits_standard", "credits_heavy", "credits_very_heavy"]))
    if int(r.get("late_registration") or 0):
        t.append("registered_late")
    elif float(r.get("registration_days_before_start") or 0) > 90:
        t.append("registered_early")

    if int(r.get("no_scores_yet") or 0):
        t.append("no_scores_yet")
    else:
        t.append(_band(r.get("weighted_score"), [40, 55, 70, 85], ["score_failing", "score_low", "score_moderate", "score_good", "score_excellent"]))
        t.append(_band(r.get("score_pct"), [0.25, 0.5, 0.75], ["score_bottom_quartile", "score_below_median", "score_above_median", "score_top_quartile"]))
        slope = r.get("score_slope")
        if slope is not None and not (isinstance(slope, float) and np.isnan(slope)):
            t.append("scores_declining" if slope < -3 else "scores_improving" if slope > 3 else "scores_steady")
    missed = float(r.get("missed_rate") or 0)
    t.append("missed_none" if missed == 0 else "missed_some" if missed < 0.5 else "missed_most")
    late = r.get("late_rate")
    if late is not None and not (isinstance(late, float) and np.isnan(late)):
        t.append("late_rarely" if late < 0.1 else "late_sometimes" if late < 0.5 else "late_often")

    if int(r.get("no_vle_activity") or 0):
        t.append("no_online_activity")
    t.append(_band(r.get("clicks_pct"), [0.2, 0.4, 0.6, 0.8], ["engagement_very_low", "engagement_low", "engagement_average", "engagement_high", "engagement_very_high"]))
    mom = float(r.get("engagement_momentum") or 0)
    t.append("engagement_declining" if mom < -0.3 else "engagement_rising" if mom > 0.3 else "engagement_stable")
    if int(r.get("inactive_weeks_last3") or 0) >= 2:
        t.append("recently_inactive")
    if float(r.get("longest_inactive_streak") or 0) >= 3:
        t.append("long_inactive_streak")
    if float(r.get("share_forum") or 0) > 0.15:
        t.append("forum_active")
    if float(r.get("share_quiz") or 0) > 0.15:
        t.append("quiz_active")
    if include_outcome and r.get("final_result"):
        t.append(f"outcome_{str(r['final_result']).lower()}")
    return t


_PHRASES = {
    "score_failing": ["failing", "fail grade", "very low score", "very poor score"],
    "score_low": ["low score", "low scores", "poor score", "poor grades", "struggling academically", "weak scores", "low marks"],
    "score_good": ["good score", "good grades", "strong scores"],
    "score_excellent": ["excellent", "top scores", "high scores", "distinction level"],
    "score_bottom_quartile": ["bottom of the class", "lowest scores", "bottom quartile"],
    "scores_declining": ["scores declining", "grades dropping", "falling grades", "declining scores", "scores dropping"],
    "scores_improving": ["improving scores", "grades improving", "scores rising"],
    "missed_some": ["missed assessment", "missed an assessment", "missing assignment", "missed deadline", "non submission"],
    "missed_most": ["missed most", "missing most", "not submitting", "stopped submitting", "missed assessments"],
    "late_often": ["late submissions", "submits late", "often late", "late assignments"],
    "engagement_very_low": ["very low engagement", "barely engaged", "hardly logs in", "minimal activity", "disengaged"],
    "engagement_low": ["low engagement", "low activity", "few clicks", "rarely online", "low vle"],
    "engagement_high": ["high engagement", "very active", "highly engaged"],
    "engagement_declining": ["declining engagement", "engagement dropping", "dropping off", "less active", "losing interest",
                             "stopped engaging", "engagement decline", "decreasing activity", "fading"],
    "engagement_rising": ["increasing engagement", "more active", "engagement rising"],
    "recently_inactive": ["recently inactive", "not logged in", "inactive recently", "stopped logging in", "gone quiet"],
    "long_inactive_streak": ["inactive for weeks", "long break", "weeks without activity", "long inactivity"],
    "no_online_activity": ["no activity", "never logged in", "no online activity"],
    "repeat_attempt": ["repeat", "retake", "second attempt", "previous attempt", "resit", "repeating"],
    "first_attempt": ["first attempt", "first time"],
    "credits_heavy": ["heavy workload", "high credits", "many credits", "overloaded"],
    "registered_late": ["registered late", "late registration", "joined late"],
    "forum_active": ["forum", "discussion"],
    "quiz_active": ["quiz", "quizzes"],
    "outcome_withdrawn": ["withdrew", "withdrawn", "dropped out", "dropout", "drop out", "left the course"],
    "outcome_fail": ["failed", "who failed", "fail outcome"],
    "outcome_pass": ["passed", "who passed"],
    "outcome_distinction": ["distinction"],
}


def parse_query(text: str) -> list[str]:
    """Map a natural-language query to descriptor tokens (plus raw words for BM25 fall-through)."""
    q = " " + re.sub(r"[^a-z0-9% ]+", " ", text.lower()) + " "
    tokens = []
    for token, phrases in _PHRASES.items():
        if any(f" {p} " in q for p in phrases):
            tokens.append(token)
    for m in re.findall(r"\b(aaa|bbb|ccc|ddd|eee|fff|ggg)\b", q):
        tokens.append(f"module_{m}")
    words = re.findall(r"[a-z_]+", q)
    known = {t for t in _all_tokens()}
    tokens += [w for w in words if w in known]
    return list(dict.fromkeys(tokens))


def _all_tokens() -> set[str]:
    base = set(_PHRASES)
    base |= {"score_moderate", "score_above_median", "score_below_median", "score_top_quartile", "scores_steady",
             "missed_none", "late_rarely", "late_sometimes", "engagement_average", "engagement_very_high",
             "engagement_stable", "credits_light", "credits_standard", "credits_very_heavy", "registered_early",
             "no_scores_yet"}
    return base


_TEXT = {
    "first_attempt": "first attempt at the module", "repeat_attempt": "repeating the module after a previous attempt",
    "credits_light": "light study load", "credits_standard": "standard study load", "credits_heavy": "heavy study load",
    "credits_very_heavy": "very heavy study load", "registered_late": "registered after the course started",
    "registered_early": "registered well before the start", "no_scores_yet": "no assessment results yet",
    "score_failing": "failing assessment results", "score_low": "low assessment results", "score_moderate": "moderate assessment results",
    "score_good": "good assessment results", "score_excellent": "excellent assessment results",
    "score_bottom_quartile": "among the lowest scores in the cohort", "score_below_median": "below the cohort median",
    "score_above_median": "above the cohort median", "score_top_quartile": "among the best in the cohort",
    "scores_declining": "with grades getting worse", "scores_improving": "with grades improving", "scores_steady": "with steady grades",
    "missed_none": "has submitted every due assessment", "missed_some": "has missed some due assessments",
    "missed_most": "has missed most due assessments", "late_rarely": "rarely submits late", "late_sometimes": "sometimes submits late",
    "late_often": "often submits late", "no_online_activity": "no online learning activity at all",
    "engagement_very_low": "very little online activity compared with peers", "engagement_low": "below-average online activity",
    "engagement_average": "average online activity", "engagement_high": "above-average online activity",
    "engagement_very_high": "very high online activity", "engagement_declining": "online activity has been declining",
    "engagement_rising": "online activity has been increasing", "engagement_stable": "online activity is stable",
    "recently_inactive": "inactive in recent weeks", "long_inactive_streak": "had a long period without activity",
    "forum_active": "participates in forums", "quiz_active": "uses practice quizzes",
}


def narrative(r: dict, include_outcome: bool = True) -> str:
    toks = descriptor_tokens(r, include_outcome=False)
    parts = [_TEXT[t] for t in toks if t in _TEXT]
    text = f"Student on module {r.get('code_module')}: " + "; ".join(parts) + "."
    if include_outcome and r.get("final_result"):
        text += f" Final outcome: {str(r['final_result']).lower()}."
    return text


def build_corpus_text(df: pd.DataFrame, include_outcome: bool = True) -> tuple[list[list[str]], list[str]]:
    recs = df.to_dict(orient="records")
    return [descriptor_tokens(r, include_outcome) for r in recs], [narrative(r, include_outcome) for r in recs]
