"""Labelled plain-English case-search queries, shared by the search-understanding evaluation (did the parser
extract the right filters?) and the typed-search retrieval benchmark (did search return the right cases?).

Every query carries ``expected`` filters and ``descriptors``: behaviour signals a matching case must show. Each
descriptor entry is a set of acceptable index tokens (any one counts), because a phrase such as "low scores"
legitimately covers more than one score band.
"""
from __future__ import annotations

import random

from ssa.data import vocab

BEHAVIOURS = {  # templated behaviour phrase -> acceptable descriptor tokens
    "with declining engagement": ["engagement_declining"],
    "with low scores": ["score_low", "score_failing"],
    "who missed assessments": ["missed_some", "missed_most"],
    "who submit late": ["late_often", "late_sometimes"],
    "": None,
}


def labelled_queries(n: int = 60, seed: int = 5) -> list[dict]:
    rng = random.Random(seed)
    out = []
    regions = {r: r.replace(" Region", "") for r in vocab.REGIONS}
    for _ in range(n):
        f, parts = {}, []
        if rng.random() < 0.7:
            m = rng.choice(vocab.MODULES)
            f["code_module"] = m
            parts.append(f"in {m}")
        if rng.random() < 0.35:
            r = rng.choice(list(regions))
            f["region"] = r
            parts.append(f"from {regions[r]}")
        if rng.random() < 0.3:
            f["num_of_prev_attempts"] = {"min": 1}
            parts.append("who are repeating the module")
        if rng.random() < 0.3:
            c = rng.choice([60, 90, 120])
            f["studied_credits"] = {"min": float(c + 1)}
            parts.append(f"studying more than {c} credits")
        if rng.random() < 0.4:
            o, phrase = rng.choice([("Withdrawn", "who dropped out"), ("Fail", "who failed"), ("Pass", "who passed")])
            f["final_result"] = o
            parts.append(phrase)
        behaviour = rng.choice(["with declining engagement", "with low scores", "who missed assessments", "who submit late", ""])
        rng.shuffle(parts)
        desc = BEHAVIOURS[behaviour]
        out.append({"query": ("students " + " ".join(parts) + " " + behaviour).strip(), "expected": f,
                    "descriptors": [desc] if desc else [], "kind": "templated"})
    return out


PARAPHRASES = [  # hand-written, phrased differently from the rule patterns
    {"query": "learners on module DDD who quit partway and were juggling a big credit load (over 120)",
     "expected": {"code_module": "DDD", "final_result": "Withdrawn", "studied_credits": {"min": 121.0}}},
    {"query": "people retaking GGG that went quiet online", "expected": {"code_module": "GGG", "num_of_prev_attempts": {"min": 1}},
     "descriptors": [["recently_inactive", "engagement_declining", "long_inactive_streak", "engagement_very_low", "engagement_low"]]},
    {"query": "Scottish students in CCC who did not make it to the end", "expected": {"code_module": "CCC", "region": "Scotland",
                                                                                         "final_result": "Withdrawn"}},
    {"query": "students in the Welsh cohort of EEE who ended up failing", "expected": {"code_module": "EEE", "region": "Wales",
                                                                                         "final_result": "Fail"}},
    {"query": "BBB folks attempting it for the first time who got a distinction", "expected": {"code_module": "BBB",
                                                                                                 "num_of_prev_attempts": {"max": 0},
                                                                                                 "final_result": "Distinction"}},
    {"query": "anyone in FFF with a postgraduate background who struggled and left",
     "expected": {"code_module": "FFF", "highest_education": "Post Graduate Qualification", "final_result": "Withdrawn"}},
]

# Behaviour-led paraphrases that avoid the rule parser's phrase list: the job the dense (meaning) channel exists for.
# Written before any benchmark run; labels are the descriptor bands a human would accept.
BEHAVIOUR_PARAPHRASES = [
    {"query": "students who stopped turning in their coursework", "expected": {},
     "descriptors": [["missed_most", "missed_some"]]},
    {"query": "learners in DDD whose marks kept slipping", "expected": {"code_module": "DDD"},
     "descriptors": [["scores_declining"]]},
    {"query": "people who barely touched the online materials and later dropped the course",
     "expected": {"final_result": "Withdrawn"}, "descriptors": [["engagement_very_low", "engagement_low", "no_online_activity"]]},
    {"query": "students in FFF who kept handing work in after the deadline", "expected": {"code_module": "FFF"},
     "descriptors": [["late_often", "late_sometimes"]]},
    {"query": "learners who were logging in less and less each week", "expected": {},
     "descriptors": [["engagement_declining"]]},
    {"query": "BBB students who disappeared from the virtual learning environment for a month", "expected": {"code_module": "BBB"},
     "descriptors": [["long_inactive_streak", "recently_inactive"]]},
    {"query": "students getting really poor marks on their tutor-marked assignments", "expected": {},
     "descriptors": [["score_failing", "score_low"]]},
    {"query": "students in CCC who took on a huge study load and failed", "expected": {"code_module": "CCC", "final_result": "Fail"},
     "descriptors": [["credits_heavy", "credits_very_heavy"]]},
]


def text_search_queries() -> list[dict]:
    """The ~40-query typed-search benchmark set: 26 templated + 6 paraphrases + 8 behaviour paraphrases."""
    para = [{"descriptors": [], **p, "kind": "paraphrase"} for p in PARAPHRASES]
    beh = [{**p, "kind": "behaviour_paraphrase"} for p in BEHAVIOUR_PARAPHRASES]
    return labelled_queries(26, seed=5) + para + beh


# Tuning-only natural-wording queries (written after the first benchmark run showed a templated-only tuning set cannot
# see what the dense channel is for). Disjoint from, and worded differently to, the reported set above.
TUNING_PARAPHRASES = [
    {"query": "students who never handed in most of their assignments", "expected": {}, "descriptors": [["missed_most", "missed_some"]]},
    {"query": "learners in EEE whose grades went downhill", "expected": {"code_module": "EEE"}, "descriptors": [["scores_declining"]]},
    {"query": "people who hardly ever opened the course website", "expected": {},
     "descriptors": [["engagement_very_low", "engagement_low", "no_online_activity"]]},
    {"query": "students in GGG who were always behind with deadlines", "expected": {"code_module": "GGG"},
     "descriptors": [["late_often", "late_sometimes"]]},
    {"query": "learners whose online activity tailed off and who then left", "expected": {"final_result": "Withdrawn"},
     "descriptors": [["engagement_declining"]]},
    {"query": "students in AAA who vanished from the course site for several weeks", "expected": {"code_module": "AAA"},
     "descriptors": [["long_inactive_streak", "recently_inactive"]]},
    {"query": "people scoring well under the pass mark on their assignments", "expected": {},
     "descriptors": [["score_failing", "score_low"]]},
    {"query": "DDD learners carrying an unusually large number of credits", "expected": {"code_module": "DDD"},
     "descriptors": [["credits_heavy", "credits_very_heavy"]]},
    {"query": "students who were doing brilliantly on their marked work", "expected": {}, "descriptors": [["score_excellent", "score_good"]]},
    {"query": "learners in BBB who were very busy on the online platform", "expected": {"code_module": "BBB"},
     "descriptors": [["engagement_high", "engagement_very_high"]]},
]


def text_tuning_queries() -> list[dict]:
    """Queries for tuning text fusion weights, never used for the reported numbers: templated queries from a
    different seed plus the tuning-only natural-wording paraphrases."""
    return labelled_queries(30, seed=101) + [{**p, "kind": "tuning_paraphrase"} for p in TUNING_PARAPHRASES]
