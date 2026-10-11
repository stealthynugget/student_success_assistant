"""Synthetic OULAD-shaped data for tests / CI only (written with a ``.synthetic_fixture`` marker).

A small latent-variable simulator (ability, motivation, withdrawal hazard) produces the seven
OULAD CSVs with realistic structure: withdrawals before start / before the snapshot / after it,
declining activity before withdrawal, late and missed submissions, banked results, the
``10-20`` IMD quirk and missing scores. Metrics on this data are never official evidence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ssa.data import vocab

PRESENTATIONS = [("2013B", 240), ("2013J", 268), ("2014B", 241), ("2014J", 269)]
MODULES = ["AAA", "BBB"]
ACT_TYPES = ["oucontent", "resource", "forumng", "quiz", "homepage", "subpage", "url"]


def _sig(x):
    return 1 / (1 + np.exp(-x))


def write_synthetic_oulad(out: Path, n_per_presentation: int = 150, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    courses, assessments, vle, info, reg, sa, sv = [], [], [], [], [], [], []
    aid, site, sid = 1000, 500000, 10000
    for module in MODULES:
        for pres, length in PRESENTATIONS:
            courses.append([module, pres, length])
            ass_rows = []
            for k, frac in enumerate([0.12, 0.30, 0.50, 0.72]):
                aid += 1
                ass_rows.append([module, pres, aid, "TMA", int(length * frac), 25])
            for frac in (0.20, 0.45):
                aid += 1
                ass_rows.append([module, pres, aid, "CMA", int(length * frac), 0])
            aid += 1
            ass_rows.append([module, pres, aid, "Exam", None, 100])
            assessments += ass_rows
            sites = []
            for t in ACT_TYPES:
                for _ in range(3):
                    site += 1
                    sites.append((site, t))
                    vle.append([site, module, pres, t, None, None])
            site_ids = np.array([s for s, _ in sites])
            n = n_per_presentation
            ability = rng.normal(0, 1, n)
            motiv = rng.normal(0, 1, n)
            prev = rng.choice([0, 0, 0, 0, 1, 2], n)
            credits = rng.choice([30, 60, 60, 60, 90, 120, 150], n)
            hazard = _sig(-1.3 - 1.1 * motiv - 0.4 * ability + 0.35 * prev + 0.004 * (credits - 60))
            withdraw = rng.random(n) < hazard
            w_day = np.where(withdraw, rng.uniform(-40, length * 0.85, n).round(), np.nan)
            reg_day = rng.integers(-150, 10, n)
            for i in range(n):
                sid += 1
                g = rng.choice(vocab.GENDERS)
                region = rng.choice(vocab.REGIONS)
                edu = rng.choice(vocab.EDUCATION, p=[0.02, 0.38, 0.42, 0.16, 0.02])
                imd = rng.choice(list(vocab.IMD_BANDS[:1]) + ["10-20"] + list(vocab.IMD_BANDS[2:]) + [None])
                age = rng.choice(vocab.AGE_BANDS, p=[0.7, 0.28, 0.02])
                dis = rng.choice(vocab.DISABILITY, p=[0.9, 0.1])
                end_day = w_day[i] if withdraw[i] else length
                # VLE: weekly activity with decay before withdrawal
                base = np.exp(2.3 + 0.55 * motiv[i] + 0.15 * ability[i])
                for day in range(-14, int(min(end_day, length))):
                    if rng.random() > _sig(0.2 + 0.8 * motiv[i]) * 0.55:
                        continue
                    decay = 1.0
                    if withdraw[i] and end_day - day < 35:
                        decay = max(0.05, (end_day - day) / 35)
                    clicks = rng.poisson(base * decay / 2) + 1
                    s_idx = rng.integers(0, len(site_ids))
                    sv.append([module, pres, sid, int(site_ids[s_idx]), day, int(clicks)])
                # assessments
                late_prop = _sig(-1.5 - 0.9 * motiv[i])
                missed_pen = 0
                for a in ass_rows:
                    if a[3] == "Exam":
                        if not withdraw[i] and rng.random() < 0.9:
                            continue
                        continue
                    due = a[4]
                    if due >= end_day:
                        continue
                    if rng.random() > _sig(1.8 + 1.2 * motiv[i] + 0.3 * ability[i]):
                        missed_pen += 1
                        continue
                    late = rng.random() < late_prop
                    submitted = due + (rng.integers(1, 10) if late else -rng.integers(0, 6))
                    score = float(np.clip(rng.normal(66 + 13 * ability[i] + 4 * motiv[i], 10), 0, 100))
                    banked = int(prev[i] > 0 and rng.random() < 0.05)
                    sa.append([a[2], sid, int(submitted), banked, None if rng.random() < 0.004 else round(score)])
                if withdraw[i]:
                    result = "Withdrawn"
                else:
                    perf = 66 + 13 * ability[i] + 4 * motiv[i] - 8 * missed_pen + rng.normal(0, 8)
                    result = "Distinction" if perf > 84 else "Fail" if perf < 50 else "Pass"
                info.append([module, pres, sid, g, region, edu, imd, age, int(prev[i]), int(credits[i]), dis, result])
                reg.append([module, pres, sid, int(reg_day[i]), None if not withdraw[i] else int(w_day[i])])
    pd.DataFrame(courses, columns=["code_module", "code_presentation", "module_presentation_length"]).to_csv(out / "courses.csv", index=False)
    pd.DataFrame(assessments, columns=["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"]).to_csv(
        out / "assessments.csv", index=False)
    pd.DataFrame(vle, columns=["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"]).to_csv(
        out / "vle.csv", index=False)
    pd.DataFrame(info, columns=["code_module", "code_presentation", "id_student", "gender", "region", "highest_education", "imd_band",
                                "age_band", "num_of_prev_attempts", "studied_credits", "disability", "final_result"]).to_csv(
        out / "studentInfo.csv", index=False)
    pd.DataFrame(reg, columns=["code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"]).to_csv(
        out / "studentRegistration.csv", index=False)
    pd.DataFrame(sa, columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]).to_csv(
        out / "studentAssessment.csv", index=False)
    pd.DataFrame(sv, columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]).to_csv(
        out / "studentVle.csv", index=False)
    (out / ".synthetic_fixture").write_text("Synthetic OULAD-shaped fixture (ssa.data.synthetic). Never report metrics as official.\n")
