"""OULAD -> per-enrollment early-warning snapshot.

Every predictor is computed only from information available on ``snapshot_day``
(= ``cutoff_fraction`` x planned presentation length). Key corrections vs. v4:

1. **Population scoping.** v4 kept students who had *already unregistered* by the
   snapshot (72.7 % of all withdrawals, 3,089 before day 0). The model then "predicted"
   withdrawal from the absence of activity of people who had already left, inflating
   ROC-AUC from ~0.70 to ~0.90. Each row now carries ``withdrawn_before_snapshot``;
   modelling, evaluation, retrieval and prioritisation use ``in_scope`` rows only.
   ``date_unregistration`` is used *only* to define that scope, never as a feature.
2. **Missing != zero.** Students with no scored assessment yet keep ``NaN`` scores plus a
   ``no_scores_yet`` flag (v4 silently filled 0, then "fillna(50)" was dead code).
3. **Dense weekly grid.** Engagement trends are computed over *every* week up to the
   snapshot, including zero-activity weeks (v4 only looked at weeks with activity, so a
   student who disappeared for six weeks looked "stable").
4. **Missed assessments.** Assessments due before the snapshot with no submission are an
   explicit, legitimate early signal.
5. **Peer-relative signals.** Percentiles within the same presentation (computed among
   in-scope peers) make clicks and scores comparable across modules.
6. **Memory.** ``studentVle.csv`` (10.6 M rows) is read with compact dtypes and joined via
   integer enrollment keys, so the full build runs in < 2 GB RAM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ssa.config import Settings, get_settings
from ssa.data import vocab
from ssa.utils import get_logger, write_json

log = get_logger("ssa.pipeline")

REQUIRED_FILES = (
    "assessments.csv", "courses.csv", "studentAssessment.csv", "studentInfo.csv",
    "studentRegistration.csv", "studentVle.csv", "vle.csv",
)
KEY_COLS = ["code_module", "code_presentation", "id_student"]
MAX_WEEKS = 40
FEATURE_VERSION = "5.0"

# Columns that must never be used as model inputs (labels, scope, identifiers, audit-only).
LABEL_COLUMNS = ["final_result", "withdrawn", "poor_outcome"]
SCOPE_COLUMNS = ["date_unregistration", "withdrawn_before_snapshot", "withdrawn_before_start", "in_scope"]
ID_COLUMNS = ["enrollment_key", "id_student", "code_module", "code_presentation"]


def validate_raw_data(raw_dir: Path | None = None) -> dict:
    s = get_settings()
    raw_dir = Path(raw_dir or s.raw_dir)
    missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
    synthetic = (raw_dir / ".synthetic_fixture").exists()
    return {"ready": not missing, "official": (not missing) and not synthetic, "missing": missing,
            "synthetic_fixture": synthetic, "directory": str(raw_dir)}


@dataclass
class BuildResult:
    features: pd.DataFrame
    weekly: np.ndarray
    weekly_keys: np.ndarray
    metadata: dict


def _ols_slope(n, sx, sy, sxy, sxx):
    denom = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where((n >= 2) & (denom > 0), (n * sxy - sx * sy) / denom, 0.0)
    return slope


def _read_inputs(raw: Path):
    info = pd.read_csv(raw / "studentInfo.csv")
    reg = pd.read_csv(raw / "studentRegistration.csv")
    courses = pd.read_csv(raw / "courses.csv")
    assessments = pd.read_csv(raw / "assessments.csv")
    sa = pd.read_csv(raw / "studentAssessment.csv", na_values=["?", ""])
    vle = pd.read_csv(raw / "vle.csv")
    for col in ("week_from", "week_to"):
        if col not in vle:
            vle[col] = np.nan
    return info, reg, courses, assessments, sa, vle


def _presentation_table(courses: pd.DataFrame, cutoff_fraction: float) -> pd.DataFrame:
    length_col = "module_presentation_length" if "module_presentation_length" in courses.columns else "length"
    if length_col not in courses.columns:
        raise ValueError("courses.csv must contain module_presentation_length")
    pres = courses[["code_module", "code_presentation", length_col]].rename(columns={length_col: "length"}).copy()
    pres["length"] = pd.to_numeric(pres["length"], errors="coerce").fillna(260).astype(int)
    pres = pres.sort_values(["code_module", "code_presentation"]).reset_index(drop=True)
    pres["pid"] = np.arange(len(pres), dtype=np.int64)
    pres["snapshot_day"] = np.maximum(7, np.round(pres["length"] * cutoff_fraction)).astype(int)
    pres["snapshot_week"] = (pres["snapshot_day"] // 7).astype(int)
    return pres


def _enrollment_key(pid: np.ndarray, id_student: np.ndarray) -> np.ndarray:
    return pid.astype(np.int64) * 10_000_000 + id_student.astype(np.int64)


def _assessment_features(base, pres, assessments, sa) -> pd.DataFrame:
    a = assessments.merge(pres[["code_module", "code_presentation", "pid", "snapshot_day"]],
                          on=["code_module", "code_presentation"], how="inner")
    a["date"] = pd.to_numeric(a["date"], errors="coerce")
    a = a[a["assessment_type"] != "Exam"].copy()
    a["due_by_snapshot"] = a["date"].notna() & (a["date"] <= a["snapshot_day"])
    due_counts = a[a["due_by_snapshot"]].groupby("pid").size().rename("assessments_due")

    due_by_type = a[a["due_by_snapshot"]].groupby(["pid", "assessment_type"]).size().unstack(fill_value=0)
    s = sa.merge(a[["id_assessment", "pid", "date", "weight", "snapshot_day", "due_by_snapshot", "assessment_type"]],
                 on="id_assessment", how="inner")
    s["date_submitted"] = pd.to_numeric(s["date_submitted"], errors="coerce")
    s["score"] = pd.to_numeric(s["score"], errors="coerce")
    s = s[s["date_submitted"] <= s["snapshot_day"]].copy()  # known at snapshot
    s["enrollment_key"] = _enrollment_key(s["pid"].to_numpy(), s["id_student"].to_numpy())
    s = s[s["enrollment_key"].isin(base["enrollment_key"])]
    banked = s.get("is_banked", pd.Series(0, index=s.index)).fillna(0).astype(int) == 1
    s["banked"] = banked.astype(int)
    s["late"] = ((~banked) & s["date"].notna() & (s["date_submitted"] > s["date"])).astype(float)
    s["days_early"] = np.where(banked, np.nan, (s["date"] - s["date_submitted"]).clip(-30, 60))
    s["w"] = pd.to_numeric(s["weight"], errors="coerce").fillna(0.0)
    s["has_score"] = s["score"].notna()
    s["sw"] = np.where(s["has_score"], s["score"] * s["w"], 0.0)
    s["wv"] = np.where(s["has_score"], s["w"], 0.0)
    s = s.sort_values(["enrollment_key", "date_submitted"])
    s["order"] = s.groupby("enrollment_key").cumcount().astype(float)
    sc = s[s["has_score"]].copy()
    sc["xy"] = sc["order"] * sc["score"]
    sc["xx"] = sc["order"] ** 2

    g = s.groupby("enrollment_key")
    out = pd.DataFrame({
        "submitted_count": g.size(),
        "submitted_due_count": g["due_by_snapshot"].sum(),
        "late_rate": g["late"].mean(),
        "mean_days_early": g["days_early"].mean(),
        "banked_count": g["banked"].sum(),
        "sum_sw": g["sw"].sum(),
        "sum_w": g["wv"].sum(),
    })
    gs = sc.groupby("enrollment_key")
    score_stats = pd.DataFrame({
        "mean_score": gs["score"].mean(),
        "score_std": gs["score"].std(ddof=0),
        "min_score": gs["score"].min(),
        "max_score": gs["score"].max(),
        "last_score": gs["score"].last(),
        "_n": gs.size().astype(float), "_sx": gs["order"].sum(), "_sy": gs["score"].sum(),
        "_sxy": gs["xy"].sum(), "_sxx": gs["xx"].sum(),
    })
    score_stats["score_slope"] = _ols_slope(score_stats["_n"], score_stats["_sx"], score_stats["_sy"],
                                             score_stats["_sxy"], score_stats["_sxx"])
    out = out.join(score_stats.drop(columns=[c for c in score_stats.columns if c.startswith("_")]), how="left")
    out["weighted_score"] = np.where(out["sum_w"] > 0, out["sum_sw"] / out["sum_w"].replace(0, np.nan), out["mean_score"])
    out = out.drop(columns=["sum_sw", "sum_w"])

    # Tutor-marked (TMA) vs computer-marked (CMA) assessments behave differently: separate signals.
    for typ in ("TMA", "CMA"):
        st_ = sc[sc["assessment_type"] == typ].groupby("enrollment_key")["score"].mean().rename(f"{typ.lower()}_mean_score")
        sub_ = s[(s["assessment_type"] == typ) & s["due_by_snapshot"]].groupby("enrollment_key").size().rename(f"_{typ}_sub")
        out = out.join(st_, how="left").join(sub_, how="left")
    res = base[["enrollment_key", "pid"]].merge(out, left_on="enrollment_key", right_index=True, how="left")
    for typ in ("TMA", "CMA"):
        due = res["pid"].map(due_by_type[typ] if typ in due_by_type else pd.Series(dtype=float)).fillna(0)
        res[f"{typ.lower()}_due"] = due
        res[f"{typ.lower()}_missed"] = (due - res[f"_{typ}_sub"].fillna(0)).clip(lower=0)
        res = res.drop(columns=[f"_{typ}_sub"])
    res = res.merge(due_counts, left_on="pid", right_index=True, how="left")
    res["assessments_due"] = res["assessments_due"].fillna(0)
    for c in ["submitted_count", "submitted_due_count", "banked_count"]:
        res[c] = res[c].fillna(0)
    res["missed_count"] = (res["assessments_due"] - res["submitted_due_count"]).clip(lower=0)
    res["missed_rate"] = np.where(res["assessments_due"] > 0, res["missed_count"] / res["assessments_due"].replace(0, np.nan), 0.0)
    res["no_scores_yet"] = res["mean_score"].isna().astype(int)
    return res.drop(columns=["pid"])


COURSE_WEEKS = 40


def _vle_features(base, pres, vle, raw: Path):
    t0 = time.time()
    pres_codes = sorted(pres["code_presentation"].unique(), key=vocab.presentation_sort_key)
    mod_codes = sorted(pres["code_module"].unique())
    sv = pd.read_csv(
        raw / "studentVle.csv",
        usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        dtype={"code_module": pd.CategoricalDtype(mod_codes), "code_presentation": pd.CategoricalDtype(pres_codes),
               "id_student": np.int32, "id_site": np.int32, "date": np.float32, "sum_click": np.float32},
    )
    log.info(f"studentVle loaded: {len(sv):,} rows in {time.time() - t0:.1f}s")
    lut = np.full((len(mod_codes), len(pres_codes)), -1, dtype=np.int64)
    for r in pres.itertuples():
        lut[mod_codes.index(r.code_module), pres_codes.index(r.code_presentation)] = r.pid
    pid = lut[sv["code_module"].cat.codes.to_numpy(), sv["code_presentation"].cat.codes.to_numpy()]
    sv = sv.drop(columns=["code_module", "code_presentation"])
    sv["pid"] = pid
    sv = sv[sv["pid"] >= 0]
    sv["enrollment_key"] = _enrollment_key(sv["pid"].to_numpy(), sv["id_student"].to_numpy())
    snap = pres.set_index("pid")["snapshot_day"]
    sv["snapshot_day"] = sv["pid"].map(snap).astype(np.float32)

    # Full-course weekly engagement by outcome among students enrolled at the snapshot
    # (historical trend analysis; never a predictor). Pre-start activity is excluded.
    scoped = base.loc[base["in_scope"] == 1, ["enrollment_key", "pid", "final_result"]]
    outcome = scoped.set_index("enrollment_key")["final_result"]
    cur = sv[(sv["date"] >= 0) & sv["enrollment_key"].isin(outcome.index)]
    # Complete-course weekly matrix for every enrolment (used to tell precedents' full stories).
    keys_all = base["enrollment_key"].to_numpy()
    row_all = pd.Series(np.arange(len(keys_all)), index=keys_all)
    full = sv[sv["date"] >= 0]
    fw = full.assign(w=(full["date"] // 7).astype(np.int16)).groupby(["enrollment_key", "w"])["sum_click"].sum().reset_index()
    fw = fw[(fw["w"] < COURSE_WEEKS) & fw["enrollment_key"].isin(row_all.index)]
    course_weekly = np.zeros((len(keys_all), COURSE_WEEKS), dtype=np.float32)
    course_weekly[row_all.loc[fw["enrollment_key"]].to_numpy(), fw["w"].to_numpy()] = fw["sum_click"].to_numpy()
    course_len_w = (base["length"].to_numpy() // 7 + 1)
    course_weekly[np.arange(COURSE_WEEKS)[None, :] > course_len_w[:, None]] = np.nan
    last_course_day = full.groupby("enrollment_key")["date"].max()
    total_course_clicks = full.groupby("enrollment_key")["sum_click"].sum()
    del fw, full

    trend = cur.assign(week_full=(cur["date"] // 7).astype(np.int16)).groupby(
        ["pid", "week_full", "enrollment_key"], observed=True)["sum_click"].sum().reset_index()
    trend["final_result"] = trend["enrollment_key"].map(outcome)
    weekly_sum = trend.groupby(["pid", "week_full", "final_result"])["sum_click"].sum().reset_index()
    group_sizes = scoped.groupby(["pid", "final_result"]).size().rename("n_students").reset_index()
    course_trends = weekly_sum.merge(group_sizes, on=["pid", "final_result"], how="left")
    course_trends["mean_clicks_per_student"] = course_trends["sum_click"] / course_trends["n_students"]
    course_trends = course_trends.merge(pres[["pid", "code_module", "code_presentation", "snapshot_week"]], on="pid")
    del trend, cur

    sv = sv[sv["date"] <= sv["snapshot_day"]]  # snapshot boundary
    sv = sv[sv["enrollment_key"].isin(base["enrollment_key"])]
    # Planned-schedule coverage: share of materials scheduled (week_from) up to the snapshot week
    # that the student has opened. OULAD only schedules ~18% of materials, so this is often NaN.
    planned = vle.dropna(subset=["week_from"]).merge(pres[["code_module", "code_presentation", "pid", "snapshot_week"]],
                                                    on=["code_module", "code_presentation"])
    planned = planned[planned["week_from"] <= planned["snapshot_week"]]
    planned_n = planned.groupby("pid")["id_site"].nunique().rename("planned_materials_due")
    hit = sv[sv["id_site"].isin(planned["id_site"])].groupby("enrollment_key")["id_site"].nunique().rename("_planned_hit")
    act = vle.drop_duplicates("id_site").set_index("id_site")["activity_type"]
    sv["activity_type"] = sv["id_site"].map(act).fillna("unknown")
    group_of = {t: g for g, types in vocab.ACTIVITY_GROUPS.items() for t in types}
    sv["agroup"] = sv["activity_type"].map(group_of).fillna("other")

    g = sv.groupby("enrollment_key")
    feats = pd.DataFrame({
        "total_clicks": g["sum_click"].sum(),
        "active_days": g["date"].nunique(),
        "resources_used": g["id_site"].nunique(),
        "activity_types_used": g["activity_type"].nunique(),
        "last_activity_day": g["date"].max(),
    })
    pre = sv[sv["date"] < 0].groupby("enrollment_key")["sum_click"].sum().rename("pre_start_clicks")
    feats = feats.join(pre, how="left").join(hit, how="left")
    by_group = sv.pivot_table(index="enrollment_key", columns="agroup", values="sum_click", aggfunc="sum", fill_value=0)
    for grp in list(vocab.ACTIVITY_GROUPS) + ["other"]:
        feats[f"clicks_{grp}"] = by_group[grp] if grp in by_group.columns else 0.0

    # Dense weekly grid (weeks 0..snapshot_week) including zero weeks.
    sv["week"] = (sv["date"].clip(lower=0) // 7).astype(np.int16)
    wk = sv.groupby(["enrollment_key", "week"])["sum_click"].sum().reset_index()
    keys = base["enrollment_key"].to_numpy()
    row_of = pd.Series(np.arange(len(keys)), index=keys)
    weekly = np.zeros((len(keys), MAX_WEEKS), dtype=np.float32)
    wk = wk[wk["week"] < MAX_WEEKS]
    weekly[row_of.loc[wk["enrollment_key"]].to_numpy(), wk["week"].to_numpy()] = wk["sum_click"].to_numpy()
    snapshot_week = base["snapshot_week"].to_numpy()
    valid = np.arange(MAX_WEEKS)[None, :] <= snapshot_week[:, None]
    weekly[~valid] = np.nan
    log.info(f"VLE features built in {time.time() - t0:.1f}s")
    course = {"weekly": course_weekly, "keys": keys_all, "last_course_day": last_course_day,
              "total_course_clicks": total_course_clicks, "planned_n": planned_n}
    return feats, weekly, keys, course_trends, course


def _weekly_trend_features(weekly: np.ndarray, snapshot_week: np.ndarray) -> pd.DataFrame:
    n_weeks = snapshot_week + 1
    w = np.nan_to_num(weekly, nan=0.0)
    valid = ~np.isnan(weekly)
    x = np.arange(weekly.shape[1], dtype=float)[None, :] * valid
    n = valid.sum(1).astype(float)
    sx, sy = x.sum(1), w.sum(1)
    sxy, sxx = (x * w).sum(1), (x * x).sum(1)
    slope = _ols_slope(n, sx, sy, sxy, sxx)
    mean = sy / np.maximum(n, 1)
    std = np.sqrt(np.maximum(((w - mean[:, None]) ** 2 * valid).sum(1) / np.maximum(n, 1), 0))
    idx = np.arange(len(w))
    early = (w[:, 0] + w[:, 1]) / 2.0
    last = n_weeks - 1
    recent = (w[idx, last] + w[idx, np.maximum(last - 1, 0)]) / 2.0
    active = (w > 0) & valid
    weeks_active = active.sum(1)
    # longest run of inactive weeks within the valid window
    longest = np.zeros(len(w), dtype=int)
    run = np.zeros(len(w), dtype=int)
    for j in range(w.shape[1]):
        inactive = valid[:, j] & ~active[:, j]
        run = np.where(inactive, run + 1, 0)
        longest = np.maximum(longest, run)
    last3 = np.zeros(len(w), dtype=int)
    for k in range(3):
        col = np.maximum(last - k, 0)
        last3 += (~active[idx, col]).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = np.where(early > 0, (early - recent) / early, 0.0)
    return pd.DataFrame({
        "weeks_observed": n,
        "weeks_active": weeks_active,
        "engagement_weekly_mean": mean,
        "engagement_slope": slope,
        "engagement_slope_rel": slope / (mean + 1.0),
        "engagement_early_mean": early,
        "engagement_recent_mean": recent,
        "engagement_momentum": (recent - early) / (early + 1.0),
        "engagement_cv": std / (mean + 1.0),
        "longest_inactive_streak": longest,
        "inactive_weeks_last3": last3,
        "engagement_drop_ratio": np.clip(drop, 0, 1),
    })


def build_features(raw_dir: Path | None = None, cutoff_fraction: float | None = None,
                   settings: Settings | None = None, write: bool = True) -> BuildResult:
    s = settings or get_settings()
    raw = Path(raw_dir or s.raw_dir)
    cutoff = float(cutoff_fraction if cutoff_fraction is not None else s.cutoff_fraction)
    if not 0.05 <= cutoff <= 0.95:
        raise ValueError("cutoff_fraction must be within [0.05, 0.95]")
    status = validate_raw_data(raw)
    if not status["ready"]:
        raise FileNotFoundError(f"Missing OULAD files in {raw}: {status['missing']}")
    t0 = time.time()
    info, reg, courses, assessments, sa, vle = _read_inputs(raw)
    pres = _presentation_table(courses, cutoff)

    base = info.merge(reg, on=KEY_COLS, how="left").merge(
        pres[["code_module", "code_presentation", "pid", "length", "snapshot_day", "snapshot_week"]],
        on=["code_module", "code_presentation"], how="inner")
    base["enrollment_key"] = _enrollment_key(base["pid"].to_numpy(), base["id_student"].to_numpy())
    base = base.drop_duplicates("enrollment_key").reset_index(drop=True)

    # Normalise categorical vocab
    base["imd_band"] = base["imd_band"].map(vocab.normalise_imd)
    for col, allowed in [("gender", vocab.GENDERS), ("region", vocab.REGIONS), ("highest_education", vocab.EDUCATION),
                         ("age_band", vocab.AGE_BANDS), ("disability", vocab.DISABILITY)]:
        if col not in base:
            base[col] = vocab.UNKNOWN
        base[col] = base[col].where(base[col].isin(allowed), vocab.UNKNOWN)

    # Registration (known at enrolment time)
    base["date_registration"] = pd.to_numeric(base["date_registration"], errors="coerce")
    base["registration_date_missing"] = base["date_registration"].isna().astype(int)
    base["registration_days_before_start"] = (-base["date_registration"]).fillna(0)
    base["late_registration"] = (base["date_registration"] > 0).astype(int)

    # Scope (NOT features)
    base["date_unregistration"] = pd.to_numeric(base["date_unregistration"], errors="coerce")
    base["withdrawn_before_snapshot"] = (base["date_unregistration"] <= base["snapshot_day"]).fillna(False).astype(int)
    base["withdrawn_before_start"] = (base["date_unregistration"] <= 0).fillna(False).astype(int)
    base["in_scope"] = (1 - base["withdrawn_before_snapshot"]).astype(int)

    # Labels (target-only)
    base["withdrawn"] = (base["final_result"] == "Withdrawn").astype(int)
    base["poor_outcome"] = base["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    ass = _assessment_features(base, pres, assessments, sa)
    vfe, weekly, keys, course_trends, course = _vle_features(base, pres, vle, raw)
    trend = _weekly_trend_features(weekly, base["snapshot_week"].to_numpy())

    out = base.merge(ass, on="enrollment_key", how="left")
    out = out.merge(vfe, left_on="enrollment_key", right_index=True, how="left")
    out = pd.concat([out.reset_index(drop=True), trend.reset_index(drop=True)], axis=1)

    count_cols = ["total_clicks", "active_days", "resources_used", "activity_types_used", "pre_start_clicks"] + \
                 [c for c in out.columns if c.startswith("clicks_")]
    for c in count_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["no_vle_activity"] = (out["total_clicks"] <= 0).astype(int)
    # Days since last in-course activity, bounded to [0, snapshot_day]; students with no
    # in-course activity get snapshot_day (and no_vle_activity distinguishes them).
    last_day = out["last_activity_day"].fillna(0).clip(lower=0)
    out["days_since_last_activity"] = (out["snapshot_day"] - last_day).clip(lower=0, upper=out["snapshot_day"])
    out = out.drop(columns=["last_activity_day"])
    out["clicks_per_active_day"] = out["total_clicks"] / out["active_days"].replace(0, np.nan)
    out["clicks_per_active_day"] = out["clicks_per_active_day"].fillna(0)
    for grp in list(vocab.ACTIVITY_GROUPS):
        out[f"share_{grp}"] = (out[f"clicks_{grp}"] / out["total_clicks"].replace(0, np.nan)).fillna(0)
    out["log_total_clicks"] = np.log1p(out["total_clicks"])
    out["planned_materials_due"] = out["presentation_id" if "presentation_id" in out else "pid"].map(course["planned_n"])
    out["planned_coverage"] = np.where(out["planned_materials_due"] > 0,
                                       out["_planned_hit"].fillna(0) / out["planned_materials_due"].replace(0, np.nan), np.nan)
    out = out.drop(columns=["_planned_hit"])

    # Peer-relative percentiles within presentation, among in-scope students (known at snapshot).
    scope = out["in_scope"] == 1
    for col, name in [("total_clicks", "clicks_pct"), ("weighted_score", "score_pct"), ("active_days", "active_days_pct"),
                      ("missed_rate", "missed_rate_pct"), ("engagement_momentum", "momentum_pct")]:
        out[name] = np.nan
        out.loc[scope, name] = out.loc[scope].groupby("pid")[col].rank(pct=True, method="average")

    out = out.rename(columns={"pid": "presentation_id"})
    meta = {
        "feature_version": FEATURE_VERSION,
        "cutoff_fraction": cutoff,
        "records": int(len(out)),
        "in_scope_records": int(out["in_scope"].sum()),
        "withdrawn_total": int(out["withdrawn"].sum()),
        "withdrawn_before_snapshot": int(out["withdrawn_before_snapshot"].sum()),
        "withdrawn_before_start": int(out["withdrawn_before_start"].sum()),
        "withdrawn_after_snapshot_in_scope": int(out.loc[scope, "withdrawn"].sum()),
        "presentations": sorted(out["code_presentation"].unique().tolist(), key=vocab.presentation_sort_key),
        "synthetic_fixture": status["synthetic_fixture"],
        "build_seconds": round(time.time() - t0, 1),
        "leakage_controls": [
            "assessment and VLE rows filtered to date <= snapshot_day",
            "students unregistered on/before snapshot_day excluded from modelling scope (in_scope=0)",
            "date_unregistration used for scoping only; never a feature",
            "final_result used as label only",
            "peer percentiles computed from in-scope peers at snapshot time",
        ],
    }
    if write:
        s.ensure_dirs()
        out.to_csv(s.features_path, index=False, compression="gzip")
        np.savez_compressed(s.weekly_path, keys=keys, weekly=weekly, snapshot_week=base["snapshot_week"].to_numpy())
        write_json(s.metrics_dir / "data_build.json", meta)
        if s.is_primary:
            course_trends.to_csv(s.course_trends_path, index=False)
            _write_course_level(s, base, out, assessments, sa, course, pres)
    log.info(f"features: {meta['records']:,} rows ({meta['in_scope_records']:,} in scope) in {meta['build_seconds']}s")
    return BuildResult(out, weekly, keys, meta)


def _write_course_level(s: Settings, base, out, assessments, sa, course, pres) -> None:
    """Complete-record artifacts (checkpoint-independent): full-course weekly activity, final coursework and
    exam results, withdrawal day, plus assessment-level analytics. Historical precedents use them to show
    what happened *after* the point the advisor is looking at."""
    a = assessments.copy()
    a["date"] = pd.to_numeric(a["date"], errors="coerce")
    s_all = sa.merge(a[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]], on="id_assessment")
    s_all = s_all.merge(base[["code_module", "code_presentation", "id_student", "enrollment_key"]],
                        on=["code_module", "code_presentation", "id_student"])
    s_all["score"] = pd.to_numeric(s_all["score"], errors="coerce")
    exam = s_all[s_all["assessment_type"] == "Exam"].groupby("enrollment_key")["score"].max().rename("exam_score")
    cw = s_all[(s_all["assessment_type"] != "Exam") & s_all["score"].notna()].copy()
    cw["sw"] = cw["score"] * cw["weight"]
    g = cw.groupby("enrollment_key")
    coursework = (g["sw"].sum() / g["weight"].sum().replace(0, np.nan)).fillna(g["score"].mean()).rename("final_coursework_score")
    det = base[["enrollment_key", "id_student", "code_module", "code_presentation", "final_result", "date_unregistration", "length"]].copy()
    det = det.join(exam, on="enrollment_key").join(coursework, on="enrollment_key")
    det = det.join(cw.groupby("enrollment_key").size().rename("submissions_total"), on="enrollment_key")
    det = det.join(course["total_course_clicks"].rename("total_course_clicks"), on="enrollment_key")
    det = det.join(course["last_course_day"].rename("last_course_day"), on="enrollment_key")
    det["withdrawal_week"] = (det["date_unregistration"] // 7).where(det["date_unregistration"].notna())
    det.to_csv(s.outcome_details_path, index=False, compression="gzip")
    np.savez_compressed(s.course_weekly_path, keys=course["keys"], weekly=course["weekly"])

    # Assessment analytics: which assessments are hard, missed, and most predictive of poor outcomes.
    from sklearn.metrics import roc_auc_score
    poor = base.set_index("enrollment_key")["poor_outcome"]
    unreg = base.set_index("enrollment_key")["date_unregistration"]
    rows = []
    for r in a.itertuples():
        enrolled = base[(base["code_module"] == r.code_module) & (base["code_presentation"] == r.code_presentation)]
        if pd.notna(r.date):
            enrolled = enrolled[~(unreg.reindex(enrolled["enrollment_key"]).to_numpy() <= r.date)]
        subs = s_all[s_all["id_assessment"] == r.id_assessment]
        subs = subs[subs["enrollment_key"].isin(enrolled["enrollment_key"])]
        n = len(enrolled)
        if n == 0:
            continue
        submitted = enrolled["enrollment_key"].isin(subs["enrollment_key"])
        y = poor.reindex(enrolled["enrollment_key"]).to_numpy()
        scored = subs.dropna(subset=["score"])
        yp = poor.reindex(scored["enrollment_key"]).to_numpy()
        late = (pd.to_numeric(subs["date_submitted"], errors="coerce") > r.date).mean() if pd.notna(r.date) and len(subs) else None
        rows.append({
            "id_assessment": int(r.id_assessment), "code_module": r.code_module, "code_presentation": r.code_presentation,
            "assessment_type": r.assessment_type, "deadline_day": None if pd.isna(r.date) else int(r.date),
            "deadline_week": None if pd.isna(r.date) else int(r.date // 7), "weight": float(r.weight),
            "enrolled_at_deadline": int(n), "submission_rate": float(submitted.mean()),
            "mean_score": float(scored["score"].mean()) if len(scored) else None,
            "fail_rate_below_40": float((scored["score"] < 40).mean()) if len(scored) else None,
            "late_rate": None if late is None else float(late),
            "non_submission_auc_for_poor_outcome": float(roc_auc_score(y, ~submitted.to_numpy())) if len(set(y)) > 1 and submitted.nunique() > 1 else None,
            "score_auc_for_poor_outcome": float(roc_auc_score(yp, -scored["score"].to_numpy())) if len(set(yp)) > 1 and len(scored) > 20 else None,
        })
    write_json(s.analysis_dir / "assessment_analytics.json", {
        "note": "Computed from complete historical records. AUC = how well missing / low-scoring this assessment separates "
                "students who later failed or withdrew (0.5 = no signal).", "assessments": rows})


def load_course_level(settings: Settings | None = None):
    s = settings or get_settings()
    det = pd.read_csv(s.outcome_details_path)
    z = np.load(s.course_weekly_path)
    return det, z["keys"], z["weekly"]


def load_features(settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    if not s.features_path.exists():
        raise FileNotFoundError(f"{s.features_path} missing. Run `python -m ssa.cli prepare` first.")
    return pd.read_csv(s.features_path, low_memory=False)


def load_weekly(settings: Settings | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = settings or get_settings()
    z = np.load(s.weekly_path)
    return z["keys"], z["weekly"], z["snapshot_week"]
