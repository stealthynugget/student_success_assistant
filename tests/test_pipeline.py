import numpy as np
import pandas as pd

from ssa.data.pipeline import LABEL_COLUMNS, SCOPE_COLUMNS, build_features
from ssa.data.synthetic import write_synthetic_oulad
from ssa.models.features import NUMERIC, FeatureEncoder


def _raw(tmp_path):
    raw = tmp_path / "raw"
    write_synthetic_oulad(raw, n_per_presentation=40, seed=3)
    return raw


def test_snapshot_ignores_information_after_cutoff(tmp_path, ssa_home):
    raw = _raw(tmp_path)
    before = build_features(raw_dir=raw, write=False).features.set_index("enrollment_key")
    sv = pd.read_csv(raw / "studentVle.csv")
    sa = pd.read_csv(raw / "studentAssessment.csv")
    victim = before[before["in_scope"] == 1].iloc[0]
    late_day = int(victim["snapshot_day"]) + 5
    sv = pd.concat([sv, pd.DataFrame([[victim["code_module"], victim["code_presentation"], int(victim["id_student"]),
                                       int(sv["id_site"].iloc[0]), late_day, 99999]], columns=sv.columns)])
    sv.to_csv(raw / "studentVle.csv", index=False)
    a = pd.read_csv(raw / "assessments.csv")
    a_id = int(a[(a.code_module == victim["code_module"]) & (a.code_presentation == victim["code_presentation"])].iloc[0]["id_assessment"])
    sa = pd.concat([sa, pd.DataFrame([[a_id, int(victim["id_student"]), late_day, 0, 1]], columns=sa.columns)])
    sa.to_csv(raw / "studentAssessment.csv", index=False)
    after = build_features(raw_dir=raw, write=False).features.set_index("enrollment_key")
    cols = [c for c in NUMERIC if c not in ("clicks_pct", "score_pct", "active_days_pct", "missed_rate_pct")]
    pd.testing.assert_series_equal(before.loc[victim.name, cols], after.loc[victim.name, cols], check_names=False)


def test_scope_flags_and_labels_are_not_features(tmp_path, ssa_home):
    f = build_features(raw_dir=_raw(tmp_path), write=False).features
    early = f["date_unregistration"] <= f["snapshot_day"]
    assert (f.loc[early, "in_scope"] == 0).all()
    assert (f.loc[~early, "in_scope"] == 1).all()
    enc = FeatureEncoder(include_sensitive=False)
    forbidden = set(LABEL_COLUMNS) | set(SCOPE_COLUMNS) | {"final_result", "gender", "region", "imd_band", "age_band",
                                                           "highest_education", "disability"}
    assert not forbidden & set(enc.groups)


def test_missing_scores_stay_missing(tmp_path, ssa_home):
    f = build_features(raw_dir=_raw(tmp_path), write=False).features
    none = f[f["no_scores_yet"] == 1]
    assert len(none) > 0
    assert none["weighted_score"].isna().all()
    assert (f.loc[f["no_scores_yet"] == 0, "weighted_score"].notna()).all()


def test_weekly_grid_includes_zero_weeks(tmp_path, ssa_home):
    r = build_features(raw_dir=_raw(tmp_path), write=False)
    f = r.features
    assert (f["weeks_observed"] == f["snapshot_week"] + 1).all()
    assert (f["weeks_active"] <= f["weeks_observed"]).all()
    assert (f["days_since_last_activity"].between(0, f["snapshot_day"])).all()
    valid = ~np.isnan(r.weekly)
    assert (valid.sum(1) == f["weeks_observed"].to_numpy()).all()


def test_imd_quirk_normalised(features):
    assert "10-20" not in set(features["imd_band"].astype(str))
