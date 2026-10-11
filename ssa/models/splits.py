"""Temporal train / validation / test splitting by course presentation.

With OULAD's four presentations: train = 2013B + 2013J, validation = 2014B (early stopping,
calibration, threshold and weight selection), test = 2014J (untouched until final scoring).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ssa.data.vocab import presentation_sort_key


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    info: dict


def temporal_split(df: pd.DataFrame, label: str, seed: int = 42) -> Split:
    pres = sorted(df["code_presentation"].astype(str).unique(), key=presentation_sort_key)
    idx = np.arange(len(df))
    p = df["code_presentation"].astype(str).to_numpy()
    y = df[label].to_numpy()

    def ok(mask):
        return mask.sum() > 0 and len(np.unique(y[mask])) >= 2

    if len(pres) >= 3:
        test_m, val_m = p == pres[-1], p == pres[-2]
        train_m = ~(test_m | val_m)
        if ok(test_m) and ok(val_m) and ok(train_m):
            return Split(idx[train_m], idx[val_m], idx[test_m],
                         {"method": "temporal", "train": pres[:-2], "validation": [pres[-2]], "test": [pres[-1]]})
    if len(pres) == 2:
        test_m = p == pres[-1]
        if ok(test_m) and ok(~test_m):
            tr, va = train_test_split(idx[~test_m], test_size=0.25, random_state=seed, stratify=y[~test_m])
            return Split(np.sort(tr), np.sort(va), idx[test_m],
                         {"method": "temporal_test_random_validation", "train": [pres[0]], "validation": [pres[0]],
                          "test": [pres[-1]]})
    strat = y if len(np.unique(y)) > 1 else None
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    strat_tr = y[tr] if strat is not None else None
    tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=strat_tr)
    return Split(np.sort(tr), np.sort(va), np.sort(te), {"method": "stratified_random_fallback", "presentations": pres})
