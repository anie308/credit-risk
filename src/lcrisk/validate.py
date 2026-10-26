"""Validation tables: master scale, calibration tests, stability, recalibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lcrisk import config, loss, metrics
from lcrisk.pipeline import CATEGORICAL, SCORED, numeric_features

TABLES = config.ROOT / "reports" / "tables"
MODELS = {"scorecard": "pd_scorecard", "lightgbm": "pd_lightgbm"}

# Master scale: fixed lifetime-PD boundaries, roughly geometric, set before looking at OOT
MASTER_SCALE = [0.0, 0.04, 0.06, 0.08, 0.11, 0.15, 0.20, 0.27, 1.0]
GRADES = [f"R{i}" for i in range(1, len(MASTER_SCALE))]


def rating(pd_: pd.Series) -> pd.Series:
    return pd.cut(pd_, MASTER_SCALE, labels=GRADES, right=False)


def load_scored() -> pd.DataFrame:
    return pd.read_parquet(SCORED)


def calibration_tables(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-grade binomial/Jeffreys tests, and Hosmer-Lemeshow per sample, for both models."""
    grades, hl = [], []
    for name, g in scored.groupby("sample", sort=False):
        for model, col in MODELS.items():
            t = metrics.calibration_by_grade(g["bad"], g[col], rating(g[col]))
            grades.append(t.reset_index().assign(sample=name, model=model))
            stat, p = metrics.hosmer_lemeshow(g["bad"], g[col])
            hl.append({"sample": name, "model": model, "hl_stat": stat, "hl_p": p,
                       "mean_pd": g[col].mean(), "odr": g["bad"].mean()})
    return pd.concat(grades, ignore_index=True), pd.DataFrame(hl)


def by_quarter(scored: pd.DataFrame) -> pd.DataFrame:
    """Predicted vs observed default rate and score PSI, per issue quarter, against train."""
    train = scored[scored["sample"] == "train"]
    later = scored[scored["sample"] != "train"]
    edges = np.unique(np.quantile(train["score"], np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for q, g in later.groupby("issue_q"):
        rows.append({"issue_q": q, "n": len(g), "odr": g["bad"].mean(),
                     "pd_scorecard": g["pd_scorecard"].mean(),
                     "pd_lightgbm": g["pd_lightgbm"].mean(),
                     "psi_score": metrics.psi(train["score"], g["score"], edges=edges),
                     "psi_lightgbm": metrics.psi(train["pd_lightgbm"], g["pd_lightgbm"]),
                     "gini_scorecard": metrics.gini(g["bad"], g["pd_scorecard"]),
                     "gini_lightgbm": metrics.gini(g["bad"], g["pd_lightgbm"])})
    t = pd.DataFrame(rows)
    t["psi_light"] = t["psi_score"].map(metrics.psi_light)
    return t


def feature_stability(pop: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """CSI (PSI per feature) of each OOT year against the training sample."""
    s = scored[["id", "sample"]].merge(pop, on="id")
    train = s[s["sample"] == "train"]
    rows = []
    for f in numeric_features() + CATEGORICAL:
        row = {"feature": f}
        for y in config.OOT_YEARS:
            oot = s[s["sample"] == f"oot_{y}"]
            if f in CATEGORICAL:
                row[f"csi_{y}"] = metrics.psi_categorical(train[f], oot[f])
            else:
                a, b = train[f].astype(float), oot[f].astype(float)
                # Missing gets its own bucket: a shift in missingness is a shift
                fill = np.nanmin(a) - 1
                row[f"csi_{y}"] = metrics.psi(a.fillna(fill), b.fillna(fill))
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(f"csi_{config.OOT_YEARS[-1]}", ascending=False)
    for y in config.OOT_YEARS:
        t[f"light_{y}"] = t[f"csi_{y}"].map(metrics.psi_light)
    return t.reset_index(drop=True)


def recalibrate(scored: pd.DataFrame, fit_on: str = "oot_2014", apply_to: str = "oot_2015"
                ) -> tuple[pd.DataFrame, dict]:
    """Intercept-only recalibration, the standard fix for level drift with stable ranking.

    Fitted on the latest fully observed year (as a bank would when monitoring) and tested on
    the next one, so the fix itself is out of time.
    """
    ref, tgt = scored[scored["sample"] == fit_on], scored[scored["sample"] == apply_to]
    shifts, rows = {}, []
    for model, col in MODELS.items():
        shifts[model] = loss.calibrate_shift(ref[col], ref["bad"].mean())
        new = loss.odds_shift(tgt[col], shifts[model])
        for label, p in [("before", tgt[col]), ("after", new)]:
            stat, hl_p = metrics.hosmer_lemeshow(tgt["bad"], p)
            t = metrics.calibration_by_grade(tgt["bad"], p, rating(p))
            rows.append({"model": model, "stage": label, "mean_pd": float(np.mean(p)),
                         "odr": tgt["bad"].mean(), "hl_stat": stat, "hl_p": hl_p,
                         "grades_rejected_jeffreys": int((t["jeffreys_p"] < 0.01).sum()),
                         "grades": len(t)})
    return pd.DataFrame(rows), shifts


def run() -> dict[str, pd.DataFrame]:
    scored = load_scored()
    pop = pd.read_parquet(config.PROCESSED / "population.parquet")
    grades, hl = calibration_tables(scored)
    recal, shifts = recalibrate(scored)
    out = {
        "performance": pd.read_csv(config.PROCESSED / "performance.csv"),
        "calibration_grades": grades,
        "hosmer_lemeshow": hl,
        "by_quarter": by_quarter(scored),
        "feature_stability": feature_stability(pop, scored),
        "recalibration": recal.assign(shift=recal["model"].map(shifts)),
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, t in out.items():
        t.to_csv(TABLES / f"{name}.csv", index=False)
    return out
