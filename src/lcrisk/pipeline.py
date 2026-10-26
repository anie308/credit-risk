"""Fit the benchmark scorecard and LightGBM challenger, score every sample."""

from __future__ import annotations

import pickle

import pandas as pd
from sklearn.metrics import brier_score_loss

from lcrisk import challenger, config, metrics, samples, woe
from lcrisk import scorecard as sc
from lcrisk.profile import feature_columns

CATEGORICAL = ["home_ownership", "verification_status", "purpose"]
# Constant in the population (individual applications only)
DROP = {"application_type"}
MODELS = config.PROCESSED / "models.pkl"
SCORED = config.PROCESSED / "scored.parquet"


def numeric_features() -> list[str]:
    return [c for c in feature_columns() if c not in CATEGORICAL and c not in DROP]


def fit_scorecard(train: pd.DataFrame) -> tuple[sc.Scorecard, pd.DataFrame]:
    y = train["bad"]
    binners = woe.fit_binners(train, y, numeric_features(), CATEGORICAL)
    iv = woe.iv_summary(binners)
    W = woe.woe_frame(train, binners)
    result, feats = sc.fit_logit(W, y, sc.select_features(W, iv))
    iv["selected"] = iv["feature"].isin(feats)
    return sc.Scorecard(binners, feats, result), iv


def fit_challenger(train: pd.DataFrame) -> tuple[challenger.Challenger, pd.DataFrame]:
    fit_part, hold = samples.time_holdout(train)
    tuning = challenger.tune(fit_part, hold, numeric_features(), CATEGORICAL, challenger.grid())
    best = tuning.iloc[0][["num_leaves", "min_child_samples", "learning_rate"]].dropna().to_dict()
    best = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in best.items()}
    return challenger.fit(fit_part, hold, numeric_features(), CATEGORICAL, best), tuning


def lc_grade_rank(df: pd.DataFrame) -> pd.Series:
    """Lending Club sub-grade as an ordinal risk rank (A1 = 0 ... G5 = 34)."""
    s = df["sub_grade"].astype("string")
    grade = s.str[0].map(dict(zip("ABCDEFG", range(7), strict=True)))
    return grade * 5 + s.str[1].astype(int) - 1


def performance(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in scored.groupby("sample", sort=False):
        y = g["bad"]
        for model, col in [("scorecard", "pd_scorecard"), ("lightgbm", "pd_lightgbm"),
                           ("lc_subgrade", "lc_rank")]:
            lo, hi = metrics.gini_ci(y, g[col], n_boot=100)
            row = {"sample": name, "model": model, "n": len(g), "odr": y.mean(),
                   "gini": metrics.gini(y, g[col]), "gini_lo": lo, "gini_hi": hi,
                   "ks": metrics.ks(y, g[col])}
            if col != "lc_rank":
                row |= {"mean_pd": g[col].mean(), "brier": brier_score_loss(y, g[col])}
            rows.append(row)
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    pop = pd.read_parquet(config.PROCESSED / "population.parquet")
    parts = samples.split(pop)
    card, iv = fit_scorecard(parts["train"])
    gbm, tuning = fit_challenger(parts["train"])

    scored = []
    for name, df in parts.items():
        scored.append(pd.DataFrame({
            "sample": name, "id": df["id"], "issue_q": df["issue_q"], "vintage": df["vintage"],
            "bad": df["bad"], "funded_amnt": df["funded_amnt"], "sub_grade": df["sub_grade"],
            "lc_rank": lc_grade_rank(df), "score": card.score(df),
            "pd_scorecard": card.predict_pd(df), "pd_lightgbm": gbm.predict_pd(df),
        }))
    scored = pd.concat(scored, ignore_index=True)
    scored.to_parquet(SCORED, index=False)
    with open(MODELS, "wb") as f:
        pickle.dump({"scorecard": card, "lightgbm": gbm, "iv": iv, "tuning": tuning}, f)

    perf = performance(scored)
    perf.to_csv(config.PROCESSED / "performance.csv", index=False)
    return perf


def load_models() -> dict:
    with open(MODELS, "rb") as f:
        return pickle.load(f)

