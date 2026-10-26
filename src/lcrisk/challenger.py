"""LightGBM challenger on raw application features, with monotone constraints."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from lcrisk import config

# +1: PD rises with the feature, -1: PD falls. Only where the business logic is unambiguous;
# everything else is left free (0).
MONOTONE = {
    "fico": -1, "dti": 1, "inq_last_6mths": 1, "revol_util": 1, "annual_inc": -1,
    "credit_hist_months": -1, "delinq_2yrs": 1, "pub_rec": 1, "pub_rec_bankruptcies": 1,
    "mort_acc": -1, "bc_util": 1, "acc_open_past_24mths": 1, "num_tl_op_past_12m": 1,
    "mths_since_recent_inq": -1, "percent_bc_gt_75": 1, "pct_tl_nvr_dlq": -1,
    "loan_to_income": 1,
}

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "monotone_constraints_method": "advanced",
    "verbosity": -1,
    "seed": config.SEED,
}


def prepare(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    X = df[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in categorical:
        X[c] = X[c].astype("string").fillna("Missing").astype("category")
    return X


@dataclass
class Challenger:
    booster: lgb.Booster
    numeric: list[str]
    categorical: list[str]
    categories: dict[str, pd.Index]

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = prepare(df, self.numeric, self.categorical)
        # Scoring must use the training category codes, or LightGBM silently remaps them
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype("string"), categories=self.categories[c])
        return X

    def predict_pd(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.booster.predict(self._X(df)), index=df.index)

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions in log-odds; the last column is the expected value."""
        c = self.booster.predict(self._X(df), pred_contrib=True)
        return pd.DataFrame(c, index=df.index, columns=[*self.numeric, *self.categorical, "base"])

    def importance(self) -> pd.Series:
        gain = self.booster.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=self.booster.feature_name()).sort_values(
            ascending=False)


def fit(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], categorical: list[str],
        params: dict | None = None, max_rounds: int = 3000) -> Challenger:
    """Fit with early stopping on `valid`, which must be later in time than `train`."""
    params = {**PARAMS, **(params or {})}
    Xt = prepare(train, numeric, categorical)
    categories = {c: Xt[c].cat.categories for c in categorical}
    Xv = prepare(valid, numeric, categorical)
    for c in categorical:
        Xv[c] = pd.Categorical(Xv[c].astype("string"), categories=categories[c])
    params["monotone_constraints"] = [MONOTONE.get(c, 0) for c in Xt.columns]
    dtrain = lgb.Dataset(Xt, train["bad"], categorical_feature=categorical)
    dvalid = lgb.Dataset(Xv, valid["bad"], reference=dtrain)
    booster = lgb.train(params, dtrain, max_rounds, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return Challenger(booster, numeric, categorical, categories)


def tune(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], categorical: list[str],
         grid: list[dict]) -> pd.DataFrame:
    """Small grid search scored by Gini on the later `valid` vintage."""
    from lcrisk.metrics import gini

    rows = []
    for g in grid:
        m = fit(train, valid, numeric, categorical, g)
        rows.append({**g, "rounds": m.booster.best_iteration,
                     "gini_valid": gini(valid["bad"], m.predict_pd(valid))})
    return pd.DataFrame(rows).sort_values("gini_valid", ascending=False).reset_index(drop=True)


def grid() -> list[dict]:
    return [{"num_leaves": nl, "min_child_samples": mc}
            for nl in (15, 31, 63) for mc in (200, 1000)] + [{"learning_rate": 0.05}]


def check_monotone(model: Challenger, df: pd.DataFrame, n: int = 2000) -> pd.Series:
    """Fraction of sampled loans whose PD moves the constrained way as each feature is raised."""
    sample = df.sample(min(n, len(df)), random_state=config.SEED)
    out = {}
    for f, sign in MONOTONE.items():
        if f not in model.numeric:
            continue
        lo, hi = sample.copy(), sample.copy()
        q = np.nanquantile(pd.to_numeric(df[f], errors="coerce"), [0.1, 0.9])
        lo[f], hi[f] = q[0], q[1]
        diff = (model.predict_pd(hi) - model.predict_pd(lo)) * sign
        out[f] = float((diff >= -1e-12).mean())
    return pd.Series(out)
