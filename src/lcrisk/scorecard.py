"""Logistic regression on WoE features, scaled to scorecard points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lcrisk import woe

# Score scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def select_features(woe_train: pd.DataFrame, iv: pd.DataFrame, min_iv: float = 0.02,
                    max_corr: float = 0.6) -> list[str]:
    """Keep features with IV >= min_iv, dropping the weaker of any pair correlated above
    max_corr."""
    ranked = iv.loc[iv["iv"] >= min_iv, "feature"].tolist()
    corr = woe_train[ranked].corr().abs()
    kept: list[str] = []
    for f in ranked:
        if all(corr.loc[f, k] <= max_corr for k in kept):
            kept.append(f)
    return kept


def fit_logit(woe_train: pd.DataFrame, y: pd.Series, features: list[str]):
    """Fit, then drop any feature with a positive (wrong-sign) or insignificant coefficient."""
    features = list(features)
    while True:
        res = sm.Logit(y.to_numpy(), sm.add_constant(woe_train[features])).fit(disp=0)
        coefs, pvals = res.params.drop("const"), res.pvalues.drop("const")
        bad = coefs[coefs >= 0].index.tolist() or pvals[pvals > 0.01].index.tolist()
        if not bad:
            return res, features
        worst = pvals[bad].idxmax()
        features.remove(worst)


@dataclass
class Scorecard:
    binners: dict
    features: list[str]
    result: object

    def log_odds_bad(self, df: pd.DataFrame) -> pd.Series:
        w = woe.woe_frame(df, {f: self.binners[f] for f in self.features})
        return pd.Series(self.result.predict(sm.add_constant(w[self.features], has_constant="add"),
                                             which="linear"), index=df.index)

    def predict_pd(self, df: pd.DataFrame) -> pd.Series:
        return 1 / (1 + np.exp(-self.log_odds_bad(df)))

    def score(self, df: pd.DataFrame) -> pd.Series:
        # Score is on good:bad odds, so it rises as risk falls
        return OFFSET - FACTOR * self.log_odds_bad(df)

    def points_table(self) -> pd.DataFrame:
        """Points per bin. Each feature carries an equal share of the intercept."""
        b0 = self.result.params["const"]
        k = len(self.features)
        rows = []
        for f in self.features:
            beta = self.result.params[f]
            for label, r in self.binners[f].table.iterrows():
                pts = OFFSET / k - FACTOR * (beta * r["woe"] + b0 / k)
                rows.append({"feature": f, "bin": label, "n": int(r["n"]),
                             "bad_rate": r["bad_rate"], "woe": r["woe"],
                             "points": round(float(pts))})
        return pd.DataFrame(rows)

    def coef_table(self) -> pd.DataFrame:
        r = self.result
        return pd.DataFrame({"coef": r.params, "se": r.bse, "z": r.tvalues, "p": r.pvalues})
