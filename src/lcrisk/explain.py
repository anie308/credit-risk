"""Model comparison beyond Gini: multicollinearity, explanations, and value at a cut-off."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from lcrisk import config, woe
from lcrisk.pipeline import load_models
from lcrisk.validate import TABLES, load_scored


def vif(card, train: pd.DataFrame) -> pd.DataFrame:
    W = woe.woe_frame(train, {f: card.binners[f] for f in card.features})
    X = np.column_stack([np.ones(len(W)), W.to_numpy()])
    return pd.DataFrame({"feature": card.features,
                         "vif": [variance_inflation_factor(X, i + 1)
                                 for i in range(len(card.features))]})


def importance(card, gbm, sample: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute contribution to the log-odds, per feature, for both models."""
    W = woe.woe_frame(sample, {f: card.binners[f] for f in card.features})
    sc = (W * card.result.params[card.features]).abs().mean()
    sh = gbm.contributions(sample).drop(columns="base").abs().mean()
    t = pd.DataFrame({"scorecard": sc, "lightgbm": sh}).fillna(0.0)
    t = t / t.sum()
    return t.sort_values("lightgbm", ascending=False).rename_axis("feature").reset_index()


def reason_codes(card, gbm, sample: pd.DataFrame) -> dict:
    """How often both models give the same top adverse-action reason for a declined loan.

    The scorecard's reasons are the features that cost the applicant the most points against
    the best bin; LightGBM's are the largest positive SHAP contributions.
    """
    W = woe.woe_frame(sample, {f: card.binners[f] for f in card.features})
    beta = card.result.params[card.features]
    best = pd.Series({f: card.binners[f].table["woe"].max() for f in card.features})
    # Log-odds penalty vs the best bin (beta < 0, so lower WoE adds risk)
    penalty = (W - best) * beta
    sc_top = penalty.idxmax(axis=1)
    contrib = gbm.contributions(sample).drop(columns="base")
    gbm_top = contrib.idxmax(axis=1)
    gbm_top3 = contrib.apply(lambda r: set(r.nlargest(3).index), axis=1)
    return {"loans": len(sample),
            "same_top_reason": float((sc_top == gbm_top).mean()),
            "scorecard_top_in_lgbm_top3": float(pd.Series(
                [a in b for a, b in zip(sc_top, gbm_top3, strict=True)]).mean()),
            "lgbm_top_not_in_scorecard": float((~gbm_top.isin(card.features)).mean())}


def cutoff_value(scored: pd.DataFrame, sample: str = "oot_2015",
                 reject_rates=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Bad rate and defaults among approved loans when the riskiest X% are declined."""
    g = scored[scored["sample"] == sample]
    rows = []
    for r in reject_rates:
        row = {"reject_rate": r}
        for model, col in [("scorecard", "pd_scorecard"), ("lightgbm", "pd_lightgbm"),
                           ("lc_subgrade", "lc_rank")]:
            accepted = g[col] <= g[col].quantile(1 - r)
            row[f"bad_rate_{model}"] = g.loc[accepted, "bad"].mean()
            row[f"defaults_avoided_{model}"] = int(g.loc[~accepted, "bad"].sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    t["extra_defaults_avoided_lgbm"] = (t["defaults_avoided_lightgbm"]
                                        - t["defaults_avoided_scorecard"])
    return t


def run() -> dict[str, pd.DataFrame]:
    m = load_models()
    card, gbm = m["scorecard"], m["lightgbm"]
    scored = load_scored()
    pop = pd.read_parquet(config.PROCESSED / "population.parquet")
    train = scored.loc[scored["sample"] == "train", ["id"]].merge(pop, on="id")
    oot = scored.loc[scored["sample"] == "oot_2015", ["id", "pd_scorecard"]].merge(pop, on="id")
    sample = oot.sample(20_000, random_state=config.SEED)
    # Adverse-action reasons only matter for declines: the riskiest 20% by scorecard PD
    declined = oot[oot["pd_scorecard"] >= oot["pd_scorecard"].quantile(0.8)].sample(
        10_000, random_state=config.SEED)
    out = {"vif": vif(card, train), "importance": importance(card, gbm, sample),
           "reason_codes": pd.DataFrame([reason_codes(card, gbm, declined)]),
           "cutoff_value": cutoff_value(scored),
           "scorecard_points": card.points_table(),
           "scorecard_coefficients": card.coef_table().rename_axis("term").reset_index(),
           "iv": m["iv"], "tuning": m["tuning"]}
    for name, t in out.items():
        t.to_csv(TABLES / f"{name}.csv", index=False)
    return out
