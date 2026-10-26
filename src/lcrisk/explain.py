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


# Reason-code families: overlapping bureau fields that describe the same thing to an applicant.
# A declined applicant is told "too many recent credit applications", not which of four
# correlated fields carried the weight.
FAMILIES = {
    "Credit score": ["fico"],
    "Income and affordability": ["annual_inc", "loan_to_income", "dti", "revol_to_income",
                                 "loan_amnt"],
    "Credit limits and balances": ["tot_hi_cred_lim", "bc_open_to_buy", "total_bc_limit",
                                   "total_rev_hi_lim", "tot_cur_bal", "avg_cur_bal",
                                   "total_bal_ex_mort", "revol_bal"],
    "Credit card utilisation": ["revol_util", "bc_util", "percent_bc_gt_75"],
    "Recent credit seeking": ["acc_open_past_24mths", "num_tl_op_past_12m", "inq_last_6mths",
                              "mths_since_recent_inq"],
    "Length and depth of credit history": ["mo_sin_old_rev_tl_op", "credit_hist_months",
                                           "total_acc", "open_acc", "num_actv_rev_tl",
                                           "mort_acc"],
    "Delinquencies and public records": ["delinq_2yrs", "mths_since_last_delinq",
                                         "mths_since_last_record", "pub_rec",
                                         "pub_rec_bankruptcies", "tax_liens",
                                         "collections_12_mths_ex_med", "acc_now_delinq",
                                         "pct_tl_nvr_dlq"],
    "Application details": ["purpose", "home_ownership", "verification_status",
                            "emp_length_yrs"],
}
FAMILY_OF = {f: fam for fam, feats in FAMILIES.items() for f in feats}


def _agreement(sc: pd.DataFrame, gbm: pd.DataFrame) -> dict:
    """Top-1 agreement and scorecard top-1 inside LightGBM top-3, on per-loan reason scores."""
    sc_top, gbm_top = sc.idxmax(axis=1), gbm.idxmax(axis=1)
    gbm_top3 = gbm.apply(lambda r: set(r.nlargest(3).index), axis=1)
    return {"same_top": float((sc_top == gbm_top).mean()),
            "scorecard_top_in_lgbm_top3": float(pd.Series(
                [a in b for a, b in zip(sc_top, gbm_top3, strict=True)]).mean()),
            "lgbm_top": gbm_top, "sc_top": sc_top}


def reason_codes(card, gbm, sample: pd.DataFrame) -> dict:
    """How often both models give the same top adverse-action reason for a declined loan,
    feature by feature and grouped into reason-code families.

    The scorecard's reasons are the features that cost the applicant the most points against
    the best bin; LightGBM's are the largest positive SHAP contributions. Family scores sum
    the members' contributions.
    """
    W = woe.woe_frame(sample, {f: card.binners[f] for f in card.features})
    beta = card.result.params[card.features]
    best = pd.Series({f: card.binners[f].table["woe"].max() for f in card.features})
    # Log-odds penalty vs the best bin (beta < 0, so lower WoE adds risk)
    penalty = (W - best) * beta
    contrib = gbm.contributions(sample).drop(columns="base")
    unmapped = set(contrib.columns) - set(FAMILY_OF)
    assert not unmapped, f"features without a reason-code family: {unmapped}"
    by_feature = _agreement(penalty, contrib)
    by_family = _agreement(penalty.T.groupby(FAMILY_OF).sum().T,
                           contrib.T.groupby(FAMILY_OF).sum().T)
    sc_families = {FAMILY_OF[f] for f in card.features}
    return {"loans": len(sample),
            "same_top_reason": by_feature["same_top"],
            "scorecard_top_in_lgbm_top3": by_feature["scorecard_top_in_lgbm_top3"],
            "lgbm_top_not_in_scorecard": float((~by_feature["lgbm_top"].isin(card.features))
                                               .mean()),
            "same_top_family": by_family["same_top"],
            "scorecard_top_family_in_lgbm_top3": by_family["scorecard_top_in_lgbm_top3"],
            "lgbm_top_family_not_in_scorecard": float(
                (~by_family["lgbm_top"].isin(sc_families)).mean())}


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
