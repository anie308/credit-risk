"""Expected loss backtest and stress scenarios.

Risk parameters (LGD, EAD ratio, share of lifetime defaults inside 12 months) are estimated per
master-scale grade on the development window only, whose workouts are complete by the data
snapshot. They are then applied out of time.
"""

from __future__ import annotations

import pandas as pd

from lcrisk import config, loss
from lcrisk.validate import GRADES, MODELS, TABLES, load_scored, rating

CRISIS_YEARS = (2007, 2009)


def _months_to_default(df: pd.DataFrame) -> pd.Series:
    """Month of the first missed payment, counted from issue (last payment month + 1)."""
    last = df["last_pymnt_d"].fillna(df["issue_d"])
    return ((last.dt.year - df["issue_d"].dt.year) * 12
            + last.dt.month - df["issue_d"].dt.month + 1)


def parameters(dev: pd.DataFrame, pd_col: str = "pd_scorecard") -> pd.DataFrame:
    """LGD, EAD ratio and 12-month default share per grade, from development-window defaults."""
    d = dev.assign(grade_r=rating(dev[pd_col]))
    r = loss.realised_loss(d).join(d[["grade_r", "bad"]])
    r["within_12m"] = (_months_to_default(d) <= 12).where(d["bad"] == 1)
    r = r[r["ead"] > 0]
    g = r.groupby("grade_r", observed=True)
    t = pd.DataFrame({"defaults": g.size(), "lgd": g["loss"].sum() / g["ead"].sum(),
                      "ead_ratio": g["ead_ratio"].mean(), "share_12m": g["within_12m"].mean()})
    return t.reindex(GRADES)


def downturn_lgd_factor(pop: pd.DataFrame) -> float:
    """Crisis-vintage LGD relative to development-window LGD (exposure weighted)."""
    lo, hi = CRISIS_YEARS
    crisis = loss.lgd_by_segment(pop[pop["vintage"].between(lo, hi)].assign(all=1), "all")
    dev = pop[pop["issue_d"].between(pd.Timestamp(config.DEV_START), pd.Timestamp(config.DEV_END))]
    base = loss.lgd_by_segment(dev.assign(all=1), "all")
    return float(crisis["lgd"].iloc[0] / base["lgd"].iloc[0])


def crisis_shift(pop: pd.DataFrame) -> dict:
    """Log-odds shift that turns development-window default rates into crisis ones.

    Each 2007-09 loan gets its LC sub-grade's development-window default rate as a PD; the
    shift is what lifts the mean of those PDs to the crisis vintages' observed rate. Holding
    sub-grade fixed strips out the mix change, leaving the downturn itself. LC re-calibrated
    its grades over time, so treat this as an approximate severity.
    """
    lo, hi = CRISIS_YEARS
    dev = pop[pop["issue_d"].between(pd.Timestamp(config.DEV_START), pd.Timestamp(config.DEV_END))]
    rate = dev.groupby("sub_grade")["bad"].mean()
    crisis = pop[pop["vintage"].between(lo, hi)]
    pd_ = crisis["sub_grade"].map(rate).dropna()
    odr = crisis.loc[pd_.index, "bad"].mean()
    return {"loans": len(pd_), "pd_at_dev_rates": float(pd_.mean()), "odr": float(odr),
            "shift": loss.calibrate_shift(pd_.clip(1e-4, 0.9999), odr)}


def _apply(book: pd.DataFrame, params: pd.DataFrame, pd_: pd.Series,
           lgd_factor: float = 1.0) -> pd.DataFrame:
    g = rating(pd_)
    lgd = (g.map(params["lgd"]).astype(float) * lgd_factor).clip(upper=1.0)
    ead = book["funded_amnt"] * g.map(params["ead_ratio"]).astype(float)
    return pd.DataFrame({"pd": pd_, "lgd": lgd, "ead": ead, "el": pd_ * lgd * ead},
                        index=book.index)


def backtest(scored: pd.DataFrame, pop: pd.DataFrame, params: dict,
             shifts: dict) -> pd.DataFrame:
    """Predicted lifetime EL vs realised loss per OOT year and model, raw and recalibrated."""
    rows = []
    for y in config.OOT_YEARS:
        book = scored[scored["sample"] == f"oot_{y}"].merge(
            pop[["id", "total_rec_prncp", "recoveries", "collection_recovery_fee"]], on="id")
        funded = book["funded_amnt"].sum()
        realised = loss.realised_loss(book)["loss"].sum()
        for model, col in MODELS.items():
            for stage, p in [("raw", book[col]),
                             ("recalibrated", loss.odds_shift(book[col], shifts[model]))]:
                el = _apply(book, params[model], p)["el"].sum()
                rows.append({"year": y, "model": model, "pd": stage, "funded": funded,
                             "expected_loss": el, "realised_loss": realised,
                             "el_rate": el / funded, "loss_rate": realised / funded,
                             "ratio": el / realised})
    return pd.DataFrame(rows)


def stress(scored: pd.DataFrame, params: pd.DataFrame, recal_shift: float, crisis: dict,
           lgd_factor: float, col: str = "pd_scorecard", year: int = 2015) -> pd.DataFrame:
    """Lifetime EL and one-year Basel-style capital on the latest vintage, under scenarios."""
    book = scored[scored["sample"] == f"oot_{year}"]
    base_pd = loss.odds_shift(book[col], recal_shift)
    scenarios = {
        "baseline": (base_pd, 1.0),
        "downturn PD": (loss.odds_shift(base_pd, crisis["shift"]), 1.0),
        "downturn PD + LGD": (loss.odds_shift(base_pd, crisis["shift"]), lgd_factor),
    }
    rows = []
    funded = book["funded_amnt"].sum()
    for name, (p, f) in scenarios.items():
        r = _apply(book, params, p, f)
        rows.append({"scenario": name, "mean_pd": p.mean(),
                     "mean_lgd": (r["lgd"] * r["ead"]).sum() / r["ead"].sum(),
                     "lifetime_el": r["el"].sum(), "el_rate": r["el"].sum() / funded})
    # One-year view on the baseline: empirical 12-month PD, downturn LGD, EAD = funded amount
    g = rating(base_pd)
    pd_12m = base_pd * g.map(params["share_12m"]).astype(float)
    lgd_dt = (g.map(params["lgd"]).astype(float) * lgd_factor).clip(upper=1.0)
    for q in (0.999, 0.99):
        c = loss.irb_capital(pd_12m, lgd_dt, book["funded_amnt"], q)
        rows.append({"scenario": f"Vasicek {q:.1%} one-year", "mean_pd": c["pd_stressed"].mean(),
                     "mean_lgd": float((lgd_dt * book["funded_amnt"]).sum() / funded),
                     "one_year_el": c["el_1y"].sum(), "loss_at_q": c["loss_q"].sum(),
                     "capital": c["capital"].sum(), "capital_rate": c["capital"].sum() / funded})
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    scored = load_scored()
    pop = pd.read_parquet(config.PROCESSED / "population.parquet")
    dev_ids = scored.loc[scored["sample"].isin(["train", "test"]), ["id", *MODELS.values()]]
    dev = dev_ids.merge(pop, on="id")
    params = {m: parameters(dev, col) for m, col in MODELS.items()}
    recal = pd.read_csv(TABLES / "recalibration.csv").drop_duplicates("model")
    shifts = dict(zip(recal["model"], recal["shift"], strict=True))
    crisis = crisis_shift(pop)
    factor = downturn_lgd_factor(pop)
    out = {
        "el_parameters": pd.concat(params, names=["model"]).reset_index(),
        "el_backtest": backtest(scored, pop, params, shifts),
        "stress": stress(scored, params["scorecard"], shifts["scorecard"], crisis, factor),
        "stress_calibration": pd.DataFrame([{**crisis, "downturn_lgd_factor": factor}]),
    }
    for name, t in out.items():
        t.to_csv(TABLES / f"{name}.csv", index=False)
    return out
