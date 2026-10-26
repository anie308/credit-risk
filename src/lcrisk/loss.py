"""EAD, LGD, expected loss and stress.

PDs are lifetime (36-month) PDs, which is what the models predict, except in the Vasicek
functions, which take a one-year PD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

TERM_YEARS = 3


def realised_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Per-loan EAD at default, net recoveries and LGD (for defaulted loans; NaN otherwise)."""
    bad = df["bad"] == 1
    ead = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0).where(bad)
    net_rec = (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0).where(bad)
    lgd = (1 - net_rec / ead.replace(0, np.nan)).clip(0, 1)
    return pd.DataFrame({"ead": ead, "ead_ratio": ead / df["funded_amnt"], "net_recovery": net_rec,
                         "lgd": lgd, "loss": (ead * lgd).fillna(0.0)}, index=df.index)


def lgd_by_segment(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Exposure-weighted LGD and mean EAD ratio among defaults, per segment."""
    r = realised_loss(df).join(df[[by]])
    r = r[r["ead"].notna() & (r["ead"] > 0)]
    g = r.groupby(by, observed=True)
    return pd.DataFrame({"defaults": g.size(),
                         "lgd": g["loss"].sum() / g["ead"].sum(),
                         "ead_ratio": g["ead_ratio"].mean()})


def expected_loss(pd_life: pd.Series, funded: pd.Series, lgd: float | pd.Series,
                  ead_ratio: float | pd.Series) -> pd.Series:
    """Lifetime EL per loan: PD x LGD x EAD, with EAD = funded x expected balance at default."""
    return pd_life * lgd * funded * ead_ratio


def annual_pd(pd_life, years: int = TERM_YEARS):
    """Constant-hazard one-year PD from a lifetime PD."""
    return 1 - (1 - np.asarray(pd_life)) ** (1 / years)


def odds_shift(pd_, shift: float):
    """Move PDs by a constant on the log-odds scale (positive = riskier)."""
    lo = np.log(pd_ / (1 - pd_)) + shift
    return 1 / (1 + np.exp(-lo))


def calibrate_shift(pd_, observed_rate: float) -> float:
    """Log-odds shift that makes the mean PD equal an observed default rate.

    Applied to the crisis vintages, it measures the part of the downturn that the borrower mix
    doesn't explain, and that becomes the stress scenario.
    """
    pd_ = np.asarray(pd_)
    return float(optimize.brentq(lambda s: odds_shift(pd_, s).mean() - observed_rate, -5, 5))


def retail_correlation(pd_1y):
    """Basel IRB asset correlation for 'other retail' exposures."""
    w = (1 - np.exp(-35 * pd_1y)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def vasicek_pd(pd_1y, rho, q: float = 0.999):
    """One-year PD conditional on a systematic factor at its q-quantile."""
    return stats.norm.cdf((stats.norm.ppf(pd_1y) + np.sqrt(rho) * stats.norm.ppf(q))
                          / np.sqrt(1 - rho))


def irb_capital(pd_1y, lgd, ead, q: float = 0.999) -> pd.DataFrame:
    """Basel-style one-year EL, conditional loss at q, and unexpected loss (capital) per loan."""
    pd1 = np.clip(np.asarray(pd_1y, float), 1e-4, 0.9999)
    rho = retail_correlation(pd1)
    cond = vasicek_pd(pd1, rho, q)
    return pd.DataFrame({"pd_1y": pd1, "rho": rho, "pd_stressed": cond,
                         "el_1y": pd1 * lgd * ead, "loss_q": cond * lgd * ead,
                         "capital": (cond - pd1) * lgd * ead})
