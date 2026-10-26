"""Discrimination, calibration and stability measures used in validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def gini_ci(y, p, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap percentile interval for Gini."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            draws.append(gini(y[i], p[i]))
    return tuple(np.quantile(draws, [alpha / 2, 1 - alpha / 2]))


def ks(y, p) -> float:
    """Maximum gap between the cumulative score distributions of bads and goods."""
    y, p = np.asarray(y), np.asarray(p)
    return float(stats.ks_2samp(p[y == 1], p[y == 0]).statistic)


def psi(expected, actual, bins: int = 10, edges=None) -> float:
    """Population stability index of `actual` against `expected`, on expected's deciles."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if edges is None:
        # Right-closed bins with the minimum kept as a cut, so a zero-inflated count still
        # splits into (-inf, 0] and (0, inf) instead of one bin that hides any shift
        q = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges = np.array([-np.inf, *q[:-1], np.inf])
    e = np.bincount(np.searchsorted(edges, expected, side="left"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, actual, side="left"), minlength=len(edges) + 1)
    e, a = e / len(expected), a / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual) -> float:
    e = pd.Series(expected).astype("string").fillna("NA").value_counts(normalize=True)
    a = pd.Series(actual).astype("string").fillna("NA").value_counts(normalize=True)
    e, a = e.align(a, fill_value=0)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_light(value: float) -> str:
    return "green" if value < 0.1 else "amber" if value < 0.25 else "red"


def hosmer_lemeshow(y, p, groups: int = 10) -> tuple[float, float]:
    """HL statistic and p-value over PD deciles."""
    df = pd.DataFrame({"y": y, "p": p})
    df["g"] = pd.qcut(df["p"], groups, duplicates="drop", labels=False)
    t = df.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    stat = (((t["obs"] - t["exp"]) ** 2) / (t["exp"] * (1 - t["exp"] / t["n"]))).sum()
    return float(stat), float(stats.chi2.sf(stat, len(t) - 2))


def calibration_by_grade(y, p, grade) -> pd.DataFrame:
    """Per-grade binomial and Jeffreys tests (ECB TRIM style).

    Both are one-sided: a low p-value says the observed default rate is above the predicted PD,
    i.e. the grade's PD is underestimated.
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "grade": np.asarray(grade)})
    t = df.groupby("grade", observed=True).agg(n=("y", "size"), defaults=("y", "sum"),
                                               pd_mean=("p", "mean"))
    t["odr"] = t["defaults"] / t["n"]
    # Size of the miss, not just its significance: observed / predicted, with a 95% Jeffreys
    # interval on the observed rate (with n in the tens of thousands the tests reject tiny gaps)
    a, b = t["defaults"] + 0.5, t["n"] - t["defaults"] + 0.5
    t["obs_pred"] = t["odr"] / t["pd_mean"]
    t["obs_pred_lo"] = stats.beta.ppf(0.025, a, b) / t["pd_mean"]
    t["obs_pred_hi"] = stats.beta.ppf(0.975, a, b) / t["pd_mean"]
    t["binomial_p"] = stats.binom.sf(t["defaults"] - 1, t["n"], t["pd_mean"])
    t["jeffreys_p"] = stats.beta.cdf(t["pd_mean"], t["defaults"] + 0.5,
                                     t["n"] - t["defaults"] + 0.5)
    return t


def reliability(y, p, bins: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop", labels=False)
    return df.groupby("bin").agg(pd_mean=("p", "mean"), odr=("y", "mean"), n=("y", "size"))
