import numpy as np
import pandas as pd
import pytest

from lcrisk import metrics, woe


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n = 20_000
    fico = rng.normal(700, 30, n)
    pd_true = 1 / (1 + np.exp(0.05 * (fico - 640)))
    y = pd.Series((rng.random(n) < pd_true).astype(int))
    x = pd.Series(fico)
    x[rng.random(n) < 0.03] = np.nan
    return x, y, pd_true


def test_numeric_bins_monotonic(synthetic):
    x, y, _ = synthetic
    b = woe.NumericBinner("fico").fit(x, y)
    assert b.table.index[-1] == woe.MISSING
    t = b.table.drop(index=woe.MISSING)
    # Bins come in increasing FICO; higher FICO, lower bad rate, so WoE increases
    lows = [float(s.split(",")[0].strip("(")) for s in t.index]
    assert lows == sorted(lows)
    assert t["woe"].is_monotonic_increasing
    assert (t["n"] >= 0.05 * x.notna().sum()).all()
    assert woe.MISSING in b.table.index
    assert b.iv > 0.1


def test_tiny_missing_group_scores_neutral():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(0, 1, 5_000))
    y = pd.Series((rng.random(5_000) < 0.2).astype(int))
    x.iloc[:3] = np.nan
    b = woe.NumericBinner("x").fit(x, y)
    assert woe.MISSING not in b.table.index
    assert b.transform(pd.Series([np.nan])).iloc[0] == 0.0


def test_zero_inflated_counts_split_at_zero():
    rng = np.random.default_rng(3)
    x = pd.Series(np.where(rng.random(10_000) < 0.85, 0, rng.integers(1, 5, 10_000)))
    y = pd.Series((rng.random(10_000) < np.where(x > 0, 0.25, 0.1)).astype(int))
    b = woe.NumericBinner("pub_rec").fit(x, y)
    assert len(b.table) == 2
    assert b.iv > 0.05


def test_transform_unseen_missing_is_neutral():
    x = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100)
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 1, 1] * 100)
    b = woe.NumericBinner("x", max_bins=4, min_share=0.1).fit(x, y)
    out = b.transform(pd.Series([np.nan, 1.0]))
    assert out.iloc[0] == 0.0
    assert out.iloc[1] > 0


def test_categorical_rare_grouped():
    x = pd.Series(["a"] * 500 + ["b"] * 480 + ["c"] * 20)
    y = pd.Series([0, 1] * 250 + [0] * 470 + [1] * 10 + [1] * 20)
    b = woe.CategoricalBinner("cat", min_share=0.05).fit(x, y)
    assert b.mapping["c"] == "Other"
    assert b.transform(pd.Series(["zzz"])).iloc[0] == b.table.loc["Other", "woe"]


def test_gini_ks_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert metrics.gini(y, np.arange(100)) == pytest.approx(1.0)
    assert metrics.ks(y, np.arange(100)) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert abs(metrics.gini(rng.integers(0, 2, 5000), rng.random(5000))) < 0.05


def test_psi():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 50_000)
    assert metrics.psi(a, rng.normal(0, 1, 50_000)) < 0.01
    assert metrics.psi(a, rng.normal(0.5, 1, 50_000)) > 0.2
    # Zero-inflated: 3% non-zero -> 15% non-zero must register
    z0 = (rng.random(50_000) < 0.03).astype(float)
    z1 = (rng.random(50_000) < 0.15).astype(float)
    assert metrics.psi(z0, z1) > 0.2


def test_calibration_tests_flag_underestimate(synthetic):
    _, y, pd_true = synthetic
    grade = pd.qcut(pd_true, 5, labels=False)
    good = metrics.calibration_by_grade(y, pd_true, grade)
    assert (good["binomial_p"] > 0.001).all()
    bad = metrics.calibration_by_grade(y, pd_true * 0.5, grade)
    assert (bad["jeffreys_p"] < 0.01).sum() >= 3
    _, p_ok = metrics.hosmer_lemeshow(y, pd_true)
    assert p_ok > 0.01
