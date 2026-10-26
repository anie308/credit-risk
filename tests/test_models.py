import numpy as np
import pandas as pd
import pytest

from lcrisk import challenger, loss, woe
from lcrisk import scorecard as sc


@pytest.fixture(scope="module")
def loans():
    rng = np.random.default_rng(0)
    n = 30_000
    df = pd.DataFrame({
        "fico": rng.normal(700, 30, n),
        "dti": rng.uniform(0, 40, n),
        "noise": rng.normal(0, 1, n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
    })
    df["fico_dup"] = df["fico"] + rng.normal(0, 2, n)
    logit = -2 - 0.04 * (df.fico - 700) + 0.04 * (df.dti - 20) + 0.4 * (df.home_ownership == "RENT")
    df["bad"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def test_scorecard(loans):
    y = loans["bad"]
    b = woe.fit_binners(loans, y, ["fico", "dti", "noise", "fico_dup"], ["home_ownership"])
    W = woe.woe_frame(loans, b)
    feats = sc.select_features(W, woe.iv_summary(b))
    assert "noise" not in feats
    assert not {"fico", "fico_dup"} <= set(feats)  # near-duplicates pruned
    res, feats = sc.fit_logit(W, y, feats)
    card = sc.Scorecard(b, feats, res)
    assert (card.coef_table().drop("const")["coef"] < 0).all()

    p, s = card.predict_pd(loans), card.score(loans)
    np.testing.assert_allclose(p, res.predict(), atol=1e-10)
    assert p.mean() == pytest.approx(y.mean(), abs=1e-3)
    # Score falls as PD rises, and 50:1 odds sit at the base score
    assert np.corrcoef(s, np.log(p / (1 - p)))[0, 1] == pytest.approx(-1)
    assert sc.OFFSET + sc.FACTOR * np.log(50) == pytest.approx(600)

    # Bin points add up to the score, up to per-bin rounding
    pts = card.points_table()
    total = 0
    for f in feats:
        rows = pts[pts.feature == f]
        total = total + b[f].transform(loans[f]).map(
            dict(zip(rows["woe"], rows["points"], strict=True)))
    assert (total - s).abs().max() <= len(feats) * 0.5 + 1e-9


def test_challenger(loans):
    train, valid = loans.iloc[:20_000], loans.iloc[20_000:]
    m = challenger.fit(train, valid, ["fico", "dti", "noise"], ["home_ownership"],
                       max_rounds=300)
    p = m.predict_pd(valid)
    assert p.between(0, 1).all()
    contrib = m.contributions(valid.head(50))
    lo = np.log(p.head(50) / (1 - p.head(50)))
    np.testing.assert_allclose(contrib.sum(axis=1), lo, atol=1e-6)
    assert (challenger.check_monotone(m, valid) == 1.0).all()
    # Unseen category at scoring time must not crash
    odd = valid.head(5).assign(home_ownership="NONE")
    assert m.predict_pd(odd).notna().all()


def test_realised_loss():
    df = pd.DataFrame({"bad": [1, 1, 0], "funded_amnt": [10_000.0, 5_000, 8_000],
                       "total_rec_prncp": [4_000.0, 5_000, 8_000],
                       "recoveries": [1_000.0, 0, 0], "collection_recovery_fee": [100.0, 0, 0]})
    r = loss.realised_loss(df)
    assert r.loc[0, "ead"] == 6_000
    assert r.loc[0, "lgd"] == pytest.approx(1 - 900 / 6_000)
    assert np.isnan(r.loc[1, "lgd"])  # zero exposure at default
    assert np.isnan(r.loc[2, "ead"]) and r.loc[2, "loss"] == 0


def test_shift_and_vasicek():
    pd_ = np.array([0.05, 0.1, 0.2])
    s = loss.calibrate_shift(pd_, 0.2)
    assert loss.odds_shift(pd_, s).mean() == pytest.approx(0.2)
    assert loss.annual_pd(0.271) == pytest.approx(0.1, abs=1e-3)
    rho = loss.retail_correlation(np.array([1e-4, 0.5]))
    assert rho[0] == pytest.approx(0.16, abs=1e-3) and rho[1] == pytest.approx(0.03, abs=1e-3)
    assert loss.vasicek_pd(0.02, 0.0) == pytest.approx(0.02)
    cap = loss.irb_capital(np.array([0.035]), 0.9, np.array([1000.0]))
    assert (cap["pd_stressed"] > cap["pd_1y"]).all() and (cap["capital"] > 0).all()
