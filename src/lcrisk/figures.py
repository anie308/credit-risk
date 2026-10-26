"""Figures for the model documentation and blog post.

Model colours are the first three slots of a validated categorical palette (all pairs pass the
colour-blind checks). LC's aqua is low-contrast on white, so every chart carries a legend and
the documentation repeats the numbers in tables. Grades are ordinal, so they get one
sequential hue rather than seven categorical ones.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from lcrisk import config, metrics  # noqa: E402
from lcrisk.validate import TABLES, load_scored, rating  # noqa: E402

SCORECARD, LIGHTGBM, LC, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a8a"
MODEL_STYLE = {
    "pd_scorecard": ("Scorecard", SCORECARD, "-"),
    "pd_lightgbm": ("LightGBM", LIGHTGBM, "-"),
    "lc_rank": ("LC sub-grade", LC, "--"),
}
GRADE_RAMP = ["#c6dbf7", "#9ec2ef", "#76a8e6", "#4f8edd", "#2a78d6", "#1f5ca6", "#143f73"]
DEV_SPAN = ("2012Q4", "2013Q4")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
})


def _save(fig, name: str):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _pct(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=None)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _shade_dev(ax, quarters: list[str]):
    i0, i1 = quarters.index(DEV_SPAN[0]), quarters.index(DEV_SPAN[1])
    ax.axvspan(i0 - 0.5, i1 + 0.5, color="#eef3fb", lw=0, zorder=0)
    ax.text((i0 + i1) / 2, ax.get_ylim()[1], "development", ha="center", va="top",
            fontsize=7.5, color="#555")


def default_by_grade(pop: pd.DataFrame):
    """Observed default rate by LC grade and vintage: the drift the models have to survive."""
    t = pop.pivot_table(index="vintage", columns="grade", values="bad", aggfunc="mean")
    t = t.loc[2009:]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axvspan(2012.75, 2013.5, color="#eef3fb", lw=0, zorder=0)
    ax.axvspan(2013.5, 2015.5, color="#fbf1ec", lw=0, zorder=0)
    for g, color in zip(t.columns, GRADE_RAMP, strict=True):
        ax.plot(t.index, t[g], color=color, lw=2, marker="o", ms=4)
        ax.text(2015.12, t[g].iloc[-1], g, va="center", fontsize=8, color=INK)
    ax.text(2013.12, 0.47, "dev", ha="center", fontsize=7.5, color="#555")
    ax.text(2014.5, 0.47, "out of time", ha="center", fontsize=7.5, color="#555")
    ax.set_xlim(2008.7, 2015.4)
    ax.set_ylim(0, 0.5)
    _pct(ax)
    ax.set_ylabel("Default rate (36-month loans)")
    ax.set_xlabel("Issue year")
    ax.set_title("Default rate by Lending Club grade: rising within every grade after 2013",
                 loc="left")
    return _save(fig, "fig1_default_by_grade")


def roc(scored: pd.DataFrame, sample: str = "oot_2015"):
    g = scored[scored["sample"] == sample]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":")
    for col, (label, color, ls) in MODEL_STYLE.items():
        fpr, tpr, _ = roc_curve(g["bad"], g[col])
        gini = metrics.gini(g["bad"], g[col])
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f"{label}  (Gini {gini:.3f})")
    ax.set_xlabel("Share of good loans flagged")
    ax.set_ylabel("Share of defaults caught")
    ax.set_title(f"ROC, out-of-time {sample[-4:]} ({len(g):,} loans)", loc="left")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    return _save(fig, "fig2_roc_oot")


def drift(by_q: pd.DataFrame, scored: pd.DataFrame):
    """Predicted vs observed default rate by quarter, with score PSI underneath."""
    # The in-window quarters include the train rows too, so recompute them on all dev loans
    dev = scored[scored["sample"].isin(["train", "test"])].groupby("issue_q").agg(
        odr=("bad", "mean"), pd_scorecard=("pd_scorecard", "mean"),
        pd_lightgbm=("pd_lightgbm", "mean"))
    t = by_q.set_index("issue_q")
    t.loc[dev.index, ["odr", "pd_scorecard", "pd_lightgbm"]] = dev
    q = t.index.tolist()
    x = np.arange(len(q))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(x, t["odr"], color=INK, lw=2, marker="o", ms=4, label="Observed default rate")
    a1.plot(x, t["pd_scorecard"], color=SCORECARD, lw=2, label="Scorecard mean PD")
    a1.plot(x, t["pd_lightgbm"], color=LIGHTGBM, lw=2, label="LightGBM mean PD")
    a1.set_ylim(0.10, 0.165)
    _pct(a1)
    _shade_dev(a1, q)
    a1.set_ylabel("Default rate")
    a1.legend(loc="lower right", ncol=1)
    a1.set_title("The population looks the same, but defaults more", loc="left")

    a2.bar(x, t["psi_score"], width=0.6, color=SCORECARD)
    a2.axhline(0.1, color=GREY, lw=0.8, ls="--")
    a2.text(len(q) - 0.6, 0.1, "0.10 amber", fontsize=7, va="bottom", ha="right", color="#555")
    a2.set_ylim(0, 0.12)
    a2.set_ylabel("Score PSI\nvs train")
    a2.set_xticks(x, q, rotation=45, ha="right")
    return _save(fig, "fig3_drift_psi")


def reliability(scored: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    for ax, sample, title in [(axes[0], "test", "Test (in time)"),
                              (axes[1], "oot_2015", "Out of time, 2015")]:
        g = scored[scored["sample"] == sample]
        ax.plot([0, 0.45], [0, 0.45], color=GREY, lw=0.8, ls=":")
        for col in ["pd_scorecard", "pd_lightgbm"]:
            label, color, _ = MODEL_STYLE[col]
            r = metrics.reliability(g["bad"], g[col], bins=20)
            ax.plot(r["pd_mean"], r["odr"], color=color, lw=2, marker="o", ms=4, label=label)
        ax.set_xlim(0, 0.45)
        ax.set_ylim(0, 0.45)
        _pct(ax)
        _pct(ax, "x")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Predicted PD (20 equal-count bins)")
    axes[0].set_ylabel("Observed default rate")
    axes[0].legend(loc="upper left")
    return _save(fig, "fig4_reliability")


def master_scale(scored: pd.DataFrame, col: str = "pd_scorecard", sample: str = "oot_2015"):
    g = scored[scored["sample"] == sample]
    t = metrics.calibration_by_grade(g["bad"], g[col], rating(g[col]))
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - 0.18, t["pd_mean"], width=0.34, color=SCORECARD, label="Predicted PD")
    ax.bar(x + 0.18, t["odr"], width=0.34, color=INK, label="Observed default rate")
    for i, (_, r) in enumerate(t.iterrows()):
        if r["jeffreys_p"] < 0.01:
            ax.text(i + 0.18, r["odr"] + 0.008, "✕", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{gr}\n{n:,}" for gr, n in zip(t.index, t["n"], strict=True)])
    _pct(ax)
    ax.set_ylabel("Default rate")
    ax.set_title("Scorecard master scale, 2015: ✕ = Jeffreys test rejects the PD at 1%",
                 loc="left")
    ax.legend(loc="upper left")
    return _save(fig, "fig5_master_scale")


def gini_by_quarter(by_q: pd.DataFrame, scored: pd.DataFrame):
    lc = scored[scored["sample"] != "train"].groupby("issue_q").apply(
        lambda g: metrics.gini(g["bad"], g["lc_rank"]), include_groups=False)
    t = by_q.set_index("issue_q")
    q = t.index.tolist()
    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(x, t["gini_scorecard"], color=SCORECARD, lw=2, marker="o", ms=4, label="Scorecard")
    ax.plot(x, t["gini_lightgbm"], color=LIGHTGBM, lw=2, marker="o", ms=4, label="LightGBM")
    ax.plot(x, lc.reindex(q), color=LC, lw=2, ls="--", marker="s", ms=4, label="LC sub-grade")
    ax.set_ylim(0.2, 0.42)
    _shade_dev(ax, q)
    ax.set_xticks(x, q, rotation=45, ha="right")
    ax.set_ylabel("Gini")
    ax.set_title("Ranking power by issue quarter (dev quarters: test sample only)", loc="left")
    ax.legend(loc="lower right", ncol=3)
    return _save(fig, "fig6_gini_by_quarter")


def woe_profiles(card):
    feats = card.features
    cols = 5
    rows = int(np.ceil(len(feats) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.1 * rows))
    for ax, f in zip(axes.flat, feats, strict=False):
        t = card.binners[f].table
        colors = [GREY if b == "Missing" else SCORECARD for b in t.index]
        ax.bar(range(len(t)), t["bad_rate"], width=0.7, color=colors)
        ax.set_title(f, fontsize=8, loc="left")
        ax.set_xticks([])
        _pct(ax)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(feats):]:
        ax.axis("off")
    fig.suptitle("Bad rate by WoE bin, scorecard features (bins in increasing feature value; "
                 "grey = missing)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig7_woe_profiles")


def run() -> list:
    from lcrisk.pipeline import load_models

    scored = load_scored()
    pop = pd.read_parquet(config.PROCESSED / "population.parquet",
                          columns=["vintage", "grade", "bad"])
    # Dev quarters in by_quarter are test-sample rows only, which is the fair basis for Gini
    by_q = pd.read_csv(TABLES / "by_quarter.csv")
    card = load_models()["scorecard"]
    return [default_by_grade(pop), roc(scored), drift(by_q, scored), reliability(scored),
            master_scale(scored), gini_by_quarter(by_q, scored), woe_profiles(card)]
