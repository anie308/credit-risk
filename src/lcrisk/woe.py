"""Monotonic weight-of-evidence binning and information value.

WoE follows the Siddiqi convention, ln(share of goods / share of bads): positive WoE means
lower risk. With the target coded bad = 1, logistic coefficients on WoE should all be negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "Missing"
EPS = 0.5  # added to good and bad counts so an empty cell doesn't give an infinite WoE


def _woe_table(groups: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"g": groups.to_numpy(), "y": y.to_numpy()}).groupby(
        "g", observed=True, sort=False
    )["y"].agg(n="size", bad="sum")
    t["good"] = t["n"] - t["bad"]
    dist_good = (t["good"] + EPS) / (t["good"].sum() + EPS * len(t))
    dist_bad = (t["bad"] + EPS) / (t["bad"].sum() + EPS * len(t))
    t["bad_rate"] = t["bad"] / t["n"]
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t


def _monotone_merge(edges: list[float], x: np.ndarray, y: np.ndarray,
                    min_share: float) -> list[float]:
    """Merge adjacent bins until bad rate is monotonic and every bin holds min_share of rows."""
    n_min = max(1, int(min_share * len(x)))
    while len(edges) > 2:
        idx = np.digitize(x, edges[1:-1], right=True)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        bads = np.bincount(idx, weights=y, minlength=len(edges) - 1)
        rates = bads / np.maximum(counts, 1)
        # Direction follows the overall trend between the first and last bin
        increasing = rates[-1] >= rates[0]
        diffs = np.diff(rates) if increasing else -np.diff(rates)
        small = np.flatnonzero(counts < n_min)
        if small.size:
            i = int(small[0])
            # Merge a small bin into the neighbour with the closer bad rate
            if i == 0:
                j = 1
            elif i == len(counts) - 1:
                j = i - 1
            else:
                j = i - 1 if abs(rates[i] - rates[i - 1]) <= abs(rates[i] - rates[i + 1]) else i + 1
            drop = max(i, j)
        elif (diffs < 0).any():
            # Remove the boundary at the worst violation
            drop = int(np.argmin(diffs)) + 1
        else:
            break
        del edges[drop]
    return edges


@dataclass
class NumericBinner:
    name: str
    max_bins: int = 10
    min_share: float = 0.05
    min_missing_share: float = 0.005
    edges: list[float] = field(default_factory=list)
    table: pd.DataFrame | None = None

    def fit(self, x: pd.Series, y: pd.Series) -> NumericBinner:
        x = pd.to_numeric(x, errors="coerce").astype(float)
        present = x.notna().to_numpy()
        xv, yv = x.to_numpy()[present], y.to_numpy()[present].astype(float)
        qs = np.unique(np.nanquantile(xv, np.linspace(0, 1, self.max_bins + 1)))
        # Keep the minimum as a cut: for zero-inflated counts (pub_rec, delinquencies) every
        # quantile is 0, and (-inf, 0] vs (0, inf) is the only split there is
        edges = [-np.inf, *qs[:-1].tolist(), np.inf]
        self.edges = _monotone_merge(edges, xv, yv, self.min_share)
        t = _woe_table(self._labels(x), y)
        # Bins in increasing feature value, missing last
        # (pd.cut's own labels, which are formatted differently from str(Interval))
        order = pd.cut(pd.Series([], dtype=float), self.edges).cat.categories.astype(str)
        self.table = t.reindex([b for b in [*order, MISSING] if b in t.index])
        assert len(self.table) == len(t), f"{self.name}: bin labels lost in reordering"
        # A tiny missing group would set its WoE from a handful of loans; drop it so missing
        # values score neutral (WoE 0) instead
        n_missing = self.table["n"].get(MISSING, 0)
        if 0 < n_missing < self.min_missing_share * len(x):
            self.table = self.table.drop(index=MISSING)
        return self

    def _labels(self, x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce").astype(float)
        cut = pd.cut(x, self.edges, right=True).astype(str)
        return cut.where(x.notna(), MISSING)

    def transform(self, x: pd.Series) -> pd.Series:
        woe = self.table["woe"]
        # Missing values in scoring, but not in development, get a neutral WoE of 0
        return self._labels(x).map(woe).fillna(0.0).astype(float)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())


@dataclass
class CategoricalBinner:
    name: str
    min_share: float = 0.02
    mapping: dict[str, str] = field(default_factory=dict)
    table: pd.DataFrame | None = None

    def fit(self, x: pd.Series, y: pd.Series) -> CategoricalBinner:
        x = x.astype("string").fillna(MISSING)
        share = x.value_counts(normalize=True)
        rare = set(share[share < self.min_share].index)
        self.mapping = {c: ("Other" if c in rare else c) for c in share.index}
        self.table = _woe_table(x.map(self.mapping), y).sort_values("woe")
        return self

    def transform(self, x: pd.Series) -> pd.Series:
        groups = x.astype("string").fillna(MISSING).map(self.mapping).fillna("Other")
        return groups.map(self.table["woe"]).fillna(0.0).astype(float)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())


def fit_binners(df: pd.DataFrame, y: pd.Series, numeric: list[str],
                categorical: list[str]) -> dict[str, NumericBinner | CategoricalBinner]:
    binners: dict[str, NumericBinner | CategoricalBinner] = {}
    for c in numeric:
        binners[c] = NumericBinner(c).fit(df[c], y)
    for c in categorical:
        binners[c] = CategoricalBinner(c).fit(df[c], y)
    return binners


def iv_summary(binners: dict) -> pd.DataFrame:
    return (pd.DataFrame({"feature": list(binners), "iv": [b.iv for b in binners.values()],
                          "bins": [len(b.table) for b in binners.values()]})
            .sort_values("iv", ascending=False).reset_index(drop=True))


def woe_frame(df: pd.DataFrame, binners: dict) -> pd.DataFrame:
    return pd.DataFrame({c: b.transform(df[c]) for c, b in binners.items()}, index=df.index)
