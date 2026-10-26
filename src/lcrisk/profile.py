"""Population profile: feature coverage and default rates by vintage."""

from __future__ import annotations

import pandas as pd

from lcrisk.ingest import APPLICATION_COLS

DERIVED = ["credit_hist_months", "emp_length_yrs", "fico", "loan_to_income", "revol_to_income"]
RAW_DROPPED = {"emp_length", "earliest_cr_line", "fico_range_low", "fico_range_high"}


def feature_columns() -> list[str]:
    return [c for c in APPLICATION_COLS if c not in RAW_DROPPED] + DERIVED


def coverage_by_vintage(pop: pd.DataFrame) -> pd.DataFrame:
    """Share of non-missing values per feature and vintage."""
    return pop.groupby("vintage")[feature_columns()].agg(lambda s: s.notna().mean()).T.round(3)


def default_rates(pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_vintage = pop.groupby("vintage").agg(loans=("bad", "size"), bad_rate=("bad", "mean"))
    by_grade = pop.pivot_table(index="grade", columns="vintage", values="bad", aggfunc="mean")
    return by_vintage, by_grade.round(3)
