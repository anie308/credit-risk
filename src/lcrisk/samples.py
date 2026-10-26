"""Development / test / out-of-time samples."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from lcrisk import config


def split(pop: pd.DataFrame, test_size: float = 0.3) -> dict[str, pd.DataFrame]:
    """Development window split at random into train/test (stratified on the target); later
    vintages are out-of-time, one sample per year."""
    start, end = pd.Timestamp(config.DEV_START), pd.Timestamp(config.DEV_END)
    dev = pop[pop["issue_d"].between(start, end)]
    train, test = train_test_split(dev, test_size=test_size, stratify=dev["bad"],
                                   random_state=config.SEED)
    out = {"train": train, "test": test}
    for y in config.OOT_YEARS:
        out[f"oot_{y}"] = pop[pop["vintage"] == y]
    return out


def time_holdout(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last development quarter held out from train, for early stopping and tuning."""
    last = train["issue_q"].max()
    return train[train["issue_q"] < last], train[train["issue_q"] == last]
