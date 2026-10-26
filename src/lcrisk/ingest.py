"""Raw Lending Club CSV -> typed parquet of the modelling population."""

from __future__ import annotations

import pandas as pd

from lcrisk import config

# Known at application. Post-origination fields (payments, last FICO, hardship, settlement)
# are leakage and never enter this list.
APPLICATION_COLS = [
    "loan_amnt", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "application_type", "pub_rec_bankruptcies",
    "collections_12_mths_ex_med", "acc_now_delinq", "tax_liens",
    # Bureau fields LC added later; kept only if coverage holds across every vintage
    "tot_cur_bal", "total_rev_hi_lim", "mort_acc", "num_actv_rev_tl", "bc_util",
    "mo_sin_old_rev_tl_op", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "mths_since_recent_inq", "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
]

# Lending Club's own pricing: a benchmark, never a feature
LC_MODEL_COLS = ["grade", "sub_grade", "int_rate"]

# Loan identity, outcome and loss fields (for target, EAD and LGD only)
OUTCOME_COLS = [
    "id", "issue_d", "term", "loan_status", "policy_code", "funded_amnt", "installment",
    "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "debt_settlement_flag",
]

# Excluded on purpose: geography (addr_state, zip_code) for fair-lending reasons, and free text
# (emp_title, desc, title), which needs its own review.

BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}


def read_raw(path=config.RAW_CSV) -> pd.DataFrame:
    cols = OUTCOME_COLS + LC_MODEL_COLS + APPLICATION_COLS
    df = pd.read_csv(path, usecols=cols, engine="pyarrow")
    return df[cols]


def _month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["issue_d"] = _month(df["issue_d"])
    df["last_pymnt_d"] = _month(df["last_pymnt_d"])
    earliest = _month(df["earliest_cr_line"])
    # Credit history length at application, in months
    df["credit_hist_months"] = (
        (df["issue_d"].dt.year - earliest.dt.year) * 12
        + (df["issue_d"].dt.month - earliest.dt.month)
    )
    df = df.drop(columns="earliest_cr_line")
    df["term_months"] = pd.to_numeric(
        df["term"].astype("string").str.extract(r"(\d+)")[0], errors="coerce"
    )
    df = df.drop(columns="term")
    df["emp_length_yrs"] = (
        df["emp_length"].astype("string")
        .str.replace("< 1 year", "0", regex=False)
        .str.extract(r"(\d+)")[0]
        .astype("Float64")
    )
    df = df.drop(columns="emp_length")
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high"])
    # Affordability. Not installment: it is computed from LC's interest rate, so it would
    # carry LC's grade into the model.
    income = df["annual_inc"].where(df["annual_inc"] > 0)
    df["loan_to_income"] = df["loan_amnt"] / income
    df["revol_to_income"] = df["revol_bal"] / income
    for c in ["int_rate", "revol_util"]:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype("string").str.rstrip("%"), errors="coerce")
    return df


def select_population(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply population filters. Returns (population, waterfall of exclusions)."""
    steps = []

    def step(name: str, mask: pd.Series) -> None:
        nonlocal df
        before = len(df)
        df = df[mask]
        steps.append({"step": name, "rows_in": before, "excluded": before - len(df),
                      "rows_out": len(df)})

    step("valid loan id and issue date", df["id"].notna() & df["issue_d"].notna())
    step(f"{config.TERM_MONTHS}-month term", df["term_months"] == config.TERM_MONTHS)
    step(f"issued on or before {config.LAST_ISSUE[:7]}",
         df["issue_d"] <= pd.Timestamp(config.LAST_ISSUE))
    step("individual application", df["application_type"].fillna("Individual") == "Individual")
    step("final outcome (paid or charged off)",
         df["loan_status"].isin(BAD_STATUSES | GOOD_STATUSES))

    df = df.copy()
    df["bad"] = df["loan_status"].isin(BAD_STATUSES).astype("int8")
    df["vintage"] = df["issue_d"].dt.year
    df["issue_q"] = df["issue_d"].dt.to_period("Q").astype(str)
    return df.reset_index(drop=True), pd.DataFrame(steps)


def build(path=config.RAW_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    pop, waterfall = select_population(clean(read_raw(path)))
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    pop.to_parquet(config.PROCESSED / "population.parquet", index=False)
    waterfall.to_csv(config.PROCESSED / "waterfall.csv", index=False)
    return pop, waterfall
