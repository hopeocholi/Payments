"""
Cleaning and behavioural feature engineering for the credit-default task.

Targets the "Default of Credit Card Clients" schema: a credit limit, four
demographic fields, six months of repayment-status history (PAY_0..PAY_6),
six months of bill amounts and six of payment amounts, and a binary
next-month default target. The engineered features capture repayment
behaviour that the raw columns only imply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def clean_credit_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented EDUCATION and MARRIAGE codes into their 'other' bucket.

    The documented EDUCATION codes are 1-4; codes 0, 5 and 6 are undocumented
    and are folded into 4 ("others"). MARRIAGE code 0 is folded into 3.
    """
    df = raw.copy()
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def engineer_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural features derived from repayment and billing history.

    Adds utilisation ratio, months delinquent, average payment delay, six-month
    totals paid and billed, and a payment-to-bill ratio. Extreme ratios are
    clipped to keep them robust to divide-by-small-balance artefacts.
    """
    df = df.copy()

    df["UTILISATION_RATIO"] = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(
        lower=-2, upper=3
    )
    df["MONTHS_DELINQUENT"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["AVG_PAY_DELAY"] = df[PAY_COLS].clip(lower=0).mean(axis=1)
    df["TOTAL_PAID_6M"] = df[PAY_AMT_COLS].sum(axis=1)
    df["TOTAL_BILLED_6M"] = df[BILL_COLS].sum(axis=1)
    df["PAYMENT_TO_BILL_RATIO"] = (
        df["TOTAL_PAID_6M"] / df["TOTAL_BILLED_6M"].replace(0, np.nan)
    ).clip(upper=5).fillna(0)

    return df


def feature_columns():
    """Return the ordered feature list used for modelling."""
    return (
        ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
        + PAY_COLS
        + BILL_COLS
        + PAY_AMT_COLS
        + [
            "UTILISATION_RATIO",
            "MONTHS_DELINQUENT",
            "AVG_PAY_DELAY",
            "TOTAL_PAID_6M",
            "TOTAL_BILLED_6M",
            "PAYMENT_TO_BILL_RATIO",
        ]
    )
