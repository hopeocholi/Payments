"""Unit tests for the crypto-fines, credit-risk and Merton modules."""

import numpy as np
import pandas as pd
import pytest

from src.credit_risk import (
    clean_credit_data,
    engineer_behavioural_features,
    feature_columns,
)
from src.crypto_fines import clean_fine, classify_entity, outcome_family, tag_era
from src.merton import (
    credit_spread,
    distance_to_default,
    merton_default_probability,
    merton_equity_value,
)


# --- crypto_fines --------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$0.7 million", 0.7),
        ("$1.2 billion", 1200.0),
        ("$0", 0.0),
        ("$4.5 million", 4.5),
    ],
)
def test_clean_fine_amounts(raw, expected):
    assert clean_fine(raw) == pytest.approx(expected)


def test_clean_fine_non_monetary_is_nan():
    assert np.isnan(clean_fine("pending litigation"))
    assert np.isnan(clean_fine("seized assets under seal"))
    assert np.isnan(clean_fine(None))


def test_classify_entity():
    assert classify_entity("Ripple Labs Inc.", "Civil") == "Corporate Entity"
    assert classify_entity("Alexander Vinnik", "Money Laundering") == "Individual"
    assert (
        classify_entity("SEC Crypto Task Force (Policy Action)", "Policy Reform")
        == "Policy / Legislative Event"
    )


def test_outcome_family():
    assert outcome_family("Settled") == "Settled / Consent Order"
    assert outcome_family("Case dismissed") == "Dismissed / Closed (No Penalty)"
    assert outcome_family("Convicted at trial") == "Convicted / Default Judgment"


def test_tag_era_boundary():
    assert tag_era("5-May-15") == "Pre-Trump (Biden/Gensler)"
    assert tag_era("1-Jun-25") == "Trump Administration"
    assert pd.isna(tag_era("not a date"))


def test_clean_fine_on_real_dataset():
    df = pd.read_csv("data/DataCryptoFines.csv")
    parsed = df["Amount of Fine"].apply(clean_fine)
    # a clear majority of rows carry a parseable numeric fine
    assert parsed.notna().sum() > len(df) * 0.7


# --- credit_risk ---------------------------------------------------------
def _toy_credit_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {
        "LIMIT_BAL": rng.integers(10000, 500000, n),
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(0, 7, n),
        "MARRIAGE": rng.integers(0, 4, n),
        "AGE": rng.integers(21, 70, n),
    }
    for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        data[c] = rng.integers(-2, 9, n)
    for i in range(1, 7):
        data[f"BILL_AMT{i}"] = rng.integers(-5000, 200000, n)
        data[f"PAY_AMT{i}"] = rng.integers(0, 50000, n)
    data["default payment next month"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_clean_credit_data_folds_codes():
    df = clean_credit_data(_toy_credit_frame())
    assert set(df["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert 0 not in df["MARRIAGE"].unique()


def test_engineer_behavioural_features():
    df = engineer_behavioural_features(_toy_credit_frame())
    for col in [
        "UTILISATION_RATIO",
        "MONTHS_DELINQUENT",
        "AVG_PAY_DELAY",
        "PAYMENT_TO_BILL_RATIO",
    ]:
        assert col in df.columns
    assert (df["MONTHS_DELINQUENT"] >= 0).all()
    assert (df["MONTHS_DELINQUENT"] <= 6).all()
    assert df["PAYMENT_TO_BILL_RATIO"].max() <= 5


def test_feature_columns_count():
    # 5 demographic + 6 pay + 6 bill + 6 pay_amt + 6 engineered = 29
    assert len(feature_columns()) == 29


# --- merton --------------------------------------------------------------
def test_merton_well_capitalised_low_default():
    """A 1:1+ reserved, low-volatility issuer has tiny default probability."""
    pd_ = merton_default_probability(1.01, 1.00, 0.04, 0.02, 1.0)
    assert 0 < pd_ < 0.05


def test_merton_default_prob_rises_with_volatility():
    low = merton_default_probability(1.05, 1.00, 0.04, 0.05, 1.0)
    high = merton_default_probability(1.05, 1.00, 0.04, 0.40, 1.0)
    assert high > low


def test_merton_equity_plus_debt_equals_assets():
    V, D, r, sigma, T = 1.20, 1.00, 0.03, 0.25, 2.0
    equity = merton_equity_value(V, D, r, sigma, T)
    risky_debt = V - equity
    assert equity + risky_debt == pytest.approx(V, abs=1e-9)


def test_distance_to_default_higher_when_safer():
    safe = distance_to_default(1.50, 1.00, 0.03, 0.20, 1.0)
    risky = distance_to_default(1.05, 1.00, 0.03, 0.20, 1.0)
    assert safe > risky


def test_credit_spread_nonnegative():
    spread = credit_spread(1.10, 1.00, 0.03, 0.30, 1.0)
    assert spread >= 0
