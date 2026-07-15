"""
Payments and Transaction Services CA1 — analysis library.

Reusable, tested components behind a three-part study:

- Section 1: a U.S. crypto-enforcement fines database, cleaned and feature-
  engineered for exploratory analysis and a decision-tree model.
- Section 2: a credit-default risk pipeline on the "Default of Credit Card
  Clients" data, with behavioural feature engineering.
- Section 3: the Merton structural credit-risk model applied to a GENIUS-Act
  payment-stablecoin issuer.

Modules
-------
crypto_fines    Fine-amount parsing, entity and outcome classification, era tagging.
credit_risk     Credit-card cleaning and behavioural feature engineering.
merton          Merton model: equity-as-call, default probability, distance to default.
"""

__version__ = "1.0.0"
__author__ = "Hope Eneojo Ocholi"

from .crypto_fines import clean_fine, classify_entity, outcome_family, tag_era
from .credit_risk import clean_credit_data, engineer_behavioural_features
from .merton import (
    merton_equity_value,
    merton_default_probability,
    distance_to_default,
)

__all__ = [
    "clean_fine",
    "classify_entity",
    "outcome_family",
    "tag_era",
    "clean_credit_data",
    "engineer_behavioural_features",
    "merton_equity_value",
    "merton_default_probability",
    "distance_to_default",
    "__version__",
]
