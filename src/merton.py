"""
Merton structural credit-risk model.

Underpins Section 3, which applies the Merton framework to a payment-stablecoin
issuer under the GENIUS Act. A leveraged firm's equity is a call option on its
assets struck at the face value of debt: equity holders are wiped out if asset
value falls below liabilities at the horizon. The GENIUS Act's 1:1 reserve
requirement maps onto the asset-value and default-threshold terms here.
"""

from __future__ import annotations

from math import exp, log, sqrt

from scipy.stats import norm


def _d1_d2(asset_value, debt_face, r, sigma, T):
    d1 = (log(asset_value / debt_face) + (r + 0.5 * sigma ** 2) * T) / (
        sigma * sqrt(T)
    )
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def merton_equity_value(asset_value, debt_face, r, sigma, T):
    """Equity value as a call option on the firm's assets (Merton 1974).

    Parameters
    ----------
    asset_value : float
        Market value of the firm's assets, ``V``.
    debt_face : float
        Face value of debt (the strike), ``D``.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Asset volatility.
    T : float
        Horizon in years.
    """
    d1, d2 = _d1_d2(asset_value, debt_face, r, sigma, T)
    return asset_value * norm.cdf(d1) - debt_face * exp(-r * T) * norm.cdf(d2)


def distance_to_default(asset_value, debt_face, r, sigma, T):
    """Distance to default: the ``d2`` term, in standard deviations."""
    _, d2 = _d1_d2(asset_value, debt_face, r, sigma, T)
    return d2


def merton_default_probability(asset_value, debt_face, r, sigma, T):
    """Risk-neutral probability of default, ``N(-d2)``.

    This is the probability that asset value falls below the debt face value
    at the horizon under the risk-neutral measure.
    """
    _, d2 = _d1_d2(asset_value, debt_face, r, sigma, T)
    return norm.cdf(-d2)


def credit_spread(asset_value, debt_face, r, sigma, T):
    """Merton credit spread over the risk-free rate, in continuous terms.

    Returns the yield spread implied by the risky-debt value; zero (or near
    zero) for a well-collateralised issuer such as a 1:1-reserved stablecoin.
    """
    equity = merton_equity_value(asset_value, debt_face, r, sigma, T)
    risky_debt = asset_value - equity
    if risky_debt <= 0:
        return 0.0
    implied_yield = -log(risky_debt / debt_face) / T
    return implied_yield - r
