"""
Cleaning and classification for the U.S. crypto-enforcement fines database.

The raw dataset records enforcement actions by the SEC, CFTC, DOJ, FinCEN,
OFAC, NYDFS and others. These helpers turn free-text fields into analysable
categories: a robust fine-amount parser that preserves the distinction between
"no monetary fine reported" and "explicitly zero paid", an entity-type
classifier, an outcome-family classifier, and an era tag around the January
2025 change in federal crypto-regulatory posture.
"""

from __future__ import annotations

import re

import numpy as np

# The federal regulatory posture shifts at the January 2025 inauguration.
TRUMP_ERA_START = "2025-01-20"


def clean_fine(x):
    """Parse a numeric fine in USD millions from a free-text amount field.

    Preserves the analytically important distinction between a genuinely
    non-monetary entry (returned as NaN) and an action that explicitly
    resulted in zero paid (returned as 0.0).
    """
    if not isinstance(x, str):
        return np.nan
    x_clean = x.replace(",", "")

    m = re.search(r"\$\s*([\d.]+)\s*million", x_clean, flags=re.I)
    if m:
        return float(m.group(1))

    m = re.search(r"\$\s*([\d.]+)\s*billion", x_clean, flags=re.I)
    if m:
        return float(m.group(1)) * 1000.0

    if re.search(
        r"n/a|disgorgement and penalties|under seal|pending|seized",
        x_clean,
        flags=re.I,
    ):
        return np.nan

    if "$0" in x_clean or x_clean.strip() == "$0":
        return 0.0

    m = re.search(r"\$\s*([\d.]+)\b", x_clean)
    if m:
        val = float(m.group(1))
        return val / 1_000_000.0 if val >= 1000 else val

    return np.nan


def classify_entity(name, category):
    """Classify a respondent as a corporate, individual, or policy event.

    Checks both the entity name and the category text, because policy and
    legislative events carry varied category labels but consistent
    parenthetical markers on the name.
    """
    name_l = str(name).lower()
    cat_l = str(category).lower()

    policy_keywords = [
        "policy", "reform", "regulatory pause", "legislation enacted",
        "annual report", "rule amendment",
    ]
    policy_name_markers = [
        "(policy action", "(public law)", "(rules of practice",
        "(rule amendment", "(annual report", "(statement",
    ]
    if any(k in cat_l for k in policy_keywords) or any(
        m in name_l for m in policy_name_markers
    ):
        return "Policy / Legislative Event"

    corporate_markers = [
        " inc", " llc", " ltd", " corp", " labs", " group", " foundation",
        " trading", " holdings", " network", " capital", " technologies",
        " technology", " platform", " exchange", " finance", " markets",
    ]
    if any(k in name_l for k in corporate_markers):
        return "Corporate Entity"

    return "Individual"


def outcome_family(outcome):
    """Collapse a free-text outcome into a small set of outcome families."""
    o_l = str(outcome).lower()
    if any(k in o_l for k in ["dismiss", "closed", "no enforcement", "no action"]):
        return "Dismissed / Closed (No Penalty)"
    if any(k in o_l for k in ["convict", "guilty", "sentenced", "default judgment"]):
        return "Convicted / Default Judgment"
    if "settle" in o_l or "consent" in o_l:
        return "Settled / Consent Order"
    if ("charge" in o_l and "pending" in o_l) or "indicted" in o_l or "ongoing" in o_l:
        return "Charged / Litigation Ongoing"
    if any(k in o_l for k in ["policy", "guidance", "rule", "report", "enacted", "announce"]):
        return "Policy / Procedural Outcome"
    return "Other / Final Judgment"


def tag_era(date, cutoff=TRUMP_ERA_START):
    """Tag a record as pre- or post- the January 2025 posture change.

    Accepts anything ``pandas.to_datetime`` understands. Returns one of
    ``"Pre-Trump (Biden/Gensler)"`` or ``"Trump Administration"``; unparseable
    dates return NaN.
    """
    import pandas as pd

    parsed = pd.to_datetime(date, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return (
        "Trump Administration"
        if parsed >= pd.to_datetime(cutoff)
        else "Pre-Trump (Biden/Gensler)"
    )
