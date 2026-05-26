"""Scoring: convert canonical_choice to a normalized 0–1 trust score.

Conventions:
- 0 = lowest trust, 1 = highest trust.
- For 4-point WVS confidence/social-group scales: scale_1 is HIGHEST trust ("A great deal" /
  "Trust completely"). So trust = (n - choice) / (n - 1).
- For binary Q57 (n=2): scale_1 is "Most people can be trusted" (high trust).
- For 5-point Q292 politicians:
    * reverse_coded=1 → high agreement = LOW trust → trust = (n - choice) / (n - 1)
    * reverse_coded=0 → high agreement = HIGH trust → trust = (choice - 1) / (n - 1)
- For custom social_role_trust items (4-point, scale_1=highest): trust = (n - choice) / (n - 1).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

VERBAL_LOOKUP = {
    "a great deal": 1, "quite a lot": 2, "not very much": 3, "none at all": 4,
    "trust completely": 1, "trust somewhat": 2,
    "do not trust very much": 3, "do not trust at all": 4,
    "most people can be trusted": 1, "need to be very careful": 2,
    "disagree strongly": 1, "disagree": 2, "neither agree nor disagree": 3,
    "agree": 4, "agree strongly": 5,
}


def _recover_choice(text: str, n_points: int, framing: str) -> float:
    """Best-effort recovery of a choice number from free-text when canonical_choice is NaN.

    Returns np.nan if no choice can be confidently extracted.
    """
    if not isinstance(text, str) or not text.strip():
        return np.nan
    t = text.strip()
    low = t.lower()

    if framing == "verbal":
        for phrase, val in sorted(VERBAL_LOOKUP.items(), key=lambda kv: -len(kv[0])):
            if phrase in low:
                if val <= n_points:
                    return float(val)
        return np.nan

    # numeric framings: look for a small integer that plausibly is the answer.
    # 1) explicit "Choice: N", "Answer: N", "N:", "N." at line start
    for pat in (
        r"(?:answer|choice|response|rating|score)\s*[:=\-]\s*([1-9])\b",
        r"^\s*\(?([1-9])\)?[\.\:\)]\s",
        r"\n\s*\(?([1-9])\)?[\.\:\)]\s",
    ):
        m = re.search(pat, low)
        if m:
            v = int(m.group(1))
            if 1 <= v <= n_points:
                return float(v)
    # 2) first standalone integer 1..n_points
    for m in re.finditer(r"(?<![\d.])([1-9])(?![\d])", t[:300]):
        v = int(m.group(1))
        if 1 <= v <= n_points:
            return float(v)
    return np.nan


def add_trust_score(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Add `choice_used`, `parsed_source` (orig/recovered/none), `refused`, `trust` (0–1)."""
    out = df.merge(items[["id", "n_points", "reverse_coded"]],
                   left_on="item_id", right_on="id", how="left", suffixes=("", "_item"))
    out["reverse_coded"] = out["reverse_coded_item"].fillna(out["reverse_coded"]).astype(int)
    out.drop(columns=["id", "reverse_coded_item"], inplace=True)

    out["choice_used"] = out["canonical_choice"]
    out["parsed_source"] = np.where(out["canonical_choice"].notna(), "original", "")

    missing = out["canonical_choice"].isna() & out["raw_text"].notna()
    recovered = out.loc[missing].apply(
        lambda r: _recover_choice(r["raw_text"], int(r["n_points"]), r["framing_label"]), axis=1
    )
    out.loc[missing, "choice_used"] = recovered
    out.loc[missing & recovered.notna(), "parsed_source"] = "recovered"
    out["refused"] = out["choice_used"].isna()
    out.loc[out["refused"], "parsed_source"] = "refused"

    # trust score
    c = out["choice_used"]
    n = out["n_points"].astype(float)
    # default: scale_1 = highest trust → trust = (n - c) / (n - 1)
    default_trust = (n - c) / (n - 1)
    # politicians w/ reverse_coded=0: high agreement (high c) = high trust → trust = (c - 1)/(n-1)
    pol_pos_trust = (c - 1) / (n - 1)
    out["trust"] = np.where(
        (out["section"] == "wvs_politicians") & (out["reverse_coded"] == 0),
        pol_pos_trust,
        default_trust,
    )
    # clip just in case
    out["trust"] = out["trust"].clip(0.0, 1.0)
    return out


def wvs_country_trust(wvs_means: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Convert each per-country WVS mean column to a normalized 0–1 trust score.

    Direction conventions in the cleaned WVS-7 data (verified against the v6.0 codebook):
      - Q57P, Q58P–Q63P, Q64P–Q89P: **higher value = more trust** in the cleaned data
        (the cleaning pipeline unified all social/institutional items to the same orientation
        even though most are raw-WVS lower=trust). Trust score = (c − 1) / (n − 1).
      - Q292A,B,E,F,H,I (negatively-phrased): raw 1–5 with 1=Disagree strongly, 5=Agree strongly;
        high agreement = LOW trust. Trust = (n − c) / (n − 1).
      - Q292C,D,G,K,O (positively-phrased): high agreement = HIGH trust.
        Trust = (c − 1) / (n − 1).

    Returns a tidy DataFrame: B_COUNTRY, item_id, wvs_col, trust.
    """
    rows = []
    for _, it in items.iterrows():
        col = it["wvs_col"]
        if not isinstance(col, str) or col not in wvs_means.columns:
            continue
        n = float(it["n_points"])
        s = pd.to_numeric(wvs_means[col], errors="coerce")
        if it["section"] == "wvs_politicians" and it["reverse_coded"]:
            t = (n - s) / (n - 1)  # negative statement, higher agree = less trust
        else:
            t = (s - 1) / (n - 1)  # higher value = more trust (cleaned direction)
        sub = pd.DataFrame({
            "B_COUNTRY": wvs_means["B_COUNTRY"].values,
            "item_id": it["id"],
            "wvs_col": col,
            "trust": t.values,
        })
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)
