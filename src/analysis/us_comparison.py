"""US-population vs AI comparison + WVS-shortlist comparison.

Builds on `loader.py`, `scoring.py`, `aggregate.py`, `wvs.py`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .aggregate import bootstrap_ci, cross_model_item_matrix
from .loader import (
    COUNTRY_CODE_TO_NAME,
    REPO,
    load_items,
    load_wvs_country_means,
    load_wvs_respondents,
)
from .scoring import wvs_country_trust

US_COUNTRY_CODE = 840

# 12 shortlist institutions (user's spec). Map to item_id via the items table.
SHORTLIST_INSTITUTIONS = [
    "The armed forces",
    "Universities",
    "The United Nations (UN)",
    "Elections",
    "The courts",
    "Banks",
    "The police",
    "The churches",
    "Major companies",
    "The press",
    "The government",
    "Political parties",
]

POLITICAL_BUCKETS = {
    "Liberal (Q240 1-4)":      lambda x: (x >= 1) & (x <= 4),
    "Centrist (Q240 5)":       lambda x: (x == 5),
    "Conservative (Q240 6-10)": lambda x: (x >= 6) & (x <= 10),
}
POLITICAL_COLORS = {
    "Liberal (Q240 1-4)":       "#1f77b4",
    "Centrist (Q240 5)":        "#7f7f7f",
    "Conservative (Q240 6-10)": "#d62728",
    "US (all)":                 "#222222",
}


def _wvs_col_for(item: pd.Series) -> str | None:
    c = item["wvs_col"]
    return c if isinstance(c, str) else None


def _trust_from_value(value: float, item: pd.Series) -> float:
    """Map a single respondent value to a 0–1 trust score using the cleaned-WVS direction.

    Mirrors `wvs_country_trust` row-by-row.
    """
    if pd.isna(value):
        return np.nan
    n = float(item["n_points"])
    if item["section"] == "wvs_politicians" and item["reverse_coded"]:
        return float(np.clip((n - value) / (n - 1), 0, 1))
    return float(np.clip((value - 1) / (n - 1), 0, 1))


@lru_cache(maxsize=1)
def _us_respondents() -> pd.DataFrame:
    cols_needed = ["B_COUNTRY", "W_WEIGHT", "Q240"]
    cols_needed += [f"Q{q}P" for q in list(range(57, 64)) + list(range(64, 90))]
    cols_needed += ["Q292A", "Q292B", "Q292C", "Q292D", "Q292E",
                    "Q292F", "Q292G", "Q292H", "Q292I", "Q292K", "Q292O"]
    df = load_wvs_respondents(usecols=cols_needed)
    return df[df["B_COUNTRY"] == US_COUNTRY_CODE].copy()


def us_item_means(items: pd.DataFrame | None = None,
                  political_split: bool = False,
                  bootstrap_n: int = 500) -> pd.DataFrame:
    """Per-item weighted trust mean and bootstrap CI for the US sample.

    Returns columns: group, item_id, wvs_col, trust_mean, trust_lo, trust_hi, n.
    If political_split=True, groups are {Liberal, Centrist, Conservative}, otherwise {US (all)}.
    """
    items = items if items is not None else load_items()
    us = _us_respondents()
    if political_split:
        groups = {g: us[mask(us["Q240"])].copy() for g, mask in POLITICAL_BUCKETS.items()}
    else:
        groups = {"US (all)": us.copy()}

    rng = np.random.default_rng(2026)
    rows = []
    for gname, gdf in groups.items():
        if len(gdf) == 0:
            continue
        w = gdf["W_WEIGHT"].fillna(1.0).clip(lower=0).values
        for _, it in items.iterrows():
            col = _wvs_col_for(it)
            if col is None or col not in gdf.columns:
                continue
            vals = gdf[col].values
            mask_ok = ~np.isnan(vals)
            if mask_ok.sum() < 5:
                continue
            v = vals[mask_ok]
            ww = w[mask_ok]
            n = float(it["n_points"])
            if it["section"] == "wvs_politicians" and it["reverse_coded"]:
                trust = (n - v) / (n - 1)
            else:
                trust = (v - 1) / (n - 1)
            trust = np.clip(trust, 0, 1)
            wmean = float((trust * ww).sum() / ww.sum())
            # bootstrap of the weighted mean
            nbts = bootstrap_n
            idx = rng.integers(0, len(trust), size=(nbts, len(trust)))
            bts = (trust[idx] * ww[idx]).sum(axis=1) / ww[idx].sum(axis=1)
            lo, hi = np.percentile(bts, [2.5, 97.5])
            rows.append({
                "group": gname,
                "item_id": it["id"],
                "wvs_col": col,
                "trust_mean": wmean,
                "trust_lo": float(lo),
                "trust_hi": float(hi),
                "n": int(mask_ok.sum()),
            })
    return pd.DataFrame(rows)


def profile_similarity(scored_llm: pd.DataFrame,
                        items: pd.DataFrame | None = None,
                        section: str = "wvs_confidence") -> pd.DataFrame:
    """Spearman and RMSE between each AI model and each reference group/country.

    Reference groups include US (all), US-Lib/Cent/Cons, WVS pooled, and every WVS-7 country
    that has data for the section.

    Returns columns: model, ref_label, ref_kind, spearman, rmse, n_items.
    """
    items = items if items is not None else load_items()
    items_in = items[items["section"] == section]
    item_ids = list(items_in["id"])

    # AI side
    mat = cross_model_item_matrix(scored_llm).reindex(item_ids)

    # US reference rows
    us_all = us_item_means(items_in, political_split=False).set_index("item_id")["trust_mean"]
    us_pol = us_item_means(items_in, political_split=True)
    us_pol_wide = us_pol.pivot(index="item_id", columns="group", values="trust_mean")

    # WVS country reference
    wvs_means = load_wvs_country_means()
    wvs_t = wvs_country_trust(wvs_means, items_in).pivot(
        index="item_id", columns="B_COUNTRY", values="trust")
    wvs_pooled = wvs_t.mean(axis=1)

    refs = {("US (all)", "US-aggregate"): us_all}
    for col in us_pol_wide.columns:
        refs[(col, "US-political")] = us_pol_wide[col]
    refs[("WVS pooled", "WVS")] = wvs_pooled
    for c in wvs_t.columns:
        cname = COUNTRY_CODE_TO_NAME.get(c, str(c))
        refs[(cname, "WVS-country")] = wvs_t[c]

    rows = []
    for model in mat.columns:
        ai_vec = mat[model]
        for (ref_label, ref_kind), ref_vec in refs.items():
            common = pd.concat([ai_vec, ref_vec], axis=1).dropna()
            if len(common) < 5:
                continue
            rho, _ = spearmanr(common.iloc[:, 0], common.iloc[:, 1])
            rmse = float(np.sqrt(((common.iloc[:, 0] - common.iloc[:, 1]) ** 2).mean()))
            rows.append({
                "model": model, "ref_label": ref_label, "ref_kind": ref_kind,
                "spearman": rho, "rmse": rmse, "n_items": len(common),
            })
    return pd.DataFrame(rows)


def shortlist_item_order(items: pd.DataFrame | None = None) -> list[str]:
    """Return item_ids for the 12-institution shortlist, in user-specified order."""
    items = items if items is not None else load_items()
    by_inst = items.set_index("institution")["id"]
    out = []
    for inst in SHORTLIST_INSTITUTIONS:
        if inst in by_inst.index:
            out.append(by_inst.loc[inst])
    return out
