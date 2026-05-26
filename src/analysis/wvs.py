"""Build WVS-side reference summaries aligned to LLM item ids."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import COUNTRY_CODE_TO_NAME, add_country_names, load_items, load_wvs_country_means
from .scoring import wvs_country_trust


def wvs_summary(items: pd.DataFrame | None = None) -> pd.DataFrame:
    """For every item with a WVS counterpart, return:
        wvs_mean   — pooled mean across countries (weighted by N_respondents)
        wvs_median — country median
        wvs_iqr_lo/hi — country 25/75 percentile
        wvs_min/max — country extremes
        n_countries — countries with valid data
    """
    items = items if items is not None else load_items()
    wvs_means = load_wvs_country_means()
    tidy = wvs_country_trust(wvs_means, items)
    n_resp = wvs_means.set_index("B_COUNTRY")["N_respondents"]
    rows = []
    for item_id, g in tidy.groupby("item_id"):
        valid = g.dropna(subset=["trust"]).copy()
        if valid.empty:
            continue
        valid["w"] = valid["B_COUNTRY"].map(n_resp).fillna(1.0)
        wmean = float((valid["trust"] * valid["w"]).sum() / valid["w"].sum())
        rows.append({
            "item_id": item_id,
            "wvs_col": valid["wvs_col"].iloc[0],
            "wvs_mean": wmean,
            "wvs_median": float(valid["trust"].median()),
            "wvs_iqr_lo": float(valid["trust"].quantile(0.25)),
            "wvs_iqr_hi": float(valid["trust"].quantile(0.75)),
            "wvs_min": float(valid["trust"].min()),
            "wvs_max": float(valid["trust"].max()),
            "n_countries": int(valid["trust"].notna().sum()),
        })
    return pd.DataFrame(rows)


def country_level_trust(items: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tidy long form: B_COUNTRY × item_id × trust, with country names."""
    items = items if items is not None else load_items()
    wvs_means = load_wvs_country_means()
    tidy = wvs_country_trust(wvs_means, items)
    return add_country_names(tidy)


def llm_country_resemblance(scored: pd.DataFrame, items: pd.DataFrame,
                             use_section: str = "wvs_confidence") -> pd.DataFrame:
    """For each LLM, find which WVS country best matches its per-item profile.

    Returns one row per (model, country) with Spearman correlation + RMSE on a
    common item set restricted to `use_section`. The argmax over countries gives
    the closest match.
    """
    from scipy.stats import spearmanr
    items_in = items[items["section"] == use_section]
    country_trust = country_level_trust(items_in).dropna(subset=["trust"])
    llm_means = (scored[scored["item_id"].isin(items_in["id"])]
                 .dropna(subset=["trust"])
                 .groupby(["model", "item_id"])["trust"].mean().reset_index())
    rows = []
    for model, g in llm_means.groupby("model"):
        llm_vec = g.set_index("item_id")["trust"]
        for country, cg in country_trust.groupby("country"):
            cv = cg.set_index("item_id")["trust"].reindex(llm_vec.index).dropna()
            if len(cv) < 5:
                continue
            common = llm_vec.reindex(cv.index)
            rho, p = spearmanr(common.values, cv.values)
            rmse = float(np.sqrt(((common.values - cv.values) ** 2).mean()))
            rows.append({"model": model, "country": country,
                         "B_COUNTRY": cg["B_COUNTRY"].iloc[0],
                         "spearman": rho, "p": p, "rmse": rmse,
                         "n_items": len(cv)})
    return pd.DataFrame(rows)
