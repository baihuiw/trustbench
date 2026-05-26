"""Aggregations over LLM trust scores: model × item × framing means with bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Return (mean, lo, hi) — percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = rng or np.random.default_rng(42)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = v[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def model_item_means(scored: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean trust + bootstrap CI, grouped by `by` (default: model, item_id)."""
    by = by or ["model", "item_id"]
    rng = np.random.default_rng(42)
    rows = []
    for keys, g in scored.dropna(subset=["trust"]).groupby(by, sort=False):
        m, lo, hi = bootstrap_ci(g["trust"].values, n_boot=1000, rng=rng)
        d = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        d.update({"trust_mean": m, "trust_lo": lo, "trust_hi": hi, "n": len(g)})
        rows.append(d)
    return pd.DataFrame(rows)


def model_section_means(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, section) mean trust + CI."""
    return model_item_means(scored, by=["model", "section"])


def framing_robustness(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, item, framing) mean — used to test if response moves with framing."""
    return model_item_means(scored, by=["model", "item_id", "framing_label"])


def cross_model_item_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    """Wide matrix: rows=item_id, cols=model, values=mean trust."""
    g = scored.dropna(subset=["trust"]).groupby(["model", "item_id"])["trust"].mean().reset_index()
    return g.pivot(index="item_id", columns="model", values="trust")


def refusal_rates(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (scored.groupby(by)["refused"].mean() * 100).rename("refusal_pct").reset_index()
