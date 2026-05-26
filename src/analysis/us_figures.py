"""Publication figures for the AI vs US-population (and WVS) comparison.

Output: figures/us_comparison/{f1..f4}.{png,pdf}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .aggregate import cross_model_item_matrix, model_item_means
from .figures import MODEL_COLORS, _save as _eda_save, _style
from .loader import COUNTRY_CODE_TO_NAME, MODEL_ORDER, load_items, load_wvs_country_means
from .scoring import wvs_country_trust
from .us_comparison import (
    POLITICAL_BUCKETS,
    POLITICAL_COLORS,
    SHORTLIST_INSTITUTIONS,
    profile_similarity,
    shortlist_item_order,
    us_item_means,
)

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "figures" / "us_comparison"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------- F1: AI vs US institutional trust profile (with political split) ----------

def f1_ai_vs_us(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    # 26 institutional (Q64-Q89) + 7 social (Q57-Q63)
    items_inst = items[items["section"].isin(["wvs_confidence", "wvs_social_general", "wvs_social_groups"])].copy()
    item_ids = items_inst["id"].tolist()

    # AI
    llm_means = model_item_means(scored, by=["model", "item_id"])
    llm_means = llm_means[llm_means["item_id"].isin(item_ids)]

    # US (all) and political split
    us_all = us_item_means(items_inst, political_split=False).set_index("item_id")
    us_pol = us_item_means(items_inst, political_split=True)

    # Sort by US-all trust mean ascending (low at top, high at bottom)
    sort_ids = us_all["trust_mean"].sort_values(ascending=True).index.tolist()
    # Build labels with star marker for shortlist
    short_ids = set(shortlist_item_order(items))
    items_lab = items.set_index("id")
    labels = []
    for iid in sort_ids:
        it = items_lab.loc[iid]
        text = it["institution"] if isinstance(it["institution"], str) else "People in general"
        star = "★ " if iid in short_ids else "   "
        labels.append(f"{star}{text}")

    fig, ax = plt.subplots(figsize=(11.5, 11))
    y = np.arange(len(sort_ids))

    # US political bands as horizontal bars from Lib to Cons (range)
    for yi, iid in enumerate(sort_ids):
        sub = us_pol[us_pol["item_id"] == iid].set_index("group")["trust_mean"]
        if {"Liberal (Q240 1-4)", "Conservative (Q240 6-10)"}.issubset(sub.index):
            lo = min(sub["Liberal (Q240 1-4)"], sub["Conservative (Q240 6-10)"])
            hi = max(sub["Liberal (Q240 1-4)"], sub["Conservative (Q240 6-10)"])
            ax.plot([lo, hi], [yi, yi], color="#bbbbbb", lw=4.5, solid_capstyle="round",
                    zorder=1, alpha=0.7)

    # US political points
    for grp, color in [("Liberal (Q240 1-4)", "#1f77b4"),
                       ("Centrist (Q240 5)", "#7f7f7f"),
                       ("Conservative (Q240 6-10)", "#d62728")]:
        sub = us_pol[us_pol["group"] == grp].set_index("item_id").reindex(sort_ids)
        ax.scatter(sub["trust_mean"], y, marker="|", s=140, lw=2.0,
                   color=color, zorder=3, label=grp.split(" (")[0])

    # US (all) — dark diamond with error bar
    sub = us_all.reindex(sort_ids)
    ax.errorbar(sub["trust_mean"], y,
                xerr=[sub["trust_mean"] - sub["trust_lo"], sub["trust_hi"] - sub["trust_mean"]],
                fmt="D", markersize=5.5, color="black", ecolor="black", elinewidth=1.0,
                capsize=0, zorder=4, label="US (all)")

    # AI models
    for j, model in enumerate(MODEL_ORDER):
        sub = llm_means[llm_means["model"] == model].set_index("item_id").reindex(sort_ids)
        offset = (j - 1) * 0.20
        ax.errorbar(sub["trust_mean"], y + offset,
                    xerr=[sub["trust_mean"] - sub["trust_lo"], sub["trust_hi"] - sub["trust_mean"]],
                    fmt="o", markersize=5.5, color=MODEL_COLORS[model],
                    ecolor=MODEL_COLORS[model], elinewidth=1.2, capsize=0,
                    zorder=5, label=model)

    # Bold shortlist labels
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    for tick, iid in zip(ax.get_yticklabels(), sort_ids):
        if iid in short_ids:
            tick.set_fontweight("bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Trust score   (0 = no trust,  1 = full trust)")
    ax.set_title("Figure 1 · AI vs US-population trust across all institutional + social-trust items",
                 fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.4)

    # custom legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["Claude Opus 4.7"], markersize=7, label="Claude Opus 4.7"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["GPT-5.5"],         markersize=7, label="GPT-5.5"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["Gemini 3.1"],      markersize=7, label="Gemini 3.1"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",                          markersize=7, label="US (all WVS-7)"),
        plt.Line2D([0], [0], marker="|", color="#1f77b4", markersize=12, lw=2,                        label="US-Liberal (Q240 1-4)"),
        plt.Line2D([0], [0], marker="|", color="#7f7f7f", markersize=12, lw=2,                        label="US-Centrist (Q240 5)"),
        plt.Line2D([0], [0], marker="|", color="#d62728", markersize=12, lw=2,                        label="US-Conservative (Q240 6-10)"),
        plt.Line2D([0], [0], color="#bbbbbb", lw=4,                                                    label="Liberal–Conservative span"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              fontsize=8.5, ncol=4, frameon=False)
    ax.text(0.02, 1.005, "★ = institution in headline shortlist",
            transform=ax.transAxes, fontsize=8.5, color="#555")
    _save(fig, "f1_ai_vs_us_profile")


# ---------- F2: AI vs WVS-66-country distributions, 12-item shortlist ----------

def f2_ai_vs_wvs_shortlist(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    item_ids = shortlist_item_order(items)
    items_lab = items.set_index("id")
    wvs_means = load_wvs_country_means()
    wvs_t = wvs_country_trust(wvs_means, items).set_index(["item_id", "B_COUNTRY"])["trust"]
    us_all = us_item_means(items, political_split=False).set_index("item_id")["trust_mean"]
    llm = model_item_means(scored, by=["model", "item_id"]).set_index(["model", "item_id"])

    n = len(item_ids)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), sharex=True)
    axes = axes.flatten()

    for i, iid in enumerate(item_ids):
        ax = axes[i]
        it = items_lab.loc[iid]
        title = it["institution"]
        # WVS country dots (stripped, jittered y for visibility)
        country_vals = wvs_t.loc[iid].dropna()
        ax.scatter(country_vals.values,
                   np.full(len(country_vals), 0.5) + np.random.default_rng(i).uniform(-0.18, 0.18, len(country_vals)),
                   s=22, alpha=0.35, color="#888", edgecolors="none", zorder=1)
        # WVS pooled mean
        pooled = float(country_vals.mean())
        ax.axvline(pooled, color="black", lw=1.4, alpha=0.7, zorder=2)
        ax.text(pooled, 1.06, "pooled", fontsize=7, color="#222", ha="center")
        # US marker (red border)
        us_val = float(us_all.loc[iid])
        ax.scatter([us_val], [0.5], s=160, marker="D", color="white", edgecolor="black",
                   linewidths=1.6, zorder=4)
        ax.text(us_val, 0.07, "US", fontsize=7.5, ha="center", color="black", fontweight="bold")
        # AI markers
        for j, model in enumerate(MODEL_ORDER):
            if (model, iid) in llm.index:
                row = llm.loc[(model, iid)]
                yoff = 0.5 + (j - 1) * 0.18
                ax.errorbar([row["trust_mean"]], [yoff],
                            xerr=[[row["trust_mean"] - row["trust_lo"]],
                                  [row["trust_hi"] - row["trust_mean"]]],
                            fmt="o", markersize=8, color=MODEL_COLORS[model],
                            ecolor=MODEL_COLORS[model], elinewidth=1.2, capsize=0, zorder=5)
        ax.set_yticks([])
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("")
        ax.grid(axis="x", alpha=0.35)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[(nrows - 1) * ncols: nrows * ncols]:
        ax.set_xlabel("Trust score (0–1)")

    # global legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=MODEL_COLORS[m], markersize=9, label=m)
               for m in MODEL_ORDER]
    handles += [
        plt.Line2D([0], [0], marker="D", color="w", markeredgecolor="black",
                   markerfacecolor="white", markersize=11, label="US (all)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                   alpha=0.55, markersize=9, label="WVS-7 country (n=66)"),
        plt.Line2D([0], [0], color="black", lw=1.4, label="WVS pooled mean"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Figure 2 · AI vs WVS-7 country distributions — 12-institution shortlist",
                 fontweight="bold", y=1.04)
    plt.tight_layout()
    _save(fig, "f2_ai_vs_wvs_shortlist")


# ---------- F3: AI vs WVS politicians (Q292) ----------

def f3_politicians_ai_vs_wvs(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    pol = items[items["section"] == "wvs_politicians"].copy()
    wvs_means = load_wvs_country_means()
    wvs_t = wvs_country_trust(wvs_means, items)
    wvs_pooled = wvs_t.dropna(subset=["trust"]).groupby("item_id")["trust"].mean()
    n_ctry = wvs_t.dropna(subset=["trust"]).groupby("item_id")["B_COUNTRY"].nunique()
    llm_means = model_item_means(scored[scored["item_id"].isin(pol["id"])],
                                  by=["model", "item_id"])

    # Order: reverse-coded first (top), then positive (bottom)
    pol_sorted = pol.sort_values(["reverse_coded", "id"], ascending=[False, True])
    y = np.arange(len(pol_sorted))
    labels = []
    for _, it in pol_sorted.iterrows():
        wc = it["wvs_col"]
        rev = " [R]" if it["reverse_coded"] else ""
        st = it["statement"].split(". ", 1)[-1]
        nc = int(n_ctry.get(it["id"], 0))
        labels.append(f"{wc}{rev}  · {st[:58]}{'…' if len(st)>58 else ''}  (n={nc})")

    fig, ax = plt.subplots(figsize=(11.5, 6))
    # WVS pooled marker
    for yi, iid in enumerate(pol_sorted["id"]):
        if iid in wvs_pooled.index:
            ax.plot([wvs_pooled.loc[iid]], [yi], marker="D", markersize=8,
                    markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4,
                    zorder=4)
    for j, model in enumerate(MODEL_ORDER):
        sub = llm_means[llm_means["model"] == model].set_index("item_id").reindex(pol_sorted["id"])
        offset = (j - 1) * 0.20
        ax.errorbar(sub["trust_mean"], y + offset,
                    xerr=[sub["trust_mean"] - sub["trust_lo"], sub["trust_hi"] - sub["trust_mean"]],
                    fmt="o", markersize=6, color=MODEL_COLORS[model],
                    ecolor=MODEL_COLORS[model], elinewidth=1.3, capsize=0,
                    zorder=5, label=model)
    # divider
    n_rev = int(pol_sorted["reverse_coded"].sum())
    ax.axhline(n_rev - 0.5, color="lightgray", lw=1, linestyle="--")
    ax.text(0.99, n_rev - 0.5 + 0.15, "← negatively phrased    |   positively phrased ↓",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color="#555")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Trust score   (reverse-coded items semantically flipped)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=m)
               for m, c in MODEL_COLORS.items()]
    handles += [plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                            markeredgecolor="black", markeredgewidth=1.3, markersize=9,
                            label="WVS-7 pooled mean (13-country subset)")]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True,
              framealpha=0.95, edgecolor="lightgray")
    ax.set_title("Figure 3 · Trust in politicians & government (Q292) — AI vs WVS pooled",
                 fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.4)
    _save(fig, "f3_politicians_ai_vs_wvs")


# ---------- F4: Profile similarity heatmap ----------

def f4_similarity_heatmap(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    ps = profile_similarity(scored, items, section="wvs_confidence")
    # pick references to show: US groups + top-12 WVS countries by max similarity to any model
    us_refs = ps[ps["ref_kind"].isin(["US-aggregate", "US-political"])]["ref_label"].unique().tolist()
    wvs_pool = ["WVS pooled"]
    # top WVS countries: pick top-12 by max Spearman across models
    wvs_only = ps[ps["ref_kind"] == "WVS-country"]
    top_countries = (wvs_only.groupby("ref_label")["spearman"].max()
                     .sort_values(ascending=False).head(12).index.tolist())
    ref_order = us_refs + wvs_pool + top_countries

    wide = (ps[ps["ref_label"].isin(ref_order)]
            .pivot(index="ref_label", columns="model", values="spearman")
            .reindex(ref_order)[MODEL_ORDER])

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(wide, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-0.2, vmax=0.85,
                linewidths=0.5, linecolor="white", cbar_kws=dict(label="Spearman ρ"),
                ax=ax)
    # group separators
    sep1 = len(us_refs)
    sep2 = sep1 + len(wvs_pool)
    ax.axhline(sep1, color="black", lw=1.4)
    ax.axhline(sep2, color="black", lw=1.4)
    # Group brackets and labels at far left
    bracket_x = -0.28
    label_x = -0.32
    for (y0, y1, label) in [
        (0, sep1, "US (WVS-7)"),
        (sep1, sep2, "WVS pooled"),
        (sep2, len(ref_order), "Top WVS-7\ncountries"),
    ]:
        ax.plot([bracket_x, bracket_x], [y0 + 0.05, y1 - 0.05], color="black", lw=1.2,
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([bracket_x, bracket_x + 0.02], [y0 + 0.05, y0 + 0.05], color="black", lw=1.2,
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([bracket_x, bracket_x + 0.02], [y1 - 0.05, y1 - 0.05], color="black", lw=1.2,
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.text(label_x, (y0 + y1) / 2, label, va="center", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=9, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Figure 4 · Profile similarity (Q64–Q89, Spearman ρ)\n"
                 "Which reference does each AI most resemble?",
                 fontweight="bold", loc="left")
    _save(fig, "f4_similarity_heatmap")


def make_all(scored: pd.DataFrame, items: pd.DataFrame):
    f1_ai_vs_us(scored, items)
    f2_ai_vs_wvs_shortlist(scored, items)
    f3_politicians_ai_vs_wvs(scored, items)
    f4_similarity_heatmap(scored, items)
