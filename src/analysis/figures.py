"""Top-conference-quality figures for TrustBench EDA.

All figures are saved as both PNG (300 dpi) and PDF to `figures/eda/`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .aggregate import (
    bootstrap_ci,
    cross_model_item_matrix,
    framing_robustness,
    model_item_means,
    model_section_means,
    refusal_rates,
)
from .loader import MODEL_ORDER
from .wvs import country_level_trust, llm_country_resemblance, wvs_summary

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe palette (Wong 2011): orange, blue, green for the 3 models
MODEL_COLORS = {
    "Claude Opus 4.7": "#E69F00",  # orange
    "GPT-5.5":          "#0072B2",  # blue
    "Gemini 3.1":       "#009E73",  # green
}
WVS_COLOR = "#555555"
WVS_FILL  = "#cccccc"

SECTION_LABELS = {
    "wvs_confidence":     "Confidence in institutions (Q64–Q89)",
    "wvs_politicians":    "Politicians & government (Q292)",
    "wvs_social_general": "Generalized social trust (Q57)",
    "wvs_social_groups":  "Trust in social groups (Q58–Q63)",
    "social_role_trust":  "Social roles (TrustBench custom)",
}


def _style():
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


# ---------- Fig 1: Master institutional trust profile ----------

def fig1_trust_profile(scored: pd.DataFrame, items: pd.DataFrame):
    """Forest plot: institutions × LLM trust scores with WVS country distribution overlay."""
    _style()
    item_means = model_item_means(scored, by=["model", "item_id"])
    wvs_glob = wvs_summary(items).set_index("item_id")
    wvs_country = country_level_trust(items).dropna(subset=["trust"])

    sections_panels = [
        ("wvs_confidence",    "A   Confidence in institutions"),
        ("wvs_politicians",   "B   Politicians & government"),
        ("wvs_social_groups", "C   Trust in social groups"),
        ("social_role_trust", "D   Social roles (no WVS)"),
    ]

    fig = plt.figure(figsize=(13.5, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[26, 11], hspace=0.32, wspace=0.55)
    axes_map = {
        "wvs_confidence":    fig.add_subplot(gs[0, :]),
        "wvs_politicians":   fig.add_subplot(gs[1, 0]),
        "wvs_social_groups": fig.add_subplot(gs[1, 1]),
    }

    for section, title in [("wvs_confidence", "A   Confidence in institutions (Q64–Q89)"),
                            ("wvs_politicians", "B   Politicians & government (Q292)"),
                            ("wvs_social_groups", "C   Trust in social groups (Q58–Q63)")]:
        ax = axes_map[section]
        items_sec = items[items["section"] == section].copy()
        # order by WVS pooled mean (asc)
        items_sec["wvs_mean"] = items_sec["id"].map(wvs_glob["wvs_mean"])
        items_sec = items_sec.sort_values("wvs_mean", na_position="last")

        labels = []
        for _, it in items_sec.iterrows():
            iid = it["id"]
            if section == "wvs_politicians":
                # use statement letter only
                lab = it["statement"].split(". ", 1)[-1][:55] + ("…" if len(it["statement"]) > 60 else "")
                rev = "  (R)" if it["reverse_coded"] else ""
                labels.append(f"Q292{it['wvs_col'][-1]} {lab}{rev}")
            else:
                labels.append(it["institution"] or it["statement"])

        y = np.arange(len(items_sec))

        # WVS country strip
        for yi, (_, it) in enumerate(zip(y, items_sec.iterrows())):
            iid = items_sec.iloc[yi]["id"]
            cd = wvs_country[wvs_country["item_id"] == iid]["trust"].dropna()
            if len(cd) > 0:
                ax.scatter(cd, np.full(len(cd), yi) + 0.0,
                          s=10, alpha=0.35, color=WVS_COLOR, zorder=1,
                          edgecolors="none", label="_nolegend_")
                wm = wvs_glob.loc[iid, "wvs_mean"] if iid in wvs_glob.index else np.nan
                if not np.isnan(wm):
                    ax.plot([wm, wm], [yi - 0.32, yi + 0.32], color="black", lw=2, zorder=4)

        for j, model in enumerate(MODEL_ORDER):
            sub = item_means[(item_means["model"] == model) &
                              (item_means["item_id"].isin(items_sec["id"]))].set_index("item_id")
            xs = [sub.loc[iid, "trust_mean"] if iid in sub.index else np.nan for iid in items_sec["id"]]
            lo = [sub.loc[iid, "trust_lo"] if iid in sub.index else np.nan for iid in items_sec["id"]]
            hi = [sub.loc[iid, "trust_hi"] if iid in sub.index else np.nan for iid in items_sec["id"]]
            offset = (j - 1) * 0.20
            ax.errorbar(xs, y + offset,
                        xerr=[np.array(xs) - np.array(lo), np.array(hi) - np.array(xs)],
                        fmt="o", markersize=5, color=MODEL_COLORS[model],
                        ecolor=MODEL_COLORS[model], elinewidth=1.4, capsize=0,
                        label=model, zorder=5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Trust score   (0 = no trust,  1 = full trust)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0.4)

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=7, label=m)
               for m, c in MODEL_COLORS.items()]
    handles += [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=WVS_COLOR,
                            alpha=0.6, markersize=7, label="WVS-7 country mean"),
                plt.Line2D([0], [0], color='black', lw=2, label="WVS pooled mean")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995),
               frameon=False)
    fig.suptitle("Figure 1 · Institutional and social-group trust: three LLMs vs WVS-7 humans",
                 y=1.005, fontsize=13, fontweight="bold")
    _save(fig, "fig1_trust_profile")


# ---------- Fig 2: Model-vs-WVS calibration scatter ----------

def fig2_llm_vs_wvs_scatter(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    item_means = model_item_means(scored, by=["model", "item_id"])
    wvs_glob = wvs_summary(items).set_index("item_id")

    items_with_wvs = items[items["wvs_col"].notna()].copy()
    sec_palette = {
        "wvs_confidence":     "#0072B2",
        "wvs_politicians":    "#D55E00",
        "wvs_social_general": "#CC79A7",
        "wvs_social_groups":  "#009E73",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        sub = item_means[item_means["model"] == model].set_index("item_id")
        xs, ys, secs = [], [], []
        for _, it in items_with_wvs.iterrows():
            iid = it["id"]
            if iid in sub.index and iid in wvs_glob.index:
                xs.append(wvs_glob.loc[iid, "wvs_mean"])
                ys.append(sub.loc[iid, "trust_mean"])
                secs.append(it["section"])
        xs, ys, secs = np.array(xs), np.array(ys), np.array(secs)
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        for sec in np.unique(secs):
            mask = secs == sec
            ax.scatter(xs[mask], ys[mask],
                      s=42, alpha=0.85, color=sec_palette.get(sec, "#666"),
                      edgecolor="white", linewidth=0.8, label=SECTION_LABELS.get(sec, sec))
        rho, p = spearmanr(xs, ys)
        rmse = np.sqrt(np.mean((xs - ys) ** 2))
        bias = float(np.mean(ys - xs))
        ax.text(0.04, 0.96,
                f"ρ = {rho:.2f}\nRMSE = {rmse:.2f}\nbias (LLM−WVS) = {bias:+.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="lightgray", boxstyle="round"))
        ax.set_title(model, color=MODEL_COLORS[model])
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("WVS-7 pooled human trust")
    axes[0].set_ylabel("LLM trust")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Figure 2 · Calibration of LLM trust against WVS-7 pooled human mean", fontweight="bold")
    _save(fig, "fig2_llm_vs_wvs_scatter")


# ---------- Fig 3: Framing robustness ----------

def fig3_framing_robustness(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    fr = framing_robustness(scored)
    wide = fr.pivot_table(index=["model", "item_id"], columns="framing_label",
                          values="trust_mean").reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, model in zip(axes, MODEL_ORDER):
        sub = wide[wide["model"] == model].dropna(subset=["numeric-original", "numeric-reversed"])
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.scatter(sub["numeric-original"], sub["numeric-reversed"],
                   s=48, color=MODEL_COLORS[model], alpha=0.75, edgecolor="white", lw=0.8,
                   label="numeric: original vs reversed")
        # verbal vs numeric (if present)
        sub2 = wide[wide["model"] == model].dropna(subset=["numeric-original", "verbal"])
        ax.scatter(sub2["numeric-original"], sub2["verbal"],
                   s=48, marker="s", facecolor="none",
                   edgecolor=MODEL_COLORS[model], lw=1.2, alpha=0.8,
                   label="numeric vs verbal labels")
        # correlations
        r1 = sub[["numeric-original", "numeric-reversed"]].corr().iloc[0, 1]
        if not sub2.empty:
            r2 = sub2[["numeric-original", "verbal"]].corr().iloc[0, 1]
        else:
            r2 = np.nan
        ax.text(0.04, 0.96,
                f"r(num↔rev) = {r1:.2f}\nr(num↔verbal) = {r2:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="lightgray", boxstyle="round"))
        ax.set_title(model, color=MODEL_COLORS[model])
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Trust under numeric-original framing")
    axes[0].set_ylabel("Trust under alternate framing")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Figure 3 · Framing robustness: numeric reversal and verbal-label invariance",
                 fontweight="bold")
    _save(fig, "fig3_framing_robustness")


# ---------- Fig 4: Cross-model agreement ----------

def fig4_cross_model(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    mat = cross_model_item_matrix(scored)
    pairs = [("Claude Opus 4.7", "GPT-5.5"),
             ("Claude Opus 4.7", "Gemini 3.1"),
             ("GPT-5.5", "Gemini 3.1")]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.5),
                              gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6]))
    plt.subplots_adjust(wspace=0.34)
    sec_color = items.set_index("id")["section"].map({
        "wvs_confidence":     "#0072B2",
        "wvs_politicians":    "#D55E00",
        "wvs_social_general": "#CC79A7",
        "wvs_social_groups":  "#009E73",
        "social_role_trust":  "#999999",
    })
    for ax, (a, b) in zip(axes[:3], pairs):
        sub = mat[[a, b]].dropna()
        sc = sec_color.reindex(sub.index)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.scatter(sub[a], sub[b], s=45, c=sc.values, alpha=0.8,
                   edgecolor="white", lw=0.7)
        rho, _ = spearmanr(sub[a], sub[b])
        rmse = np.sqrt(((sub[a] - sub[b]) ** 2).mean())
        ax.text(0.04, 0.96, f"ρ = {rho:.2f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="lightgray", boxstyle="round"))
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(a, color=MODEL_COLORS[a])
        ax.set_ylabel(b, color=MODEL_COLORS[b])
        ax.set_aspect("equal")
    # disagreement panel: items where models most disagree (std)
    ax = axes[3]
    mat_arr = mat[MODEL_ORDER].to_numpy()
    sd = pd.Series(np.nanstd(mat_arr, axis=1, ddof=1), index=mat.index)
    disagree = sd.sort_values(ascending=False).head(15)
    items_lab = items.set_index("id")
    lab = []
    for iid in disagree.index:
        it = items_lab.loc[iid]
        text = it["institution"] if isinstance(it["institution"], str) else (it["statement"] if isinstance(it["statement"], str) else iid)
        lab.append(text[:34])
    ypos = np.arange(len(disagree))[::-1]
    ax.barh(ypos, disagree.values, color="#777", alpha=0.85)
    # overlay model dots
    for j, model in enumerate(MODEL_ORDER):
        ax.scatter(mat.loc[disagree.index, model].values, ypos,
                   s=42, color=MODEL_COLORS[model], edgecolor="white", lw=0.6, zorder=3,
                   label=model)
    ax.set_yticks(ypos); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Trust (model dots) / inter-model SD (gray bar)")
    ax.set_title("Top 15 items with largest model disagreement", fontsize=10)
    fig.suptitle("Figure 4 · Cross-model agreement and disagreement", fontweight="bold")
    _save(fig, "fig4_cross_model")


# ---------- Fig 5: Country resemblance ----------

def fig5_country_resemblance(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    res_all = llm_country_resemblance(scored, items, "wvs_confidence")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    plt.subplots_adjust(wspace=0.55)
    for ax, model in zip(axes, MODEL_ORDER):
        res = res_all[res_all["model"] == model].sort_values("spearman", ascending=False).head(15)
        ypos = np.arange(len(res))[::-1]
        ax.barh(ypos, res["spearman"].values, color=MODEL_COLORS[model], alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_yticks(ypos); ax.set_yticklabels(res["country"].values, fontsize=9)
        ax.set_xlim(0, 1.25)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Spearman ρ on Q64–Q89 profile")
        ax.set_title(f"{model}\nclosest WVS-7 countries", color=MODEL_COLORS[model])
        for i, (rho, rm) in enumerate(zip(res["spearman"].values, res["rmse"].values)):
            ax.text(rho + 0.02, ypos[i], f"RMSE {rm:.2f}",
                    fontsize=7.5, va="center", color="#666")
    fig.suptitle("Figure 5 · Which WVS-7 country does each LLM resemble most on institutional trust?",
                 fontweight="bold")
    _save(fig, "fig5_country_resemblance")


# ---------- Fig 6: Section-level summary ----------

def fig6_section_summary(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    sec_means = model_section_means(scored)
    # WVS pooled per section
    wsum = wvs_summary(items)
    wsum = wsum.merge(items[["id", "section"]], left_on="item_id", right_on="id")
    wvs_sec_pooled = wsum.groupby("section")["wvs_mean"].mean()

    sec_order = ["wvs_confidence", "wvs_politicians", "wvs_social_general",
                 "wvs_social_groups", "social_role_trust"]
    sec_order = [s for s in sec_order if s in sec_means["section"].unique()]
    width = 0.24
    x = np.arange(len(sec_order))

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for j, model in enumerate(MODEL_ORDER):
        ms = sec_means[sec_means["model"] == model].set_index("section").reindex(sec_order)
        ax.bar(x + (j - 1) * width, ms["trust_mean"], width,
               yerr=[ms["trust_mean"] - ms["trust_lo"], ms["trust_hi"] - ms["trust_mean"]],
               color=MODEL_COLORS[model], label=model, alpha=0.92, capsize=2.5,
               error_kw=dict(ecolor="black", lw=0.8))
    # WVS pooled per section as black notches
    for xi, sec in enumerate(sec_order):
        if sec in wvs_sec_pooled.index:
            ax.plot([xi - 1.5 * width, xi + 1.5 * width],
                    [wvs_sec_pooled[sec], wvs_sec_pooled[sec]],
                    color="black", lw=2, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([SECTION_LABELS.get(s, s).split(" (")[0] for s in sec_order],
                       rotation=15, ha="right")
    ax.set_ylabel("Mean trust (0–1)")
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", lw=2, label="WVS-7 pooled mean"))
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.set_title("Figure 6 · Section-level trust summary: LLMs vs WVS-7 humans", fontweight="bold")
    _save(fig, "fig6_section_summary")


# ---------- Fig 7: Refusal & parse-recovery ----------

def fig7_refusal(scored: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                              gridspec_kw=dict(width_ratios=[1, 1.2]))

    # Panel A: refusal % by model × framing
    ax = axes[0]
    rr = refusal_rates(scored, ["model", "framing_label"])
    framings = ["numeric-original", "numeric-reversed", "verbal"]
    width = 0.24
    x = np.arange(len(framings))
    for j, model in enumerate(MODEL_ORDER):
        vals = [rr[(rr["model"] == model) & (rr["framing_label"] == f)]["refusal_pct"].iloc[0]
                if not rr[(rr["model"] == model) & (rr["framing_label"] == f)].empty else 0
                for f in framings]
        ax.bar(x + (j - 1) * width, vals, width, color=MODEL_COLORS[model], label=model, alpha=0.92)
    ax.set_xticks(x); ax.set_xticklabels(framings, rotation=15, ha="right")
    ax.set_ylabel("Refusal / unparseable (%)")
    ax.set_title("(A) Refusal rate by framing", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # Panel B: parsed_source breakdown per model
    ax = axes[1]
    src_counts = (scored.groupby(["model", "parsed_source"]).size().unstack(fill_value=0))
    src_counts = src_counts.reindex(MODEL_ORDER)
    pct = src_counts.div(src_counts.sum(axis=1), axis=0) * 100
    cols = ["original", "recovered", "refused"]
    pct = pct.reindex(columns=cols, fill_value=0)
    colors = ["#4daf4a", "#984ea3", "#e41a1c"]
    bottom = np.zeros(len(pct))
    for col, c in zip(cols, colors):
        ax.barh(pct.index, pct[col], left=bottom, color=c, label=col, alpha=0.92)
        bottom += pct[col].values
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of responses (%)")
    ax.set_title("(B) Choice parse source by model", loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Figure 7 · Refusal & response-parse recovery", fontweight="bold")
    _save(fig, "fig7_refusal")


# ---------- Fig 8: Politicians Q292 deep dive ----------

def fig8_politicians(scored: pd.DataFrame, items: pd.DataFrame):
    _style()
    pol = items[items["section"] == "wvs_politicians"]
    means = model_item_means(scored[scored["item_id"].isin(pol["id"])],
                              by=["model", "item_id"])
    wvs_glob = wvs_summary(items).set_index("item_id")
    wvs_country = country_level_trust(items)

    # Order items: reverse-coded first, then positive
    pol_sorted = pol.sort_values(["reverse_coded", "id"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(pol_sorted))
    labels = []
    for _, it in pol_sorted.iterrows():
        wc = it["wvs_col"]
        rev_tag = " [R]" if it["reverse_coded"] else ""
        st = it["statement"].split(". ", 1)[-1]
        labels.append(f"{wc}{rev_tag}  · {st[:60]}{'…' if len(st)>60 else ''}")
        # country strip
        cd = wvs_country[wvs_country["item_id"] == it["id"]]["trust"].dropna()
        if len(cd) > 0:
            yi = list(pol_sorted["id"]).index(it["id"])
            ax.scatter(cd, np.full(len(cd), yi), s=14, alpha=0.5,
                      color=WVS_COLOR, edgecolors="none", zorder=1)
            wm = wvs_glob.loc[it["id"], "wvs_mean"] if it["id"] in wvs_glob.index else np.nan
            if not np.isnan(wm):
                ax.plot([wm, wm], [yi - 0.32, yi + 0.32], color="black", lw=2, zorder=4)

    for j, model in enumerate(MODEL_ORDER):
        sub = means[means["model"] == model].set_index("item_id")
        xs = [sub.loc[iid, "trust_mean"] if iid in sub.index else np.nan
              for iid in pol_sorted["id"]]
        lo = [sub.loc[iid, "trust_lo"] if iid in sub.index else np.nan for iid in pol_sorted["id"]]
        hi = [sub.loc[iid, "trust_hi"] if iid in sub.index else np.nan for iid in pol_sorted["id"]]
        offset = (j - 1) * 0.20
        ax.errorbar(xs, y + offset,
                    xerr=[np.array(xs) - np.array(lo), np.array(hi) - np.array(xs)],
                    fmt="o", markersize=6, color=MODEL_COLORS[model],
                    ecolor=MODEL_COLORS[model], elinewidth=1.4, capsize=0,
                    label=model, zorder=5)
    # Divider between reverse and non-reverse blocks
    n_rev = int(pol_sorted["reverse_coded"].sum())
    ax.axhline(n_rev - 0.5, color="lightgray", lw=1, linestyle="--")
    ax.text(0.99, n_rev - 0.5 + 0.15, "← reverse-phrased   |   positive-phrased ↓",
            transform=ax.get_yaxis_transform(),
            ha="right", fontsize=8, color="#555")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Trust score (after applying semantic reverse-coding for negatively-phrased items)")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=7, label=m)
               for m, c in MODEL_COLORS.items()]
    handles += [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=WVS_COLOR,
                            markersize=7, alpha=0.6, label="WVS country mean (n=13)"),
                plt.Line2D([0], [0], color='black', lw=2, label="WVS pooled mean")]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.set_title("Figure 8 · Q292 politicians & government — item-level deep dive",
                 fontweight="bold")
    _save(fig, "fig8_politicians")


def make_all(scored: pd.DataFrame, items: pd.DataFrame):
    fig1_trust_profile(scored, items)
    fig2_llm_vs_wvs_scatter(scored, items)
    fig3_framing_robustness(scored, items)
    fig4_cross_model(scored, items)
    fig5_country_resemblance(scored, items)
    fig6_section_summary(scored, items)
    fig7_refusal(scored)
    fig8_politicians(scored, items)
