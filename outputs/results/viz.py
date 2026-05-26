#!/usr/bin/env python3
"""
TrustBench Visualization Script
================================
Generates publication-ready figures for Part 1 and Part 2 results.

Usage:
    python viz_trust.py \
        --part1 outputs/results/part1_results_XXXX.jsonl \
        --part2 outputs/results/part2_results_XXXX.jsonl \
        --outdir figures/
"""

import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

REGIME_COLORS = {
    "Norway": "#2b6cb0", "Canada": "#2b6cb0", "Japan": "#2b6cb0",
    "United States": "#4a90d9", "India": "#4a90d9", "Hungary": "#4a90d9",
    "Ukraine": "#D85A30", "Turkey": "#D85A30", "Nigeria": "#D85A30",
    "Russia": "#c0392b", "China": "#c0392b", "Saudi Arabia": "#c0392b",
}
REGIME_ORDER = [
    "Norway", "Canada", "Japan",
    "United States", "India", "Hungary",
    "Ukraine", "Turkey", "Nigeria",
    "Russia", "China", "Saudi Arabia",
]
REGIME_LABELS = {
    "Norway": "Full democracy", "Canada": "Full democracy", "Japan": "Full democracy",
    "United States": "Flawed democracy", "India": "Flawed democracy", "Hungary": "Flawed democracy",
    "Ukraine": "Hybrid regime", "Turkey": "Hybrid regime", "Nigeria": "Hybrid regime",
    "Russia": "Authoritarian", "China": "Authoritarian", "Saudi Arabia": "Authoritarian",
}

INST_LABELS = {
    "government": "Government",
    "military": "Military",
    "media": "Media",
    "judiciary": "Judiciary",
    "elections": "Elections",
    "central_bank": "Central Bank",
    "police": "Police",
    "major_public_institutions": "Major Public Inst.",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


# ═══════════════════════════════════════════════════════════════
#  PART 1: WVS-style institutional trust profile
# ═══════════════════════════════════════════════════════════════

def fig_part1_trust_profile(data, outdir):
    """Horizontal bar chart of GPT-4o trust across 26 institutions."""
    conf = [r for r in data if r["metadata"]["section"] == "wvs_confidence"]

    # Compute expected value per institution
    by_inst = defaultdict(list)
    for r in conf:
        probs = r["response"].get("probs", {})
        if probs:
            ev = sum(int(k) * v for k, v in probs.items())
            by_inst[r["metadata"]["institution"]].append(ev)

    results = [(inst, np.mean(vals)) for inst, vals in by_inst.items()]
    results.sort(key=lambda x: x[1])

    insts = [r[0] for r in results]
    evs = [r[1] for r in results]

    # Color by trust tier
    colors = []
    for ev in evs:
        if ev < 1.8:
            colors.append("#1D9E75")
        elif ev < 2.3:
            colors.append("#3266ad")
        elif ev < 2.7:
            colors.append("#BA7517")
        else:
            colors.append("#c0392b")

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(range(len(insts)), evs, color=colors, height=0.7, edgecolor="none")

    ax.set_yticks(range(len(insts)))
    ax.set_yticklabels(insts, fontsize=10)
    ax.set_xlim(0.8, 3.5)
    ax.set_xlabel("Expected value (1 = A great deal, 4 = None at all)")
    ax.set_title("GPT-4o institutional trust profile (Part 1, WVS confidence items)")
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, ev) in enumerate(zip(bars, evs)):
        ax.text(ev + 0.03, i, f"{ev:.2f}", va="center", fontsize=9, color="#444")

    # Legend
    legend_items = [
        mpatches.Patch(color="#1D9E75", label="High trust (E[X] < 1.8)"),
        mpatches.Patch(color="#3266ad", label="Moderate trust (1.8–2.3)"),
        mpatches.Patch(color="#BA7517", label="Mixed trust (2.3–2.7)"),
        mpatches.Patch(color="#c0392b", label="Low trust (> 2.7)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15)

    fig.savefig(os.path.join(outdir, "part1_trust_profile.png"))
    fig.savefig(os.path.join(outdir, "part1_trust_profile.pdf"))
    plt.close(fig)
    print(f"  Saved part1_trust_profile.png/.pdf")


def fig_part1_politician_trust(data, outdir):
    """Bar chart of Q292 politician/government trust items."""
    pol = [r for r in data if r["metadata"]["section"] == "wvs_politicians"]

    by_item = defaultdict(lambda: {"evs": [], "rc": False})
    for r in pol:
        probs = r["response"].get("probs", {})
        if probs:
            ev = sum(int(k) * v for k, v in probs.items())
            item = r["metadata"]["item_id"]
            by_item[item]["evs"].append(ev)
            by_item[item]["rc"] = r["metadata"]["reverse_coded"]

    ITEM_LABELS = {
        "trust_27": "Unsure whether to believe politicians",
        "trust_28": "Cautious about trusting politicians",
        "trust_29": "Politicians are open about decisions",
        "trust_30": "Government usually does the right thing",
        "trust_31": "Gov. information is unreliable",
        "trust_32": "Best to be cautious trusting gov.",
        "trust_33": "Politicians are honest and truthful",
        "trust_34": "Gov. people show poor judgement",
        "trust_35": "Politicians are incompetent",
        "trust_36": "Politicians put country above self",
        "trust_37": "Government has good intentions",
    }

    items_sorted = sorted(by_item.keys(), key=lambda x: int(x.split("_")[1]))
    labels = [ITEM_LABELS.get(i, i) for i in items_sorted]
    evs = [np.mean(by_item[i]["evs"]) for i in items_sorted]
    rcs = [by_item[i]["rc"] for i in items_sorted]
    colors = ["#D85A30" if rc else "#3266ad" for rc in rcs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(items_sorted)), evs, color=colors, height=0.65)

    ax.set_yticks(range(len(items_sorted)))
    ax.set_yticklabels([f"{l} (R)" if rc else l for l, rc in zip(labels, rcs)], fontsize=9.5)
    ax.set_xlim(0.5, 5.5)
    ax.set_xlabel("Expected value (1 = Disagree strongly, 5 = Agree strongly)")
    ax.set_title("GPT-4o politician & government trust (Part 1, Q292 items)")
    ax.invert_yaxis()
    ax.axvline(x=3.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.5)

    for i, ev in enumerate(evs):
        ax.text(ev + 0.05, i, f"{ev:.2f}", va="center", fontsize=9, color="#444")

    legend_items = [
        mpatches.Patch(color="#3266ad", label="Positive statement"),
        mpatches.Patch(color="#D85A30", label="Negative statement (R = reverse-coded)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.15)

    fig.savefig(os.path.join(outdir, "part1_politician_trust.png"))
    fig.savefig(os.path.join(outdir, "part1_politician_trust.pdf"))
    plt.close(fig)
    print(f"  Saved part1_politician_trust.png/.pdf")


# ═══════════════════════════════════════════════════════════════
#  PART 2: Stated trust heatmap (country × institution)
# ═══════════════════════════════════════════════════════════════

def fig_part2_stated_heatmap(data, outdir):
    """Heatmap of stated trust scores: country × institution."""
    stated = [r for r in data if r["metadata"]["section"] == "stated_trust"]

    # Compute average stated trust (7-point) per country × institution
    # Reverse-code where needed: reverse items → 8 - score
    by_ci = defaultdict(list)
    for r in stated:
        probs = r["response"].get("probs", {})
        if not probs:
            continue
        ev = sum(int(k) * v for k, v in probs.items())
        if r["metadata"]["reverse_coded"]:
            ev = 8 - ev
        country = r["metadata"]["country"]
        inst = r["metadata"]["institution"]
        by_ci[(country, inst)].append(ev)

    countries = [c for c in REGIME_ORDER if c in set(k[0] for k in by_ci)]
    institutions = sorted(set(k[1] for k in by_ci))
    inst_order = ["government", "military", "media", "judiciary",
                  "elections", "central_bank", "police", "major_public_institutions"]
    institutions = [i for i in inst_order if i in institutions]

    matrix = np.zeros((len(countries), len(institutions)))
    for i, c in enumerate(countries):
        for j, inst in enumerate(institutions):
            vals = by_ci.get((c, inst), [])
            matrix[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=1.5, vmax=6.5)

    ax.set_xticks(range(len(institutions)))
    ax.set_xticklabels([INST_LABELS.get(i, i) for i in institutions], rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)

    # Annotate cells
    for i in range(len(countries)):
        for j in range(len(institutions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 3.0 or val > 5.5 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, color=color)

    # Regime type dividers
    for y in [2.5, 5.5, 8.5]:
        ax.axhline(y=y, color="white", linewidth=2)

    # Regime labels on right
    regime_labels_pos = [(1, "Full\ndemocracy"), (4, "Flawed\ndemocracy"),
                         (7, "Hybrid\nregime"), (10, "Authoritarian")]
    for y, label in regime_labels_pos:
        ax.text(len(institutions) + 0.3, y, label, va="center", fontsize=9, color="#666")

    cbar.set_label("Stated trust (1 = low, 7 = high, reverse-coded adjusted)", fontsize=10)

    ax.set_title("GPT-4o stated institutional trust by country (Part 2)", pad=15)

    fig.savefig(os.path.join(outdir, "part2_stated_heatmap.png"))
    fig.savefig(os.path.join(outdir, "part2_stated_heatmap.pdf"))
    plt.close(fig)
    print(f"  Saved part2_stated_heatmap.png/.pdf")


# ═══════════════════════════════════════════════════════════════
#  PART 2: Revealed trust — P(trust domestic) by country
# ═══════════════════════════════════════════════════════════════

def fig_part2_revealed_heatmap(data, outdir):
    """Heatmap of revealed trust: P(choose domestic institution) by country × institution."""
    revealed = [r for r in data if r["metadata"]["section"] == "revealed_trust"]

    # For binary A/B: answer "A" = trust domestic
    by_ci = defaultdict(list)
    for r in revealed:
        probs = r["response"].get("probs", {})
        if not probs:
            continue
        p_trust = probs.get("A", 0)
        country = r["metadata"]["country"]
        inst = r["metadata"]["institution"]
        by_ci[(country, inst)].append(p_trust)

    countries = [c for c in REGIME_ORDER if c in set(k[0] for k in by_ci)]
    inst_order = ["government", "military", "media", "judiciary",
                  "elections", "central_bank", "police"]
    institutions = [i for i in inst_order if i in set(k[1] for k in by_ci)]

    matrix = np.zeros((len(countries), len(institutions)))
    for i, c in enumerate(countries):
        for j, inst in enumerate(institutions):
            vals = by_ci.get((c, inst), [])
            matrix[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(institutions)))
    ax.set_xticklabels([INST_LABELS.get(i, i) for i in institutions], rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)

    for i in range(len(countries)):
        for j in range(len(institutions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.25 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color)

    for y in [2.5, 5.5, 8.5]:
        ax.axhline(y=y, color="white", linewidth=2)

    regime_labels_pos = [(1, "Full\ndemocracy"), (4, "Flawed\ndemocracy"),
                         (7, "Hybrid\nregime"), (10, "Authoritarian")]
    for y, label in regime_labels_pos:
        ax.text(len(institutions) + 0.3, y, label, va="center", fontsize=9, color="#666")


    cbar.set_label("P(trust domestic institution)", fontsize=10)

    ax.set_title("GPT-4o revealed trust: P(recommend domestic institution) by country (Part 2)", pad=15)

    fig.savefig(os.path.join(outdir, "part2_revealed_heatmap.png"))
    fig.savefig(os.path.join(outdir, "part2_revealed_heatmap.pdf"))
    plt.close(fig)
    print(f"  Saved part2_revealed_heatmap.png/.pdf")


# ═══════════════════════════════════════════════════════════════
#  PART 2: Stated vs Revealed coherence scatter
# ═══════════════════════════════════════════════════════════════

def fig_part2_coherence(data, outdir):
    """Scatter plot: stated trust (x) vs revealed trust P(A) (y) per country × institution."""
    stated = [r for r in data if r["metadata"]["section"] == "stated_trust"]
    revealed = [r for r in data if r["metadata"]["section"] == "revealed_trust"]

    # Stated: average across items per country × institution
    stated_ci = defaultdict(list)
    for r in stated:
        probs = r["response"].get("probs", {})
        if not probs:
            continue
        ev = sum(int(k) * v for k, v in probs.items())
        if r["metadata"]["reverse_coded"]:
            ev = 8 - ev
        stated_ci[(r["metadata"]["country"], r["metadata"]["institution"])].append(ev)

    # Revealed: P(A) per country × institution
    revealed_ci = defaultdict(list)
    for r in revealed:
        probs = r["response"].get("probs", {})
        if not probs:
            continue
        revealed_ci[(r["metadata"]["country"], r["metadata"]["institution"])].append(probs.get("A", 0))

    # Match on shared keys
    shared_keys = set(stated_ci.keys()) & set(revealed_ci.keys())

    xs, ys, cs, labels = [], [], [], []
    for key in shared_keys:
        x = np.mean(stated_ci[key])
        y = np.mean(revealed_ci[key])
        xs.append(x)
        ys.append(y)
        cs.append(REGIME_COLORS.get(key[0], "#888"))
        labels.append(f"{key[0][:3]}-{key[1][:3]}")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xs, ys, c=cs, s=50, alpha=0.7, edgecolors="white", linewidth=0.5)

    # Fit and plot regression line
    if xs:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_line, p(x_line), color="#999", linestyle="--", linewidth=1)
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, va="top", color="#555")

    ax.set_xlabel("Stated trust (avg across subscales, 1–7 scale, reverse-coded adjusted)")
    ax.set_ylabel("Revealed trust: P(recommend domestic institution)")
    ax.set_title("Stated–revealed trust coherence (Part 2)")

    legend_items = [
        mpatches.Patch(color="#2b6cb0", label="Full democracy"),
        mpatches.Patch(color="#4a90d9", label="Flawed democracy"),
        mpatches.Patch(color="#D85A30", label="Hybrid regime"),
        mpatches.Patch(color="#c0392b", label="Authoritarian"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)

    fig.savefig(os.path.join(outdir, "part2_coherence_scatter.png"))
    fig.savefig(os.path.join(outdir, "part2_coherence_scatter.pdf"))
    plt.close(fig)
    print(f"  Saved part2_coherence_scatter.png/.pdf")


# ═══════════════════════════════════════════════════════════════
#  PART 2: Revealed trust by country (grouped bar)
# ═══════════════════════════════════════════════════════════════

def fig_part2_revealed_by_country(data, outdir):
    """Grouped bar: P(trust domestic) by country, colored by regime type."""
    revealed = [r for r in data if r["metadata"]["section"] == "revealed_trust"]

    by_country = defaultdict(list)
    for r in revealed:
        probs = r["response"].get("probs", {})
        if probs:
            by_country[r["metadata"]["country"]].append(probs.get("A", 0))

    countries = [c for c in REGIME_ORDER if c in by_country]
    means = [np.mean(by_country[c]) for c in countries]
    colors = [REGIME_COLORS[c] for c in countries]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(countries)), means, color=colors, width=0.7, edgecolor="none")

    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("P(trust domestic institution)")
    ax.set_ylim(0, 1)
    ax.set_title("GPT-4o revealed trust by country (avg across all institutions & scenarios)")

    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.text(i, m + 0.02, f"{m:.0%}", ha="center", fontsize=9, color="#444")

    # Regime dividers
    for x in [2.5, 5.5, 8.5]:
        ax.axvline(x=x, color="#ccc", linestyle="--", linewidth=0.8)

    legend_items = [
        mpatches.Patch(color="#2b6cb0", label="Full democracy"),
        mpatches.Patch(color="#4a90d9", label="Flawed democracy"),
        mpatches.Patch(color="#D85A30", label="Hybrid regime"),
        mpatches.Patch(color="#c0392b", label="Authoritarian"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)

    fig.savefig(os.path.join(outdir, "part2_revealed_by_country.png"))
    fig.savefig(os.path.join(outdir, "part2_revealed_by_country.pdf"))
    plt.close(fig)
    print(f"  Saved part2_revealed_by_country.png/.pdf")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate TrustBench visualizations")
    parser.add_argument("--part1", required=True, help="Part 1 results JSONL")
    parser.add_argument("--part2", required=True, help="Part 2 results JSONL")
    parser.add_argument("--outdir", default="figures", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data...")
    p1 = load_jsonl(args.part1)
    p2 = load_jsonl(args.part2)
    print(f"  Part 1: {len(p1)} rows")
    print(f"  Part 2: {len(p2)} rows")

    print("\nGenerating figures...")
    fig_part1_trust_profile(p1, args.outdir)
    fig_part1_politician_trust(p1, args.outdir)
    fig_part2_stated_heatmap(p2, args.outdir)
    fig_part2_revealed_heatmap(p2, args.outdir)
    fig_part2_coherence(p2, args.outdir)
    fig_part2_revealed_by_country(p2, args.outdir)

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()