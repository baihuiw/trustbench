# TrustBench v2 EDA — short methods note

Python: `/Users/wangbaihui/anaconda3/bin/python` (pandas 2.1, numpy 1.26, matplotlib 3.10, seaborn 0.13, scipy 1.11).
Modules live in `src/analysis/`. Outputs in `figures/eda/`.

---

## Part 1 · WVS-pooled deliverable (`figures/eda/wvs-pooled/`)

**Step 1 · Inventory inputs.** Three models × three framings (numeric-original, numeric-reversed, verbal) × 30 reps = 4,500 rows/model. Verified `_all.jsonl = base ∪ verbal`.

**Step 2 · Verify scale direction (critical).** The cleaned WVS file re-orients Q57–Q89 so *higher value = more trust* (confirmed against the v6.0 codebook — Q58 family value 4 = "trust completely", not "none at all"). Q292 left in raw codebook direction (1=disagree, 5=agree).

**Step 3 · `loader.py`.** Pure I/O — loads jsonl results, item dictionary, WVS country means and respondent file, builds the item→WVS-column mapping (`Q64P`, `Q292A`, …).

**Step 4 · `scoring.py`.** Maps every response to a 0–1 trust score (0 = no trust, 1 = full trust). LLM and WVS use *opposite* raw-to-trust formulas because their cleaned scales point in opposite directions; six Q292 items (A, B, E, F, H, I) reverse-coded for semantic negativity.

LLM answer c=1 means "A great deal" → should produce trust=1.0
trust_llm = (n - c) / (n - 1)
c=1: (4-1)/3 = 1.0  ✓ highest trust
c=4: (4-4)/3 = 0.0  ✓ lowest trust

WVS side (scoring.py::wvs_country_trust)
WVS cleaned c=4 means "A great deal" → should produce trust=1.0
trust_wvs = (c - 1) / (n - 1)
c=1: (1-1)/3 = 0.0  ✓ lowest trust
c=4: (4-1)/3 = 1.0  ✓ highest trust

**Step 5 · `aggregate.py`.** Bootstrap CIs on the trust score for each (model × item) and (model × section) cell. Chose bootstrap over parametric CIs because many model responses are mode-collapsed (zero variance).

**Step 6 · `wvs.py`.** Builds pooled WVS means + 25/50/75-percentile country distributions per item, plus the model↔country resemblance ranker (Spearman ρ on the 26-item Q64–Q89 profile).

**Step 7 · `figures.py`.** Eight publication figures (Wong colorblind palette, 300 dpi PNG + PDF). The notebook `eda.ipynb` orchestrates everything; the report writeup is `EDA_report.md`.

### Figures

| Fig | What it shows | What it means |
|---|---|---|
| **F1** Trust profile | 50 items × {3 AI, WVS country distribution, WVS pooled mean} forest plot | Where each AI sits vs the human range, item by item. Headline figure. |
| **F2** Calibration scatter | AI mean vs WVS pooled mean per item, with identity line | How well-calibrated each model is to global human trust (Spearman + RMSE + bias). |
| **F3** Framing robustness | Trust under numeric-original vs reversed vs verbal labels | Whether the model gives the same answer when option order or label format changes. |
| **F4** Cross-model agreement | Pairwise scatters + top-15 items by inter-model SD | Where the three models converge / diverge. |
| **F5** Country resemblance | Top-15 WVS-7 countries each model is closest to (Spearman) | Which human population each AI's profile most resembles. |
| **F6** Section summary | Bar chart by section × model with WVS pooled overlay | High-level section-level deviations. |
| **F7** Refusal & parse | Refusal % by framing + parsed-source breakdown | How often each model refused or had to be recovery-parsed. |
| **F8** Q292 deep-dive | 11 politicians items, AI vs WVS-13 pool | Whether the models reverse-code Q292 correctly and how skeptical they are about politicians. |

---

## Part 2 · US-comparison deliverable (`figures/eda/us_comparison/`)

**Step 1 · Sample audit.** US n=2,596 in WVS-7; Q240 (left–right 1–10) present so we can split Lib/Cent/Cons; **Q292 not administered in the US** so politicians AI-vs-US is impossible (falls back to WVS-13 pool).

**Step 2 · `us_comparison.py`.** Computes weighted US means (using `W_WEIGHT`) per item, with three groups via Q240 tri-split: Liberal (1–4, n=890) / Centrist (5, n=608) / Conservative (6–10, n=1,013). Bootstrap CI by resampling respondents.

**Step 3 · Profile similarity.** Spearman ρ + RMSE on the 26 Q64–Q89 items between each AI and {US-all, US-Lib, US-Cent, US-Cons, WVS pooled, every WVS country}. Spearman over Pearson because trust is bounded and many AI responses are mode-collapsed.

**Step 4 · `us_figures.py`.** Reuses the EDA's style + palette so the two deliverables visually match. Four figures (PNG + PDF). Notebook `us_comparison.ipynb` + writeup `REPORT.md`.

### Figures

| Fig | What it shows | What it means |
|---|---|---|
| **F1** AI vs US profile | 33 items (Q57–Q89) × {3 AI, US-all, US Lib/Cent/Cons} | The headline US figure: where each AI lands relative to the US public and its ideological wings. |
| **F2** AI vs WVS shortlist | 12 institutions × WVS country distribution + US + 3 AI | Whether AI deviations from US opinion are still inside the global human range. |
| **F3** Politicians (Q292) | 11 items, AI vs WVS-13 pool | AI political cynicism level (US dropped — no Q292 data). |
| **F4** Similarity heatmap | Spearman ρ for each AI × 16 reference groups | Quantifies which group each AI's profile is closest to. **Lead finding lives here.** |

---

## Headline findings

1. **WVS calibration is modest.** Spearman ρ vs WVS pooled: Gemini 0.50 > Claude 0.41 > GPT-5.5 0.36.
2. **AI over-trusts outgroups** (strangers, other-religion, other-nationality) by +0.25 to +0.34 across all three models.
3. **AI under-trusts churches** by −0.16 to −0.27 across all three models (only universal under-trust).
4. **AI leans US-Liberal.** Spearman ρ of GPT vs US-Liberal = 0.64, vs US-Conservative = 0.17 (a 4× gap). Gemini similar (0.57 vs 0.27).
5. **Group A (over-trust):** universities, elections, UN, press. **Group C (under-trust):** churches. Everything else near-parity.
6. **Mode collapse.** Many model means are exactly 0.33 or 0.67 — a single deterministic answer despite 30 reps.
7. **Framing fragility.** GPT-5.5 stable across framings (r=0.94), Gemini next (0.87), Claude least (0.60).
8. **Refusal.** Claude refuses 18 % under verbal labels; Gemini ~11 % across all framings; GPT essentially never refuses.

