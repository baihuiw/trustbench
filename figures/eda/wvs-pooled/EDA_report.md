# TrustBench EDA — Three frontier LLMs vs WVS-7 humans

**Models:** Claude Opus 4.7, GPT-5.5, Gemini 3.1
**Items:** 50 (26 WVS institutional confidence Q64–Q89; 11 politicians/government Q292; 1 generalized social trust Q57; 6 social-group trust Q58–Q63; 6 TrustBench custom social-role items)
**Framings per item:** numeric labels in original order, numeric labels in reversed order, verbal labels (e.g. *"A great deal"*) — 30 repetitions each.
**Total responses:** 13,500 (4,500 / model).
**Human reference:** WVS-7 v6.0 weighted country means across 66 countries (Q292 only available for 13).
**Trust score:** every item is mapped to a 0–1 score (0 = no trust, 1 = full trust). For Q292, the six negatively-phrased items (A, B, E, F, H, I) are semantically reverse-coded so the score has the same orientation as positive items. The WVS-7 cleaning pipeline already unifies Q57–Q89 to *higher value = more trust*; Q292 is left raw and reverse-coded at score-time. Verified against the v6.0 codebook.

> All figures referenced below live in `figures/eda/` (PNG + PDF). The pipeline is in `src/analysis/{loader, scoring, aggregate, wvs, figures}.py`; the runnable notebook is `src/analysis/eda.ipynb`.

---

## Headline findings

**F1 — Trust profile is highly model-specific, especially on institutional confidence.**
On Q64–Q89, GPT-5.5 averages **0.60**, Gemini 3.1 **0.57**, and Claude Opus 4.7 **0.43**. WVS-7 humans (pooled) sit at **0.49**. Claude is *less* trusting of institutions than the human baseline; GPT and Gemini are *more* trusting (Figure 6).

**F2 — All three LLMs systematically over-trust outgroups.**
"People you meet for the first time", "people of another religion/nationality" are over-trusted by **+0.25 to +0.34** relative to WVS pooled means across all three models (Figure 1C; Figure 2 — green points pulled above identity line). This is the single largest, most consistent bias and looks like an RLHF artifact emphasising tolerance/inclusivity.

**F3 — All three LLMs under-trust churches.**
LLM−WVS gap of **−0.16 to −0.27** on Q64 churches, consistent across models. Cross-cuts the otherwise-positive bias on institutions for GPT and Gemini, so it's not simply "trust everything more".

**F4 — Claude refuses the binary Q57 entirely.**
Out of 90 valid responses on the generalized-trust item, Claude picks *"Need to be very careful"* 100% of the time → trust = 0.00. GPT and Gemini are also more cautious than humans on Q57 but do produce a mix (mean ≈ 0.31, vs WVS pooled 0.24). This is a sharp model-specific behavior worth a paragraph in the paper.

**F5 — Framing robustness ranks GPT-5.5 ≫ Gemini 3.1 ≫ Claude Opus 4.7.**
Per-item trust under numeric-original vs numeric-reversed framings: Pearson r = **0.94 / 0.87 / 0.60**. Claude's responses change substantially when the option order is reversed — a reliability concern (Figure 3).

**F6 — Calibration to WVS is modest for all models; Gemini is best.**
Item-level Spearman ρ of LLM mean vs WVS pooled mean across all 44 WVS items: **0.50 (Gemini) > 0.41 (Claude) > 0.36 (GPT-5.5)**. RMSE 0.13–0.17. Bias: Claude −0.05, GPT +0.07, Gemini +0.07 (Figure 2).

**F7 — Refusal is highly framing-sensitive for Claude, uniform for Gemini, near-zero for GPT.**
| Model            | num-orig | num-rev | verbal |
|------------------|---------:|--------:|-------:|
| Claude Opus 4.7  |   0.73 % |  0.73 % | **18.40 %** |
| GPT-5.5          |   0.40 % |  0.27 % |   0.20 % |
| Gemini 3.1       |  10.73 % | 11.80 % |  10.87 % |

The verbal-label framing alone triggers an 18× jump in Claude's refusal rate. Most "refusals" surface as `"As an AI, I don't have personal confidence…"` preamble with no extracted choice (Figure 7A; recovery parser salvaged 229 Opus and 40 Gemini answers where the choice was embedded in narrative — Figure 7B).

**F8 — Cross-model agreement is moderate-to-high; Claude is the outlier.**
Pairwise Spearman ρ on item-mean trust: GPT × Gemini = **0.80**, Opus × Gemini = 0.72, Opus × GPT = 0.67 (Figure 4). The items with the largest cross-model disagreement are *courts*, *people in general (Q57)*, *United Nations*, *European Union*, *World Bank* — i.e., politically-laden global institutions.

**F9 — Country resemblance: East-Asian / developed-democracy profile.**
On the Q64–Q89 profile, GPT-5.5 most resembles South Korea (ρ=0.55), Gemini 3.1 resembles Canada / Taiwan / Netherlands (ρ ≈ 0.46–0.63), Claude resembles Taiwan / Slovakia / Armenia. None of the three models resembles low-trust countries (e.g. Colombia, Nicaragua, Lebanon, Iraq, Peru — all outside the top-15 for every model; Figure 5).

**F10 — Q292 politicians: orientation is correct, magnitude differs.**
All three models correctly score positive Q292 statements higher than negatively-phrased ones after semantic reverse-coding, so they're not blindly agreeing with everything (good news for the methodology). But absolute trust scores cluster well below the WVS pooled means for the 13 Q292 countries — LLMs are *less* trusting of politicians than the available human sample (Figure 8).

---

## Methodology notes worth flagging in the paper

1. **WVS direction.** Q57 and Q58–Q63 in the cleaned WVS-7 file have already been re-oriented so higher = more trust (verified against the v6.0 codebook: Q57 Andorra raw = 25.5 % trust / 74.2 % careful, but the cleaned mean Q57P = 1.26 implies 1 = careful / 2 = trust). Q64–Q89 similarly re-oriented. Q292 left raw; semantic reverse-coding applied at scoring time inside `src/analysis/scoring.py::wvs_country_trust`.
2. **Reverse-coding fork on Q292.** The cleaned WVS file flags only Q292A, B, I as reverse-scored, but six items (A, B, E, F, H, I) are semantically negative. We reverse all six in both the LLM and WVS sides for an apples-to-apples trust score. The other three items (E, F, H) follow exactly the same logic — flagging them is the safer choice.
3. **Parse recovery.** A meaningful share of "missing" canonical_choice values for Claude and Gemini are actually parser failures, not refusals — the model put the choice mid-paragraph instead of leading. We recovered 269 of these (229 Opus + 40 Gemini) using a permissive line-and-verbal-token scan. Genuine refusals are the residual.
4. **Q292 has 13 countries, not 9.** The cleaned means file has 13 non-null Q292 country rows (Andorra is non-null with a single item missing — overall ≈13 across A–O). Treat the politicians comparison cautiously; per-country variation is large with so few units.
5. **Open question for next stage.** Whether to include `social_role_trust` (the TrustBench custom items: doctor, teacher, restaurant manager, neighbor, stranger asking for directions, people in general) in headline tables — there is no WVS reference for them, but they show a striking GPT/Gemini ≈ 0.71 vs Claude 0.53 split. Possibly worth a separate panel rather than mixing into Figure 1.

---

## Figure index

| File | What it shows |
|---|---|
| `fig1_trust_profile.png` | Per-item forest plot — LLMs with bootstrap CIs against the WVS-7 country-mean distribution, sorted by WVS pooled mean. Three panels: confidence in institutions / politicians / social groups. **Headline figure.** |
| `fig2_llm_vs_wvs_scatter.png` | Per-model calibration scatter against WVS pooled mean. Identity line, points colored by section. ρ / RMSE / bias annotated. |
| `fig3_framing_robustness.png` | Numeric-original vs numeric-reversed and verbal-labels for each model. Lower self-correlation = more framing-sensitive. |
| `fig4_cross_model.png` | Pairwise model agreement (3 scatter panels) + top-15 items with largest inter-model SD. |
| `fig5_country_resemblance.png` | Top-15 WVS-7 countries each LLM most resembles on Q64–Q89 (Spearman ρ, RMSE labelled). |
| `fig6_section_summary.png` | Section-level bar chart with WVS pooled mean overlays. |
| `fig7_refusal.png` | (A) refusal % by framing, (B) original / recovered / refused breakdown per model. |
| `fig8_politicians.png` | Q292 item-level deep-dive, separating reverse- and positive-phrased items, with WVS country strip + pooled mean. |

## Next analytic steps (paper-shaped)

1. **Anchor experiment integration.** `anchor_results_20260414_140054.jsonl` is sitting outside this EDA — fold in once we agree on how the anchor maps to TrustBench items.
2. **Demographic conditioning of WVS comparison.** Right now we compare LLM responses to *country-mean* WVS scores. The respondent-level file (97k rows × 93 columns) supports conditioning on age / education / urbanicity / regime type — a stronger headline would be "LLMs resemble high-education urban respondents in democratic countries" or refute it.
3. **Sub-scale construction for Q292.** With reverse-coding correct, build the four-factor stated-trust scale (OT/BI/CO/IH per `part2.py`) and compare to revealed-trust (`anchor_experiment.py`) — that's the v2-→v3 path the proposal hints at.
4. **Refusal as a signal, not noise.** Claude's verbal-label refusal spike + Gemini's uniform refusal rate are model-specific behaviors that probably deserve their own section/figure rather than being shown as a parse-rate footnote.
