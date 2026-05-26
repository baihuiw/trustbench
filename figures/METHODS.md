# TrustBench v2 EDA — methods walkthrough


Python: `/Users/wangbaihui/anaconda3/bin/python` (pandas 2.1, numpy 1.26, matplotlib 3.10, seaborn 0.13, scipy 1.11).
Modules live in `src/analysis/`. Outputs in `figures/eda/`.

---

## Part 1 · WVS-pooled deliverable

### Step 1 — Inventory the raw inputs

Before touching code, I confirmed the structure of every input.

```bash
# LLM result files — three models × three variants
wc -l outputs/results/v2_{opus47,gpt55,gemini31}{,_all,_verbal}.jsonl
# → base = 3 000, all = 4 500, verbal = 1 500 per model
```

```python
# Sample one record to learn the schema
import json
with open('outputs/results/v2_opus47_all.jsonl') as f:
    r = json.loads(f.readline())
print(r['metadata'].keys())   # prompt_id, item_id, section, institution,
                              # condition, numbering, framing, response_type,
                              # reverse_coded, order_mapping
print(r['response'].keys())   # model, repetition, raw_text, choice_from_text,
                              # canonical_choice, justification, error
```

**Key facts established at this step:**
- 50 items per model × 3 framings (numeric-original, numeric-reversed, verbal) × 30 reps.
- `_all` = `base ∪ verbal` exactly (verified by set intersection — zero overlap, union equals `_all`).
- `canonical_choice` is already direction-normalized for reversed-scale framings, but it is NaN whenever the response leads with prose instead of a number/label.

### Step 2 — Verify WVS scale direction (this was the most important step)

The cleaned WVS CSV could not be trusted to follow the published codebook orientation, so I checked empirically.

```python
import pandas as pd
df = pd.read_csv('data/(all)wvs7_trustbench_clean.csv',
                 usecols=['B_COUNTRY','Q57P','Q58P','Q64P','Q73P','Q292A','Q292C'],
                 nrows=200000)
for q in ['Q57P','Q58P','Q63P','Q64P','Q73P','Q292A','Q292C']:
    print(q, df[q].value_counts(dropna=False).head().to_dict())
```

Result: Q58P (family trust) had value 4 appearing 75 456 times. If 4 = "Do not trust at all" (codebook direction), 75k people in 66 countries refuse to trust their family — impossible. So the cleaning pipeline must have re-oriented it. I confirmed against the v6.0 codebook PDF using `pdftotext -layout` and searching for "Q57-" / "Q64-":

```bash
pdftotext -layout "F00010763-WVS_Results_By_Country_2017-2022_v6.0.0.pdf" /tmp/wvs.txt
grep -n "^Q57-\|^Q64-" /tmp/wvs.txt
# → Q57 line 15320, Q64 line 16049
awk 'NR>=15320 && NR<=15340' /tmp/wvs.txt
```

Codebook Andorra Q57: 25.5 % trust / 74.2 % careful → expected raw mean ≈ 1.74.
Cleaned Q57P Andorra mean = 1.26. The only way to get 1.26 is if 1 = careful, 2 = trust — i.e. the data has been re-oriented to *higher = more trust*. Same for Q58–Q63 and (after a Q64 Andorra check: codebook 6.2/22.2/36.8/34.4 % across the four points, cleaned Q64P Andorra = 2.00) for Q64–Q89.

**Decision recorded at this step:** in the cleaned WVS file, *all* social and institutional items are oriented so higher = more trust. Q292 is left raw (codebook direction). This rule drove every scoring formula downstream.

### Step 3 — Build `src/analysis/loader.py`


```python
def load_items() -> pd.DataFrame:
    items = pd.read_csv(ITEMS_CSV)
    scale_cols = [c for c in items.columns if c.startswith("scale_")]
    items["n_points"] = items[scale_cols].notna().sum(axis=1)
    # Map every item to its WVS column name
    items["wvs_col"] = items["org_question_number"].where(
        items["org_question_number"].str.startswith("Q"), other=None)
    items.loc[items.section == "wvs_politicians", "wvs_col"] = (
        "Q292" + items.loc[items.section == "wvs_politicians", "statement"].str[0])
    # Confidence/social cols have a "P" suffix; Q292 does not
    items.loc[items.section.str.startswith("wvs_") &
              items.wvs_col.str.match(r"^Q[0-9]+$"), "wvs_col"] += "P"
    return items

def load_llm_results(variant: str = "all") -> pd.DataFrame:
    suffix = {"all": "_all", "base": "", "verbal": "_verbal"}[variant]
    rows = []
    for label, slug in MODELS.items():
        with open(RESULTS_DIR / f"v2_{slug}{suffix}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                rows.append({...})
    return pd.DataFrame(rows)
```

Plus a `COUNTRY_CODE_TO_NAME` dict (ISO-3166-numeric → country name) so output figures can use readable labels.

### Step 4 — Build `src/analysis/scoring.py`: the trust-score transform

Two functions matter.

**4a — `_recover_choice`**: a permissive parser for responses where `canonical_choice` is NaN. The pattern was that ~12 % of Opus and Gemini responses had the choice buried mid-paragraph ("As an AI I don't have personal confidence, but if I must answer, 2: Quite a lot..."). The recovery parser scans for:

```python
# Explicit "Answer: N", "Choice: N", "N:", "N." patterns
for pat in (r"(?:answer|choice|response|rating|score)\s*[:=\-]\s*([1-9])\b",
            r"^\s*\(?([1-9])\)?[\.\:\)]\s",
            r"\n\s*\(?([1-9])\)?[\.\:\)]\s"):
    m = re.search(pat, text.lower())
    if m: return float(m.group(1))
# Verbal framing: look for scale labels
for phrase, val in sorted(VERBAL_LOOKUP.items(), key=lambda kv: -len(kv[0])):
    if phrase in text.lower(): return float(val)
# Numeric fallback: first standalone 1..n in first 300 chars
```

This recovered 229 Opus and 40 Gemini responses; the residual is recorded as `refused` and reported transparently (Figure 7B).

**4b — `add_trust_score`**: normalize every response to [0, 1] where higher = more trust.

```python
c = out["choice_used"]; n = out["n_points"].astype(float)
default_trust = (n - c) / (n - 1)       # scale_1 is highest trust
pol_pos_trust = (c - 1) / (n - 1)       # positively-phrased Q292: high agree = trust
out["trust"] = np.where(
    (out.section == "wvs_politicians") & (out.reverse_coded == 0),
    pol_pos_trust, default_trust)
```

The WVS side gets the *opposite* default formula because of the Step-2 finding:

```python
def wvs_country_trust(wvs_means, items):
    # For Q57P, Q58P-Q63P, Q64P-Q89P: cleaned data has higher=trust
    # → trust = (c - 1) / (n - 1)
    # For Q292 negatively-phrased: trust = (n - c) / (n - 1)
    # For Q292 positively-phrased: trust = (c - 1) / (n - 1)
```

**Reverse-coding fork on Q292:** the cleaned WVS file flags only A, B, I as reverse-scored, but A, B, E, F, H, I are *all* semantically negative. I reverse all six in both the LLM and WVS sides for an apples-to-apples score, and documented the divergence in the methods notes.

**Sanity check after Step 4:**

```
Family (Q58P, social_2):    WVS mean 0.91   (everyone trusts family → max)
Generalized trust (Q57P):    WVS mean 0.24   (~24 % trust most people, matches WVS-7 priors)
Political parties (Q72P):    WVS mean 0.33   (universally distrusted)
```

These match published WVS-7 numbers, confirming the scoring is right.

### Step 5 — Build `src/analysis/aggregate.py`: model × item × CI

```python
def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    v = np.asarray(values, dtype=float); v = v[~np.isnan(v)]
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return v.mean(), lo, hi

def model_item_means(scored, by=["model","item_id"]):
    rows = []
    for keys, g in scored.dropna(subset=["trust"]).groupby(by):
        m, lo, hi = bootstrap_ci(g["trust"].values, n_boot=1000)
        rows.append({**dict(zip(by, keys)), "trust_mean": m,
                     "trust_lo": lo, "trust_hi": hi, "n": len(g)})
    return pd.DataFrame(rows)
```

Bootstrap chosen over normal-approximation CIs because trust scores are bounded [0, 1] and many model responses are mode-collapsed (zero variance), where a parametric CI would degenerate.

### Step 6 — Build `src/analysis/wvs.py`: country-distribution context

```python
def wvs_summary(items):
    # Pooled (weighted by N_respondents) + median + IQR per item across 66 countries
    tidy = wvs_country_trust(load_wvs_country_means(), items)
    rows = []
    for iid, g in tidy.groupby("item_id"):
        valid = g.dropna(subset=["trust"])
        w = valid["B_COUNTRY"].map(n_resp)
        wmean = (valid["trust"] * w).sum() / w.sum()
        rows.append({..., "wvs_mean": wmean,
                     "wvs_iqr_lo": valid["trust"].quantile(0.25),
                     "wvs_iqr_hi": valid["trust"].quantile(0.75)})
    return pd.DataFrame(rows)

def llm_country_resemblance(scored, items, use_section="wvs_confidence"):
    # For each model × country, Spearman ρ on the 26-item Q64–Q89 profile
    from scipy.stats import spearmanr
    llm_vec = scored.groupby(["model","item_id"])["trust"].mean()
    for model in models:
        for country in countries:
            rho, p = spearmanr(llm_vec[model], country_trust[country])
            rmse = np.sqrt(((llm_vec[model] - country_trust[country])**2).mean())
```

### Step 7 — Build `src/analysis/figures.py`: publication style

A common style block applied to every figure:

```python
def _style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", ... })

MODEL_COLORS = {  # Wong 2011 colorblind-safe
    "Claude Opus 4.7": "#E69F00",
    "GPT-5.5":          "#0072B2",
    "Gemini 3.1":       "#009E73"}
```

Every figure saved as both PNG (300 dpi) and PDF.

**Figure-by-figure design decisions:**

| Fig | What it shows | Design choice & why |
|---|---|---|
| F1 | Forest plot: items × (3 AI + WVS country distribution + pooled mean) | Items sorted ascending by WVS pooled mean within each section. AI markers as small error-bar dots at three vertical offsets (one per model). WVS country values as gray scatter on the row; pooled mean as a heavy black tick. **Headline figure** — combines pooled overlay + country variation in one view per the user's choice. |
| F2 | LLM mean vs WVS pooled mean scatter (1 panel × 3 models) | Identity line for visual calibration check; points colored by section (Confidence / Politicians / Social-general / Social-groups). Spearman ρ, RMSE, bias inline. |
| F3 | Framing robustness scatter | Per-model, two scatters overlaid: num-original vs num-reversed (filled dots) and num-original vs verbal (open squares). Correlation r in the corner of each panel. |
| F4 | Cross-model agreement | 3 pairwise scatters + a fourth panel showing the top-15 items with largest inter-model SD as horizontal bars with each model's score overlaid. |
| F5 | Country resemblance | One horizontal bar chart per model, top-15 countries by Spearman ρ. RMSE labeled to the right of each bar. |
| F6 | Section means with WVS overlay | Grouped bar chart (5 sections × 3 models) with bootstrap CIs; WVS pooled mean as a heavy black notch per section. |
| F7 | Refusal & parse recovery | Two-panel: (A) refusal % by framing as grouped bars, (B) stacked horizontal bar of `parsed_source` (original / recovered / refused) per model. |
| F8 | Q292 deep-dive | Same forest-plot grammar as F1 but only the 11 politicians items, divided by a dashed line into reverse-coded (top) and positively-phrased (bottom) blocks. |

Pitfalls encountered and fixed during this step:
- `DataFrame.std(axis=1)` and `.max(axis=1)` raised `RecursionError` on a frame whose column index was a `pandas.Index` of strings (model names) — pandas internals tried to coerce numeric dtype. Workaround: drop to numpy (`np.nanstd(mat.to_numpy(), axis=1)`) before applying reductions.
- Country codes 462 (Maldives) and 909 (Northern Ireland) were missing from my initial dict, surfacing as numeric strings in F5. Added.
- F4's right-panel labels initially overlapped the third scatter panel — fixed with `gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6])` and explicit `wspace`.

### Step 8 — Notebook + report

`src/analysis/eda.ipynb` is a thin orchestration layer (~10 code cells) that imports everything from the modules and prints the tables/figures inline. Validated end-to-end by concatenating all code cells and `exec`-ing them as one script (saves on installing a Jupyter kernel inside the anaconda env).

`figures/eda/wvs-pooled/EDA_report.md` is the narrative: 10 numbered findings keyed to specific figures, plus a methodology-notes section.

---

## Part 2 · US-comparison deliverable

### Step 1 — Check what the WVS-7 US sample supports

Before writing any code, three things had to be confirmed against the cleaned WVS file.

```python
us = pd.read_csv('data/(all)wvs7_trustbench_clean.csv',
                 usecols=['B_COUNTRY','W_WEIGHT','Q240','Q241','Q223',
                          'Q64P','Q73P','Q292A','Q292C'])
us = us[us['B_COUNTRY'] == 840]
print('n=', len(us))                       # 2 596
print(us['Q240'].value_counts(dropna=False).sort_index())
print('Q292 non-null:', us['Q292A'].notna().sum())  # 0 ←
```

**Findings that shaped the design:**
1. n = 2 596 US respondents.
2. Q223 (party voted for) is **not** in the cleaned columns. Q240 (left–right 1–10) is present with full spread (248 strong-left, 615 centrist, 182 strong-right). → Use Q240 as the Rep/Dem proxy.
3. Q292 (politicians) has **0 non-null US rows** — that module wasn't run in the US. → AI-vs-US politicians figure is impossible from this dataset; the politicians figure falls back to AI vs WVS-13 pool only.

These three constraints were turned into the AskUserQuestion at the start of the US deliverable to confirm the user's preferred handling.

### Step 2 — Build `src/analysis/us_comparison.py`

```python
POLITICAL_BUCKETS = {
    "Liberal (Q240 1-4)":       lambda x: (x >= 1) & (x <= 4),
    "Centrist (Q240 5)":        lambda x: (x == 5),
    "Conservative (Q240 6-10)": lambda x: (x >= 6) & (x <= 10),
}  # standard Pew/Gallup convention

@lru_cache(maxsize=1)
def _us_respondents() -> pd.DataFrame:
    cols = ["B_COUNTRY","W_WEIGHT","Q240"] + [f"Q{q}P" for q in
            list(range(57,64)) + list(range(64,90))] + Q292_COLS
    df = load_wvs_respondents(usecols=cols)
    return df[df["B_COUNTRY"] == US_COUNTRY_CODE].copy()
```

`@lru_cache` because the respondent-level file is the only slow load in the whole pipeline (~20 MB CSV).

**Weighted mean with bootstrap CI** — has to be weighted because the WVS sampling design includes design weights:

```python
def us_item_means(items, political_split=False, bootstrap_n=500):
    us = _us_respondents()
    groups = ({g: us[mask(us["Q240"])] for g, mask in POLITICAL_BUCKETS.items()}
              if political_split else {"US (all)": us})
    rng = np.random.default_rng(2026)
    for gname, gdf in groups.items():
        w = gdf["W_WEIGHT"].fillna(1.0).clip(lower=0).values
        for it in items.itertuples():
            vals = gdf[it.wvs_col].values
            ok = ~np.isnan(vals); v, ww = vals[ok], w[ok]
            # Apply the same scoring rule as in scoring.wvs_country_trust:
            if it.section == "wvs_politicians" and it.reverse_coded:
                trust = (it.n_points - v) / (it.n_points - 1)
            else:
                trust = (v - 1) / (it.n_points - 1)
            trust = np.clip(trust, 0, 1)
            wmean = (trust * ww).sum() / ww.sum()
            # bootstrap of the weighted mean
            idx = rng.integers(0, len(trust), size=(bootstrap_n, len(trust)))
            bts = (trust[idx] * ww[idx]).sum(axis=1) / ww[idx].sum(axis=1)
            lo, hi = np.percentile(bts, [2.5, 97.5])
```

The bootstrap **resamples respondents with replacement and recomputes the weighted mean each draw** — that gives a valid CI in the presence of design weights.

### Step 3 — Profile similarity (Spearman + RMSE)

```python
def profile_similarity(scored_llm, items, section="wvs_confidence"):
    items_in = items[items.section == section]
    mat = cross_model_item_matrix(scored_llm).reindex(items_in["id"])
    refs = {("US (all)", "US-aggregate"): us_item_means(items_in).set_index("item_id")["trust_mean"]}
    for col in us_pol_wide.columns:
        refs[(col, "US-political")] = us_pol_wide[col]      # Lib / Cent / Cons
    refs[("WVS pooled", "WVS")] = wvs_pooled
    for c in wvs_t.columns:                                  # 66 WVS countries
        refs[(COUNTRY_CODE_TO_NAME[c], "WVS-country")] = wvs_t[c]
    for model in mat.columns:
        for (ref_label, ref_kind), ref_vec in refs.items():
            common = pd.concat([mat[model], ref_vec], axis=1).dropna()
            rho, _ = spearmanr(common.iloc[:, 0], common.iloc[:, 1])
            rmse = np.sqrt(((common.iloc[:,0] - common.iloc[:,1])**2).mean())
```

Spearman over Pearson because trust scores can be bounded near 0 or 1 (mode-collapsed responses) and we care about *ranking* of institutions, not absolute fit. RMSE complements ρ by capturing absolute miscalibration.

### Step 4 — Build `src/analysis/us_figures.py`

Reused the same `_style()` and `MODEL_COLORS` from the EDA module to keep visual consistency across both deliverables.

| Fig | What it shows | Design choice |
|---|---|---|
| F1 | 33 items × {3 AI, US-all, US Lib/Cent/Cons} forest plot | Items sorted ascending by US-all trust mean. Three AI models as colored dots at vertical offsets. US-all as a black diamond with bootstrap CI. Lib/Cent/Cons as colored vertical ticks with a gray bar spanning the Lib–Cons range. Shortlist labels bolded with a ★ prefix. |
| F2 | 3×4 grid, 12 shortlist institutions | Each panel: gray jitter for 66 WVS countries, vertical black line for WVS pooled mean, white-fill black-edge diamond for US (with "US" label below), AI models as colored error-bar dots at vertical offsets. Compact, lets the reader scan one institution at a time. |
| F3 | 11 Q292 items, AI vs WVS-13 | Same forest grammar as EDA F8; US dropped because no Q292 data. Negative/positive split shown with a dashed horizontal divider. |
| F4 | Spearman ρ heatmap, 16 references × 3 models | `seaborn.heatmap(annot=True)` with `cmap="vlag"` centered on 0. Three reference groups separated by black `axhline`s and labeled with custom brackets drawn in `get_yaxis_transform` coordinates so the bracket sits outside the heatmap regardless of figure size. |

### Step 5 — US notebook + report

`src/analysis/us_comparison.ipynb` mirrors the EDA notebook structure: load → US means → AI vs US gap → Lib/Cons gap → profile similarity → generate figures. Validated by `exec`-ing the concatenated code cells.

`figures/eda/us_comparison/REPORT.md` follows the user's safety/legitimacy framing: H1 (Liberal-leaning AI profile), H2 (three regimes), H3 (mode collapse), H4 (where Lib–Cons disagree, AI tracks Liberal), H5 (Q292 politicians — corrected after a finding that the original "less trusting on every item" claim wasn't true, only ~half of items).

---

## How to reproduce, top to bottom

```bash
cd /Users/wangbaihui/trustbench
PY=/Users/wangbaihui/anaconda3/bin/python

# WVS-pooled deliverable
$PY -c "
import sys; sys.path.insert(0, '.')
from src.analysis.loader   import load_items, load_llm_results
from src.analysis.scoring  import add_trust_score
from src.analysis.figures  import make_all
items, llm = load_items(), load_llm_results('all')
scored = add_trust_score(llm, items)
make_all(scored, items)            # writes 8 figures to figures/eda/wvs-pooled/
"

# US-comparison deliverable
$PY -c "
import sys; sys.path.insert(0, '.')
from src.analysis.loader   import load_items, load_llm_results
from src.analysis.scoring  import add_trust_score
from src.analysis.us_figures import make_all
items, llm = load_items(), load_llm_results('all')
scored = add_trust_score(llm, items)
make_all(scored, items)            # writes 4 figures to figures/eda/us_comparison/
"
```

Or run the notebooks end to end:
- `src/analysis/eda.ipynb` (WVS deliverable)
- `src/analysis/us_comparison.ipynb` (US deliverable)

---

## What is and is not in scope of these two deliverables

**In scope:**
- Stated-trust elicitation only (no revealed-trust scenarios from `anchor_experiment.py` yet).
- Three models × three framings × thirty repetitions.
- 50 items (26 Q64–Q89 + 11 Q292 + 1 Q57 + 6 Q58–Q63 + 6 custom social-role).
- Comparison to WVS-7 country means (66 countries; 13 for Q292) and US respondent-level data.
- Ideological split via Q240 left–right tri-split (1–4 / 5 / 6–10).

**Not yet:**
- Anchor experiment integration.
- Demographic conditioning beyond political ID (age × education × urbanicity).
- The four-factor Q292 sub-scale (OT/BI/CO/IH per `part2.py`).
- Revealed-trust delegation scenarios.
- Any per-country LLM prompt-context manipulation.
