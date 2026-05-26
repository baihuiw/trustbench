"""Loaders for TrustBench v2 LLM results, item dictionary, and WVS data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "outputs" / "results"
DATA_DIR = REPO / "data"

MODELS = {
    "Claude Opus 4.7": "opus47",
    "GPT-5.5": "gpt55",
    "Gemini 3.1": "gemini31",
}

MODEL_ORDER = ["Claude Opus 4.7", "GPT-5.5", "Gemini 3.1"]

WVS_MEANS = DATA_DIR / "(means)wvs7_trustbench_clean_country_means.csv"
WVS_ALL = DATA_DIR / "(all)wvs7_trustbench_clean.csv"
ITEMS_CSV = DATA_DIR / "trust_items_v2.csv"

REVERSE_FLAG_COLS = ["Q292A_REVERSE_FLAG", "Q292B_REVERSE_FLAG", "Q292I_REVERSE_FLAG"]


def load_items() -> pd.DataFrame:
    """Item dictionary: id, org_question_number (e.g. Q64), institution, section, reverse_coded, n-point scale."""
    items = pd.read_csv(ITEMS_CSV)
    scale_cols = [c for c in items.columns if c.startswith("scale_")]
    items["n_points"] = items[scale_cols].notna().sum(axis=1).astype(int)
    items["wvs_col"] = items["org_question_number"].where(
        items["org_question_number"].str.startswith("Q"), other=None
    )
    items.loc[items["section"] == "wvs_politicians", "wvs_col"] = (
        "Q292" + items.loc[items["section"] == "wvs_politicians", "statement"].str[0]
    )
    items.loc[items["section"].str.startswith("wvs_") & items["wvs_col"].str.match(r"^Q[0-9]+$"), "wvs_col"] += "P"
    return items


def load_llm_results(variant: str = "all") -> pd.DataFrame:
    """Load LLM JSONL results for all three models.

    variant: 'all' (base + verbal, 4500 rows/model), 'base' (3000), or 'verbal' (1500).
    """
    suffix = {"all": "_all", "base": "", "verbal": "_verbal"}[variant]
    rows = []
    for label, slug in MODELS.items():
        path = RESULTS_DIR / f"v2_{slug}{suffix}.jsonl"
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                m = r["metadata"]
                resp = r["response"]
                rows.append({
                    "model": label,
                    "model_slug": slug,
                    "prompt_id": m["prompt_id"],
                    "item_id": m["item_id"],
                    "section": m["section"],
                    "institution": m.get("institution", ""),
                    "condition": m["condition"],
                    "numbering": m.get("numbering"),
                    "framing": m.get("framing"),
                    "response_type": m.get("response_type"),
                    "reverse_coded": m.get("reverse_coded", False),
                    "repetition": resp.get("repetition"),
                    "raw_text": resp.get("raw_text", ""),
                    "canonical_choice": resp.get("canonical_choice"),
                    "error": resp.get("error"),
                })
    df = pd.DataFrame(rows)
    df["canonical_choice"] = pd.to_numeric(df["canonical_choice"], errors="coerce")
    df["framing_label"] = df["condition"].map({
        "original_survey_choice_justify": "numeric-original",
        "reversed_survey_choice_justify": "numeric-reversed",
        "verbal_survey_choice_justify": "verbal",
    })
    return df


def load_wvs_country_means() -> pd.DataFrame:
    """WVS-7 country-level weighted means. Index by B_COUNTRY (numeric ISO)."""
    df = pd.read_csv(WVS_MEANS)
    return df


def load_wvs_respondents(usecols: list[str] | None = None,
                        nrows: int | None = None) -> pd.DataFrame:
    """Respondent-level WVS (97k rows). Pass usecols to keep memory low."""
    return pd.read_csv(WVS_ALL, usecols=usecols, nrows=nrows)


COUNTRY_CODE_TO_NAME = {
    8: "Albania", 20: "Andorra", 32: "Argentina", 36: "Australia", 50: "Bangladesh",
    51: "Armenia", 68: "Bolivia", 76: "Brazil", 104: "Myanmar", 124: "Canada",
    152: "Chile", 156: "China", 158: "Taiwan", 170: "Colombia", 196: "Cyprus",
    203: "Czechia", 218: "Ecuador", 231: "Ethiopia", 268: "Georgia", 276: "Germany",
    300: "Greece", 320: "Guatemala", 344: "Hong Kong", 356: "India", 360: "Indonesia",
    364: "Iran", 368: "Iraq", 392: "Japan", 398: "Kazakhstan", 400: "Jordan",
    404: "Kenya", 410: "South Korea", 417: "Kyrgyzstan", 422: "Lebanon",
    434: "Libya", 446: "Macao", 458: "Malaysia", 462: "Maldives", 484: "Mexico",
    496: "Mongolia", 504: "Morocco", 528: "Netherlands", 554: "New Zealand",
    558: "Nicaragua", 566: "Nigeria", 578: "Norway", 586: "Pakistan", 604: "Peru",
    608: "Philippines", 616: "Poland", 630: "Puerto Rico", 642: "Romania",
    643: "Russia", 688: "Serbia", 702: "Singapore", 703: "Slovakia", 704: "Vietnam",
    716: "Zimbabwe", 752: "Sweden", 762: "Tajikistan", 764: "Thailand",
    780: "Trinidad and Tobago", 788: "Tunisia", 792: "Turkey", 804: "Ukraine",
    818: "Egypt", 826: "Great Britain", 840: "United States", 858: "Uruguay",
    860: "Uzbekistan", 862: "Venezuela", 909: "Northern Ireland",
}


def add_country_names(df: pd.DataFrame, col: str = "B_COUNTRY") -> pd.DataFrame:
    df = df.copy()
    df["country"] = df[col].map(COUNTRY_CODE_TO_NAME).fillna(df[col].astype(str))
    return df
