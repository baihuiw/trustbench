"""
Analysis Module
===============
Post-processing of raw API results:
  - Parse & score Likert and binary responses
  - Reverse-code where needed
  - Aggregate by model × country × institution × subscale
  - Export tidy CSV for downstream analysis (R, Python viz, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_results(path: str | Path) -> pd.DataFrame:
    """Load JSONL results into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    logger.info("Loaded %d records from %s", len(df), path)
    return df


def extract_prompt_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested prompt_item dict into top-level columns."""
    prompt_fields = pd.json_normalize(df["prompt_item"])
    # Prefix to avoid collisions
    prompt_fields = prompt_fields.add_prefix("p_")
    df = pd.concat([df.drop(columns=["prompt_item"]), prompt_fields], axis=1)
    return df


def score_responses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert parsed responses to numeric scores.
    - Likert: 1-4 or 1-5 or 1-7 as-is
    - Binary A/B: A=1 (trust domestic), B=0 (bypass)
    - Reverse-code where flagged
    """
    df = df.copy()

    # Numeric score
    def _to_numeric(row):
        val = row.get("parsed_value")
        if val is None:
            return None
        if val in ("A", "B"):
            return 1 if val == "A" else 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    df["score"] = df.apply(_to_numeric, axis=1)

    # Reverse coding
    def _reverse(row):
        if not row.get("p_reverse_coded", False):
            return row["score"]
        if row["score"] is None:
            return None
        section = row.get("p_section", "")
        if section == "wvs_confidence":
            # 4-point: 1↔4, 2↔3
            return 5 - row["score"]
        elif section == "wvs_politicians":
            # 5-point: 1↔5, 2↔4
            return 6 - row["score"]
        elif section == "stated_trust":
            # 7-point: 1↔7, 2↔6, etc.
            return 8 - row["score"]
        return row["score"]

    df["score_reversed"] = df.apply(_reverse, axis=1)
    return df


def aggregate_scores(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Create summary tables:
      - part1_summary: model × institution mean scores
      - stated_summary: model × country × institution × subscale
      - revealed_summary: model × country × institution P(trust)
    """
    summaries = {}

    # ── Part 1 ─────────────────────────────────────────────────
    p1 = df[df["p_part"] == 1].copy()
    if len(p1):
        summaries["part1_summary"] = (
            p1.groupby(["model_label", "p_section", "p_institution", "p_item_id"])
            .agg(
                mean_score=("score_reversed", "mean"),
                std_score=("score_reversed", "std"),
                n=("score_reversed", "count"),
                n_parsed=("parsed_value", lambda x: x.notna().sum()),
            )
            .reset_index()
        )

    # ── Part 2: Stated ─────────────────────────────────────────
    stated = df[df["p_section"] == "stated_trust"].copy()
    if len(stated):
        summaries["stated_summary"] = (
            stated.groupby([
                "model_label", "p_country", "p_institution",
                "p_metadata",  # contains subscale
            ])
            .agg(
                mean_score=("score_reversed", "mean"),
                std_score=("score_reversed", "std"),
                n=("score_reversed", "count"),
            )
            .reset_index()
        )

    # ── Part 2: Revealed ───────────────────────────────────────
    revealed = df[df["p_section"] == "revealed_trust"].copy()
    if len(revealed):
        summaries["revealed_summary"] = (
            revealed.groupby([
                "model_label", "p_country", "p_institution", "p_item_id",
            ])
            .agg(
                p_trust=("score", "mean"),   # proportion choosing A
                n=("score", "count"),
                n_parsed=("parsed_value", lambda x: x.notna().sum()),
            )
            .reset_index()
        )

    return summaries


def export_summaries(
    summaries: dict[str, pd.DataFrame],
    output_dir: str | Path,
):
    """Save each summary as a CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in summaries.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("Exported %s (%d rows) to %s", name, len(df), path)


def extract_logprob_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-response probability distributions from top_logprobs.

    For each response, finds the logprobs of scale-relevant tokens
    (e.g., "1","2","3","4" for 4-point; "A","B" for binary) among
    the top-k returned tokens, converts to probabilities via softmax
    over the relevant subset, and stores as new columns.

    Returns a DataFrame with columns like:
        logprob_token_1, logprob_token_2, ..., prob_token_1, prob_token_2, ...
    """
    import math

    records = []
    for _, row in df.iterrows():
        top_lp = row.get("top_logprobs")
        scale_labels = row.get("p_scale_labels", [])
        rec = {"_idx": row.name}

        if not top_lp or not isinstance(top_lp, list):
            records.append(rec)
            continue

        # Build lookup: token_str -> logprob
        token_lp_map = {}
        for entry in top_lp:
            if isinstance(entry, dict):
                token_lp_map[entry.get("token", "").strip()] = entry.get("logprob", None)

        # Determine which tokens are scale-relevant
        if isinstance(scale_labels, list) and scale_labels == ["A", "B"]:
            relevant_tokens = ["A", "B"]
        elif isinstance(scale_labels, list):
            relevant_tokens = [str(i) for i in range(1, len(scale_labels) + 1)]
        else:
            records.append(rec)
            continue

        # Extract logprobs for relevant tokens
        found_logprobs = {}
        for tok in relevant_tokens:
            lp = token_lp_map.get(tok)
            rec[f"logprob_token_{tok}"] = lp
            if lp is not None:
                found_logprobs[tok] = lp

        # Convert to probabilities (softmax over found relevant tokens)
        if found_logprobs:
            max_lp = max(found_logprobs.values())
            exp_vals = {t: math.exp(lp - max_lp) for t, lp in found_logprobs.items()}
            total_exp = sum(exp_vals.values())
            for tok in relevant_tokens:
                if tok in exp_vals:
                    rec[f"prob_token_{tok}"] = exp_vals[tok] / total_exp
                else:
                    rec[f"prob_token_{tok}"] = None
        else:
            for tok in relevant_tokens:
                rec[f"prob_token_{tok}"] = None

        records.append(rec)

    lp_df = pd.DataFrame(records).set_index("_idx")
    return lp_df


def run_analysis(results_path: str | Path, output_dir: str | Path):
    """Full analysis pipeline."""
    output_dir = Path(output_dir)
    df = load_results(results_path)
    df = extract_prompt_fields(df)
    df = score_responses(df)

    # Extract logprob distributions
    has_logprobs = df["top_logprobs"].apply(
        lambda x: x is not None and isinstance(x, list) and len(x) > 0
    ).sum()
    logger.info("Responses with logprobs: %d / %d", has_logprobs, len(df))

    if has_logprobs > 0:
        lp_df = extract_logprob_distributions(df)
        df = df.join(lp_df)
        # Save full logprobs separately (they're large)
        logprob_path = output_dir / "logprob_distributions.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        lp_cols = [c for c in df.columns if c.startswith(("logprob_token_", "prob_token_"))]
        id_cols = ["prompt_id", "model_id", "model_label", "repetition",
                   "parsed_value", "p_country", "p_institution", "p_item_id", "p_section"]
        available_id_cols = [c for c in id_cols if c in df.columns]
        df[available_id_cols + lp_cols].to_csv(logprob_path, index=False)
        logger.info("Saved logprob distributions to %s", logprob_path)

    # Save scored data (drop the large logprobs/top_logprobs columns for CSV)
    scored_path = output_dir / "scored_responses.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    drop_cols = [c for c in ["logprobs", "top_logprobs"] if c in df.columns]
    df.drop(columns=drop_cols, errors="ignore").to_csv(scored_path, index=False)
    logger.info("Saved scored responses to %s", scored_path)

    summaries = aggregate_scores(df)
    export_summaries(summaries, output_dir)

    # Quick stats
    total = len(df)
    parsed = df["parsed_value"].notna().sum()
    logger.info(
        "Parse rate: %d / %d (%.1f%%)",
        parsed, total, 100 * parsed / total if total else 0,
    )

    return df, summaries
