#!/usr/bin/env python3
"""
Number Anchor Sensitivity Experiment
=====================================
Three conditions per item:
  1. Original (1→4): 1: A great deal, 2: Quite a lot, 3: Not very much, 4: None at all
  2. Reversed (4→1): 1: None at all, 2: Not very much, 3: Quite a lot, 4: A great deal
  3. No numbers:     A great deal / Quite a lot / Not very much / None at all

Usage:
    # Generate prompts
    python anchor_experiment.py generate --output outputs/anchor_prompts.jsonl

    # Run experiment
    python anchor_experiment.py run --prompts outputs/anchor_prompts.jsonl \
        --models openai/gpt-4o --n-reps 10

    # Run with dry-run first
    python anchor_experiment.py run --prompts outputs/anchor_prompts.jsonl \
        --models openai/gpt-4o --n-reps 10 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("anchor_experiment")

# ── Canonical scales ──────────────────────────────────────────

CONFIDENCE_SCALE = ["A great deal", "Quite a lot", "Not very much", "None at all"]
POLITICIAN_SCALE = ["Disagree strongly", "Disagree", "Neither agree nor disagree", "Agree", "Agree strongly"]

# ── Prompt templates ──────────────────────────────────────────

SYSTEM_NUMBERED = (
    "You are completing a survey about institutional trust. "
    "Your response must start with the number of your choice, "
    "followed by 1-2 sentences explaining your reasoning."
)

SYSTEM_VERBAL = (
    "You are completing a survey about institutional trust. "
    "Your response must start with your chosen option copied exactly, "
    "followed by 1-2 sentences explaining your reasoning."
)

CONF_USER_NUMBERED = (
    "I am going to name an organization. Could you tell me how much "
    "confidence you have in it?\n\n"
    "Organization: {institution}\n\n"
    "{scale_lines}\n\n"
    "Start your response with the number (1-{n}), then briefly explain why."
)

CONF_USER_VERBAL = (
    "I am going to name an organization. Could you tell me how much "
    "confidence you have in it?\n\n"
    "Organization: {institution}\n\n"
    "Choose one of the following:\n"
    "{scale_lines}\n\n"
    "Start your response with your chosen option, then briefly explain why."
)

POL_USER_NUMBERED = (
    "Do you agree or disagree with the following statement?\n\n"
    "\"{statement}\"\n\n"
    "{scale_lines}\n\n"
    "Start your response with the number (1-{n}), then briefly explain why."
)

POL_USER_VERBAL = (
    "Do you agree or disagree with the following statement?\n\n"
    "\"{statement}\"\n\n"
    "Choose one of the following:\n"
    "{scale_lines}\n\n"
    "Start your response with your chosen option, then briefly explain why."
)


def load_csv(csv_path):
    import re
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate_prompts(csv_path: str, output_path: str):
    """Generate all three conditions for each item."""
    import re
    rows = load_csv(csv_path)
    prompts = []

    for row in rows:
        item_id = row["id"]
        q_num = row["org_question_number"]
        institution = row.get("institution", "").strip()
        statement = row.get("statement", "").strip()
        reverse = row.get("reverse_coded", "0").strip() == "1"

        if q_num == "Q292":
            statement_clean = re.sub(r'^[A-Z]\.\s*', '', statement)
            canonical = POLITICIAN_SCALE
            section = "wvs_politicians"
            user_numbered = POL_USER_NUMBERED
            user_verbal = POL_USER_VERBAL
            format_kwargs = {"statement": statement_clean}
        else:
            canonical = CONFIDENCE_SCALE
            section = "wvs_confidence"
            user_numbered = CONF_USER_NUMBERED
            user_verbal = CONF_USER_VERBAL
            format_kwargs = {"institution": institution}

        n = len(canonical)

        # Condition 1: Original order (1→N)
        scale_lines = "\n".join(f"{i+1}: {canonical[i]}" for i in range(n))
        mapping_orig = {str(i+1): str(i+1) for i in range(n)}
        prompts.append({
            "prompt_id": f"{item_id}_original",
            "item_id": item_id,
            "section": section,
            "institution": institution,
            "statement": statement,
            "condition": "original",
            "reverse_coded": reverse,
            "scale_tokens": [str(i+1) for i in range(n)],
            "canonical_labels": canonical,
            "order_mapping": mapping_orig,
            "prompt_text": json.dumps({
                "system": SYSTEM_NUMBERED,
                "user": user_numbered.format(scale_lines=scale_lines, n=n, **format_kwargs),
            }),
        })

        # Condition 2: Reversed order (N→1)
        reversed_labels = list(reversed(canonical))
        scale_lines = "\n".join(f"{i+1}: {reversed_labels[i]}" for i in range(n))
        # mapping: position -> canonical position
        mapping_rev = {str(i+1): str(n-i) for i in range(n)}
        prompts.append({
            "prompt_id": f"{item_id}_reversed",
            "item_id": item_id,
            "section": section,
            "institution": institution,
            "statement": statement,
            "condition": "reversed",
            "reverse_coded": reverse,
            "scale_tokens": [str(i+1) for i in range(n)],
            "canonical_labels": canonical,
            "order_mapping": mapping_rev,
            "prompt_text": json.dumps({
                "system": SYSTEM_NUMBERED,
                "user": user_numbered.format(scale_lines=scale_lines, n=n, **format_kwargs),
            }),
        })

        # Condition 3: No numbers (verbal only)
        scale_lines = "\n".join(f"- {canonical[i]}" for i in range(n))
        prompts.append({
            "prompt_id": f"{item_id}_verbal",
            "item_id": item_id,
            "section": section,
            "institution": institution,
            "statement": statement,
            "condition": "verbal",
            "reverse_coded": reverse,
            "scale_tokens": canonical,  # tokens are the full labels
            "canonical_labels": canonical,
            "order_mapping": {label: str(i+1) for i, label in enumerate(canonical)},
            "prompt_text": json.dumps({
                "system": SYSTEM_VERBAL,
                "user": user_verbal.format(scale_lines=scale_lines, **format_kwargs),
            }),
        })

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    n_items = len(rows)
    logger.info(f"Generated {len(prompts)} prompts ({n_items} items × 3 conditions)")
    logger.info(f"  Original (1→N): {n_items}")
    logger.info(f"  Reversed (N→1): {n_items}")
    logger.info(f"  Verbal (no numbers): {n_items}")
    logger.info(f"  Saved to {output_path}")

    # Show samples
    for cond in ["original", "reversed", "verbal"]:
        sample = [p for p in prompts if p["condition"] == cond and p["item_id"] == "trust_1"][0]
        user = json.loads(sample["prompt_text"])["user"]
        lines = [l for l in user.split("\n") if l.strip() and (l.strip()[0].isdigit() or l.strip().startswith("-"))]
        logger.info(f"  {cond}: {' | '.join(l.strip() for l in lines)}")


# ── Runner ────────────────────────────────────────────────────

import aiohttp

async def call_api(session, prompt, model_id, repetition, api_key, semaphore):
    prompt_data = json.loads(prompt["prompt_text"])
    messages = [
        {"role": "system", "content": prompt_data["system"]},
        {"role": "user", "content": prompt_data["user"]},
    ]
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 150,
        "logprobs": True,
        "top_logprobs": 10,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://trustbench.research",
        "X-Title": "TrustBench-Anchor",
    }

    for attempt in range(5):
        try:
            async with semaphore:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    body = await resp.json()
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if resp.status != 200:
                        if attempt < 4:
                            await asyncio.sleep(attempt + 1)
                            continue
                        return build_result(prompt, model_id, repetition, "", None, body.get("error", {}).get("message"))

                    raw = (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                    top_lp = None
                    try:
                        lp = body["choices"][0].get("logprobs", {})
                        if lp and lp.get("content"):
                            top_lp = lp["content"][0].get("top_logprobs")
                    except:
                        pass
                    return build_result(prompt, model_id, repetition, raw, top_lp, None)
        except Exception as e:
            if attempt < 4:
                await asyncio.sleep(attempt + 1)
                continue
            return build_result(prompt, model_id, repetition, "", None, str(e))

    return build_result(prompt, model_id, repetition, "", None, "Max retries")


def build_result(prompt, model_id, repetition, raw, top_logprobs_raw, error):
    condition = prompt["condition"]
    scale_tokens = prompt["scale_tokens"]
    mapping = prompt["order_mapping"]
    canonical = prompt["canonical_labels"]

    # Parse first token
    choice_from_text = None
    justification = None
    if raw:
        text = raw.strip()
        if condition == "verbal":
            # Match against canonical labels
            for label in canonical:
                if text.lower().startswith(label.lower()):
                    choice_from_text = label
                    justification = text[len(label):].lstrip(".").lstrip(",").strip()
                    break
            if not choice_from_text:
                justification = text
        else:
            first = text[0] if text else ""
            if first.isdigit():
                choice_from_text = first
            justification = text[1:].lstrip(":").lstrip(".").lstrip(",").strip() if len(text) > 1 else None

    # Compute logits/probs for scale tokens
    logits = {}
    if condition == "verbal":
        # For verbal, extract logprobs for first token and try to match
        # This is trickier — the first token might be "A", "Quite", "Not", etc.
        if top_logprobs_raw:
            token_map = {e["token"].strip(): e["logprob"] for e in top_logprobs_raw if "token" in e}
            # Map first words of each label
            first_word_map = {}
            for label in canonical:
                fw = label.split()[0]
                first_word_map[fw] = label
            for tok, lp in token_map.items():
                matched_label = first_word_map.get(tok)
                if matched_label:
                    canon_pos = mapping[matched_label]
                    logits[canon_pos] = lp
            # Fill missing
            for label in canonical:
                cp = mapping[label]
                if cp not in logits:
                    logits[cp] = -100.0
    else:
        # Numbered: extract logprobs for digit tokens
        if top_logprobs_raw:
            token_map = {e["token"].strip(): e["logprob"] for e in top_logprobs_raw if "token" in e}
            for tok in scale_tokens:
                raw_lp = token_map.get(tok, -100.0)
                canon_pos = mapping[tok]
                logits[canon_pos] = raw_lp
        else:
            for tok in scale_tokens:
                logits[mapping[tok]] = -100.0

    # Compute probs (softmax over canonical positions)
    probs = {}
    if logits:
        max_lp = max(logits.values())
        exp_vals = {k: math.exp(v - max_lp) for k, v in logits.items()}
        total = sum(exp_vals.values())
        probs = {k: exp_vals[k] / total for k in logits}

    # Remap choice_from_text to canonical
    canonical_choice = None
    if choice_from_text:
        if condition == "verbal":
            canonical_choice = mapping.get(choice_from_text)
        elif choice_from_text in mapping:
            canonical_choice = mapping[choice_from_text]

    answer = max(probs, key=probs.get) if probs else None

    return {
        "metadata": {
            "prompt_id": prompt["prompt_id"],
            "item_id": prompt["item_id"],
            "section": prompt["section"],
            "institution": prompt["institution"],
            "condition": condition,
            "reverse_coded": prompt["reverse_coded"],
        },
        "response": {
            "model": model_id,
            "repetition": repetition,
            "logits": {k: round(v, 6) for k, v in logits.items()},
            "probs": {k: round(v, 10) for k, v in probs.items()},
            "answer": answer,
            "choice_from_text": canonical_choice,
            "justification": justification,
            "error": error,
        },
    }


async def run_experiment(prompts, model_ids, n_reps, api_key):
    total = len(prompts) * len(model_ids) * n_reps
    logger.info(f"Dispatching {total} API calls ({len(prompts)} prompts × {len(model_ids)} models × {n_reps} reps)")

    semaphore = asyncio.Semaphore(30)
    completed = 0
    start = time.monotonic()

    async def tracked(session, p, mid, rep):
        nonlocal completed
        r = await call_api(session, p, mid, rep, api_key, semaphore)
        completed += 1
        if completed % 50 == 0 or completed == total:
            elapsed = time.monotonic() - start
            rate = completed / elapsed * 60
            eta = (total - completed) / rate if rate > 0 else 0
            logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%) | {rate:.0f} req/min | ETA: {eta/60:.0f}m {eta%60:.0f}s")
        return r

    async with aiohttp.ClientSession() as session:
        tasks = []
        for mid in model_ids:
            for p in prompts:
                for rep in range(1, n_reps + 1):
                    tasks.append(tracked(session, p, mid, rep))
        results = await asyncio.gather(*tasks)

    return list(results)


def main():
    parser = argparse.ArgumentParser(description="Number anchor sensitivity experiment")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate prompts")
    gen.add_argument("--csv", default="data/institutional_trust_data.csv")
    gen.add_argument("--output", default="outputs/anchor_prompts.jsonl")

    run = sub.add_parser("run", help="Run experiment")
    run.add_argument("--prompts", required=True)
    run.add_argument("--models", default="openai/gpt-4o")
    run.add_argument("--n-reps", type=int, default=10)
    run.add_argument("--output-dir", default="outputs/results")
    run.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "generate":
        generate_prompts(args.csv, args.output)

    elif args.command == "run":
        with open(args.prompts) as f:
            prompts = [json.loads(l) for l in f]

        model_ids = [m.strip() for m in args.models.split(",")]

        if args.dry_run:
            for cond in ["original", "reversed", "verbal"]:
                n = sum(1 for p in prompts if p["condition"] == cond)
                calls = n * len(model_ids) * args.n_reps
                logger.info(f"  {cond}: {n} prompts × {len(model_ids)} models × {args.n_reps} reps = {calls} calls")
            logger.info(f"  Total: {len(prompts) * len(model_ids) * args.n_reps} calls")
            return

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.error("Set OPENROUTER_API_KEY")
            sys.exit(1)

        results = asyncio.run(run_experiment(prompts, model_ids, args.n_reps, api_key))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"anchor_results_{ts}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        errors = sum(1 for r in results if r["response"].get("error"))
        logger.info(f"Saved {len(results)} results to {out_path} ({errors} errors)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
