#!/usr/bin/env python3
"""
TrustBench V2 Runner
=====================
Async OpenRouter runner with checkpoint/resume.

Features:
  - Saves results incrementally to JSONL (every 50 completions)
  - On restart, loads existing results and skips completed (prompt_id, model, rep) combos
  - No logprobs — discrete choice + justification only
  - Progress bar with ETA

Usage:
    # Generate prompts first
    python generate_prompts_v2.py --csv data/trust_items_v2.csv

    # Run all 3 models, 30 reps
    python run_v2.py --prompts outputs/prompts/part1_v2_prompts.jsonl \
        --models openai/gpt-4o,anthropic/claude-sonnet-4,google/gemini-2.5-pro-preview-06-05 \
        --n-reps 30

    # Run one model at a time
    python run_v2.py --prompts outputs/prompts/part1_v2_prompts.jsonl \
        --models openai/gpt-4o --n-reps 30

    # Resume interrupted run (same command — auto-detects checkpoint)
    python run_v2.py --prompts outputs/prompts/part1_v2_prompts.jsonl \
        --models openai/gpt-4o --n-reps 30

    # Dry run
    python run_v2.py --prompts outputs/prompts/part1_v2_prompts.jsonl \
        --models openai/gpt-4o --n-reps 30 --dry-run

    # Custom output file
    python run_v2.py --prompts outputs/prompts/part1_v2_prompts.jsonl \
        --models openai/gpt-4o --n-reps 30 --output outputs/results/gpt4o_results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_v2")


# ── Checkpoint ────────────────────────────────────────────────

def load_checkpoint(output_path: str) -> tuple[list[dict], set[str]]:
    """Load existing results and build set of completed keys."""
    results = []
    completed = set()
    if Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    results.append(r)
                    key = _result_key(r)
                    completed.add(key)
                except json.JSONDecodeError:
                    continue
    return results, completed


def _result_key(r: dict) -> str:
    """Unique key for a completed result."""
    meta = r.get("metadata", {})
    resp = r.get("response", {})
    return f"{meta.get('prompt_id')}|{resp.get('model')}|{resp.get('repetition')}"


def _task_key(prompt_id: str, model_id: str, rep: int) -> str:
    return f"{prompt_id}|{model_id}|{rep}"


# ── API caller ────────────────────────────────────────────────

async def call_api(
    session: aiohttp.ClientSession,
    prompt: dict,
    model_id: str,
    repetition: int,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 150,
    temperature: float = 1.0,
) -> dict:
    """Call OpenRouter and return a result dict."""
    prompt_data = json.loads(prompt["prompt_text"])
    messages = [
        {"role": "system", "content": prompt_data["system"]},
        {"role": "user", "content": prompt_data["user"]},
    ]
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://trustbench.research",
        "X-Title": "TrustBench-V2",
    }

    for attempt in range(5):
        try:
            async with semaphore:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    body = await resp.json()

                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        if attempt < 4:
                            await asyncio.sleep(attempt + 1)
                            continue
                        return _build_result(
                            prompt, model_id, repetition, "",
                            body.get("error", {}).get("message", f"HTTP {resp.status}")
                        )

                    raw = (body.get("choices", [{}])[0]
                           .get("message", {})
                           .get("content") or "").strip()
                    return _build_result(prompt, model_id, repetition, raw, None)

        except asyncio.TimeoutError:
            if attempt < 4:
                await asyncio.sleep(attempt + 1)
                continue
            return _build_result(prompt, model_id, repetition, "", "Timeout")
        except Exception as e:
            if attempt < 4:
                await asyncio.sleep(attempt + 1)
                continue
            return _build_result(prompt, model_id, repetition, "", str(e))

    return _build_result(prompt, model_id, repetition, "", "Max retries exceeded")


def _build_result(prompt: dict, model_id: str, repetition: int, raw: str, error: str | None) -> dict:
    """Parse raw response and build result dict."""
    condition = prompt.get("condition", "")
    canonical_labels = prompt.get("canonical_labels", [])
    mapping = prompt.get("order_mapping", {})

    choice_from_text = None
    justification = None

    if raw:
        text = raw.strip()

        if prompt.get("numbering") == "verbal":
            # Match against canonical labels
            for label in canonical_labels:
                if text.lower().startswith(label.lower()):
                    choice_from_text = label
                    justification = text[len(label):].lstrip(".,: ").strip()
                    break
            if not choice_from_text:
                # Try partial match (first word)
                first_word = text.split()[0].rstrip(".,:")  if text.split() else ""
                for label in canonical_labels:
                    if label.lower().startswith(first_word.lower()) and len(first_word) > 2:
                        choice_from_text = label
                        justification = text[len(first_word):].lstrip(".,: ").strip()
                        break
                if not choice_from_text:
                    justification = text
        else:
            # Numbered: first character should be a digit
            first = text[0] if text else ""
            if first.isdigit():
                choice_from_text = first
            justification = text[1:].lstrip(".:,) ").strip() if len(text) > 1 else None

    # Map to canonical position
    canonical_choice = None
    if choice_from_text and choice_from_text in mapping:
        canonical_choice = mapping[choice_from_text]

    return {
        "metadata": {
            "prompt_id": prompt["prompt_id"],
            "item_id": prompt["item_id"],
            "section": prompt["section"],
            "institution": prompt.get("institution", ""),
            "condition": condition,
            "numbering": prompt.get("numbering", ""),
            "framing": prompt.get("framing", ""),
            "response_type": prompt.get("response_type", ""),
            "reverse_coded": prompt.get("reverse_coded", False),
            "order_mapping": mapping,
        },
        "response": {
            "model": model_id,
            "repetition": repetition,
            "raw_text": raw[:500] if raw else "",
            "choice_from_text": choice_from_text,
            "canonical_choice": canonical_choice,
            "justification": justification[:300] if justification else None,
            "error": error,
        },
    }


# ── Main runner ───────────────────────────────────────────────

async def run_experiment(
    prompts: list[dict],
    model_ids: list[str],
    n_reps: int,
    api_key: str,
    output_path: str,
    concurrency: int = 30,
    temperature: float = 1.0,
    max_tokens: int = 150,
):
    """Run all prompts with checkpoint/resume."""

    # Load checkpoint
    existing_results, completed = load_checkpoint(output_path)
    if completed:
        logger.info(f"Checkpoint loaded: {len(completed)} results already completed")

    # Build task list, skipping completed
    tasks_todo = []
    for model_id in model_ids:
        for prompt in prompts:
            for rep in range(1, n_reps + 1):
                key = _task_key(prompt["prompt_id"], model_id, rep)
                if key not in completed:
                    tasks_todo.append((prompt, model_id, rep))

    total_planned = len(prompts) * len(model_ids) * n_reps
    total_skipped = total_planned - len(tasks_todo)
    total_todo = len(tasks_todo)

    logger.info(f"Total planned: {total_planned}")
    logger.info(f"Already done:  {total_skipped}")
    logger.info(f"Remaining:     {total_todo}")

    if total_todo == 0:
        logger.info("All tasks already completed!")
        return

    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    error_count = 0
    start_time = time.monotonic()

    # Open file in append mode for incremental saves
    buffer = []
    FLUSH_EVERY = 50

    async def tracked_call(session, prompt, model_id, rep):
        nonlocal completed_count, error_count
        result = await call_api(
            session, prompt, model_id, rep, api_key, semaphore,
            max_tokens=max_tokens, temperature=temperature,
        )
        completed_count += 1
        if result["response"].get("error"):
            error_count += 1

        buffer.append(result)

        # Flush buffer periodically
        if len(buffer) >= FLUSH_EVERY:
            flush_buffer(buffer, output_path)

        # Progress
        if completed_count % 100 == 0 or completed_count == total_todo:
            elapsed = time.monotonic() - start_time
            rate = completed_count / elapsed * 60
            eta_min = (total_todo - completed_count) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {completed_count}/{total_todo} "
                f"({100*completed_count/total_todo:.1f}%) | "
                f"{rate:.0f} req/min | "
                f"ETA: {eta_min:.0f}m | "
                f"Errors: {error_count}"
            )

        return result

    async with aiohttp.ClientSession() as session:
        # Process in batches to avoid overwhelming memory
        BATCH_SIZE = 500
        for batch_start in range(0, len(tasks_todo), BATCH_SIZE):
            batch = tasks_todo[batch_start:batch_start + BATCH_SIZE]
            coros = [
                tracked_call(session, prompt, model_id, rep)
                for prompt, model_id, rep in batch
            ]
            await asyncio.gather(*coros)

            # Flush any remaining buffer after each batch
            if buffer:
                flush_buffer(buffer, output_path)

    # Final flush
    if buffer:
        flush_buffer(buffer, output_path)

    elapsed = time.monotonic() - start_time
    logger.info(f"Done! {completed_count} calls in {elapsed/60:.1f}m ({error_count} errors)")


def flush_buffer(buffer: list[dict], output_path: str):
    """Append buffered results to file and clear buffer."""
    with open(output_path, "a") as f:
        for r in buffer:
            f.write(json.dumps(r) + "\n")
    buffer.clear()


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrustBench V2 Runner with checkpoint/resume")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL")
    parser.add_argument("--models", required=True,
                        help="Comma-separated OpenRouter model IDs")
    parser.add_argument("--n-reps", type=int, default=30, help="Repetitions per prompt (default: 30)")
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: auto-generated in outputs/results/)")
    parser.add_argument("--concurrency", type=int, default=30,
                        help="Max concurrent API requests (default: 30)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max output tokens (default: 150)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load prompts
    with open(args.prompts) as f:
        prompts = [json.loads(l) for l in f]

    model_ids = [m.strip() for m in args.models.split(",")]

    # Auto-generate output path
    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = "_".join(m.split("/")[-1] for m in model_ids)
        out_dir = Path("outputs/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"v2_{model_tag}_{ts}.jsonl")

    total = len(prompts) * len(model_ids) * args.n_reps

    logger.info("=" * 60)
    logger.info("TrustBench V2 Runner")
    logger.info(f"  Prompts:     {len(prompts)}")
    logger.info(f"  Models:      {model_ids}")
    logger.info(f"  Reps:        {args.n_reps}")
    logger.info(f"  Total calls: {total:,}")
    logger.info(f"  Concurrency: {args.concurrency}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max tokens:  {args.max_tokens}")
    logger.info(f"  Output:      {output_path}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN — no API calls")
        # Check for existing checkpoint
        _, completed = load_checkpoint(output_path)
        remaining = total - len(completed)
        logger.info(f"  Already done: {len(completed)}")
        logger.info(f"  Remaining:    {remaining}")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    asyncio.run(run_experiment(
        prompts=prompts,
        model_ids=model_ids,
        n_reps=args.n_reps,
        api_key=api_key,
        output_path=output_path,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))


if __name__ == "__main__":
    main()
