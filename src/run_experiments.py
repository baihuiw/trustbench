#!/usr/bin/env python3
"""
Step 2: Run Experiments
=======================
Reads a previously generated prompt file (JSONL) and sends all prompts
to models via OpenRouter. Can also read the manifest to auto-detect
settings, or override everything from the command line.

Usage:
    # Run from a specific prompt file (uses manifest for settings)
    python run_experiments.py --prompts outputs/prompts/part1_prompts.jsonl

    # Run Part 2
    python run_experiments.py --prompts outputs/prompts/part2_prompts.jsonl

    # Run with manifest (auto-picks reps, models, etc.)
    python run_experiments.py --manifest outputs/prompts/manifest.json --part 1

    # Override repetitions
    python run_experiments.py --prompts outputs/prompts/part1_prompts.jsonl --n-reps 50

    # Override models (comma-separated OpenRouter IDs)
    python run_experiments.py --prompts outputs/prompts/part1_prompts.jsonl \\
        --models openai/gpt-4o,anthropic/claude-sonnet-4

    # Dry run (show what would happen)
    python run_experiments.py --prompts outputs/prompts/part1_prompts.jsonl --dry-run

    # Run with Hydra overrides for API settings
    python run_experiments.py --prompts outputs/prompts/part1_prompts.jsonl \\
        -- api.requests_per_minute=60 run.batch_size=20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).parent))

# Hydra is optional here — we use it only for API/run config
# but the prompts come from the file
import yaml

from src.prompt_generators import PromptItem
from src.runners import OpenRouterRunner, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trustbench.run")


def load_prompts(path: str | Path) -> list[PromptItem]:
    """Load prompts from JSONL file back into PromptItem objects."""
    prompts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            prompts.append(PromptItem(**d))
    return prompts


def load_manifest(path: str | Path) -> dict:
    """Load the generation manifest."""
    with open(path) as f:
        return json.load(f)


def load_config(config_path: str | Path = None) -> dict:
    """Load the Hydra config as a plain dict."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve the API key from environment
    import os
    api_key = cfg.get("api", {}).get("key", "")
    if "${oc.env:OPENROUTER_API_KEY}" in str(api_key):
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        cfg["api"]["key"] = api_key

    return cfg


def build_models_list(
    model_override: str | None,
    manifest: dict | None,
    cfg: dict,
) -> list[dict]:
    """Determine which models to use (CLI override > manifest > config)."""
    if model_override:
        models = []
        for mid in model_override.split(","):
            mid = mid.strip()
            label = mid.split("/")[-1] if "/" in mid else mid
            models.append({"id": mid, "label": label})
        return models

    if manifest and "model_ids" in manifest:
        return [
            {"id": mid, "label": label}
            for mid, label in zip(manifest["model_ids"], manifest["models"])
        ]

    return cfg.get("models", {}).get("model_list", [])


def detect_n_reps(
    args_n_reps: int | None,
    prompt_path: str,
    manifest: dict | None,
    cfg: dict,
) -> int:
    """Determine repetition count (CLI override > manifest > config default)."""
    if args_n_reps is not None:
        return args_n_reps

    if manifest:
        # Try to match the prompt file to a part in the manifest
        prompt_name = Path(prompt_path).name
        if "part1" in prompt_name and "part1" in manifest.get("files", {}):
            return manifest["files"]["part1"]["n_reps"]
        if "part2" in prompt_name and "part2" in manifest.get("files", {}):
            return manifest["files"]["part2"]["n_reps"]
        # Fallback: check manifest-level
        if "part1" in prompt_name:
            return manifest.get("n_repetitions_part1", 100)
        if "part2" in prompt_name:
            return manifest.get("n_repetitions_part2", 1)

    # Config fallback
    prompt_name = Path(prompt_path).name
    if "part1" in prompt_name:
        return cfg.get("run", {}).get("n_repetitions_part1", 100)
    elif "part2" in prompt_name:
        return cfg.get("run", {}).get("n_repetitions_part2", 1)
    return 5


def main():
    parser = argparse.ArgumentParser(
        description="Run TrustBench experiments from pre-generated prompt files"
    )
    parser.add_argument(
        "--prompts", type=str, required=False,
        help="Path to a prompt JSONL file (e.g., outputs/prompts/part1_prompts.jsonl)"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to manifest.json (auto-detects prompts, models, reps)"
    )
    parser.add_argument(
        "--part", type=int, choices=[1, 2], default=None,
        help="When using --manifest, which part to run"
    )
    parser.add_argument(
        "--n-reps", type=int, default=None,
        help="Override number of repetitions per prompt"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated OpenRouter model IDs (overrides config)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: outputs/results)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without making API calls"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: configs/config.yaml)"
    )

    args = parser.parse_args()

    # ── Resolve prompt file ────────────────────────────────────
    manifest = None
    if args.manifest:
        manifest = load_manifest(args.manifest)

    prompt_path = args.prompts
    if not prompt_path and manifest and args.part:
        part_key = f"part{args.part}"
        if part_key in manifest.get("files", {}):
            prompt_path = manifest["files"][part_key]["path"]
        else:
            logger.error("Part %d not found in manifest", args.part)
            sys.exit(1)
    elif not prompt_path and manifest:
        # If no --part given, try to find the manifest dir
        manifest_dir = Path(args.manifest).parent
        available = list(manifest_dir.glob("*_prompts.jsonl"))
        if len(available) == 1:
            prompt_path = str(available[0])
        else:
            logger.error(
                "Multiple prompt files found. Use --prompts or --part to specify:\n  %s",
                "\n  ".join(str(p) for p in available)
            )
            sys.exit(1)

    if not prompt_path:
        logger.error("Must provide --prompts <file.jsonl> or --manifest <manifest.json> --part <1|2>")
        sys.exit(1)

    # Also auto-load manifest if it's in the same directory
    if not manifest:
        manifest_candidate = Path(prompt_path).parent / "manifest.json"
        if manifest_candidate.exists():
            manifest = load_manifest(manifest_candidate)
            logger.info("Auto-loaded manifest from %s", manifest_candidate)

    # ── Load config + prompts ──────────────────────────────────
    cfg = load_config(args.config)
    prompts = load_prompts(prompt_path)
    models = build_models_list(args.models, manifest, cfg)
    n_reps = detect_n_reps(args.n_reps, prompt_path, manifest, cfg)

    # Apply manifest/config settings to the runner config
    if manifest:
        cfg.setdefault("run", {})
        cfg["run"]["logprobs"] = manifest.get("logprobs", True)
        cfg["run"]["top_logprobs"] = manifest.get("top_logprobs", 10)
        cfg["run"]["temperature"] = manifest.get("temperature", 1.0)
        cfg["run"]["max_tokens"] = manifest.get("max_tokens", 1)

    total_calls = len(prompts) * len(models) * n_reps

    # ── Summary ────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("EXPERIMENT PLAN")
    logger.info("  Prompt file: %s", prompt_path)
    logger.info("  Prompts: %d", len(prompts))
    logger.info("  Models (%d): %s", len(models), [m["label"] for m in models])
    logger.info("  Repetitions: %d", n_reps)
    logger.info("  Total API calls: %d", total_calls)
    logger.info("  Logprobs: %s (top_k=%s)",
                cfg.get("run", {}).get("logprobs", True),
                cfg.get("run", {}).get("top_logprobs", 10))
    logger.info("  Max tokens: %s", cfg.get("run", {}).get("max_tokens", 1))
    logger.info("  Temperature: %s", cfg.get("run", {}).get("temperature", 1.0))
    logger.info("═" * 60)

    if args.dry_run:
        logger.info("DRY RUN — no API calls made")

        # Show a breakdown by section
        from collections import Counter
        sections = Counter(p.section for p in prompts)
        for section, count in sections.items():
            logger.info("  %s: %d prompts × %d models × %d reps = %d calls",
                         section, count, len(models), n_reps,
                         count * len(models) * n_reps)

        # Show a few example prompts
        logger.info("")
        logger.info("Sample prompts:")
        for p in prompts[:3]:
            pd = json.loads(p.prompt_text)
            logger.info("  [%s] %s", p.prompt_id, pd["user"][:100] + "...")
        return

    # ── Validate API key ───────────────────────────────────────
    if not cfg.get("api", {}).get("key"):
        logger.error("OPENROUTER_API_KEY not set. Export it first:")
        logger.error("  export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    # ── Run ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or cfg.get("paths", {}).get(
        "results_dir", "outputs/results"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = OpenRouterRunner(cfg)
    results = asyncio.run(runner.run_all(prompts, models, n_reps))

    # Determine a label from the prompt filename
    prompt_label = Path(prompt_path).stem.replace("_prompts", "")
    results_file = output_dir / f"{prompt_label}_results_{timestamp}.jsonl"
    save_results(results, results_file)

    # ── Final report ───────────────────────────────────────────
    errors = sum(1 for r in results if r["response"].get("error"))
    has_logprobs = sum(1 for r in results if r["response"].get("logits"))
    has_answer = sum(1 for r in results if r["response"].get("answer"))
    logger.info("═" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("  Total calls: %d", len(results))
    logger.info("  Errors: %d (%.1f%%)", errors,
                100 * errors / len(results) if results else 0)
    logger.info("  With logprobs: %d (%.1f%%)", has_logprobs,
                100 * has_logprobs / len(results) if results else 0)
    logger.info("  With answer: %d (%.1f%%)", has_answer,
                100 * has_answer / len(results) if results else 0)
    logger.info("  Results: %s", results_file)
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
