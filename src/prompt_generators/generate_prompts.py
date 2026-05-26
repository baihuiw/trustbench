#!/usr/bin/env python3
"""
Step 1: Generate Prompts
========================
Generates all prompts from the config and saves them as JSONL files.
Does NOT call any APIs — just creates the prompt files for review.

Usage:
    # Generate all prompts (Part 1 + Part 2)
    python generate_prompts.py

    # Part 1 only
    python generate_prompts.py run.parts=[1]

    # Part 2 only
    python generate_prompts.py run.parts=[2]

    # Override output directory
    python generate_prompts.py paths.prompt_dir=./my_prompts

Output:
    outputs/prompts/part1_prompts.jsonl   (37 items)
    outputs/prompts/part2_prompts.jsonl   (1,728 items)
    outputs/prompts/manifest.json         (summary + run plan)

Then review with:
    python preview_prompts.py --part 1 --n 0
    python preview_prompts.py --section revealed_trust --country Russia

When satisfied, run experiments with:
    python run_experiments.py --prompts outputs/prompts/part1_prompts.jsonl
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.prompt_generators import generate_part1_prompts, PromptItem
from src.prompt_generators.part2 import generate_part2_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trustbench.generate")


def flatten_countries(cfg: DictConfig) -> list[str]:
    countries = []
    for regime_type, country_list in cfg.countries.items():
        countries.extend(list(country_list))
    return countries


def flatten_institutions(cfg: DictConfig) -> list[str]:
    insts = list(cfg.institutions.core) + list(cfg.institutions.extended)
    insts.append("major_public_institutions")
    return insts


def save_prompts(prompts: list[PromptItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p.to_dict()) + "\n")
    logger.info("Saved %d prompts → %s", len(prompts), path)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info("TrustBench — Prompt Generation")

    countries = flatten_countries(cfg)
    institutions = flatten_institutions(cfg)
    models = OmegaConf.to_container(cfg.models.model_list, resolve=True)
    parts = list(cfg.run.parts)
    n_reps_p1 = cfg.run.get("n_repetitions_part1", 100)
    n_reps_p2 = cfg.run.get("n_repetitions_part2", 1)
    n_shuffles = cfg.run.get("n_shuffles", 0)

    prompt_dir = Path(cfg.paths.prompt_dir)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "countries": countries,
        "institutions": institutions,
        "models": [m["label"] for m in models],
        "model_ids": [m["id"] for m in models],
        "parts": parts,
        "n_repetitions_part1": n_reps_p1,
        "n_repetitions_part2": n_reps_p2,
        "n_shuffles": n_shuffles,
        "logprobs": cfg.run.get("logprobs", True),
        "top_logprobs": cfg.run.get("top_logprobs", 10),
        "temperature": cfg.run.get("temperature", 1.0),
        "max_tokens": cfg.run.get("max_tokens", 1),
        "files": {},
    }

    # ── Part 1 ─────────────────────────────────────────────────
    if 1 in parts:
        csv_path = Path(cfg.paths.data_dir) / "institutional_trust_data.csv"
        p1 = generate_part1_prompts(csv_path, n_shuffles=n_shuffles)
        p1_path = prompt_dir / "part1_prompts.jsonl"
        save_prompts(p1, p1_path)

        p1_calls = len(p1) * len(models) * n_reps_p1
        manifest["files"]["part1"] = {
            "path": str(p1_path),
            "n_prompts": len(p1),
            "n_reps": n_reps_p1,
            "n_models": len(models),
            "total_api_calls": p1_calls,
            "sections": {
                "wvs_confidence": sum(1 for p in p1 if p.section == "wvs_confidence"),
                "wvs_politicians": sum(1 for p in p1 if p.section == "wvs_politicians"),
            },
        }
        logger.info("Part 1: %d prompts → %d API calls (%d models × %d reps)",
                     len(p1), p1_calls, len(models), n_reps_p1)

    # ── Part 2 ─────────────────────────────────────────────────
    if 2 in parts:
        p2 = generate_part2_prompts(countries, institutions)
        p2_path = prompt_dir / "part2_prompts.jsonl"
        save_prompts(p2, p2_path)

        p2_calls = len(p2) * len(models) * n_reps_p2
        n_stated = sum(1 for p in p2 if p.section == "stated_trust")
        n_revealed = sum(1 for p in p2 if p.section == "revealed_trust")
        manifest["files"]["part2"] = {
            "path": str(p2_path),
            "n_prompts": len(p2),
            "n_reps": n_reps_p2,
            "n_models": len(models),
            "total_api_calls": p2_calls,
            "sections": {
                "stated_trust": n_stated,
                "revealed_trust": n_revealed,
            },
        }
        logger.info("Part 2: %d prompts (%d stated + %d revealed) → %d API calls",
                     len(p2), n_stated, n_revealed, p2_calls)

    # ── Save manifest ──────────────────────────────────────────
    total_calls = sum(
        f["total_api_calls"] for f in manifest["files"].values()
    )
    manifest["total_api_calls"] = total_calls

    manifest_path = prompt_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest → %s", manifest_path)

    # ── Summary ────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("PROMPT GENERATION COMPLETE")
    logger.info("  Output directory: %s", prompt_dir)
    for part_name, info in manifest["files"].items():
        logger.info("  %s: %d prompts → %s",
                     part_name, info["n_prompts"], info["path"])
    logger.info("  Total API calls when run: %d", total_calls)
    logger.info("═" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review prompts:  python preview_prompts.py --part 1 --n 0")
    logger.info("  2. Run Part 1:      python run_experiments.py --prompts %s",
                manifest["files"].get("part1", {}).get("path", "outputs/prompts/part1_prompts.jsonl"))
    logger.info("  3. Run Part 2:      python run_experiments.py --prompts %s",
                manifest["files"].get("part2", {}).get("path", "outputs/prompts/part2_prompts.jsonl"))


if __name__ == "__main__":
    main()
