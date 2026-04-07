#!/usr/bin/env python3
"""
Remap shuffled results back to canonical scale order.

Usage:
    python remap_results.py --input outputs/results/part1_results_XXXX.jsonl

    # Custom output path
    python remap_results.py --input results.jsonl --output results_canonical.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.prompt_generators import remap_to_canonical


def main():
    parser = argparse.ArgumentParser(description="Remap shuffled results to canonical order")
    parser.add_argument("--input", required=True, help="Path to results JSONL")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_canonical.jsonl)")
    args = parser.parse_args()

    output_path = args.output or args.input.replace(".jsonl", "_canonical.jsonl")

    results = []
    with open(args.input) as f:
        for line in f:
            results.append(json.loads(line))

    remapped = [remap_to_canonical(r) for r in results]

    with open(output_path, "w") as f:
        for r in remapped:
            f.write(json.dumps(r) + "\n")

    print(f"Remapped {len(remapped)} results → {output_path}")

    # Quick summary
    n_shuffled = sum(1 for r in results if r["metadata"].get("variation", 0) > 0)
    print(f"  Original order (v0): {len(results) - n_shuffled}")
    print(f"  Shuffled variations: {n_shuffled}")


if __name__ == "__main__":
    main()
