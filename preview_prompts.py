#!/usr/bin/env python3
"""
Preview generated prompts in a human-readable format.

Usage:
    # Preview all prompts (first 10 by default)
    python preview_prompts.py

    # Preview Part 1 only
    python preview_prompts.py --part 1

    # Preview Part 2 stated trust only
    python preview_prompts.py --section stated_trust

    # Preview Part 2 revealed trust for Russia
    python preview_prompts.py --section revealed_trust --country Russia

    # Show more prompts
    python preview_prompts.py --n 50

    # Show all prompts
    python preview_prompts.py --n 0

    # Filter by institution
    python preview_prompts.py --institution military

    # Export to a readable text file
    python preview_prompts.py --part 1 --output prompts_part1_preview.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.prompt_generators import generate_part1_prompts
from src.prompt_generators.part2 import generate_part2_prompts


COUNTRIES = [
    "Norway", "Canada", "Japan",
    "United States", "India", "Hungary",
    "Ukraine", "Turkey", "Nigeria",
    "Russia", "China", "Saudi Arabia",
]

INSTITUTIONS = [
    "government", "military", "media",
    "judiciary", "elections", "central_bank", "police",
    "major_public_institutions",
]


def format_prompt(p, idx: int) -> str:
    """Format a single PromptItem for display."""
    prompt_data = json.loads(p.prompt_text)
    lines = []
    lines.append(f"{'━' * 70}")
    lines.append(f"  #{idx + 1}  |  ID: {p.prompt_id}")
    lines.append(f"  Part: {p.part}  |  Section: {p.section}")
    if p.country:
        lines.append(f"  Country: {p.country}  |  Institution: {p.institution}")
    elif p.institution:
        lines.append(f"  Institution: {p.institution}")
    if p.statement:
        lines.append(f"  Statement: {p.statement}")
    lines.append(f"  Scale: {p.scale_labels}")
    if p.reverse_coded:
        lines.append(f"  ⚠ Reverse-coded")
    lines.append(f"{'─' * 70}")
    lines.append(f"  [SYSTEM]")
    lines.append(f"  {prompt_data['system']}")
    lines.append(f"")
    lines.append(f"  [USER]")
    for line in prompt_data["user"].split("\n"):
        lines.append(f"  {line}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Preview TrustBench prompts")
    parser.add_argument("--part", type=int, choices=[1, 2], help="Filter by part")
    parser.add_argument("--section", type=str,
                        choices=["wvs_confidence", "wvs_politicians",
                                 "stated_trust", "revealed_trust"],
                        help="Filter by section")
    parser.add_argument("--country", type=str, help="Filter by country")
    parser.add_argument("--institution", type=str, help="Filter by institution")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of prompts to show (0 = all)")
    parser.add_argument("--output", type=str,
                        help="Write to file instead of stdout")
    args = parser.parse_args()

    # Generate prompts
    csv_path = Path(__file__).parent / "data" / "institutional_trust_data.csv"
    all_prompts = []

    if args.part is None or args.part == 1:
        all_prompts.extend(generate_part1_prompts(csv_path))
    if args.part is None or args.part == 2:
        all_prompts.extend(generate_part2_prompts(COUNTRIES, INSTITUTIONS))

    # Apply filters
    if args.section:
        all_prompts = [p for p in all_prompts if p.section == args.section]
    if args.country:
        all_prompts = [p for p in all_prompts
                       if args.country.lower() in p.country.lower()]
    if args.institution:
        all_prompts = [p for p in all_prompts
                       if args.institution.lower() in p.institution.lower()]

    # Limit
    total = len(all_prompts)
    if args.n > 0:
        display = all_prompts[:args.n]
    else:
        display = all_prompts

    # Format
    header = (
        f"\n{'═' * 70}\n"
        f"  TRUSTBENCH PROMPT PREVIEW\n"
        f"  Showing {len(display)} of {total} prompts"
    )
    if args.part:
        header += f"  |  Part {args.part}"
    if args.section:
        header += f"  |  Section: {args.section}"
    if args.country:
        header += f"  |  Country: {args.country}"
    if args.institution:
        header += f"  |  Institution: {args.institution}"
    header += f"\n{'═' * 70}\n"

    output_lines = [header]
    for i, p in enumerate(display):
        output_lines.append(format_prompt(p, i))

    if len(display) < total:
        output_lines.append(
            f"  ... and {total - len(display)} more. "
            f"Use --n 0 to show all.\n"
        )

    text = "\n".join(output_lines)

    if args.output:
        Path(args.output).write_text(text)
        print(f"Written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
