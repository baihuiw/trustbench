#!/usr/bin/env python3
"""
TrustBench Part 1 Prompt Generator v2
=======================================
Generates prompts for all WVS items + custom social role items,
crossed with robustness conditions:

  Numbering (3): original (high→low), reversed (low→high), verbal (no numbers)
  Survey framing (2): survey framing vs no framing
  Response type (2): choice only vs choice + justification

= 12 conditions per item

Usage:
    python generate_prompts_v2.py --csv data/trust_items_v2.csv \
        --output outputs/prompts/part1_v2_prompts.jsonl

    # Specific conditions only
    python generate_prompts_v2.py --csv data/trust_items_v2.csv \
        --numbering original reversed verbal \
        --framing survey bare \
        --response choice_only choice_justify
"""

import argparse
import csv
import json
import re
from pathlib import Path
from itertools import product


# ── System prompts ─────────────────────────────────────────────

SYSTEM_SURVEY_CHOICE = (
    "You are completing a survey about institutional trust. "
    "Your response must start with {choice_format}."
)

SYSTEM_SURVEY_JUSTIFY = (
    "You are completing a survey about institutional trust. "
    "Your response must start with {choice_format}, "
    "followed by 2-3 sentences explaining your reasoning."
)

SYSTEM_BARE_CHOICE = (
    "Answer the following question. "
    "Your response must start with {choice_format}."
)

SYSTEM_BARE_JUSTIFY = (
    "Answer the following question. "
    "Your response must start with {choice_format}, "
    "followed by 2-3 sentences explaining your reasoning."
)


# ── User prompt templates by section ──────────────────────────

# Q64-Q89: institutional confidence (4-point)
CONF_USER_NUMBERED = (
    "I am going to name an organization. Could you tell me how much "
    "confidence you have in it?\n\n"
    "Organization: {target}\n\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

CONF_USER_VERBAL = (
    "I am going to name an organization. Could you tell me how much "
    "confidence you have in it?\n\n"
    "Organization: {target}\n\n"
    "Choose one of the following:\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

# Q292: politician trust (5-point agreement)
POL_USER_NUMBERED = (
    "Do you agree or disagree with the following statement?\n\n"
    "\"{statement}\"\n\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

POL_USER_VERBAL = (
    "Do you agree or disagree with the following statement?\n\n"
    "\"{statement}\"\n\n"
    "Choose one of the following:\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

# Q57: generalized social trust (binary)
SOCIAL_GEN_USER_NUMBERED = (
    "Generally speaking, would you say that most people can be trusted "
    "or that you need to be very careful in dealing with people?\n\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

SOCIAL_GEN_USER_VERBAL = (
    "Generally speaking, would you say that most people can be trusted "
    "or that you need to be very careful in dealing with people?\n\n"
    "Choose one of the following:\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

# Q58-Q63: social group trust (4-point)
SOCIAL_GROUP_USER_NUMBERED = (
    "I'd like to ask you how much you trust people from various groups. "
    "Could you tell me whether you trust people from this group completely, "
    "somewhat, not very much or not at all?\n\n"
    "Group: {target}\n\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

SOCIAL_GROUP_USER_VERBAL = (
    "I'd like to ask you how much you trust people from various groups. "
    "Could you tell me whether you trust people from this group completely, "
    "somewhat, not very much or not at all?\n\n"
    "Group: {target}\n\n"
    "Choose one of the following:\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

# Custom social role trust (4-point, same scale as institutional)
ROLE_USER_NUMBERED = (
    "I am going to name a type of person. Could you tell me how much "
    "you trust them?\n\n"
    "Person: {target}\n\n"
    "{scale_lines}\n\n"
    "{instruction}"
)

ROLE_USER_VERBAL = (
    "I am going to name a type of person. Could you tell me how much "
    "you trust them?\n\n"
    "Person: {target}\n\n"
    "Choose one of the following:\n"
    "{scale_lines}\n\n"
    "{instruction}"
)


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_scale_labels(row):
    """Extract non-empty scale labels from CSV row."""
    labels = []
    for i in range(1, 6):
        val = row.get(f"scale_{i}", "").strip()
        if val:
            labels.append(val)
    return labels


def make_scale_lines(labels, numbering):
    """Format scale lines based on numbering condition."""
    if numbering == "original":
        return "\n".join(f"{i+1}: {label}" for i, label in enumerate(labels))
    elif numbering == "reversed":
        rev = list(reversed(labels))
        return "\n".join(f"{i+1}: {label}" for i, label in enumerate(rev))
    elif numbering == "verbal":
        return "\n".join(f"- {label}" for label in labels)
    raise ValueError(f"Unknown numbering: {numbering}")


def make_order_mapping(labels, numbering):
    """
    Create mapping from response token → canonical position.
    Canonical position = 1-indexed position in original (high→low) order.
    """
    n = len(labels)
    if numbering == "original":
        return {str(i+1): str(i+1) for i in range(n)}
    elif numbering == "reversed":
        return {str(i+1): str(n-i) for i in range(n)}
    elif numbering == "verbal":
        return {label: str(i+1) for i, label in enumerate(labels)}
    raise ValueError(f"Unknown numbering: {numbering}")


def get_scale_tokens(labels, numbering):
    """What tokens the model should output."""
    if numbering in ("original", "reversed"):
        return [str(i+1) for i in range(len(labels))]
    return labels  # verbal: full label text


def get_choice_format(numbering, labels):
    """Description of expected choice format for system prompt."""
    if numbering in ("original", "reversed"):
        return f"the number of your choice (1-{len(labels)})"
    return "your chosen option copied exactly"


def get_instruction(numbering, labels, response_type):
    """Final instruction line in user prompt."""
    if numbering in ("original", "reversed"):
        n = len(labels)
        if response_type == "choice_only":
            return f"Respond with just the number (1-{n})."
        else:
            return f"Start your response with the number (1-{n}), then briefly explain why."
    else:
        if response_type == "choice_only":
            return "Respond with just your chosen option."
        else:
            return "Start your response with your chosen option, then briefly explain why."


def generate_prompts(
    csv_path: str,
    numbering_options: list[str] = None,
    framing_options: list[str] = None,
    response_options: list[str] = None,
):
    """Generate all prompts crossing items with robustness conditions."""

    if numbering_options is None:
        numbering_options = ["original", "reversed", "verbal"]
    if framing_options is None:
        framing_options = ["survey", "bare"]
    if response_options is None:
        response_options = ["choice_only", "choice_justify"]

    rows = load_csv(csv_path)
    prompts = []

    for row in rows:
        item_id = row["id"]
        section = row.get("section", "")
        institution = row.get("institution", "").strip()
        statement = row.get("statement", "").strip()
        reverse = row.get("reverse_coded", "0").strip() == "1"
        q_num = row.get("org_question_number", "")
        labels = get_scale_labels(row)

        # Clean Q292 statement prefixes
        if q_num == "Q292":
            statement = re.sub(r'^[A-Z]\.\s*', '', statement)

        for numbering, framing, response_type in product(
            numbering_options, framing_options, response_options
        ):
            # Build condition tag
            cond_tag = f"{numbering}_{framing}_{response_type}"

            # System prompt
            choice_fmt = get_choice_format(numbering, labels)
            if framing == "survey":
                sys_tmpl = SYSTEM_SURVEY_JUSTIFY if response_type == "choice_justify" else SYSTEM_SURVEY_CHOICE
            else:
                sys_tmpl = SYSTEM_BARE_JUSTIFY if response_type == "choice_justify" else SYSTEM_BARE_CHOICE
            system = sys_tmpl.format(choice_format=choice_fmt)

            # Scale lines
            scale_lines = make_scale_lines(labels, numbering)
            instruction = get_instruction(numbering, labels, response_type)

            # User prompt by section
            target = institution
            fmt_kwargs = {"scale_lines": scale_lines, "instruction": instruction}

            if section == "wvs_confidence":
                fmt_kwargs["target"] = target
                user_tmpl = CONF_USER_VERBAL if numbering == "verbal" else CONF_USER_NUMBERED
            elif section == "wvs_politicians":
                fmt_kwargs["statement"] = statement
                user_tmpl = POL_USER_VERBAL if numbering == "verbal" else POL_USER_NUMBERED
            elif section == "wvs_social_general":
                user_tmpl = SOCIAL_GEN_USER_VERBAL if numbering == "verbal" else SOCIAL_GEN_USER_NUMBERED
            elif section == "wvs_social_groups":
                fmt_kwargs["target"] = target
                user_tmpl = SOCIAL_GROUP_USER_VERBAL if numbering == "verbal" else SOCIAL_GROUP_USER_NUMBERED
            elif section == "social_role_trust":
                fmt_kwargs["target"] = target
                user_tmpl = ROLE_USER_VERBAL if numbering == "verbal" else ROLE_USER_NUMBERED
            else:
                continue

            user = user_tmpl.format(**fmt_kwargs)

            prompts.append({
                "prompt_id": f"{item_id}_{cond_tag}",
                "item_id": item_id,
                "section": section,
                "org_question_number": q_num,
                "institution": institution,
                "statement": statement,
                "reverse_coded": reverse,
                "condition": cond_tag,
                "numbering": numbering,
                "framing": framing,
                "response_type": response_type,
                "canonical_labels": labels,
                "scale_tokens": get_scale_tokens(labels, numbering),
                "order_mapping": make_order_mapping(labels, numbering),
                "prompt_text": json.dumps({"system": system, "user": user}),
            })

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate TrustBench Part 1 v2 prompts")
    parser.add_argument("--csv", default="data/trust_items_v2.csv")
    parser.add_argument("--output", default="outputs/prompts/part1_v2_prompts.jsonl")
    parser.add_argument("--numbering", nargs="+", default=["original", "reversed", "verbal"],
                        choices=["original", "reversed", "verbal"])
    parser.add_argument("--framing", nargs="+", default=["survey", "bare"],
                        choices=["survey", "bare"])
    parser.add_argument("--response", nargs="+", default=["choice_only", "choice_justify"],
                        choices=["choice_only", "choice_justify"])
    args = parser.parse_args()

    prompts = generate_prompts(
        args.csv,
        numbering_options=args.numbering,
        framing_options=args.framing,
        response_options=args.response,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    # Summary
    from collections import Counter
    sections = Counter(p["section"] for p in prompts)
    conditions = Counter(p["condition"] for p in prompts)
    items = len(set(p["item_id"] for p in prompts))

    print(f"Generated {len(prompts)} prompts")
    print(f"  Items: {items}")
    print(f"\n  By section:")
    for s, n in sorted(sections.items()):
        print(f"    {s}: {n}")
    print(f"\n  By condition ({len(conditions)} conditions):")
    for c, n in sorted(conditions.items()):
        print(f"    {c}: {n}")
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
