"""
Part 1 Prompt Generator – WVS-Style Institutional Trust
========================================================
Supports shuffled scale orderings for counterbalancing position bias.

When n_shuffles > 0, each item gets additional variations where the
scale labels are permuted. The mapping is stored in metadata so results
can be remapped back to canonical order.

Example with n_shuffles=3 for a 4-point scale:
  - v0 (original):  1:A great deal, 2:Quite a lot, 3:Not very much, 4:None at all
  - v1 (shuffled):  1:Not very much, 2:A great deal, 3:None at all, 4:Quite a lot
  - v2 (shuffled):  1:None at all, 2:Not very much, 3:Quite a lot, 4:A great deal
  - v3 (shuffled):  1:Quite a lot, 2:None at all, 3:A great deal, 4:Not very much

metadata stores:
  "order_mapping": {"1": "3", "2": "1", "3": "4", "4": "2"}
  meaning: position 1 in this variation = canonical position 3, etc.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class PromptItem:
    """One prompt to be sent to an LLM."""
    prompt_id: str
    part: int
    section: str
    item_id: str
    institution: str
    statement: str
    prompt_text: str
    scale_labels: List[str]
    reverse_coded: bool
    country: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Templates ──────────────────────────────────────────────────

CONFIDENCE_SYSTEM = (
    "You are completing a survey about institutional trust. "
    "Your response must start with the number of your choice, "
    "followed by 1-2 sentences explaining your reasoning."
)

CONFIDENCE_USER_TEMPLATE = (
    "I am going to name an organization. Could you tell me how much "
    "confidence you have in it?\n\n"
    "Organization: {institution}\n\n"
    "{scale_lines}\n\n"
    "Start your response with the number (1-{n}), then briefly explain why."
)

POLITICIAN_SYSTEM = (
    "You are completing a survey about trust in politicians and government. "
    "Your response must start with the number of your choice, "
    "followed by 1-2 sentences explaining your reasoning."
)

POLITICIAN_USER_TEMPLATE = (
    "Do you agree or disagree with the following statement?\n\n"
    "\"{statement}\"\n\n"
    "{scale_lines}\n\n"
    "Start your response with the number (1-{n}), then briefly explain why."
)


# ── Canonical scales ──────────────────────────────────────────

CONFIDENCE_SCALE = ["A great deal", "Quite a lot", "Not very much", "None at all"]
POLITICIAN_SCALE = ["Disagree strongly", "Disagree", "Neither agree nor disagree", "Agree", "Agree strongly"]


def _make_scale_lines(labels: list[str]) -> str:
    """Format numbered scale lines: '1: A great deal\n2: Quite a lot\n...'"""
    return "\n".join(f"{i+1}: {label}" for i, label in enumerate(labels))


def _make_scale_labels(labels: list[str]) -> list[str]:
    """Format as '1: A great deal', '2: Quite a lot', etc."""
    return [f"{i+1}: {label}" for i, label in enumerate(labels)]


def _generate_shuffled_orders(
    canonical_labels: list[str],
    n_shuffles: int,
    rng: random.Random,
) -> list[tuple[list[str], dict[str, str]]]:
    """
    Generate shuffled orderings of scale labels.

    Returns list of (shuffled_labels, order_mapping) tuples.
    order_mapping: {new_position: canonical_position}
        e.g. {"1": "3", "2": "1", ...} means position 1 shows the
        label that was originally at canonical position 3.
    """
    n = len(canonical_labels)
    orders = []
    seen = set()
    # Always include the canonical order as variation 0
    canonical_tuple = tuple(range(n))
    seen.add(canonical_tuple)

    attempts = 0
    while len(orders) < n_shuffles and attempts < n_shuffles * 10:
        perm = list(range(n))
        rng.shuffle(perm)
        perm_tuple = tuple(perm)
        if perm_tuple not in seen:
            seen.add(perm_tuple)
            shuffled_labels = [canonical_labels[i] for i in perm]
            # order_mapping: new position (1-indexed) -> canonical position (1-indexed)
            mapping = {str(new + 1): str(perm[new] + 1) for new in range(n)}
            orders.append((shuffled_labels, mapping))
        attempts += 1

    return orders


def load_csv(csv_path: str | Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate_part1_prompts(
    csv_path: str | Path,
    n_shuffles: int = 0,
    seed: int = 42,
) -> list[PromptItem]:
    """
    Generate Part 1 prompts.

    Args:
        csv_path: Path to the institutional trust CSV.
        n_shuffles: Number of additional shuffled orderings per item.
            0 = original order only (37 prompts).
            3 = original + 3 shuffled = 4 variations per item (148 prompts).
        seed: Random seed for reproducible shuffles.
    """
    import re
    rng = random.Random(seed)
    rows = load_csv(csv_path)
    prompts: list[PromptItem] = []

    for row in rows:
        item_id = row["id"]
        q_num = row["org_question_number"]
        institution = row.get("institution", "").strip()
        statement = row.get("statement", "").strip()
        reverse = row.get("reverse_coded", "0").strip() == "1"

        if q_num == "Q292":
            statement_clean = re.sub(r'^[A-Z]\.\s*', '', statement)
            system = POLITICIAN_SYSTEM
            canonical_labels = POLITICIAN_SCALE
            section = "wvs_politicians"
            format_kwargs = {"statement": statement_clean}
            user_template = POLITICIAN_USER_TEMPLATE
        else:
            system = CONFIDENCE_SYSTEM
            canonical_labels = CONFIDENCE_SCALE
            section = "wvs_confidence"
            format_kwargs = {"institution": institution}
            user_template = CONFIDENCE_USER_TEMPLATE

        # Build all variations: canonical + shuffled
        variations = [(canonical_labels, None)]  # (labels, mapping_or_None)
        if n_shuffles > 0:
            shuffled = _generate_shuffled_orders(canonical_labels, n_shuffles, rng)
            variations.extend(shuffled)

        for var_idx, (labels, mapping) in enumerate(variations):
            scale_lines = _make_scale_lines(labels)
            user = user_template.format(
                scale_lines=scale_lines,
                n=len(labels),
                **format_kwargs,
            )
            prompt_text = json.dumps({"system": system, "user": user})
            scale_labels_fmt = _make_scale_labels(labels)

            pid = f"p1_{item_id}_v{var_idx}" if n_shuffles > 0 else f"p1_{item_id}"

            meta = {"org_question_number": q_num, "variation": var_idx}
            if mapping is not None:
                meta["order_mapping"] = mapping
            # For canonical order, mapping is identity
            if mapping is None and n_shuffles > 0:
                meta["order_mapping"] = {
                    str(i+1): str(i+1) for i in range(len(canonical_labels))
                }

            prompts.append(PromptItem(
                prompt_id=pid,
                part=1,
                section=section,
                item_id=item_id,
                institution=institution,
                statement=statement,
                prompt_text=prompt_text,
                scale_labels=scale_labels_fmt,
                reverse_coded=reverse,
                metadata=meta,
            ))

    return prompts


def remap_to_canonical(result: dict) -> dict:
    """
    Remap a shuffled result's logits/probs back to canonical scale order.

    Uses the order_mapping in metadata. If no mapping present, returns as-is.

    Example:
        order_mapping = {"1": "3", "2": "1", "3": "4", "4": "2"}
        logits = {"1": -0.5, "2": -1.2, "3": -3.0, "4": -2.1}
        remapped = {"3": -0.5, "1": -1.2, "4": -3.0, "2": -2.1}
        (position 1 had logit -0.5, and that maps to canonical position 3)
    """
    mapping = result.get("metadata", {}).get("order_mapping")
    if not mapping:
        return result

    result = json.loads(json.dumps(result))  # deep copy

    resp = result["response"]
    for key in ("logits", "probs"):
        if resp.get(key):
            original = resp[key]
            resp[key] = {
                mapping[pos]: val for pos, val in original.items()
                if pos in mapping
            }

    # Remap answer and choice_from_text
    if resp.get("answer") and resp["answer"] in mapping:
        resp["answer"] = mapping[resp["answer"]]
    if resp.get("choice_from_text") and resp["choice_from_text"] in mapping:
        resp["choice_from_text"] = mapping[resp["choice_from_text"]]

    return result