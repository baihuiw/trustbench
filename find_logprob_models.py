#!/usr/bin/env python3
"""
Find models on OpenRouter that support logprobs.

Usage:
    python find_logprob_models.py

    # Filter by keyword
    python find_logprob_models.py --filter llama

    # Show only cheap models (< $1/M input tokens)
    python find_logprob_models.py --max-price 1.0
"""

import argparse
import json
from collections import defaultdict

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter model IDs by keyword")
    parser.add_argument("--max-price", type=float, default=None,
                        help="Max price per 1M input tokens in USD")
    args = parser.parse_args()

    print("Fetching models from OpenRouter API...")
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    models = resp.json()["data"]
    print(f"Total models: {len(models)}")

    # Find models with top_logprobs
    logprob_models = []
    for m in models:
        params = m.get("supported_parameters", [])
        if "top_logprobs" in params:
            price_in = float(m["pricing"]["prompt"]) * 1_000_000
            price_out = float(m["pricing"]["completion"]) * 1_000_000

            if args.filter and args.filter.lower() not in m["id"].lower():
                continue
            if args.max_price is not None and price_in > args.max_price:
                continue

            logprob_models.append({
                "id": m["id"],
                "name": m["name"],
                "context": m.get("context_length"),
                "price_in_per_M": round(price_in, 4),
                "price_out_per_M": round(price_out, 4),
            })

    print(f"Models with logprobs support: {len(logprob_models)}")
    print()

    # Group by provider
    by_provider = defaultdict(list)
    for m in logprob_models:
        provider = m["id"].split("/")[0] if "/" in m["id"] else "other"
        by_provider[provider].append(m)

    for provider in sorted(by_provider.keys()):
        mlist = by_provider[provider]
        print(f"{'─' * 60}")
        print(f"{provider} ({len(mlist)} models)")
        print(f"{'─' * 60}")
        for m in sorted(mlist, key=lambda x: x["price_in_per_M"]):
            print(
                f"  {m['id']:55s} "
                f"${m['price_in_per_M']:.2f}/${m['price_out_per_M']:.2f} per M"
            )
        print()

    # Summary of well-known frontier models
    print("=" * 60)
    print("FRONTIER MODELS WITH LOGPROBS:")
    print("=" * 60)
    frontier_ids = [
        "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4.1",
        "openai/gpt-4.1-mini", "openai/gpt-4.1-nano",
        "meta-llama/llama-4-maverick", "meta-llama/llama-4-scout",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat-v3-0324",  "deepseek/deepseek-r1",
        "mistralai/mistral-large-2411", "mistralai/mistral-small-3.1-24b-instruct",
        "google/gemini-2.5-pro-preview-06-05", "google/gemini-2.0-flash-001",
        "anthropic/claude-sonnet-4", "anthropic/claude-haiku-4-5",
        "qwen/qwen-2.5-72b-instruct", "qwen/qwen3-235b-a22b",
    ]
    all_ids = {m["id"] for m in logprob_models}
    for fid in frontier_ids:
        has = "✓ logprobs" if fid in all_ids else "✗ NO logprobs"
        print(f"  {fid:55s} {has}")


if __name__ == "__main__":
    main()
