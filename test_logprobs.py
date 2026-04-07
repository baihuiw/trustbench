#!/usr/bin/env python3
"""
Test Logprobs Support
=====================
Sends one test request per model to verify logprobs are returned.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python test_logprobs.py

    # Test specific models only
    python test_logprobs.py --models openai/gpt-4o,anthropic/claude-sonnet-4

    # Also test via OpenAI directly
    export OPENAI_API_KEY="sk-..."
    python test_logprobs.py --openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

# A simple test prompt
TEST_SYSTEM = (
    "You are completing a survey about institutional trust. "
    "Answer with ONLY the number corresponding to your choice. "
    "Do not explain your reasoning."
)
TEST_USER = (
    "I am going to name an organization. Could you tell me how much "
    "confidence you have in it: is it a great deal of confidence, quite a "
    "lot of confidence, not very much confidence, or none at all?\n\n"
    "Organization: The United Nations\n\n"
    "1 — A great deal\n"
    "2 — Quite a lot\n"
    "3 — Not very much\n"
    "4 — None at all\n\n"
    "Respond with just the number."
)

DEFAULT_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat-v3-0324",
]


def test_openrouter(model_id: str, api_key: str) -> dict:
    """Send one request via OpenRouter and check for logprobs."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": TEST_SYSTEM},
            {"role": "user", "content": TEST_USER},
        ],
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 10,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://trustbench.research",
        "X-Title": "TrustBench-LogprobTest",
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        body = resp.json()

        if resp.status_code != 200:
            error = body.get("error", {}).get("message", f"HTTP {resp.status_code}")
            return {"model": model_id, "status": "ERROR", "error": error}

        # Extract response
        choice = body.get("choices", [{}])[0]
        content = (choice.get("message", {}).get("content") or "").strip()
        logprobs_data = choice.get("logprobs")

        result = {
            "model": model_id,
            "response": content,
            "status": "OK",
            "has_logprobs": logprobs_data is not None,
            "raw_logprobs_keys": list(logprobs_data.keys()) if isinstance(logprobs_data, dict) else None,
        }

        if logprobs_data and logprobs_data.get("content"):
            lp_content = logprobs_data["content"]
            first_token = lp_content[0]
            result["first_token"] = first_token.get("token")
            result["first_token_logprob"] = first_token.get("logprob")
            result["n_top_logprobs"] = len(first_token.get("top_logprobs", []))
            result["top_tokens"] = [
                {"token": t["token"], "logprob": round(t["logprob"], 4)}
                for t in first_token.get("top_logprobs", [])[:5]
            ]
        else:
            result["n_top_logprobs"] = 0
            result["top_tokens"] = []

        return result

    except Exception as e:
        return {"model": model_id, "status": "EXCEPTION", "error": str(e)}


def test_openai_direct(api_key: str) -> dict:
    """Send one request directly to OpenAI API (for batch API compatibility check)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": TEST_SYSTEM},
                {"role": "user", "content": TEST_USER},
            ],
            max_tokens=1,
            temperature=1.0,
            logprobs=True,
            top_logprobs=10,
        )

        choice = resp.choices[0]
        content = choice.message.content.strip()
        lp = choice.logprobs

        result = {
            "model": "gpt-4o (direct OpenAI)",
            "response": content,
            "status": "OK",
            "has_logprobs": lp is not None,
        }

        if lp and lp.content:
            first = lp.content[0]
            result["first_token"] = first.token
            result["first_token_logprob"] = round(first.logprob, 4)
            result["n_top_logprobs"] = len(first.top_logprobs)
            result["top_tokens"] = [
                {"token": t.token, "logprob": round(t.logprob, 4)}
                for t in first.top_logprobs[:5]
            ]
        else:
            result["n_top_logprobs"] = 0
            result["top_tokens"] = []

        return result

    except ImportError:
        return {"model": "gpt-4o (direct)", "status": "ERROR",
                "error": "openai package not installed"}
    except Exception as e:
        return {"model": "gpt-4o (direct)", "status": "EXCEPTION", "error": str(e)}


def print_result(r: dict):
    model = r["model"]
    status = r["status"]

    if status != "OK":
        print(f"  ✗ {model}")
        print(f"    Status: {status}")
        print(f"    Error: {r.get('error', 'unknown')}")
        return

    has_lp = r["has_logprobs"]
    mark = "✓" if has_lp else "✗"
    print(f"  {mark} {model}")
    print(f"    Response: {r['response']}")
    print(f"    Logprobs returned: {has_lp}")

    if has_lp and r["n_top_logprobs"] > 0:
        print(f"    Top {r['n_top_logprobs']} logprobs for first token:")
        for t in r["top_tokens"]:
            import math
            prob = math.exp(t["logprob"]) * 100
            print(f"      token=\"{t['token']}\"  logprob={t['logprob']}  prob={prob:.1f}%")
    elif has_lp:
        print(f"    (logprobs object present but no top_logprobs content)")

    if r.get("raw_logprobs_keys"):
        print(f"    Raw logprobs keys: {r['raw_logprobs_keys']}")


def main():
    parser = argparse.ArgumentParser(description="Test logprobs support per model")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model IDs")
    parser.add_argument("--openai", action="store_true",
                        help="Also test OpenAI directly (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    oai_key = os.environ.get("OPENAI_API_KEY", "")

    if not or_key and not args.openai:
        print("ERROR: Set OPENROUTER_API_KEY or use --openai with OPENAI_API_KEY")
        sys.exit(1)

    models = (
        [m.strip() for m in args.models.split(",")]
        if args.models else DEFAULT_MODELS
    )

    print("=" * 60)
    print("LOGPROBS SUPPORT TEST")
    print(f"Testing {len(models)} models via OpenRouter")
    if args.openai:
        print("+ 1 model via OpenAI direct")
    print("=" * 60)
    print()

    # Test via OpenRouter
    if or_key:
        print("Via OpenRouter:")
        for model_id in models:
            r = test_openrouter(model_id, or_key)
            print_result(r)
            print()

    # Test via OpenAI direct
    if args.openai:
        if oai_key:
            print("Via OpenAI direct API:")
            r = test_openai_direct(oai_key)
            print_result(r)
            print()
        else:
            print("Skipping OpenAI direct: OPENAI_API_KEY not set")

    print("=" * 60)
    print("SUMMARY")
    print("  Models with logprobs → use batch API or run_experiments.py")
    print("  Models without logprobs → use run_experiments.py (sampling only)")
    print("=" * 60)


if __name__ == "__main__":
    main()
