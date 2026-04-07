#!/usr/bin/env python3
"""
Generate Batch Files
====================
Reads a prompt JSONL file and expands it into per-model batch JSONL files
with N repetitions, in OpenAI Batch API format.

Each line in the output is a complete request object:
{
    "custom_id": "<prompt_id>_rep<N>",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "<model_id>",
        "messages": [...],
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": true,
        "top_logprobs": 10
    }
}

Usage:
    # Generate batch files for Part 1 with 100 reps (reads from manifest)
    python generate_batch.py --prompts outputs/prompts/part1_prompts.jsonl

    # Override reps
    python generate_batch.py --prompts outputs/prompts/part1_prompts.jsonl --n-reps 50

    # Single model only
    python generate_batch.py --prompts outputs/prompts/part1_prompts.jsonl \
        --models openai/gpt-4o

    # Custom output directory
    python generate_batch.py --prompts outputs/prompts/part1_prompts.jsonl \
        --output-dir outputs/batches

Output:
    outputs/batches/
    ├── batch_gpt-4o_part1_100reps.jsonl           (3,700 requests)
    ├── batch_claude-sonnet-4_part1_100reps.jsonl   (3,700 requests)
    ├── batch_gemini-2.5-pro_part1_100reps.jsonl    (3,700 requests)
    ├── batch_llama-4-maverick_part1_100reps.jsonl   (3,700 requests)
    └── batch_manifest.json                          (summary)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_manifest(prompt_path: str) -> dict | None:
    manifest_path = Path(prompt_path).parent / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def sanitize_label(model_id: str) -> str:
    """Turn 'openai/gpt-4o' into 'gpt-4o' for filenames."""
    return model_id.split("/")[-1]


def build_batch_request(
    prompt: dict,
    model_id: str,
    repetition: int,
    temperature: float,
    max_tokens: int,
    logprobs: bool,
    top_logprobs: int,
) -> dict:
    """Build one OpenAI Batch API request line."""
    prompt_data = json.loads(prompt["prompt_text"])
    messages = [
        {"role": "system", "content": prompt_data["system"]},
        {"role": "user", "content": prompt_data["user"]},
    ]

    # Strip provider prefix for native APIs (openai/gpt-4o → gpt-4o)
    # Keep full ID in custom_id for traceability
    native_model_id = model_id.split("/")[-1] if "/" in model_id else model_id

    body = {
        "model": native_model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs

    return {
        "custom_id": f"{prompt['prompt_id']}_rep{repetition:03d}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-model batch JSONL files for OpenAI Batch API"
    )
    parser.add_argument(
        "--prompts", required=True,
        help="Path to prompt JSONL (e.g., outputs/prompts/part1_prompts.jsonl)"
    )
    parser.add_argument(
        "--n-reps", type=int, default=None,
        help="Number of repetitions per prompt (default: from manifest)"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model IDs (default: from manifest/config)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/batches",
        help="Output directory for batch files"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Temperature (default: from manifest, or 1.0)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max tokens (default: from manifest, or 1)"
    )
    parser.add_argument(
        "--no-logprobs", action="store_true",
        help="Disable logprobs in batch requests"
    )
    parser.add_argument(
        "--top-logprobs", type=int, default=None,
        help="Number of top logprobs (default: from manifest, or 10)"
    )
    args = parser.parse_args()

    # Load prompts and manifest
    prompts = load_prompts(args.prompts)
    manifest = load_manifest(args.prompts)
    prompt_stem = Path(args.prompts).stem.replace("_prompts", "")

    # Resolve settings
    if args.n_reps is not None:
        n_reps = args.n_reps
    elif manifest:
        part_key = "part1" if "part1" in prompt_stem else "part2"
        n_reps = manifest.get("files", {}).get(part_key, {}).get(
            "n_reps", manifest.get(f"n_repetitions_{part_key}", 1)
        )
    else:
        n_reps = 1

    temperature = args.temperature or (manifest or {}).get("temperature", 1.0)
    max_tokens = args.max_tokens or (manifest or {}).get("max_tokens", 1)
    logprobs = not args.no_logprobs and (manifest or {}).get("logprobs", True)
    top_logprobs = args.top_logprobs or (manifest or {}).get("top_logprobs", 10)

    # Resolve models
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    elif manifest:
        model_ids = manifest.get("model_ids", [])
    else:
        print("ERROR: No models specified. Use --models or generate from manifest.")
        sys.exit(1)

    if not model_ids:
        print("ERROR: No models found.")
        sys.exit(1)

    # Generate batch files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BATCH FILE GENERATION")
    print(f"  Prompts: {len(prompts)} from {args.prompts}")
    print(f"  Repetitions: {n_reps}")
    print(f"  Models: {model_ids}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Logprobs: {logprobs} (top_k={top_logprobs})")
    print("=" * 60)

    batch_files = {}

    for model_id in model_ids:
        label = sanitize_label(model_id)
        filename = f"batch_{label}_{prompt_stem}_{n_reps}reps.jsonl"
        filepath = output_dir / filename

        count = 0
        with open(filepath, "w") as f:
            for prompt in prompts:
                for rep in range(1, n_reps + 1):
                    request = build_batch_request(
                        prompt, model_id, rep,
                        temperature, max_tokens,
                        logprobs, top_logprobs,
                    )
                    f.write(json.dumps(request) + "\n")
                    count += 1

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  {filename}: {count:,} requests ({size_mb:.1f} MB)")
        batch_files[model_id] = {
            "file": str(filepath),
            "filename": filename,
            "n_requests": count,
            "size_mb": round(size_mb, 2),
        }

    # Save batch manifest
    batch_manifest = {
        "source_prompts": args.prompts,
        "n_prompts": len(prompts),
        "n_reps": n_reps,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "models": batch_files,
        "total_requests": sum(b["n_requests"] for b in batch_files.values()),
    }
    manifest_path = output_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(batch_manifest, f, indent=2)

    print()
    print(f"  Manifest: {manifest_path}")
    print(f"  Total requests across all models: {batch_manifest['total_requests']:,}")
    print()
    print("Next steps:")
    print("  For OpenAI models (gpt-4o):")
    print(f"    python submit_batch.py --file {batch_files.get('openai/gpt-4o', {}).get('file', 'outputs/batches/batch_gpt-4o_...')}")
    print("  Or upload manually:")
    print("    openai api files create -f <batch_file>.jsonl -p batch")
    print("    openai api batches create -i <file_id> -e /v1/chat/completions -c 24h")


if __name__ == "__main__":
    main()
