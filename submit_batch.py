#!/usr/bin/env python3
"""
Submit & Monitor Batch Jobs
============================
Submits batch JSONL files to OpenAI's Batch API and monitors progress.

Usage:
    # Submit a batch file
    python submit_batch.py --file outputs/batches/batch_gpt-4o_part1_100reps.jsonl

    # Check status of a batch
    python submit_batch.py --status <batch_id>

    # Check status of all batches
    python submit_batch.py --list

    # Download results when complete
    python submit_batch.py --download <batch_id> --output results_gpt4o.jsonl

    # Full workflow: submit, wait, download
    python submit_batch.py --file outputs/batches/batch_gpt-4o_part1_100reps.jsonl \
        --wait --output outputs/results/

Note:
    This uses OpenAI's API directly (not OpenRouter) since OpenRouter
    doesn't support batch API. Set OPENAI_API_KEY for OpenAI models.
    For other providers, use their respective batch APIs or run_experiments.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai")
    sys.exit(1)


def submit_batch(client: OpenAI, filepath: str, description: str = "") -> dict:
    """Upload file and create a batch job."""
    print(f"Uploading {filepath}...")
    with open(filepath, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {file_obj.id}")

    if not description:
        description = f"TrustBench: {Path(filepath).stem}"

    print(f"Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.status}")
    return {"batch_id": batch.id, "file_id": file_obj.id, "status": batch.status}


def check_status(client: OpenAI, batch_id: str) -> dict:
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    info = {
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "completed_at": batch.completed_at,
        "failed_at": batch.failed_at,
        "expired_at": batch.expired_at,
        "request_counts": {
            "total": batch.request_counts.total if batch.request_counts else 0,
            "completed": batch.request_counts.completed if batch.request_counts else 0,
            "failed": batch.request_counts.failed if batch.request_counts else 0,
        },
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }
    return info


def download_results(client: OpenAI, file_id: str, output_path: str):
    """Download batch results to a file."""
    print(f"Downloading {file_id} → {output_path}")
    content = client.files.content(file_id)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(content.read())
    # Count lines
    with open(output_path) as f:
        n = sum(1 for _ in f)
    print(f"  Saved {n} results to {output_path}")


def wait_for_batch(client: OpenAI, batch_id: str, poll_interval: int = 30):
    """Poll until batch completes or fails."""
    print(f"Waiting for batch {batch_id}...")
    while True:
        info = check_status(client, batch_id)
        status = info["status"]
        counts = info["request_counts"]
        progress = (
            f"{counts['completed']}/{counts['total']}"
            if counts["total"] > 0 else "..."
        )
        print(f"  [{status}] {progress} completed, {counts['failed']} failed")

        if status in ("completed", "failed", "expired", "cancelled"):
            return info

        time.sleep(poll_interval)


def list_batches(client: OpenAI, limit: int = 10):
    """List recent batches."""
    batches = client.batches.list(limit=limit)
    for b in batches.data:
        counts = b.request_counts
        total = counts.total if counts else "?"
        completed = counts.completed if counts else "?"
        print(
            f"  {b.id}  [{b.status}]  "
            f"{completed}/{total} requests  "
            f"created={b.created_at}"
        )
        if b.metadata and b.metadata.get("description"):
            print(f"    desc: {b.metadata['description']}")


def main():
    parser = argparse.ArgumentParser(description="Submit and monitor OpenAI batch jobs")
    parser.add_argument("--file", type=str, help="Batch JSONL file to submit")
    parser.add_argument("--status", type=str, help="Check status of batch ID")
    parser.add_argument("--list", action="store_true", help="List recent batches")
    parser.add_argument("--download", type=str, help="Download results for batch ID")
    parser.add_argument("--output", type=str, help="Output path for downloaded results")
    parser.add_argument("--wait", action="store_true", help="Wait for batch to complete")
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="Seconds between status polls (default: 30)"
    )
    parser.add_argument("--description", type=str, default="", help="Batch description")
    args = parser.parse_args()

    client = OpenAI()

    if args.list:
        print("Recent batches:")
        list_batches(client)
        return

    if args.status:
        info = check_status(client, args.status)
        print(json.dumps(info, indent=2, default=str))
        return

    if args.download:
        info = check_status(client, args.download)
        file_id = info.get("output_file_id")
        if not file_id:
            print(f"Batch {args.download} has no output file yet (status: {info['status']})")
            sys.exit(1)
        output = args.output or f"batch_output_{args.download}.jsonl"
        download_results(client, file_id, output)
        return

    if args.file:
        result = submit_batch(client, args.file, args.description)

        if args.wait:
            info = wait_for_batch(client, result["batch_id"], args.poll_interval)
            if info["status"] == "completed" and info["output_file_id"]:
                output = args.output or f"outputs/results/{Path(args.file).stem}_results.jsonl"
                download_results(client, info["output_file_id"], output)

                # Also download errors if any
                if info["error_file_id"]:
                    err_path = output.replace(".jsonl", "_errors.jsonl")
                    download_results(client, info["error_file_id"], err_path)
            elif info["status"] == "failed":
                print("Batch FAILED.")
                if info["error_file_id"]:
                    err_path = args.output or f"outputs/results/{Path(args.file).stem}_errors.jsonl"
                    download_results(client, info["error_file_id"], err_path)
        else:
            print()
            print("Batch submitted. Check status with:")
            print(f"  python submit_batch.py --status {result['batch_id']}")
            print(f"  python submit_batch.py --download {result['batch_id']} --output results.jsonl")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
