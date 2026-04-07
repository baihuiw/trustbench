"""
OpenRouter API Runner
=====================
Sends prompts to LLMs via OpenRouter. Extracts first-token logprobs
and justification text. Output format per row:

{
  "metadata": { "prompt_id", "item_id", "institution", ... },
  "response": {
    "status_code": 200,
    "model": "openai/gpt-4o",
    "repetition": 1,
    "logits": {"1": -0.85, "2": -0.82, "3": -2.04, "4": -6.41},
    "probs":  {"1": 0.30, "2": 0.31, "3": 0.09, "4": 0.001},
    "mean": 1.82,
    "entropy_norm": 0.45,
    "answer": "2",
    "justification": "Churches play a significant role...",
    "latency_ms": 450.3
  }
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import aiohttp

from src.prompt_generators import PromptItem

logger = logging.getLogger(__name__)


def _extract_first_token_logprobs(body: dict) -> dict | None:
    """Extract top logprobs for the first token from API response."""
    try:
        lp = body.get("choices", [{}])[0].get("logprobs")
        if lp is None:
            return None
        content = lp.get("content")
        if not content:
            return None
        return content[0].get("top_logprobs")
    except (IndexError, KeyError, TypeError):
        return None


def _compute_logits_and_probs(
    top_logprobs: list[dict],
    scale_tokens: list[str],
) -> dict:
    """
    From raw top_logprobs, extract logits and probs for scale-relevant
    tokens only (e.g., "1","2","3","4" or "A","B").

    Returns dict with keys: logits, probs, mean, entropy_norm, answer
    """
    # Build token -> logprob mapping
    token_map = {}
    for entry in top_logprobs:
        tok = entry.get("token", "").strip()
        lp = entry.get("logprob")
        if tok and lp is not None:
            token_map[tok] = lp

    # Extract logits for scale tokens
    logits = {}
    for tok in scale_tokens:
        if tok in token_map:
            logits[tok] = token_map[tok]
        else:
            logits[tok] = -100.0  # effectively zero probability

    # Convert to probabilities (softmax over scale tokens only)
    if not logits:
        return {"logits": {}, "probs": {}, "mean": None,
                "entropy_norm": None, "answer": None}

    max_lp = max(logits.values())
    exp_vals = {t: math.exp(lp - max_lp) for t, lp in logits.items()}
    total_exp = sum(exp_vals.values())
    probs = {t: exp_vals[t] / total_exp for t in logits}

    # Find the answer (highest prob token)
    answer = max(probs, key=probs.get)

    # Compute expected value (mean) for numeric scales
    mean = None
    if all(t.isdigit() for t in scale_tokens):
        mean = sum(int(t) * p for t, p in probs.items())

    # Normalized entropy
    entropy_norm = None
    n = len(scale_tokens)
    if n > 1:
        entropy = -sum(p * math.log(p + 1e-30) for p in probs.values())
        max_entropy = math.log(n)
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "logits": {t: round(v, 6) for t, v in logits.items()},
        "probs": {t: round(v, 10) for t, v in probs.items()},
        "mean": round(mean, 6) if mean is not None else None,
        "entropy_norm": round(entropy_norm, 6) if entropy_norm is not None else None,
        "answer": answer,
    }


def _get_scale_tokens(scale_labels: list[str]) -> list[str]:
    """
    Determine which tokens to extract from logprobs.
    scale_labels like ["1: A great deal", ...] -> ["1","2","3","4"]
    scale_labels like ["A", "B"] -> ["A","B"]
    """
    tokens = []
    for label in scale_labels:
        label = label.strip()
        if label and label[0].isdigit():
            tokens.append(label[0])
        elif label.upper() in ("A", "B"):
            tokens.append(label.upper())
    if not tokens:
        # Fallback: generate 1..N
        tokens = [str(i) for i in range(1, len(scale_labels) + 1)]
    return tokens


def _parse_response(raw: str, scale_tokens: list[str]) -> tuple[str | None, str | None]:
    """Parse first character as choice, rest as justification."""
    if not raw or not raw.strip():
        return None, None
    text = raw.strip()
    first_char = text[0]
    rest = text[1:].lstrip(".").lstrip(":").lstrip(")").lstrip(",").strip()
    justification = rest if rest else None

    if first_char in scale_tokens:
        return first_char, justification
    if first_char.upper() in scale_tokens:
        return first_char.upper(), justification
    return None, justification


def _build_result_row(
    prompt_item: PromptItem,
    model_id: str,
    repetition: int,
    status_code: int,
    raw: str,
    top_logprobs_raw: list[dict] | None,
    latency_ms: float,
    error: str | None,
) -> dict:
    """Build one clean result row in the target format."""
    scale_tokens = _get_scale_tokens(prompt_item.scale_labels)

    # Parse choice and justification from raw text
    choice_from_text, justification = _parse_response(raw, scale_tokens)

    # Compute logits/probs from first-token logprobs
    if top_logprobs_raw:
        lp_result = _compute_logits_and_probs(top_logprobs_raw, scale_tokens)
    else:
        lp_result = {
            "logits": {}, "probs": {}, "mean": None,
            "entropy_norm": None, "answer": None,
        }

    return {
        "metadata": {
            "prompt_id": prompt_item.prompt_id,
            "part": prompt_item.part,
            "section": prompt_item.section,
            "item_id": prompt_item.item_id,
            "institution": prompt_item.institution,
            "country": prompt_item.country,
            "reverse_coded": prompt_item.reverse_coded,
            "order_mapping": prompt_item.metadata.get("order_mapping"),
            "variation": prompt_item.metadata.get("variation", 0),
        },
        "response": {
            "status_code": status_code,
            "model": model_id,
            "repetition": repetition,
            "logits": lp_result["logits"],
            "probs": lp_result["probs"],
            "entropy_norm": lp_result["entropy_norm"],
            "answer": lp_result["answer"],
            "choice_from_text": choice_from_text,
            "justification": justification,
            "error": error,
        },
    }


class OpenRouterRunner:
    """Manages async API calls to OpenRouter."""

    def __init__(self, cfg: dict):
        self.base_url = cfg["api"]["base_url"]
        self.api_key = cfg["api"]["key"]
        self.max_retries = cfg["api"].get("max_retries", 5)
        self.retry_delay = cfg["api"].get("retry_delay", 2.0)
        self.timeout = cfg["api"].get("timeout", 120)
        self.rpm = cfg["api"].get("requests_per_minute", 200)
        self.temperature = cfg["run"].get("temperature", 1.0)
        self.max_tokens = cfg["run"].get("max_tokens", 150)
        self.seed = cfg["run"].get("seed", None)
        self.batch_size = cfg["run"].get("batch_size", 30)
        self.request_logprobs = cfg["run"].get("logprobs", True)
        self.top_logprobs_k = cfg["run"].get("top_logprobs", 10)

        self._semaphore = asyncio.Semaphore(self.batch_size)
        self._min_interval = 60.0 / self.rpm
        self._last_request_time = 0.0

    async def _rate_limit(self):
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _build_payload(self, model_id: str, messages: list[dict]) -> dict:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs_k
        return payload

    async def _call_api(
        self,
        session: aiohttp.ClientSession,
        prompt_item: PromptItem,
        model_id: str,
        repetition: int,
    ) -> dict:
        """Single API call with retries. Returns a clean result dict."""
        prompt_data = json.loads(prompt_item.prompt_text)
        messages = [
            {"role": "system", "content": prompt_data["system"]},
            {"role": "user", "content": prompt_data["user"]},
        ]
        payload = self._build_payload(model_id, messages)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://trustbench.research",
            "X-Title": "TrustBench",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._semaphore:
                    await self._rate_limit()
                    t0 = time.monotonic()
                    async with session.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        latency = (time.monotonic() - t0) * 1000
                        body = await resp.json()

                        if resp.status == 429:
                            wait = self.retry_delay * (2 ** (attempt - 1))
                            logger.warning(
                                "Rate limited on %s (attempt %d), waiting %.1fs",
                                prompt_item.prompt_id, attempt, wait,
                            )
                            await asyncio.sleep(wait)
                            continue

                        if resp.status != 200:
                            error_msg = body.get("error", {}).get(
                                "message", f"HTTP {resp.status}"
                            )
                            if attempt < self.max_retries:
                                await asyncio.sleep(self.retry_delay * attempt)
                                continue
                            return _build_result_row(
                                prompt_item, model_id, repetition,
                                resp.status, "", None, latency, error_msg,
                            )

                        raw = (
                            body.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content") or ""
                        ).strip()

                        top_lp = _extract_first_token_logprobs(body)

                        return _build_result_row(
                            prompt_item, model_id, repetition,
                            200, raw, top_lp, latency, None,
                        )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                return _build_result_row(
                    prompt_item, model_id, repetition,
                    0, "", None, 0.0, str(e),
                )

        return _build_result_row(
            prompt_item, model_id, repetition,
            0, "", None, 0.0, "Max retries exhausted",
        )

    async def run_all(
        self,
        prompts: list[PromptItem],
        models: list[dict],
        n_repetitions: int,
    ) -> list[dict]:
        """Run all prompts × models × repetitions with progress tracking."""
        total = len(prompts) * len(models) * n_repetitions
        logger.info(
            "Dispatching %d API calls (%d prompts × %d models × %d reps)",
            total, len(prompts), len(models), n_repetitions,
        )

        completed = 0
        errors = 0
        start_time = time.monotonic()

        async def _tracked_call(session, prompt, model_id, rep):
            nonlocal completed, errors
            result = await self._call_api(session, prompt, model_id, rep)
            completed += 1
            if result["response"].get("error"):
                errors += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f%%) | %.1f req/min | "
                    "errors: %d | ETA: %.0fm %.0fs",
                    completed, total, 100 * completed / total,
                    rate, errors, remaining // 60, remaining % 60,
                )
            return result

        async with aiohttp.ClientSession() as session:
            tasks = []
            for model in models:
                for prompt in prompts:
                    for rep in range(1, n_repetitions + 1):
                        tasks.append(
                            _tracked_call(session, prompt, model["id"], rep)
                        )
            results = list(await asyncio.gather(*tasks))

        elapsed = time.monotonic() - start_time
        logger.info(
            "All %d calls completed in %.1fm (%.1f req/min, %d errors)",
            total, elapsed / 60,
            total / elapsed * 60 if elapsed > 0 else 0,
            errors,
        )
        return results


def save_results(results: list[dict], path: str | Path):
    """Save results as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info("Saved %d results to %s", len(results), path)