"""
OpenWebUI automation client — sends pre-calculated data to LLM models
and parses structured responses for the consensus mechanism.

Used for MARGINAL signals: queries multiple models and aggregates their decisions.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from llm_trading_bot.config import OpenWebUIConfig
from llm_trading_bot.routing import build_llm_context
from llm_trading_bot.scoring import Direction, ScoringResult, TradeTargets


@dataclass
class LLMResponse:
    """Parsed response from a single LLM model."""
    model_id: str
    decision: str  # "LONG", "SHORT", "WAIT"
    confidence: float
    reasoning: str
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    raw_response: str = ""
    parse_error: Optional[str] = None
    latency_ms: float = 0


@dataclass
class ConsensusResult:
    """Aggregated result from multiple LLM models."""
    decision: str  # "LONG", "SHORT", "WAIT"
    agreement_pct: float
    average_confidence: float
    individual_responses: list[LLMResponse] = field(default_factory=list)
    reasoning_summary: str = ""


def _extract_json_object(raw: str) -> Optional[str]:
    """
    Extract the last balanced ``{...}`` JSON object from an LLM response.

    Robust to: reasoning-model ``<think>...</think>`` preambles, markdown code fences,
    surrounding prose, and JSON containing nested objects (which a non-greedy regex
    like ``\\{.*?\\}`` would truncate at the first inner brace). Scans for balanced
    brace pairs and returns the LAST one (the final answer, after any reasoning).
    """
    # Drop reasoning-model think blocks so their braces don't confuse the scan.
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)

    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(cleaned[start:i + 1])
                    start = -1

    # Prefer the last candidate that actually parses as JSON.
    for cand in reversed(candidates):
        try:
            json.loads(cand)
            return cand
        except json.JSONDecodeError:
            continue
    return None


def _parse_llm_response(raw: str, model_id: str) -> LLMResponse:
    """
    Parse the LLM's JSON response. Handles various formatting issues
    (markdown code blocks, reasoning preambles, nested objects, extra prose).
    """
    response = LLMResponse(model_id=model_id, decision="WAIT", confidence=0, reasoning="", raw_response=raw)

    try:
        json_str = _extract_json_object(raw)
        if json_str is None:
            response.parse_error = "No JSON found in response"
            return response

        data = json.loads(json_str)

        response.decision = str(data.get("decision", "WAIT")).upper()
        if response.decision not in ("LONG", "SHORT", "WAIT"):
            response.decision = "WAIT"

        response.confidence = float(data.get("confidence", 0))
        # Clamp to the system-wide confidence invariant [5, 95] (matches
        # scoring.confidence_min/max) so LLM confidences can't fall outside the
        # range the rest of the pipeline assumes.
        response.confidence = max(5, min(95, response.confidence))
        response.reasoning = str(data.get("reasoning", ""))
        response.entry = data.get("entry")
        response.stop_loss = data.get("stop_loss")
        response.take_profit_1 = data.get("take_profit_1")
        response.take_profit_2 = data.get("take_profit_2")

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        response.parse_error = f"Parse error: {e}"

    return response


def query_openwebui(
    config: OpenWebUIConfig,
    model_id: str,
    prompt: str,
    system_prompt: str = "You are a professional cryptocurrency trading analyst. Respond ONLY with the requested JSON format.",
) -> LLMResponse:
    """
    Send a prompt to a specific model via the OpenWebUI API.
    Returns a parsed LLMResponse.
    """
    url = f"{config.base_url.rstrip('/')}/api/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    start = time.time()
    try:
        resp = requests.post(
            url, json=payload, headers=headers, timeout=config.timeout_seconds
        )
        latency = (time.time() - start) * 1000
        resp.raise_for_status()

        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]

        result = _parse_llm_response(raw_content, model_id)
        result.latency_ms = latency
        return result

    except requests.exceptions.Timeout:
        return LLMResponse(
            model_id=model_id, decision="WAIT", confidence=0,
            reasoning="", parse_error="Request timed out",
            latency_ms=(time.time() - start) * 1000,
        )
    except requests.exceptions.RequestException as e:
        return LLMResponse(
            model_id=model_id, decision="WAIT", confidence=0,
            reasoning="", parse_error=f"Request failed: {e}",
            latency_ms=(time.time() - start) * 1000,
        )
    except (KeyError, IndexError) as e:
        return LLMResponse(
            model_id=model_id, decision="WAIT", confidence=0,
            reasoning="", parse_error=f"Response format error: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


def build_consensus(responses: list[LLMResponse]) -> ConsensusResult:
    """
    Aggregate multiple LLM responses into a consensus decision.

    Rules:
    - Majority vote wins
    - WAIT responses from parse errors are excluded from voting
    - If no clear majority, defaults to WAIT
    - Average confidence only from valid responses
    """
    valid = [r for r in responses if not r.parse_error]
    if not valid:
        return ConsensusResult(
            decision="WAIT",
            agreement_pct=0,
            average_confidence=0,
            individual_responses=responses,
            reasoning_summary="No valid LLM responses received.",
        )

    # Count votes
    votes: dict[str, int] = {"LONG": 0, "SHORT": 0, "WAIT": 0}
    confidences: dict[str, list[float]] = {"LONG": [], "SHORT": [], "WAIT": []}

    for r in valid:
        votes[r.decision] = votes.get(r.decision, 0) + 1
        confidences[r.decision].append(r.confidence)

    total_votes = sum(votes.values())
    winner = max(votes, key=votes.get)  # type: ignore
    agreement_pct = (votes[winner] / total_votes * 100) if total_votes else 0

    # Need a STRICT majority (>50%) for action, else WAIT. Using `<= 50` makes an
    # even split (e.g. a two-model 1 LONG / 1 SHORT tie = exactly 50%) resolve to
    # WAIT instead of silently picking whichever direction sorts first.
    if winner in ("LONG", "SHORT") and agreement_pct <= 50:
        winner = "WAIT"

    avg_conf = (
        sum(confidences[winner]) / len(confidences[winner])
        if confidences[winner] else 0
    )

    # Build summary
    reasons = [f"{r.model_id}: {r.decision} ({r.confidence}%) — {r.reasoning}" for r in valid]
    summary = f"Consensus: {winner} ({agreement_pct:.0f}% agreement)\n" + "\n".join(reasons)

    return ConsensusResult(
        decision=winner,
        agreement_pct=agreement_pct,
        average_confidence=avg_conf,
        individual_responses=responses,
        reasoning_summary=summary,
    )


def run_consensus(
    config: OpenWebUIConfig,
    scoring_result: ScoringResult,
    targets: Optional[TradeTargets],
) -> ConsensusResult:
    """
    Full consensus pipeline: build context, query all models, aggregate.
    """
    prompt = build_llm_context(scoring_result, targets)

    responses: list[LLMResponse] = []
    for model_id in config.model_ids:
        print(f"  Querying {model_id}...")
        resp = query_openwebui(config, model_id, prompt)
        responses.append(resp)
        if resp.parse_error:
            print(f"    Warning: {resp.parse_error}")
        else:
            print(f"    → {resp.decision} ({resp.confidence}%) in {resp.latency_ms:.0f}ms")

    return build_consensus(responses)
