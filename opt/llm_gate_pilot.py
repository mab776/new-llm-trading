"""Resumable single-model backtest for the historical MARGINAL-signal gate.

The model never receives the candle timestamp, symbol, or future data.  We first
collect actual baseline entry opportunities, deterministically sample them across
half-year regimes, cache one Ollama response per frozen prompt, then replay the fast
backtest. Small samples are sparse interventions. A full-baseline run additionally
queries newly exposed opportunities until the gated strategy reaches fixed-point closure.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import requests

import opt.driver as driver
from llm_trading_bot.openwebui_client import _parse_llm_response
from llm_trading_bot.routing import build_llm_context
from llm_trading_bot.scoring import Direction


DEFAULT_URL = "http://192.168.0.70:11435"
DEFAULT_MODEL = "qwen3.6:35b-a3b-q8_0"
DEFAULT_CACHE = Path("reports/llm_gate_qwen36_35b_q8.jsonl")

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["LONG", "SHORT", "WAIT"]},
        "confidence": {"type": "number", "minimum": 1, "maximum": 100},
        "reasoning": {"type": "string"},
        "entry": {"type": ["number", "null"]},
        "stop_loss": {"type": ["number", "null"]},
        "take_profit_1": {"type": ["number", "null"]},
        "take_profit_2": {"type": ["number", "null"]},
    },
    "required": [
        "decision", "confidence", "reasoning", "entry", "stop_loss",
        "take_profit_1", "take_profit_2",
    ],
}


def _prompt_id(model: str, timestamp: str, prompt: str, *, think: bool = False,
               num_predict: int = 500) -> str:
    # Timestamp distinguishes cache rows but is deliberately never included in the
    # inference payload (historical dates would make memorized-future leakage easier).
    inference = f"think={think}\0num_predict={num_predict}"
    return hashlib.sha256(
        f"{model}\0{inference}\0{timestamp}\0{prompt}".encode()
    ).hexdigest()[:20]


def _build_blinded_context(result, targets) -> str:
    """Build the canonical report after removing recognizable historical price levels.

    Each timeframe is independently rebased to close=100.  Relative relationships,
    percentages, oscillator values, category scores, and target R:R are preserved.
    This matters because a post-2025 model could otherwise recognize an exact BTC price
    level and recall what happened next even though the indicator computation is causal.
    """
    blinded = copy.deepcopy(result)
    primary_close = targets.entry if targets is not None else None
    for ind in blinded.indicators.values():
        scale = 100.0 / ind.close if ind.close else 1.0
        for field in (
            "close", "open", "high", "low", "ema_9", "ema_21", "ema_50",
            "ema_200", "sma_50", "sma_200", "macd_line", "macd_signal",
            "macd_histogram", "vwap", "atr_14", "bb_upper", "bb_middle",
            "bb_lower", "pivot", "support_1", "support_2", "resistance_1",
            "resistance_2", "nearest_support", "nearest_resistance",
        ):
            value = getattr(ind, field)
            if value is not None:
                setattr(ind, field, value * scale)

    blinded_targets = copy.deepcopy(targets)
    if blinded_targets is not None and primary_close:
        scale = 100.0 / primary_close
        for field in (
            "entry", "stop_loss", "take_profit_1", "take_profit_2",
            "risk_amount", "reward_1", "reward_2",
        ):
            setattr(blinded_targets, field, getattr(blinded_targets, field) * scale)
    return build_llm_context(blinded, blinded_targets)


def _load_cache(path: Path) -> dict[str, dict]:
    cached = {}
    if not path.exists():
        return cached
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cached[row["id"]] = row
    return cached


def _append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _query_ollama(base_url: str, model: str, prompt: str, timeout: int, *,
                  think: bool = False, num_predict: int = 500) -> dict:
    started = time.time()
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious cryptocurrency trading gate. Use only the "
                        "supplied frozen technical data. Respond only with valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": think,
            "format": JSON_SCHEMA,
            "options": {"temperature": 0, "seed": 7, "num_predict": num_predict},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload["message"]
    raw = message["content"]
    parsed = _parse_llm_response(raw, model)
    return {
        "decision": parsed.decision,
        "confidence": parsed.confidence,
        "reasoning": parsed.reasoning,
        "parse_error": parsed.parse_error,
        "raw_response": raw,
        "thinking": message.get("thinking", ""),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
        "latency_s": round(time.time() - started, 3),
    }


def _query_openai(base_url: str, model: str, prompt: str, timeout: int, *,
                  think: bool = False, num_predict: int = 500) -> dict:
    """Query an OpenAI-compatible endpoint (e.g. vLLM) for one gate decision.

    vLLM does not expose Ollama's ``think``/``format`` fields. Native thinking is
    requested via ``chat_template_kwargs={"enable_thinking": True}`` and we DO NOT
    constrain output with a JSON schema — guided decoding would force JSON from the
    first token and leave no room to reason. Instead the model thinks then emits a
    JSON object, which ``_parse_llm_response`` extracts from the free-form content
    (so ``num_predict`` must be large enough for thinking + JSON, e.g. ~2000).
    """
    started = time.time()
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cautious cryptocurrency trading gate. Use only the "
                    "supplied frozen technical data. After reasoning, respond with a "
                    "single JSON object with keys: decision (LONG/SHORT/WAIT), "
                    "confidence (1-100), reasoning, entry, stop_loss, take_profit_1, "
                    "take_profit_2."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0,
        "seed": 7,
        "max_tokens": num_predict,
        # Explicit both ways: a no-think run must actively DISABLE reasoning, not
        # fall back to the model's default template (Qwen3.6 defaults to thinking).
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    message = payload["choices"][0]["message"]
    raw = message.get("content") or ""
    parsed = _parse_llm_response(raw, model)
    usage = payload.get("usage") or {}
    return {
        "decision": parsed.decision,
        "confidence": parsed.confidence,
        "reasoning": parsed.reasoning,
        "parse_error": parsed.parse_error,
        "raw_response": raw,
        "thinking": message.get("reasoning_content", "") or "",
        "prompt_eval_count": usage.get("prompt_tokens"),
        "eval_count": usage.get("completion_tokens"),
        "latency_s": round(time.time() - started, 3),
    }


def _query_backend(args, prompt: str) -> dict:
    """Dispatch one gate query to the configured backend."""
    if args.backend == "openai":
        return _query_openai(
            args.url, args.model, prompt, args.timeout,
            think=args.think, num_predict=args.num_predict,
        )
    return _query_ollama(
        args.url, args.model, prompt, args.timeout,
        think=args.think, num_predict=args.num_predict,
    )


def _collect_candidates() -> list[dict]:
    candidates = []
    cfg = driver.build_config({})
    for fold, start, end in driver.HALF_FOLDS:
        def collect(timestamp, result, targets, fold=fold):
            expected = "LONG" if result.direction == Direction.BULLISH else "SHORT"
            prompt = _build_blinded_context(result, targets)
            candidates.append({
                "fold": fold,
                "timestamp": timestamp,
                "expected": expected,
                "score": round(result.raw_score, 6),
                "prompt": prompt,
            })
            return True

        driver.fb.simulate(
            driver._PRE, cfg, start, end, slip=0.0002, model_liquidation=True,
            funding_by_pos=driver._FUND, marginal_gate=collect,
        )
    return candidates


def _stratified_sample(candidates: list[dict], size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_fold = {name: [] for name, _, _ in driver.HALF_FOLDS}
    for item in candidates:
        by_fold[item["fold"]].append(item)
    for items in by_fold.values():
        rng.shuffle(items)

    picked = []
    while len(picked) < min(size, len(candidates)):
        progressed = False
        for name, _, _ in driver.HALF_FOLDS:
            if by_fold[name] and len(picked) < size:
                picked.append(by_fold[name].pop())
                progressed = True
        if not progressed:
            break
    return picked


def _growth_ratio(gated: dict, baseline: dict) -> float:
    return gated["compound_x"] / baseline["compound_x"] if baseline["compound_x"] else 0.0


def _fold_for_timestamp(timestamp: str) -> str:
    date = timestamp[:10]
    for name, start, end in driver.HALF_FOLDS:
        if start <= date <= end:
            return name
    raise ValueError(f"Timestamp outside configured half-year folds: {timestamp}")


def _query_items(items: list[dict], args, cached: dict[str, dict], label: str = "") -> list[dict]:
    rows = []
    for number, item in enumerate(items, 1):
        key = _prompt_id(
            args.model, item["timestamp"], item["prompt"],
            think=args.think, num_predict=args.num_predict,
        )
        prefix = f"{label} " if label else ""
        if key not in cached:
            print(f"{prefix}[{number}/{len(items)}] {item['fold']} {item['expected']} "
                  f"score={item['score']:+.2f}", flush=True)
            try:
                answer = _query_backend(args, item["prompt"])
            except Exception as exc:
                answer = {
                    "decision": "WAIT", "confidence": 0, "reasoning": "",
                    "parse_error": f"{type(exc).__name__}: {exc}", "raw_response": "",
                    "thinking": "", "prompt_eval_count": None, "eval_count": None,
                    "latency_s": 0,
                }
            row = {
                "id": key, "model": args.model, "fold": item["fold"],
                "timestamp": item["timestamp"], "expected": item["expected"],
                "score": item["score"], "think": args.think,
                "num_predict": args.num_predict, **answer,
            }
            _append_cache(args.cache, row)
            cached[key] = row
        else:
            print(f"{prefix}[{number}/{len(items)}] cached {item['fold']} {item['expected']}")
        rows.append(cached[key])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama",
                        help="ollama = /api/chat (native think/format); "
                             "openai = /v1/chat/completions (vLLM etc.)")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--think", action="store_true",
                        help="Enable the model's native thinking/reasoning mode")
    parser.add_argument("--num-predict", type=int, default=500,
                        help="Maximum generated tokens, including thinking")
    args = parser.parse_args()

    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.num_predict < 1:
        parser.error("--num-predict must be positive")

    driver.setup()
    candidates = _collect_candidates()
    sampled = _stratified_sample(candidates, args.sample_size, args.seed)
    cached = _load_cache(args.cache)
    print(f"Collected {len(candidates)} baseline marginal entries; sampled {len(sampled)}.")
    print(f"Model: {args.model} (single model; no consensus; think={args.think}; "
          f"num_predict={args.num_predict})")

    rows = _query_items(sampled, args, cached)

    # A full-baseline run needs fixed-point closure. Rejecting one baseline entry changes
    # slot/cooldown state and can expose a later marginal opportunity that was not eligible
    # on the baseline path. Query newly exposed cases, replay, and repeat until every
    # dynamically eligible marginal setup has a cached decision.
    full_policy = len(sampled) == len(candidates)
    if full_policy:
        for pass_number in range(1, 21):
            decisions = {row["timestamp"]: row for row in rows}
            discovered: dict[str, dict] = {}

            def discover_gate(timestamp, result, targets):
                row = decisions.get(timestamp)
                if row is not None:
                    expected = "LONG" if result.direction == Direction.BULLISH else "SHORT"
                    return not row.get("parse_error") and row["decision"] == expected
                expected = "LONG" if result.direction == Direction.BULLISH else "SHORT"
                discovered[timestamp] = {
                    "fold": _fold_for_timestamp(timestamp), "timestamp": timestamp,
                    "expected": expected, "score": round(result.raw_score, 6),
                    "prompt": _build_blinded_context(result, targets),
                }
                return True

            driver.evaluate({}, folds=driver.HALF_FOLDS, slip=0.0002, funding=True,
                            marginal_gate=discover_gate)
            if not discovered:
                print(f"Fixed-point closure reached after {pass_number - 1} expansion pass(es).")
                break
            new_items = _stratified_sample(list(discovered.values()), len(discovered), args.seed)
            print(f"Closure pass {pass_number}: querying {len(new_items)} newly exposed entries.")
            rows.extend(_query_items(new_items, args, cached, label=f"closure-{pass_number}"))
        else:
            raise RuntimeError("LLM gate did not reach fixed-point closure in 20 passes")

    decisions = {row["timestamp"]: row for row in rows}

    def gate(timestamp, result, targets):
        row = decisions.get(timestamp)
        if row is None:
            return not full_policy  # sampled pilot keeps unreviewed baseline behavior
        expected = "LONG" if result.direction == Direction.BULLISH else "SHORT"
        return not row.get("parse_error") and row["decision"] == expected

    print("\nDecisions:", dict(Counter(row["decision"] for row in rows)))
    accepted = sum(
        not row.get("parse_error") and row["decision"] == row["expected"]
        for row in rows
    )
    print("Matching/accepted:", accepted, "/", len(rows))

    for label, folds in (("TRAIN", driver.TRAIN_FOLDS), ("TEST", driver.TEST_FOLDS),
                         ("ALL", driver.HALF_FOLDS)):
        baseline = driver.evaluate({}, folds=folds, slip=0.0002, funding=True)
        gated = driver.evaluate({}, folds=folds, slip=0.0002, funding=True,
                                 marginal_gate=gate)
        print(f"\n{label} baseline:\n{driver.fmt(baseline)}")
        policy_label = "full" if full_policy else "sparse"
        print(f"{label} {policy_label} LLM gate:\n{driver.fmt(gated)}")
        print(f"{label} growth ratio gated/baseline: {_growth_ratio(gated, baseline):.4f}x")

    failures = [row for row in rows if row.get("parse_error")]
    avg_latency = sum(row.get("latency_s", 0) for row in rows) / len(rows)
    print(f"\nRun complete: {len(rows)} responses, {len(failures)} failures, "
          f"mean latency {avg_latency:.1f}s. Cache: {args.cache}")
    if not full_policy:
        print("This is a small sparse-intervention pilot; it is not sufficient to ship a policy.")


if __name__ == "__main__":
    main()
