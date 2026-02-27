"""
Tests for the OpenWebUI automation client.
Covers: response parsing, consensus building.
"""

import json

import pytest

from llm_trading_bot.openwebui_client import (
    ConsensusResult,
    LLMResponse,
    _parse_llm_response,
    build_consensus,
)


class TestParseResponse:
    def test_clean_json(self):
        raw = json.dumps({
            "decision": "LONG",
            "confidence": 75,
            "reasoning": "Strong trend alignment",
            "entry": 50000,
            "stop_loss": 49000,
            "take_profit_1": 52000,
            "take_profit_2": 54000,
        })
        resp = _parse_llm_response(raw, "test-model")
        assert resp.decision == "LONG"
        assert resp.confidence == 75
        assert resp.entry == 50000
        assert resp.parse_error is None

    def test_markdown_code_block(self):
        raw = """Here's my analysis:\n\n```json\n{"decision": "SHORT", "confidence": 60, "reasoning": "Bearish"}\n```"""
        resp = _parse_llm_response(raw, "test-model")
        assert resp.decision == "SHORT"
        assert resp.confidence == 60
        assert resp.parse_error is None

    def test_json_embedded_in_text(self):
        raw = """Based on the analysis, I recommend:
        {"decision": "WAIT", "confidence": 30, "reasoning": "Unclear"}
        Hope that helps!"""
        resp = _parse_llm_response(raw, "test-model")
        assert resp.decision == "WAIT"
        assert resp.confidence == 30

    def test_no_json_returns_error(self):
        raw = "I think you should buy some bitcoin!"
        resp = _parse_llm_response(raw, "test-model")
        assert resp.decision == "WAIT"
        assert resp.parse_error is not None

    def test_invalid_decision_defaults_to_wait(self):
        raw = json.dumps({"decision": "HODL", "confidence": 50, "reasoning": "test"})
        resp = _parse_llm_response(raw, "test-model")
        assert resp.decision == "WAIT"

    def test_confidence_clamped(self):
        raw = json.dumps({"decision": "LONG", "confidence": 150, "reasoning": "test"})
        resp = _parse_llm_response(raw, "test-model")
        assert resp.confidence == 100

        raw2 = json.dumps({"decision": "LONG", "confidence": -20, "reasoning": "test"})
        resp2 = _parse_llm_response(raw2, "test-model")
        assert resp2.confidence == 0


class TestBuildConsensus:
    def test_unanimous_long(self):
        responses = [
            LLMResponse(model_id="m1", decision="LONG", confidence=80, reasoning="Bullish"),
            LLMResponse(model_id="m2", decision="LONG", confidence=75, reasoning="Bullish"),
            LLMResponse(model_id="m3", decision="LONG", confidence=70, reasoning="Bullish"),
        ]
        result = build_consensus(responses)
        assert result.decision == "LONG"
        assert result.agreement_pct == 100

    def test_majority_wins(self):
        responses = [
            LLMResponse(model_id="m1", decision="SHORT", confidence=80, reasoning="Bear"),
            LLMResponse(model_id="m2", decision="SHORT", confidence=75, reasoning="Bear"),
            LLMResponse(model_id="m3", decision="LONG", confidence=70, reasoning="Bull"),
        ]
        result = build_consensus(responses)
        assert result.decision == "SHORT"
        assert abs(result.agreement_pct - 66.67) < 1

    def test_no_majority_defaults_to_wait(self):
        responses = [
            LLMResponse(model_id="m1", decision="LONG", confidence=80, reasoning="Bull"),
            LLMResponse(model_id="m2", decision="SHORT", confidence=75, reasoning="Bear"),
            LLMResponse(model_id="m3", decision="WAIT", confidence=50, reasoning="Unclear"),
        ]
        result = build_consensus(responses)
        # Each has 33% — no majority for action
        assert result.decision == "WAIT"

    def test_parse_errors_excluded(self):
        responses = [
            LLMResponse(model_id="m1", decision="LONG", confidence=80, reasoning="Bull"),
            LLMResponse(model_id="m2", decision="WAIT", confidence=0, reasoning="", parse_error="broken"),
            LLMResponse(model_id="m3", decision="LONG", confidence=70, reasoning="Bull"),
        ]
        result = build_consensus(responses)
        assert result.decision == "LONG"
        assert result.agreement_pct == 100  # Only valid responses counted

    def test_all_failures_returns_wait(self):
        responses = [
            LLMResponse(model_id="m1", decision="WAIT", confidence=0, reasoning="", parse_error="err1"),
            LLMResponse(model_id="m2", decision="WAIT", confidence=0, reasoning="", parse_error="err2"),
        ]
        result = build_consensus(responses)
        assert result.decision == "WAIT"
        assert result.agreement_pct == 0
