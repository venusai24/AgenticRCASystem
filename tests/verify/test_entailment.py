"""Tests for the Stage 4 layer 2 entailment verifier."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from agentic_rca.llm.base import LLMResponse, ScriptedLLM
from agentic_rca.verify.entailment import verify
from tests.helpers import Builder


@pytest.fixture
def builder(tmp_path):
    return Builder(tmp_path)


@pytest.fixture
def ledger(builder):
    return builder.led


def _build_test_state(builder):
    """Build a hypothesis link and evidence in the ledger."""
    # Insert query using builder
    qshort = builder.q([{"value": 42.0}])
    
    # Create some evidence
    e = builder.led.apply(
        "evidence_record",
        {
            "claim": "c",
            "query_id": qshort,
            "evidence_kind": "occurrence",
            "row_refs": [{"row": 0, "values": {"value": 42.0}}],
        },
        actor="lead",
    )["id"]
    e_fq = builder.led.q(e)

    # Emit a deterministic 'pass' verdict for this evidence
    builder.led.apply("evidence_verdict", {"evidence_id": e_fq, "verdict": "pass", "detail": "ok"}, actor="scaffold")

    link = {
        "cause": "High CPU",
        "effect": "High Latency",
        "evidence": [{"evidence_id": e_fq, "role": "observation"}],
    }

    h = builder.led.apply(
        "hypothesis_create", {"statement": "A", "branches": [{"links": [link]}]}, actor="lead"
    )["id"]
    return f"{builder.led.q(h)}.l1"


def test_verify_passes_when_llm_says_yes(builder, ledger):
    lid = _build_test_state(builder)
    
    # Scripted LLM that says YES
    llm = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="YES\nThe data shows 42.0 which supports the claim.", output_tokens=10))
    
    # Mock runtime (not needed for fetching rows because we use row_refs directly)
    runtime = Mock()
    
    res = verify(ledger, runtime, llm)
    
    assert res["counts"]["pass"] == 1
    assert res["counts"]["fail"] == 0
    
    # Check the ledger state
    st = ledger.state()
    assert lid in st.entailments
    ent = st.entailments[lid]
    assert ent["verdict"] == "pass"
    assert "The data shows 42.0" in ent["rationale"]


def test_verify_fails_when_llm_says_no(builder, ledger):
    lid = _build_test_state(builder)
    
    # Scripted LLM that says NO
    llm = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="NO\nThis doesn't prove latency.", output_tokens=10))
    
    runtime = Mock()
    res = verify(ledger, runtime, llm)
    
    assert res["counts"]["fail"] == 1
    st = ledger.state()
    ent = st.entailments[lid]
    assert ent["verdict"] == "fail"
    assert "This doesn't prove" in ent["rationale"]


def test_verify_skips_link_if_evidence_failed_deterministic(builder, ledger):
    # Insert query using builder
    qshort = builder.q([], empty_because="no_matches")
    
    # Create evidence
    e = ledger.apply(
        "evidence_record",
        {"claim": "c", "query_id": qshort, "evidence_kind": "occurrence", "row_refs": []},
        actor="lead",
    )["id"]
    e_fq = ledger.q(e)

    # Emit a deterministic 'fail' verdict for this evidence
    ledger.apply("evidence_verdict", {"evidence_id": e_fq, "verdict": "fail", "detail": "missing data"}, actor="scaffold")

    link = {
        "cause": "High CPU",
        "effect": "High Latency",
        "evidence": [{"evidence_id": e_fq, "role": "observation"}],
    }

    ledger.apply(
        "hypothesis_create", {"statement": "A", "branches": [{"links": [link]}]}, actor="lead"
    )
    
    llm = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="YES", output_tokens=10)) # Should not be called
    runtime = Mock()
    
    res = verify(ledger, runtime, llm)
    assert res["counts"]["pass"] == 0
    assert res["counts"]["fail"] == 0
    assert not llm.seen
