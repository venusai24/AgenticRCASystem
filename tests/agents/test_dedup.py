"""Tests for the hypothesis-dedup agent (#10)."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from agentic_rca.ledger import Ledger
from agentic_rca.llm import ScriptedLLM, LLMResponse, LLMError
from agentic_rca.agents.dedup import assign_cluster, _exact_match, _next_cluster_id

def test_dedup_exact_match(tmp_path: Path):
    """Exact match skips the LLM and assigns the same cluster."""
    ledger = Ledger.create(tmp_path / "run.db")
    client = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="", output_tokens=0))
    
    # Create first hypothesis
    short_id = ledger.apply("hypothesis_create", {
        "statement": "The database is down",
        "branches": []
    }, actor="lead")["id"]
    h1_id = f"{ledger.run_id}/{short_id}"
    
    # We assign it manually for testing since the dispatcher isn't running here
    res1 = assign_cluster(ledger, client, h1_id, 1)
    assert res1.cluster_id == "cl001"
    assert not res1.unchecked
    
    # Second hypothesis exact match
    short_id2 = ledger.apply("hypothesis_create", {
        "statement": "The database is down",
        "branches": []
    }, actor="lead")["id"]
    h2_id = f"{ledger.run_id}/{short_id2}"
    
    res2 = assign_cluster(ledger, client, h2_id, 3)
    assert res2.cluster_id == "cl001"
    assert res2.rationale == "exact match"


def test_dedup_llm_judge(tmp_path: Path):
    """LLM judges a rewording as the same cluster, and a new one as new."""
    ledger = Ledger.create(tmp_path / "run.db")
    
    # 1. First hypothesis
    short_id1 = ledger.apply("hypothesis_create", {
        "statement": "The db is down",
        "branches": []
    }, actor="lead")["id"]
    h1_id = f"{ledger.run_id}/{short_id1}"
    
    client = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="", output_tokens=0))
    res1 = assign_cluster(ledger, client, h1_id, 1)
    assert res1.cluster_id == "cl001"
    
    # 2. Second hypothesis - rewording
    short_id2 = ledger.apply("hypothesis_create", {
        "statement": "Database connection failed",
        "branches": []
    }, actor="lead")["id"]
    h2_id = f"{ledger.run_id}/{short_id2}"
    
    client = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text=json.dumps({"rationale": "same issue", "decision": "cl001"}), output_tokens=10))
    res2 = assign_cluster(ledger, client, h2_id, 3)
    assert res2.cluster_id == "cl001"
    
    # 3. Third hypothesis - new issue
    short_id3 = ledger.apply("hypothesis_create", {
        "statement": "Disk is full",
        "branches": []
    }, actor="lead")["id"]
    h3_id = f"{ledger.run_id}/{short_id3}"
    
    client = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text=json.dumps({"rationale": "different issue", "decision": "new"}), output_tokens=10))
    res3 = assign_cluster(ledger, client, h3_id, 5)
    assert res3.cluster_id == "cl002"


def test_dedup_llm_failure(tmp_path: Path):
    """LLM failure results in a new cluster marked unchecked."""
    ledger = Ledger.create(tmp_path / "run.db")
    
    short_id1 = ledger.apply("hypothesis_create", {
        "statement": "The db is down",
        "branches": []
    }, actor="lead")["id"]
    h1_id = f"{ledger.run_id}/{short_id1}"
    
    client = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="", output_tokens=0))
    assign_cluster(ledger, client, h1_id, 1)
    
    short_id2 = ledger.apply("hypothesis_create", {
        "statement": "Database connection failed",
        "branches": []
    }, actor="lead")["id"]
    h2_id = f"{ledger.run_id}/{short_id2}"
    
    def raise_err(*args, **kwargs):
        raise LLMError("API offline")
        
    client = ScriptedLLM(fn=raise_err)
    res2 = assign_cluster(ledger, client, h2_id, 3)
    assert res2.cluster_id == "cl002"
    assert res2.unchecked is True
    assert "API offline" in res2.rationale


def test_dedup_retry_on_invalid_schema(tmp_path: Path):
    """Dedup retries once if the LLM output is invalid."""
    ledger = Ledger.create(tmp_path / "run.db")
    
    short_id1 = ledger.apply("hypothesis_create", {
        "statement": "The db is down",
        "branches": []
    }, actor="lead")["id"]
    h1_id = f"{ledger.run_id}/{short_id1}"
    
    assign_cluster(ledger, ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="", output_tokens=0)), h1_id, 1)
    
    short_id2 = ledger.apply("hypothesis_create", {
        "statement": "Database connection failed",
        "branches": []
    }, actor="lead")["id"]
    h2_id = f"{ledger.run_id}/{short_id2}"
    
    responses = [
        LLMResponse(text=json.dumps({"rationale": "same", "decision": "cl005"}), output_tokens=10),
        LLMResponse(text=json.dumps({"rationale": "same", "decision": "cl001"}), output_tokens=10)
    ]
    client = ScriptedLLM(fn=lambda r, m, t: responses.pop(0))
    res2 = assign_cluster(ledger, client, h2_id, 3)
    assert res2.cluster_id == "cl001"

def test_dedup_input_constraints(tmp_path: Path):
    """AC4: Input constraints (no origin, status, evidence in judge payload)."""
    ledger = Ledger.create(tmp_path / "run.db")
    
    short_id1 = ledger.apply("hypothesis_create", {
        "statement": "The db is down",
        "branches": [],
    }, actor="lead")["id"]
    h1_id = f"{ledger.run_id}/{short_id1}"
    q_id = f"{ledger.run_id}/q123"
    ledger.con.execute("INSERT INTO queries (run_id, query_id, seq, actor, step, tool, args_json, args_sha256, input_query_ids, sql_texts, status, row_count, result_digest, empty_because) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [ledger.run_id, q_id, 1, "lead", 1, "x", "{}", "", "[]", "[]", "ok", 0, "z", "no_matches"])
    ev_id = ledger.apply("evidence_record", {"claim": "c", "query_id": q_id, "evidence_kind": "occurrence"}, actor="lead")["id"]
    ledger.apply("hypothesis_link_evidence", {"hypothesis_id": h1_id, "evidence_id": ev_id, "stance": "contradicts"}, actor="lead")
    ledger.apply("hypothesis_set_status", {"hypothesis_id": h1_id, "status": "refuted", "basis": {"evidence_ids": [ev_id], "reason": "test reason"}}, actor="lead")
    
    short_id2 = ledger.apply("hypothesis_create", {
        "statement": "Database connection failed",
        "branches": []
    }, actor="lead")["id"]
    h2_id = f"{ledger.run_id}/{short_id2}"
    
    # We need to capture the exact string sent to the LLM
    captured_messages = []
    def capture_fn(role, messages, tools):
        captured_messages.extend(messages)
        return LLMResponse(text=json.dumps({"rationale": "same", "decision": "cl001"}), output_tokens=10)
        
    assign_cluster(ledger, ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="", output_tokens=0)), h1_id, 1)
    
    client = ScriptedLLM(fn=capture_fn)
    assign_cluster(ledger, client, h2_id, ledger.state().hypotheses[h2_id]["content_seq"])
    
    # Check the user message sent to the judge
    user_msg = next((m["content"] for m in captured_messages if m["role"] == "user"), "")
    
    # Assert constraints
    assert "refuted" not in user_msg.lower()
    assert "lead" not in user_msg.lower()
    # It SHOULD contain the statement
    assert "The db is down" in user_msg
    assert "Database connection failed" in user_msg


def test_dedup_on_hypothesis_write_updates(tmp_path: Path):
    """AC5: Content updates re-judge, evidence updates don't."""
    from agentic_rca.agents.dedup import on_hypothesis_write
    
    ledger = Ledger.create(tmp_path / "run.db")
    
    short_id1 = ledger.apply("hypothesis_create", {
        "statement": "The db is down",
        "branches": [],
    }, actor="lead")["id"]
    h1_id = f"{ledger.run_id}/{short_id1}"
    
    call_count = [0]
    def count_calls(role, messages, tools):
        call_count[0] += 1
        return LLMResponse(text=json.dumps({"rationale": "new", "decision": "new"}), output_tokens=10)
        
    client = ScriptedLLM(fn=count_calls)
    
    # 1. Create triggers it (though through on_hypothesis_write, wait: we must get the seq)
    on_hypothesis_write(ledger, client, 1) # hypothesis_create is seq 1
    assert call_count[0] == 0 # 0 because no_candidates skips LLM call
    
    # Let's add a second one to ensure LLM is called
    short_id2 = ledger.apply("hypothesis_create", {
        "statement": "Disk is full",
        "branches": []
    }, actor="lead")["id"]
    h2_id = f"{ledger.run_id}/{short_id2}"
    
    on_hypothesis_write(ledger, client, 3) # seq 3
    assert call_count[0] == 1 # LLM called for the second one
    
    # 3. Evidence update (doesn't trigger)
    q_id = f"{ledger.run_id}/q124"
    ledger.con.execute("INSERT INTO queries (run_id, query_id, seq, actor, step, tool, args_json, args_sha256, input_query_ids, sql_texts, status, row_count, result_digest, empty_because) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [ledger.run_id, q_id, 2, "lead", 1, "x", "{}", "", "[]", "[]", "ok", 0, "z", "no_matches"])
    ev_id = ledger.apply("evidence_record", {
        "claim": "Logs show disk full",
        "query_id": q_id,
        "evidence_kind": "occurrence"
    }, actor="lead")["id"]
    ledger.apply("hypothesis_link_evidence", {
        "hypothesis_id": h2_id,
        "evidence_id": ev_id,
        "stance": "supports"
    }, actor="lead")
    on_hypothesis_write(ledger, client, ledger.state().last_seq) # link_evidence
    # The ev_type is hypothesis_link_evidence which is ignored
    assert call_count[0] == 1
    
    # 4. Content update (statement changes) -> triggers
    ledger.apply("hypothesis_revise", {
        "hypothesis_id": h2_id,
        "statement": "Disk is 100% full",
    }, actor="lead")
    on_hypothesis_write(ledger, client, ledger.state().last_seq) # revise
    assert call_count[0] == 2
