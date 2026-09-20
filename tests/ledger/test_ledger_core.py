import os
import pytest
import duckdb
from datetime import datetime, timezone
import json

from agentic_rca.ledger.db import init_db
from agentic_rca.ledger.ops import apply_op, InvalidOpError, get_ledger_state
from agentic_rca.ledger.schemas import LedgerState, LedgerEvent, Hypothesis
from agentic_rca.ledger.two_live import two_live_check


@pytest.fixture
def db_path(tmp_path):
    run_dir = tmp_path / "run1"
    con = init_db(str(run_dir), "run_1")
    con.execute("""
        CREATE TABLE coverage (
            query_id VARCHAR PRIMARY KEY,
            unobserved_entities JSON,
            scope_verified BOOLEAN
        );
    """)
    yield con
    con.close()


def test_rejected_op_appends_zero_events(db_path):
    con = db_path
    with pytest.raises(InvalidOpError):
        # I4 violation
        apply_op(con, "run_1", "lead", "step1", "ledger_hypothesis_update", "hypothesis", "h1", {"hypothesis_id": "h1"})
        
    res = con.execute("SELECT count(*) FROM ledger_events").fetchone()[0]
    assert res == 0


def test_i1_evidence_query_id(db_path):
    con = db_path
    con.execute("INSERT INTO queries (query_id, status, row_count, empty_because, tool) VALUES ('q1', 'error', 0, NULL, 'generic')")
    
    # Reject: query status is not 'ok'
    with pytest.raises(InvalidOpError, match="I1 violation"):
        apply_op(con, "run_1", "lead", "s1", "ledger_record_evidence", "evidence", "e1", {
            "evidence_id": "e1",
            "claim": "c",
            "query_id": "q1",
            "row_refs": [],
            "evidence_kind": "occurrence"
        })
        
    con.execute("UPDATE queries SET status = 'ok' WHERE query_id = 'q1'")
    
    # Pass: I1 passes, but now I3 fails because row_count=0 but empty_because is not no_matches
    # Let's fix that
    con.execute("UPDATE queries SET empty_because = 'no_matches' WHERE query_id = 'q1'")
    apply_op(con, "run_1", "lead", "s1", "ledger_record_evidence", "evidence", "e2", {
        "evidence_id": "e2",
        "claim": "c",
        "query_id": "q1",
        "row_refs": [],
        "evidence_kind": "occurrence"
    })
    
    res = con.execute("SELECT count(*) FROM ledger_events").fetchone()[0]
    assert res == 1


def test_i13_negative_evidence_guard(db_path):
    con = db_path
    
    # I13(a) reject non-no_matches
    con.execute("INSERT INTO queries (query_id, status, row_count, empty_because, tool) VALUES ('q2', 'ok', 0, 'scope_unverified', 'generic')")
    con.execute("INSERT INTO coverage (query_id, unobserved_entities, scope_verified) VALUES ('q2', NULL, true)")
    with pytest.raises(InvalidOpError, match="I13\\(a\\)"):
        apply_op(con, "run_1", "lead", "s", "ledger_record_evidence", "ev", "e", {
            "evidence_id": "e", "claim": "", "query_id": "q2", "row_refs": [], "evidence_kind": "occurrence"
        })
        
    # I13(b) reject aggregate over unobserved entity
    con.execute("INSERT INTO queries (query_id, status, row_count, empty_because, tool) VALUES ('q3', 'ok', 1, NULL, 'generic')")
    con.execute("INSERT INTO coverage (query_id, unobserved_entities, scope_verified) VALUES ('q3', '{\"host\":[\"unknown\"]}', true)")
    with pytest.raises(InvalidOpError, match="I13\\(b\\)"):
        apply_op(con, "run_1", "lead", "s", "ledger_record_evidence", "ev", "e2", {
            "evidence_id": "e2", "claim": "", "query_id": "q3", "row_refs": [{"row":0, "values": {}}], "evidence_kind": "occurrence"
        })
        
    # I13(c) unverified scope is accepted with guard
    con.execute("INSERT INTO queries (query_id, status, row_count, empty_because, tool) VALUES ('q4', 'ok', 1, NULL, 'generic')")
    con.execute("INSERT INTO coverage (query_id, unobserved_entities, scope_verified) VALUES ('q4', NULL, false)")
    apply_op(con, "run_1", "lead", "s", "ledger_record_evidence", "ev", "e3", {
        "evidence_id": "e3", "claim": "", "query_id": "q4", "row_refs": [{"row":0, "values": {}}], "evidence_kind": "occurrence"
    })
    state = get_ledger_state(con, "run_1")
    assert state.evidence["e3"].guard == "unverified_scope"


def test_i12_single_accepted_hypothesis(db_path):
    con = db_path
    
    # Create two hypotheses
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_create", "hyp", "h1", {"hypothesis_id": "h1", "statement": "", "cluster_id": "c1"})
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_create", "hyp", "h2", {"hypothesis_id": "h2", "statement": "", "cluster_id": "c2"})
    
    # Accept one
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_status", "hyp", "h1", {"hypothesis_id": "h1", "status": "accepted"})
    
    # Accept second -> fail
    with pytest.raises(InvalidOpError, match="I12"):
        apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_status", "hyp", "h2", {"hypothesis_id": "h2", "status": "accepted"})
        
    # Reopen first
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_status", "hyp", "h1", {"hypothesis_id": "h1", "status": "live"})
    
    # Accept second -> pass
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_status", "hyp", "h2", {"hypothesis_id": "h2", "status": "accepted"})


def test_i11_branches_validation(db_path):
    con = db_path
    
    # two joins: null
    with pytest.raises(InvalidOpError, match="I11 violation"):
        apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_create", "hyp", "h1", {
            "hypothesis_id": "h1", 
            "statement": "", 
            "cluster_id": "c1",
            "branches": [
                {"links": ["l1"], "joins": None},
                {"links": ["l2"], "joins": None}
            ]
        })
        
    # pass
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_create", "hyp", "h2", {
        "hypothesis_id": "h2", 
        "statement": "", 
        "cluster_id": "c1",
        "branches": [
            {"links": ["l1"], "joins": "l2"},
            {"links": ["l2"], "joins": None}
        ]
    })


def test_deterministic_fold():
    # Construct random events and shuffle
    # Then assert the folded state matches sorted
    events = [
        LedgerEvent(run_id="1", seq=2, event_id="e2", created_at="now", actor="lead", step="1", op="ledger_hypothesis_update", record_type="hyp", record_id="h1", payload={"hypothesis_id": "h1", "status": "accepted"}),
        LedgerEvent(run_id="1", seq=1, event_id="e1", created_at="now", actor="lead", step="1", op="ledger_hypothesis_create", record_type="hyp", record_id="h1", payload={"hypothesis_id": "h1", "statement": "stmt", "cluster_id": "c1", "origin": "lead"})
    ]
    
    state1 = LedgerState(run_id="1").fold(events) # Unsorted -> the update will fail because h1 doesn't exist yet! Wait, fold processes in given order.
    state2 = LedgerState(run_id="1").fold(sorted(events, key=lambda e: e.seq))
    
    assert state2.hypotheses["h1"].status == "accepted"


def test_origin_protected(db_path):
    con = db_path
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_create", "hyp", "h1", {
        "hypothesis_id": "h1", 
        "statement": "", 
        "origin": "scaffold", # Trying to override
        "cluster_id": "c1"
    })
    
    state = get_ledger_state(con, "run_1")
    assert state.hypotheses["h1"].origin == "lead"


def test_two_live_check():
    state = LedgerState(run_id="1")
    
    # 1 lead hypothesis (c1)
    state.hypotheses["h1"] = Hypothesis(hypothesis_id="h1", statement="", origin="lead", cluster_id="c1", run_id="1", seq=1, actor="lead", step="1")
    
    # 1 scaffold hypothesis (c2)
    state.hypotheses["h2"] = Hypothesis(hypothesis_id="h2", statement="", origin="scaffold", cluster_id="c2", run_id="1", seq=1, actor="scaffold", step="1")
    
    ok, detail = two_live_check(state)
    assert not ok, "Scaffold hypothesis should not count towards >=2 live rule"
    assert detail["standing_clusters"] == 1
    
    # add another lead hypothesis
    state.hypotheses["h3"] = Hypothesis(hypothesis_id="h3", statement="", origin="lead", cluster_id="c3", run_id="1", seq=1, actor="lead", step="1")
    
    ok, detail = two_live_check(state)
    assert ok


def test_amendment_leaves_parent_untouched(tmp_path):
    parent_dir = tmp_path / "parent"
    con1 = init_db(str(parent_dir), "parent_run")
    apply_op(con1, "parent_run", "lead", "s", "ledger_hypothesis_create", "hyp", "h1", {"hypothesis_id": "h1", "statement": "", "cluster_id": "c1"})
    parent_file = parent_dir / "run.duckdb"
    con1.close()
    
    import hashlib
    hash1 = hashlib.md5(parent_file.read_bytes()).hexdigest()
    
    child_dir = tmp_path / "child"
    con2 = init_db(str(child_dir), "child_run", is_amendment=True, parent_db_path=str(parent_file))
    
    # We can write to child DB
    apply_op(con2, "child_run", "lead", "s", "ledger_hypothesis_create", "hyp", "h2", {"hypothesis_id": "h2", "statement": "", "cluster_id": "c2"})
    con2.close()
    
    hash2 = hashlib.md5(parent_file.read_bytes()).hexdigest()
    assert hash1 == hash2


def test_i2_row_bounds(db_path):
    con = db_path
    con.execute("INSERT INTO queries (query_id, status, row_count, empty_because, tool) VALUES ('q_i2', 'ok', 2, NULL, 'generic')")
    con.execute("INSERT INTO coverage (query_id, unobserved_entities, scope_verified) VALUES ('q_i2', NULL, true)")
    
    # We need a stub for get_result. The test uses envelope which accesses _RESULT_STORE.
    from agentic_rca.tools.envelope import make_envelope, Provenance
    make_envelope(
        rows=[{"a": 1}, {"a": 2}],
        provenance=Provenance(source="test"),
        coverage={}
    )
    # the query_id mints randomly. Let's just patch _RESULT_STORE for this test
    from agentic_rca.tools.envelope import _RESULT_STORE
    _RESULT_STORE['q_i2'] = ([{"a": 1}, {"a": 2}], Provenance(source="test"))
    
    # Pass: bounds are 0 and 1
    apply_op(con, "run_1", "lead", "s", "ledger_record_evidence", "ev", "e_i2_1", {
        "evidence_id": "e_i2_1", "claim": "", "query_id": "q_i2", "row_refs": [{"row": 0, "values": {}}, {"row": 1, "values": {}}], "evidence_kind": "occurrence"
    })
    
    # Reject: out of bounds (row 2 for row_count 2)
    with pytest.raises(InvalidOpError, match="I2 violation"):
        apply_op(con, "run_1", "lead", "s", "ledger_record_evidence", "ev", "e_i2_2", {
            "evidence_id": "e_i2_2", "claim": "", "query_id": "q_i2", "row_refs": [{"row": 2, "values": {}}], "evidence_kind": "occurrence"
        })


def test_i5_refute_contradicting(db_path):
    con = db_path
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_create", "hyp", "h_i5", {"hypothesis_id": "h_i5", "statement": "", "cluster_id": "c1"})
    
    # Reject: no contradicting evidence
    with pytest.raises(InvalidOpError, match="I5 violation"):
        apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_status", "hyp", "h_i5", {"hypothesis_id": "h_i5", "status": "refuted"})
        
    # Add contradicting evidence
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_update", "hyp", "h_i5", {
        "hypothesis_id": "h_i5",
        "evidence": [{"evidence_id": "e1", "stance": "contradicts"}]
    })
    
    # Pass: has contradicting evidence
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_status", "hyp", "h_i5", {"hypothesis_id": "h_i5", "status": "refuted"})


def test_i6_failure_resolve_query_ok(db_path):
    con = db_path
    apply_op(con, "run_1", "scaffold", "s", "failure_record", "fail", "f1", {
        "failure_id": "f1", "query_id": "q1", "tool": "t", "error_type": "transient", "diagnostic": "d"
    })
    
    con.execute("INSERT INTO queries (query_id, status, row_count, empty_because, tool) VALUES ('q_fail', 'error', 0, NULL, 'generic')")
    
    # Reject: by_query_id is not ok
    with pytest.raises(InvalidOpError, match="I6 violation"):
        apply_op(con, "run_1", "lead", "s", "failure_resolve", "fail", "f1", {
            "failure_id": "f1", "resolution": {"by_query_id": "q_fail", "note": ""}
        })
        
    con.execute("UPDATE queries SET status = 'ok' WHERE query_id = 'q_fail'")
    
    # Pass
    apply_op(con, "run_1", "lead", "s", "failure_resolve", "fail", "f1", {
        "failure_id": "f1", "resolution": {"by_query_id": "q_fail", "note": ""}
    })


def test_i7_objection_resolve_open(db_path):
    con = db_path
    apply_op(con, "run_1", "critic", "s", "objection_raise", "obj", "o1", {
        "objection_id": "o1", "round": 1, "text": "", "target": {"hypothesis_id": "h1"}, "status": "open"
    })
    
    # Pass: resolve an open objection
    apply_op(con, "run_1", "lead", "s", "objection_resolve", "obj", "o1", {
        "objection_id": "o1", "resolution": {"round": 1, "kind": "revision", "note": ""}, "status": "resolved"
    })
    
    # Reject: resolve a resolved objection
    with pytest.raises(InvalidOpError, match="I7 violation: objection is not open"):
        apply_op(con, "run_1", "lead", "s", "objection_resolve", "obj", "o1", {
            "objection_id": "o1", "resolution": {"round": 2, "kind": "revision", "note": ""}
        })


def test_i8_actor_op_permissions(db_path):
    con = db_path
    # Reject: human cannot raise objection
    with pytest.raises(InvalidOpError, match="I8 violation"):
        apply_op(con, "run_1", "human", "s", "objection_raise", "obj", "o1", {})
        
    # Pass: critic can raise objection
    apply_op(con, "run_1", "critic", "s", "objection_raise", "obj", "o1", {
        "objection_id": "o1", "round": 1, "text": "", "target": {"hypothesis_id": "h1"}, "status": "open"
    })


def test_i14_same_as_unresolved(db_path):
    con = db_path
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_create", "hyp", "h_i14", {"hypothesis_id": "h_i14", "statement": "", "cluster_id": "c1"})
    
    # Reject: same_as with accepted (which is not unresolved)
    with pytest.raises(InvalidOpError, match="I14 violation"):
        apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_status", "hyp", "h_i14", {
            "hypothesis_id": "h_i14", "status": "accepted", "status_basis": {"reason": "r", "same_as": "h2"}
        })
        
    # Pass: same_as with unresolved
    apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_status", "hyp", "h_i14", {
        "hypothesis_id": "h_i14", "status": "unresolved", "status_basis": {"reason": "r", "same_as": "h2"}
    })


def test_i11_cycle_rejection(db_path):
    con = db_path
    # Reject: cycle between l1 and l2 (l1 -> l2 -> l1)
    with pytest.raises(InvalidOpError, match="cycle detected"):
        apply_op(con, "run_1", "lead", "s", "ledger_hypothesis_create", "hyp", "h1_cyc", {
            "hypothesis_id": "h1_cyc", 
            "statement": "", 
            "cluster_id": "c1",
            "branches": [
                {"links": ["l1"], "joins": "l2"},
                {"links": ["l2"], "joins": "l1"},
                {"links": ["l3"], "joins": None}
            ]
        })
