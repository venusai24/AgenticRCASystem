import json
import duckdb
import pytest
from agentic_rca.ledger.rundb import RUN_DDL
from agentic_rca.report.render import render_report, render_coverage_human

@pytest.fixture
def run_dir(tmp_path):
    run_id = "r-test"
    db_path = tmp_path / "run.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(RUN_DDL)
    con.execute("INSERT INTO run (run_id) VALUES (?)", (run_id,))
    
    events = [
        {"op": "hypothesis_create", "payload": {"hypothesis_id": "h1", "statement": "CPU is high", "origin": "human"}},
        {"op": "hypothesis_create", "payload": {"hypothesis_id": "h2", "statement": "Memory is high", "origin": "scaffold"}},
        {"op": "hypothesis_set_status", "payload": {"hypothesis_id": "h2", "status": "refuted", "basis": {"evidence_ids": ["e1"]}}},
        {"op": "run_outcome", "record_type": "scaffold_event", "payload": {"kind": "conclusive", "reason": "found it", "accepted_hypothesis_id": "h1"}},
        {"op": "done_check_result", "payload": {
            "phase": "final",
            "unmet": [{"criterion": "1", "code": "foo", "target": "bar", "detail": "missed"}],
            "observability_gaps": {"on_path": [{"attempt_reason": "check cpu", "source": "logs", "entity_kind": "host", "values": ["apache01"]}], "other": []}
        }}
    ]
    
    for i, e in enumerate(events):
        con.execute(
            "INSERT INTO ledger_events (run_id, seq, event_id, actor, step, op, record_type, payload) VALUES (?, ?, ?, 'test', 1, ?, 'test', ?)",
            (run_id, i+1, f"e{i}", e["op"], json.dumps(e.get("payload", {})))
        )
        
    con.execute(
        "INSERT INTO coverage (run_id, coverage_id, query_id, actor, kind, attempt_reason, sources, requested, row_count) VALUES (?, 'c1', 'q1', 'test', 'queried', '', '[\"metrics\"]', '{\"kind\": [\"cpu\"]}', 100)",
        (run_id,)
    )
    
    con.close()
    return str(db_path)


def test_render_report(run_dir):
    report = render_report(run_dir)
    assert "Outcome**: CONCLUSIVE" in report
    assert "Accepted Hypothesis: h1" in report
    assert "CPU is high" in report
    assert "## Refutations" in report
    assert "Memory is high" in report
    assert "## Observability Gaps" in report
    assert "check cpu | logs | host" in report
    assert "## Unmet Criteria" in report
    assert "Criterion 1" in report

def test_render_coverage_human(run_dir):
    cov = render_coverage_human(run_dir)
    assert "Coverage (Human-Facing)" in cov
    assert "SUCCESS**: Sources: [\"metrics\"] | Requested: {\"kind\": [\"cpu\"]}" in cov
