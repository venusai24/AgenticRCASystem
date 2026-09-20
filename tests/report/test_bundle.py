import os
import json
import duckdb
import pytest
from datetime import datetime, timezone
from agentic_rca.ledger.rundb import RUN_DDL
from agentic_rca.report.bundle import freeze_bundle, verify_bundle, compute_sha256, BundleError

@pytest.fixture
def run_dir(tmp_path):
    run_id = "r2026-test"
    base_dir = tmp_path / "runs"
    base_dir.mkdir()
    
    partial_dir = base_dir / f"{run_id}.partial"
    partial_dir.mkdir()
    
    # Create a stub run.duckdb
    db_path = partial_dir / "run.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(RUN_DDL)
    con.execute("INSERT INTO run (run_id) VALUES (?)", (run_id,))
    
    # Insert a dummy event so outcome is known
    payload = json.dumps({"kind": "conclusive", "reason": "found it", "accepted_hypothesis_id": "h1"})
    con.execute(
        "INSERT INTO ledger_events (run_id, seq, event_id, op, record_type, payload) VALUES (?, 1, 'e1', 'run_outcome', 'scaffold_event', ?)", 
        (run_id, payload)
    )
    
    con.close()
    
    return str(base_dir), run_id

def test_freeze_and_verify(run_dir):
    base_dir, run_id = run_dir
    
    # 1. Freeze
    partial_dir = os.path.join(base_dir, f"{run_id}.partial")
    bundle_dir, _ = freeze_bundle(partial_dir, base_dir)
    final_dir = str(bundle_dir)
    assert os.path.exists(final_dir)
    assert not os.path.exists(os.path.join(base_dir, f"{run_id}.partial"))
    
    # Check permissions (0444 for files, 0555 for dirs)
    manifest_path = os.path.join(final_dir, "manifest.json")
    assert oct(os.stat(manifest_path).st_mode)[-3:] == "444"
    assert oct(os.stat(final_dir).st_mode)[-3:] == "555"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    assert manifest["run_id"] == run_id
    assert manifest["outcome"]["kind"] == "conclusive"
    assert manifest["outcome"]["accepted_hypothesis_id"] == "h1"
    
    assert "report.md" in manifest["files"]
    assert "coverage_human.md" in manifest["files"]
    assert "run.duckdb" in manifest["files"]
    
    # 2. Verify success
    assert verify_bundle(final_dir)["ok"]
    
    # 3. Tamper with a file and verify failure
    db_path = os.path.join(final_dir, "run.duckdb")
    os.chmod(db_path, 0o644)
    os.chmod(final_dir, 0o755)
    with open(db_path, "ab") as f:
        f.write(b"tamper")
        
    with pytest.raises(BundleError, match="Hash mismatch for run.duckdb"):
        verify_bundle(final_dir)
        
def test_freeze_rejects_wal(run_dir):
    base_dir, run_id = run_dir
    partial_dir = os.path.join(base_dir, f"{run_id}.partial")
    
    # create wal
    with open(os.path.join(partial_dir, "run.duckdb.wal"), "w") as f:
        f.write("wal")
        
    with pytest.raises(ValueError, match="run.duckdb.wal exists"):
        freeze_bundle(partial_dir, base_dir)

