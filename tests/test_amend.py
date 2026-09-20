import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

import duckdb
from agentic_rca.run import RunConfig, run_investigation
from agentic_rca.agents.loop import Budget
from agentic_rca.llm import ScriptedLLM, call
from agentic_rca.ledger import Ledger
from agentic_rca.__main__ import main
import agentic_rca.report.bundle as bundle_module

@pytest.fixture
def fake_parent_bundle(tmp_path):
    # Create a fake parent bundle
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    parent_id = "parent123"
    led = Ledger.create(
        runs_dir,
        store_path=str(tmp_path / "store.duckdb"),
        dataset_digest={},
        code_version="v1",
        incident_ts=1614850000.0,
        run_id=parent_id,
        params={}
    )
    
    parent_dir = runs_dir / parent_id
    
    # Fake manifest
    manifest = {
        "run_id": parent_id,
        "depth": 0,
        "root_run_id": parent_id,
        "files": {"store.duckdb": str(tmp_path / "store.duckdb")}
    }
    with open(parent_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    outcome = {"kind": "inconclusive", "accepted_hypothesis_id": None}
    led.apply("run_outcome", outcome, actor="scaffold")
    
    # We need clock_offsets and other inputs
    led.con.execute("INSERT INTO run_inputs VALUES (?, ?, ?)", [parent_id, "clock_offsets", "[]"])
    led.con.execute("INSERT INTO run_inputs VALUES (?, ?, ?)", [parent_id, "source_ts_unit_s", "{}"])
    led.con.execute("INSERT INTO run_inputs VALUES (?, ?, ?)", [parent_id, "declaration", "{}"])
    led.close()
    
    duckdb.connect(str(tmp_path / "store.duckdb")).close()
    
    return parent_dir, runs_dir

def test_amend_addendum(fake_parent_bundle, tmp_path):
    parent_dir, runs_dir = fake_parent_bundle
    
    llm = ScriptedLLM(default=lambda r, m, t: None) # not used because we mock build_lead
    
    cfg = RunConfig(
        store_path=tmp_path / "store.duckdb",
        runs_root=runs_dir,
        incident_ts=1614850000.0,
        parent_run_dir=parent_dir,
        budget=Budget(max_steps=20)
    )
    
    with patch("agentic_rca.donecheck.precheck_for", return_value=lambda mode="": {"met": True}):
        with patch("agentic_rca.run.verify"):
            with patch("agentic_rca.run._final_check", return_value=MagicMock(ok=True)):
                with patch("agentic_rca.run._snapshot", return_value={"declaration": {}, "clock_offsets": [], "source_ts_unit_s": {}}):
                    with patch("agentic_rca.report.bundle.verify_bundle"): # mock verify_bundle on parent
                        with patch("agentic_rca.run.build_lead") as m_lead:
                            m_lead.return_value.run.return_value = MagicMock(status="terminated")
                            m_lead.return_value.budget = MagicMock(steps=1, tokens=100, extensions=0)
                            
                            def mock_state():
                                return MagicMock(latest_conclusion=lambda: {"kind": "inconclusive", "accepted_hypothesis_id": None}, objections={}, verdicts=[], evidence=[])
                            
                            with patch.object(Ledger, 'state', side_effect=mock_state):
                                with patch("agentic_rca.report.render.render_report", return_value="report"):
                                    with patch("agentic_rca.report.render.render_coverage_human", return_value="cov"):
                                        res = run_investigation(cfg, llm)
                            
    assert res["outcome"]["kind"] == "inconclusive"
    assert cfg.critic_rounds == 0 # Skipped critic because kind matched parent

def test_amend_superseding(fake_parent_bundle, tmp_path):
    parent_dir, runs_dir = fake_parent_bundle
    
    def conclude_conclusive(r, m, t):
        if r == "lead":
            return call("request_termination", kind="conclusive", accepted_hypothesis_id="hy001")
        elif r == "critic":
            return call("objection_resolve")
        return call("request_termination")
            
    llm = ScriptedLLM(default=conclude_conclusive)
    
    cfg = RunConfig(
        store_path=tmp_path / "store.duckdb",
        runs_root=runs_dir,
        incident_ts=1614850000.0,
        parent_run_dir=parent_dir,
        budget=Budget(max_steps=20)
    )
    
    with patch("agentic_rca.donecheck.precheck_for", return_value=lambda mode="": {"met": True}):
        with patch("agentic_rca.run.verify"):
            with patch("agentic_rca.run._final_check", return_value=MagicMock(ok=True)):
                with patch("agentic_rca.run._snapshot", return_value={"declaration": {}, "clock_offsets": [], "source_ts_unit_s": {}}):
                    with patch("agentic_rca.report.bundle.verify_bundle"):
                        with patch("agentic_rca.report.render.render_report", return_value="report"):
                            with patch("agentic_rca.report.render.render_coverage_human", return_value="cov"):
                                # We mock lead.run returning conclusive by mutating state directly since it's hard to full-mock
                                with patch("agentic_rca.run.build_lead") as m_lead:
                                    m_lead.return_value.run.return_value = MagicMock(status="terminated")
                                    m_lead.return_value.budget = MagicMock(steps=1, tokens=100, extensions=0)
                                    
                                    # Manually force concl
                                    def mock_state():
                                        return MagicMock(latest_conclusion=lambda: {"kind": "conclusive", "accepted_hypothesis_id": "hy001"}, objections={}, verdicts=[], evidence=[])
                                    
                                    with patch.object(Ledger, 'state', side_effect=mock_state):
                                        res = run_investigation(cfg, llm); print("RES OUTCOME:", res["outcome"])
                                        assert cfg.critic_rounds == 1 # Critic was forced because kind changed

def test_cli_amend_caps(fake_parent_bundle, tmp_path):
    parent_dir, runs_dir = fake_parent_bundle
    
    # modify manifest to have depth = 3
    with open(parent_dir / "manifest.json", "r+") as f:
        m = json.load(f)
        m["depth"] = 3
        f.seek(0)
        json.dump(m, f)
        f.truncate()
        
    with patch("agentic_rca.report.bundle.verify_bundle"):
        assert main(["amend", str(parent_dir), "--runs", str(runs_dir)]) == 1
