"""run-orchestrator acceptance (PLAN.md #8): scripted end-to-end runs on real data."""
import json
import os

import pytest

from agentic_rca.agents.loop import Budget, LoopConfig
from agentic_rca.llm import ScriptedLLM, call
from agentic_rca.report.bundle import BundleError, verify_bundle
from agentic_rca.run import RunConfig, run_investigation
from tests.conftest import DATA_DIR, INCIDENT


def last_result(messages):
    body = messages[-1]["content"].split("\n", 1)[1].rsplit("\n", 1)[0]
    return json.loads(body)


def rec(messages, tools):
    r = last_result(messages)
    row = r["preview"][0]
    return call("evidence_record", claim="spans seen", query_id=r["query_id"], evidence_kind="occurrence",
                row_refs=[{"row": row["_row"], "values": {"cmdb_id": row["cmdb_id"]}}])


def conclude(messages, tools):
    return call("request_termination", kind="inconclusive",
                symptoms=[{"text": "latency", "evidence_ids": ["ev0001"]}],
                discriminators=[{"between": ["hy002", "hy003"], "evidence_needed": "per-host timing"}])


def cfg(tmp_path, store_path, **kw):
    return RunConfig(store_path=store_path, runs_root=tmp_path / "runs", incident_ts=INCIDENT[0] + 900,
                     data_dir=DATA_DIR, loop=LoopConfig(consolidate_every=0), **kw)


def test_end_to_end_bundle(tmp_path, store_path):
    lead = [call("hypothesis_create", statement="A"), call("hypothesis_create", statement="B"),
            call("sql", query="SELECT DISTINCT cmdb_id FROM spans"), rec]
    lead += [call("hypothesis_set_status", hypothesis_id=h, status="unresolved", basis={"reason": "not separated"})
             for h in ("hy001", "hy002", "hy003")]
    lead += [conclude]
    llm = ScriptedLLM(by_role={"lead": lead, "critic": [call("finish_round")]})
    out = run_investigation(cfg(tmp_path, store_path), llm)
    assert out["outcome"] == {"kind": "inconclusive", "reason": "proposed_inconclusive", "accepted_hypothesis_id": None}
    bundle = tmp_path / "runs" / out["run_id"]
    assert verify_bundle(bundle)["ok"] and (bundle / "report.md").exists()
    assert "Candidates" in (bundle / "report.md").read_text()
    assert not (tmp_path / "runs" / ".partial" / out["run_id"]).exists()
    os.chmod(bundle, 0o755)
    os.chmod(bundle / "report.md", 0o644)
    (bundle / "report.md").write_text("tampered")
    with pytest.raises(BundleError):
        verify_bundle(bundle)


def test_budget_exhaustion_is_inconclusive(tmp_path, store_path):
    llm = ScriptedLLM(by_role={"lead": [call("hypothesis_create", statement="A"), call("ledger_view"),
                                        call("ledger_view"), call("ledger_view")]})
    out = run_investigation(cfg(tmp_path, store_path, budget=Budget(max_steps=2)), llm)
    assert out["outcome"]["kind"] == "inconclusive" and out["outcome"]["reason"] == "budget_exhausted"
    assert verify_bundle(tmp_path / "runs" / out["run_id"])["ok"]
