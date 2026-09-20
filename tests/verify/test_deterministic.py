"""verifier-deterministic acceptance (PLAN.md #5) on real data."""

from __future__ import annotations

import subprocess
import sys

import pytest
import tests.tools.test_registry  # noqa: F401 - registers the t_* test tools
from tests.conftest import INCIDENT

from agentic_rca.verify.deterministic import VerifierRefused, verify, verify_evidence

W = {"start": INCIDENT[0], "end": INCIDENT[0] + 120}


def cite(rt, env, i=0, cols=("value", "cmdb_id")):
    row = rt.handles[rt.ledger.q(env.query_id)][i]
    return {"row": row["_row"], "values": {c: row[c] for c in cols}}, row


def build(rt, *, anchor_ts_shift=0.0, anchor_host=None):
    env = rt.call("t_container", {"hosts": ["Tomcat01"], "window": W})
    ref0, r0 = cite(rt, env, 0)
    ref1, r1 = cite(rt, env, env.row_count - 1)
    led = rt.ledger
    e = led.apply(
        "evidence_record",
        {
            "claim": "c",
            "query_id": env.query_id,
            "row_refs": [ref0, ref1],
            "evidence_kind": "occurrence",
        },
        actor="lead",
    )["id"]
    link = {
        "cause": "x",
        "effect": "y",
        "evidence": [{"evidence_id": e, "role": "contrast"}],
        "cause_at": {
            "evidence_id": e,
            "row": ref0["row"],
            "ts_s": r0["timestamp_s"] + anchor_ts_shift,
            "host": anchor_host or r0["cmdb_id"],
        },
        "effect_at": {
            "evidence_id": e,
            "row": ref1["row"],
            "ts_s": r1["timestamp_s"],
            "host": r1["cmdb_id"],
        },
    }
    h = led.apply(
        "hypothesis_create", {"statement": "A", "branches": [{"links": [link]}]}, actor="lead"
    )["id"]
    led.apply(
        "hypothesis_set_status",
        {"hypothesis_id": h, "status": "accepted", "basis": {"reason": "r"}},
        actor="lead",
    )
    led.apply(
        "conclusion_propose",
        {
            "kind": "conclusive",
            "accepted_hypothesis_id": h,
            "confidence": "low",
            "confidence_basis": {"strong": [], "text": "t"},
            "sufficiency_note": "s",
            "symptoms": [{"text": "s", "evidence_ids": [e], "explained_by": f"{h}.l1"}],
        },
        actor="lead",
    )
    return led.q(e)


def test_pass_writes_verdict_events(runtime):
    eid = build(runtime)
    out = verify(runtime.ledger, runtime)
    assert out["counts"] == {"pass": 1, "fail": 0, "unverifiable": 0}
    v = runtime.ledger.state().verdicts[eid]
    assert v["verdict"] == "pass" and v["digest_match"] is True


def test_anchor_timestamp_and_host_are_checked(runtime):
    eid = build(runtime, anchor_ts_shift=5.0)
    assert "ts_s" in verify_evidence(runtime.ledger, runtime, eid)["detail"]


def test_anchor_host_checked(runtime):
    eid = build(runtime, anchor_host="Tomcat02")
    v = verify_evidence(runtime.ledger, runtime, eid)
    assert v["verdict"] == "fail" and "host" in v["detail"]


def test_value_not_in_result_fails(runtime):
    env = runtime.call("t_container", {"hosts": ["Tomcat01"], "window": W})
    qid = runtime.ledger.q(env.query_id)
    saved = runtime.handles.pop(
        qid
    )  # the handle expired: the ledger accepts, the verifier is the check
    e = runtime.ledger.apply(
        "evidence_record",
        {
            "claim": "c",
            "query_id": env.query_id,
            "evidence_kind": "occurrence",
            "row_refs": [{"row": 0, "values": {"value": -12345.0}}],
        },
        actor="lead",
    )["id"]
    runtime.handles[qid] = saved
    v = verify_evidence(runtime.ledger, runtime, runtime.ledger.q(e))
    assert v["verdict"] == "fail" and "not found" in v["detail"]


def test_non_total_order_matches_by_value(runtime):
    env = runtime.call("t_ties", {})
    e = runtime.ledger.apply(
        "evidence_record",
        {
            "claim": "c",
            "query_id": env.query_id,
            "evidence_kind": "occurrence",
            "row_refs": [{"row": 1, "values": {"v": "b"}}],
        },
        actor="lead",
    )["id"]
    runtime.handles[runtime.ledger.q(env.query_id)].reverse()  # ordinals legitimately unstable
    assert verify_evidence(runtime.ledger, runtime, runtime.ledger.q(e))["verdict"] == "pass"


def test_unverifiable_when_tool_gone(runtime):
    env = runtime.call("t_container", {"hosts": ["Tomcat01"], "window": W})
    ref, _ = cite(runtime, env)
    e = runtime.ledger.apply(
        "evidence_record",
        {"claim": "c", "query_id": env.query_id, "evidence_kind": "occurrence", "row_refs": [ref]},
        actor="lead",
    )["id"]
    runtime.ledger.con.execute("UPDATE queries SET args_json = '{\"bogus\": 1}'")
    assert (
        verify_evidence(runtime.ledger, runtime, runtime.ledger.q(e))["verdict"] == "unverifiable"
    )


def test_refuses_on_dataset_digest_change(tmp_path, store_path):
    from agentic_rca.ledger import Ledger
    from agentic_rca.tools.registry import ToolRuntime

    led = Ledger.create(tmp_path / "runs", dataset_digest={"a.csv": "111"})
    rt = ToolRuntime(store_path, led)
    with pytest.raises(VerifierRefused):
        verify(led, rt, current_dataset_digest={"a.csv": "222"})
    rt.close()
    led.close()


def test_verifier_imports_no_llm():
    code = (
        "import sys, agentic_rca.verify.deterministic; "
        "bad=[m for m in sys.modules if m.startswith('agentic_rca.llm') or m.startswith('agentic_rca.agents')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd="src").returncode == 0
