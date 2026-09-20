"""done-check acceptance (PLAN.md #4): d67 §2-§8 predicates, met / unmet /
deferred fixtures, inconclusive tiers, gaps, and the no-LLM invariant."""

from __future__ import annotations

import subprocess
import sys

import pytest

from agentic_rca.donecheck import (
    DoneCheckInput,
    done_check,
    load_coverage,
    load_queries,
    render_unmet,
)
from tests.helpers import Builder, link

UNITS = {"container_metrics": 1.0, "spans": 0.001, "app_metrics": 1.0, "logs": 1.0}
OFFSETS = [
    {"host_a": "h1", "host_b": "h2", "offset_ms": 300.0, "residual_std_ms": 50.0},
    {"host_a": "h1", "host_b": "h3", "offset_ms": float("nan"), "residual_std_ms": float("nan")},
]


def check(b, phase="pre_challenge", rounds=0):
    inp = DoneCheckInput(
        b.led.state(),
        load_coverage(b.led.con),
        load_queries(b.led.con),
        OFFSETS,
        UNITS,
        phase,
        rounds,
    )
    return done_check(inp)


def codes(res, crit=None):
    return {i["code"] for i in res.unmet if crit is None or i["criterion"] == crit}


@pytest.fixture
def world(tmp_path):
    """A conclusive case that meets every pre_challenge criterion."""
    b = Builder(tmp_path)
    q1 = b.q([{"v": 1}, {"v": 2}], requested={"host": ["h1"]})
    qc = b.q([{"v": 0}], requested={"host": ["h1"]})
    e1 = b.ev(q1, [0, 1])
    ec = b.ev(qc, [0], claim="baseline")
    b.op("standing_seed", {}, "scaffold")
    a = b.op(
        "hypothesis_create",
        {"statement": "A", "branches": [{"links": [link(e1, e1, contrast=[ec], erow=1)]}]},
    )
    bb = b.op("hypothesis_create", {"statement": "B"})
    qx = b.q([{"v": 9}], requested={"host": ["h1"]})
    ex = b.ev(qx, [0])
    b.op(
        "hypothesis_link_evidence",
        {"hypothesis_id": bb, "evidence_id": ex, "stance": "contradicts"},
    )
    b.op(
        "hypothesis_set_status",
        {"hypothesis_id": bb, "status": "refuted", "basis": {"evidence_ids": [ex], "reason": "r"}},
    )
    st = b.led.state()
    b.op(
        "hypothesis_set_status",
        {
            "hypothesis_id": st.standing_id().split("/")[1],
            "status": "unresolved",
            "basis": {"reason": "argued down only partly"},
        },
    )
    b.op(
        "hypothesis_set_status",
        {"hypothesis_id": a, "status": "accepted", "basis": {"reason": "r"}},
    )
    b.a, b.e1, b.ec = a, e1, ec
    return b


def propose(b, **over):
    c = {
        "kind": "conclusive",
        "accepted_hypothesis_id": b.a,
        "confidence": "medium",
        "confidence_basis": {"strong": [], "text": "t"},
        "sufficiency_note": "s",
        "symptoms": [{"text": "sym", "evidence_ids": [b.e1], "explained_by": f"{b.a}.l1"}],
    }
    c.update(over)
    b.op("conclusion_propose", c)


def test_all_met(world):
    propose(world)
    res = check(world)
    assert res.ok, res.unmet
    assert res.criteria["2"]["status"] == "met" and res.criteria["6"]["status"] == "not_applicable"


def test_no_conclusion(world):
    assert codes(check(world)) == {"no_conclusion_proposed"}


def _revise(b, lk):
    b.op(
        "hypothesis_set_status",
        {"hypothesis_id": b.a, "status": "live", "basis": {"reason": "reopen"}},
    )
    b.op("hypothesis_revise", {"hypothesis_id": b.a, "branches": [{"links": [lk]}]})
    b.op(
        "hypothesis_set_status",
        {"hypothesis_id": b.a, "status": "accepted", "basis": {"reason": "r"}},
    )
    lid = f"{b.a}.l2"
    propose(b, symptoms=[{"text": "s", "evidence_ids": [b.e1], "explained_by": lid}])
    return lid


def test_c2_clock_relations(world):
    b = world
    # measured pair h1-h2: bound = 1 + 0.3 + 2*0.05 = 1.4s; delta 1.0 -> within uncertainty
    _revise(b, link(b.e1, b.e1, contrast=[b.ec], ct=10, et=11, ch="h1", eh="h2", erow=1))
    res = check(b)
    item = next(i for i in res.unmet if i["criterion"] == 2)
    assert item["code"] == "ordering_within_uncertainty" and item["detail"][
        "bound_s"
    ] == pytest.approx(1.4)
    assert "uncertainty bound" in render_unmet(res, b.led.short)[0]


def test_c2_reversed_unmeasured_and_note(world):
    b = world
    _revise(b, link(b.e1, b.e1, contrast=[b.ec], ct=30, et=20, ch="h1", eh="h2", erow=1))
    assert "timestamps_reversed" in codes(check(b), 2)
    lid = _revise(b, link(b.e1, b.e1, contrast=[b.ec], ct=10, et=50, ch="h1", eh="h3", erow=1))
    assert "ordering_unmeasured_clock" in codes(check(b), 2)  # NaN residual
    propose(
        b,
        symptoms=[{"text": "s", "evidence_ids": [b.e1], "explained_by": lid.replace("l2", "l3")}],
        ordering_notes=[
            {"link_id": lid.replace("l2", "l3"), "text": "queue order", "evidence_ids": [b.e1]}
        ],
    )
    res = check(b)
    assert res.criteria["2"]["status"] == "deferred" and res.ok


def test_c2_same_source_null_host_flagged(world):
    b = world
    _revise(b, link(b.e1, b.e1, contrast=[b.ec], ct=10, et=12, ch=None, eh=None, erow=1))
    res = check(b)
    assert res.criteria["2"]["status"] == "met"
    assert any(f["code"] == "clocks_assumed_shared" for f in res.flags)


def test_c3_contrast_and_unverified_scope(world, tmp_path):
    b = world
    qg = b.q([{"n": 0}], scope_verified=False)
    eg = b.ev(qg, [0])
    _revise(b, link(b.e1, b.e1, contrast=[eg], erow=1))
    res = check(b)
    assert "no_contrast_for_initiating_condition" in codes(res, 3)
    assert any(f["code"] == "unverified_scope_evidence" for f in res.flags)


def test_c4_refutation_must_discriminate(world):
    b = world
    ex = b.led.state().hypotheses[b.led.q("hy003")]["status_basis"]["evidence_ids"][0]
    b.op(
        "hypothesis_link_evidence",
        {"hypothesis_id": b.a, "evidence_id": ex, "stance": "contradicts"},
    )
    propose(b)
    assert "refutation_not_discriminating" in codes(check(b), 4)


def test_c4_live_remaining_and_c10_human(world):
    b = world
    b.op("hypothesis_create_human", {"statement": "hunch"}, "human")
    propose(b)
    res = check(b)
    assert "live_hypothesis_remaining" in codes(res, 4)
    assert "human_hypothesis_undispositioned" in codes(res, 10)


def test_c8_claimed_content_and_failure_touch(world):
    b = world
    qcl = b.q([{"line": "x"}], requested={"host": ["h1"]})
    ecl = b.ev(qcl, [0], kind="claimed_content")
    _revise(
        b,
        {
            "cause": "c",
            "effect": "e",
            "evidence": [{"evidence_id": ecl, "role": "observation"}],
            "cause_at": {"evidence_id": ecl, "row": 0, "ts_s": 1, "host": "h1"},
            "effect_at": {"evidence_id": ecl, "row": 0, "ts_s": 5, "host": "h1"},
        },
    )
    assert "claimed_content_only" in codes(check(b), 8)
    # a timeout on the same source+host+window touches; an invalid_query never does
    for et in ("timeout", "invalid_query"):
        qf = b.q(
            [], status="error", requested={"host": ["h1"]}, attempt_reason="error", error_type=et
        )
        b.op(
            "failure_record",
            {
                "query_id": b.led.q(qf),
                "tool": "t",
                "error_type": et,
                "diagnostic": "d",
                "retried": False,
            },
            "scaffold",
        )
    res = check(b)
    touching = [i for i in res.unmet if i["code"] == "unresolved_failure_touches_link"]
    assert len(touching) == 1 and touching[0]["detail"]["failure"].endswith("fa001")


def test_c11_branch_contrast_must_be_distinct(world):
    b = world
    b.op(
        "hypothesis_set_status",
        {"hypothesis_id": b.a, "status": "live", "basis": {"reason": "reopen"}},
    )
    lk = link(b.e1, b.e1, contrast=[b.ec], erow=1)
    b.op(
        "hypothesis_revise",
        {
            "hypothesis_id": b.a,
            "branches": [{"links": [lk], "joins": {"branch": 1, "link": 0}}, {"links": [lk]}],
        },
    )
    b.op(
        "hypothesis_set_status",
        {"hypothesis_id": b.a, "status": "accepted", "basis": {"reason": "r"}},
    )
    propose(b, symptoms=[{"text": "s", "evidence_ids": [b.e1], "explained_by": f"{b.a}.l3"}])
    assert codes(check(b), 11) == {"branch_contrast_not_distinct"}


def test_final_phase_filter_and_stripping(world):
    b = world
    propose(b)
    res = check(b, "final")
    assert "verdict_missing" in codes(res, 7) and not res.ok
    for eid in b.led.state().evidence:
        b.op("evidence_verdict", {"evidence_id": eid, "verdict": "pass"}, "scaffold")
    assert check(b, "final").ok
    b.op("evidence_verdict", {"evidence_id": b.led.q(b.ec), "verdict": "fail"}, "scaffold")
    res = check(b, "final")
    assert b.led.q(b.ec) in res.stripped and "no_contrast_for_initiating_condition" in codes(res, 3)


def test_c6_after_critic_round(world):
    b = world
    b.op("objection_raise", {"text": "t", "target": {"hypothesis_id": b.a}}, "critic")
    propose(b)
    assert check(b).ok  # n/a before a critic round
    assert "objection_open" in codes(check(b, rounds=1), 6)


def test_strong_claim_must_be_met(world):
    b = world
    _revise(b, link(b.e1, b.e1, contrast=[b.ec], ct=10, et=50, ch="h1", eh="h3", erow=1))
    lid = f"{b.a}.l2"
    propose(
        b,
        confidence_basis={"strong": [2], "text": "t"},
        symptoms=[{"text": "s", "evidence_ids": [b.e1], "explained_by": lid}],
        ordering_notes=[{"link_id": lid, "text": "n", "evidence_ids": [b.e1]}],
    )
    assert "strong_claim_not_met" in codes(check(b), 9)


def test_inconclusive_tiers_and_standing_separate(tmp_path):
    b = Builder(tmp_path)
    q1 = b.q([{"v": 1}, {"v": 2}], requested={"host": ["h1"]})
    qc = b.q([{"v": 0}], requested={"host": ["h1"]})
    e1, ec = b.ev(q1, [0, 1]), b.ev(qc, [0])
    b.op("standing_seed", {}, "scaffold")
    a = b.op(
        "hypothesis_create",
        {"statement": "A", "branches": [{"links": [link(e1, e1, contrast=[ec], erow=1)]}]},
    )
    c = b.op("hypothesis_create", {"statement": "C"})
    d = b.op("hypothesis_create", {"statement": "D"})
    sid = b.led.short(b.led.state().standing_id())
    for h in (a, c, d, sid):
        b.op(
            "hypothesis_set_status",
            {"hypothesis_id": h, "status": "unresolved", "basis": {"reason": "r"}},
        )
    b.op(
        "conclusion_propose",
        {
            "kind": "inconclusive",
            "symptoms": [{"text": "s", "evidence_ids": [e1]}],
            "discriminators": [{"between": [a, c], "evidence_needed": "x"}],
        },
    )
    res = check(b)
    assert codes(res, 4) == {"candidate_without_discriminator"}
    ranked = [r for r in res.candidates if not r.get("standing")]
    assert (
        ranked[0]["hypothesis_id"].endswith(a)
        and ranked[0]["criteria_met"] > ranked[1]["criteria_met"]
    )
    assert ranked[1]["tied"] and ranked[2]["tied"]
    assert res.candidates[-1].get("standing")


def test_observability_gaps_on_path_vs_other(world):
    b = world
    b.q(
        [],
        source="container_metrics",
        requested={"host": ["h9"]},
        attempt_reason="entity_absent_from_source",
        empty_because="entity_absent_from_source",
        kind="attempt",
    )
    b.q(
        [],
        source="logs",
        requested={"host": ["h8"]},
        window=(5e10, 6e10),
        attempt_reason="entity_absent_from_source",
        empty_because="entity_absent_from_source",
        kind="attempt",
    )
    propose(b)
    gaps = check(b).observability_gaps
    assert [g["values"] for g in gaps["on_path"]] == [["h9"]]
    assert [g["values"] for g in gaps["other"]] == [["h8"]]


def test_done_check_imports_no_llm():
    code = (
        "import sys, agentic_rca.donecheck; "
        "bad=[m for m in sys.modules if m.startswith('agentic_rca.llm') or m.startswith('agentic_rca.agents')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd="src").returncode == 0
