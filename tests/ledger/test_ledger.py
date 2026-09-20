"""ledger-core acceptance (PLAN.md #1): I1-I14 reject + pass, zero-event
rejection, fold determinism, origin from actor, standing exclusion,
amendment leaves the parent byte-identical, render hides origin."""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from agentic_rca.ledger import Ledger, LedgerError, fold, render, two_live_check

ROWS = {}


def lookup(qid):
    return ROWS.get(qid)


def add_query(
    led,
    short,
    rows,
    *,
    status="ok",
    tool="sql",
    empty_because=None,
    unobserved=None,
    scope_verified=True,
    inputs=(),
):
    qid = f"{led.run_id}/{short}"
    led.con.execute(
        "INSERT INTO queries (run_id, query_id, tool, status, row_count, empty_because, "
        "unobserved_entities, scope_verified, input_query_ids) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            led.run_id,
            qid,
            tool,
            status,
            len(rows),
            empty_because,
            json.dumps(unobserved) if unobserved else None,
            scope_verified,
            json.dumps(list(inputs)),
        ],
    )
    ROWS[qid] = rows
    return short


@pytest.fixture
def led(tmp_path):
    ledger = Ledger.create(tmp_path / "runs", result_lookup=lookup)
    yield ledger
    ledger.close()


def ev(led, q="q000001", rows=None, kind="occurrence", claim="c"):
    if rows is None:
        rows = [{"row": 0, "values": {"v": 1.0}}]
    return led.apply(
        "evidence_record",
        {"claim": claim, "query_id": q, "row_refs": rows, "evidence_kind": kind},
        actor="lead",
    )["id"]


def base(led):
    add_query(
        led, "q000001", [{"ts": 10.0, "host": "a", "v": 1.0}, {"ts": 20.0, "host": "b", "v": 2.0}]
    )
    e1 = ev(led, rows=[{"row": 0, "values": {"v": 1.0}}, {"row": 1, "values": {"v": 2.0}}])
    return e1


def n_events(led):
    return led.con.execute("SELECT count(*) FROM ledger_events").fetchone()[0]


def rejects(led, rule, op, payload, actor="lead", **kw):
    before = n_events(led)
    with pytest.raises(LedgerError) as exc:
        led.apply(op, payload, actor=actor, **kw)
    assert exc.value.rule == rule, exc.value
    assert n_events(led) == before  # a rejected op appends zero events
    return exc.value


# ---- run DB --------------------------------------------------------------------


def test_run_id_and_tables(led):
    import re

    assert re.fullmatch(r"r\d{8}T\d{6}Z-[0-9a-f]{6}", led.run_id)
    assert (led.run_dir / "run.duckdb").exists()
    tables = {r[0] for r in led.con.execute("SHOW TABLES").fetchall()}
    assert {"run", "ledger_events", "queries", "coverage", "inspections", "llm_calls"} <= tables


def test_schema_rejects_unknown_and_scaffold_fields(led):
    base(led)
    rejects(led, "schema", "hypothesis_create", {"statement": "x", "origin": "human"})
    rejects(
        led,
        "schema",
        "evidence_record",
        {"claim": "c", "query_id": "q000001", "evidence_kind": "occurrence", "guard": "ok"},
    )


# ---- I1 - I3 ------------------------------------------------------------------------


def test_I1_query_must_exist_and_succeed(led):
    rejects(
        led,
        "I1",
        "evidence_record",
        {"claim": "c", "query_id": "q000009", "evidence_kind": "occurrence"},
    )
    add_query(led, "q000002", [], status="error")
    rejects(
        led,
        "I1",
        "evidence_record",
        {"claim": "c", "query_id": "q000002", "evidence_kind": "occurrence"},
    )


def test_I2_row_range_and_values(led):
    add_query(led, "q000001", [{"v": 1.0}])
    rejects(
        led,
        "I2",
        "evidence_record",
        {
            "claim": "c",
            "query_id": "q000001",
            "evidence_kind": "occurrence",
            "row_refs": [{"row": 5, "values": {}}],
        },
    )
    err = rejects(
        led,
        "I2",
        "evidence_record",
        {
            "claim": "c",
            "query_id": "q000001",
            "evidence_kind": "occurrence",
            "row_refs": [{"row": 0, "values": {"v": 9.0}}],
        },
    )
    assert "actual row" in err.diagnostic
    assert ev(led, rows=[{"row": 0, "values": {"v": 1.0 + 1e-12}}])  # within tolerance


def test_I3_empty_refs_and_derived(led):
    add_query(led, "q000001", [{"v": 1}])
    rejects(
        led,
        "I3",
        "evidence_record",
        {"claim": "c", "query_id": "q000001", "evidence_kind": "occurrence"},
    )
    rejects(
        led,
        "I3",
        "evidence_record",
        {
            "claim": "c",
            "query_id": "q000001",
            "evidence_kind": "derived",
            "row_refs": [{"row": 0, "values": {"v": 1}}],
        },
    )
    add_query(led, "q000002", [{"m": 3}], tool="python_sandbox", inputs=[f"{led.run_id}/q000001"])
    assert ev(led, q="q000002", rows=[{"row": 0, "values": {"m": 3}}], kind="derived")


# ---- I13 negative-evidence guard -------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    ["no_data_in_window", "entity_absent_from_source", "join_key_unpopulated", "scope_unverified"],
)
def test_I13a_zero_rows_need_no_matches(led, reason):
    add_query(led, "q000001", [], empty_because=reason)
    err = rejects(
        led,
        "I13",
        "evidence_record",
        {"claim": "no errors", "query_id": "q000001", "evidence_kind": "occurrence"},
    )
    assert reason in err.diagnostic


def test_I13a_no_matches_licenses_negative(led):
    add_query(led, "q000001", [], empty_because="no_matches")
    assert ev(led, rows=[])


def test_I13b_aggregate_over_unobserved_entity(led):
    add_query(led, "q000001", [{"count": 0}], unobserved={"host": ["mysql01"]})
    err = rejects(
        led,
        "I13",
        "evidence_record",
        {
            "claim": "mysql01 had zero spans",
            "query_id": "q000001",
            "evidence_kind": "occurrence",
            "row_refs": [{"row": 0, "values": {"count": 0}}],
        },
    )
    assert "mysql01" in err.diagnostic


def test_I13c_unverified_scope_is_marked(led):
    add_query(led, "q000001", [{"count": 0}], scope_verified=False)
    eid = ev(led, rows=[{"row": 0, "values": {"count": 0}}])
    assert led.state().evidence[led.q(eid)]["guard"] == "unverified_scope"
    assert "UNVERIFIED-SCOPE" in render(led.state(), led.short)


# ---- hypotheses: I4, I5, I10, I11, I12, I14 ----------------------------------------------


def chain(e, joins=None, n=1):
    return {
        "links": [
            {
                "cause": f"c{i}",
                "effect": f"e{i}",
                "evidence": [{"evidence_id": e, "role": "observation"}],
                "cause_at": {"evidence_id": e, "row": 0, "ts_s": 10.0, "host": "a"},
                "effect_at": {"evidence_id": e, "row": 1, "ts_s": 20.0, "host": "b"},
            }
            for i in range(n)
        ],
        "joins": joins,
    }


def test_origin_from_actor_and_chain_ids(led):
    e1 = base(led)
    h = led.apply(
        "hypothesis_create", {"statement": "A", "branches": [chain(e1, n=2)]}, actor="lead"
    )["id"]
    hc = led.apply("hypothesis_create", {"statement": "B"}, actor="critic")["id"]
    hh = led.apply("hypothesis_create_human", {"statement": "pool exhausted"}, actor="human")["id"]
    st = led.state()
    assert st.hypotheses[led.q(h)]["origin"] == "lead"
    assert st.hypotheses[led.q(hc)]["origin"] == "critic"
    assert st.hypotheses[led.q(hh)]["origin"] == "human"
    assert [led.short(x) for x in st.hypotheses[led.q(h)]["branches"][0]["links"]] == [
        f"{h}.l1",
        f"{h}.l2",
    ]
    rejects(led, "I8", "hypothesis_create_human", {"statement": "x"}, actor="lead")
    rejects(
        led,
        "schema",
        "hypothesis_create_human",
        {"statement": "x", "window": "10:00"},
        actor="human",
    )


def test_I4_anchor_must_be_cited(led):
    e1 = base(led)
    bad = chain(e1)
    bad["links"][0]["cause_at"]["row"] = 7
    rejects(led, "I4", "hypothesis_create", {"statement": "A", "branches": [bad]})
    rejects(led, "I4", "hypothesis_create", {"statement": "A", "branches": [chain("ev0099")]})


def test_I11_branches(led):
    e1 = base(led)
    two_terminal = [chain(e1), chain(e1)]
    rejects(led, "I11", "hypothesis_create", {"statement": "A", "branches": two_terminal})
    cycle = [chain(e1, joins={"branch": 1, "link": 0}), chain(e1, joins={"branch": 0, "link": 0})]
    rejects(led, "I11", "hypothesis_create", {"statement": "A", "branches": cycle})
    conj = [chain(e1, joins={"branch": 1, "link": 0}), chain(e1)]
    h = led.apply(
        "hypothesis_create",
        {"statement": "deploy AND pool misconfig", "branches": conj},
        actor="lead",
    )["id"]
    b = led.state().hypotheses[led.q(h)]["branches"]
    assert b[0]["joins"] == b[1]["links"][0] and b[1]["joins"] is None


def test_I5_refute_needs_linked_contradiction(led):
    e1 = base(led)
    h = led.apply("hypothesis_create", {"statement": "A"}, actor="lead")["id"]
    basis = {"evidence_ids": [e1], "reason": "contradicted"}
    rejects(
        led,
        "I5",
        "hypothesis_set_status",
        {"hypothesis_id": h, "status": "refuted", "basis": basis},
    )
    led.apply(
        "hypothesis_link_evidence",
        {"hypothesis_id": h, "evidence_id": e1, "stance": "contradicts"},
        actor="lead",
    )
    led.apply(
        "hypothesis_set_status",
        {"hypothesis_id": h, "status": "refuted", "basis": basis},
        actor="lead",
    )
    assert led.state().hypotheses[led.q(h)]["status"] == "refuted"
    rejects(
        led,
        "I8",
        "hypothesis_set_status",
        {"hypothesis_id": h, "status": "live", "basis": basis},
        actor="critic",
    )


def test_I10_withdraw_needs_active_link(led):
    e1 = base(led)
    h = led.apply("hypothesis_create", {"statement": "A"}, actor="lead")["id"]
    rejects(
        led,
        "I10",
        "hypothesis_withdraw_evidence",
        {"hypothesis_id": h, "evidence_id": e1, "reason": "r"},
    )
    led.apply(
        "hypothesis_link_evidence",
        {"hypothesis_id": h, "evidence_id": e1, "stance": "supports"},
        actor="lead",
    )
    led.apply(
        "hypothesis_withdraw_evidence",
        {"hypothesis_id": h, "evidence_id": e1, "reason": "misread"},
        actor="lead",
    )
    assert led.state().hypotheses[led.q(h)]["evidence"][0]["withdrawn"] == "misread"


def test_I12_one_accepted(led):
    base(led)
    a = led.apply("hypothesis_create", {"statement": "A"}, actor="lead")["id"]
    b = led.apply("hypothesis_create", {"statement": "B"}, actor="lead")["id"]
    ok = {"evidence_ids": [], "reason": "best supported"}
    led.apply(
        "hypothesis_set_status",
        {"hypothesis_id": a, "status": "accepted", "basis": ok},
        actor="lead",
    )
    err = rejects(
        led, "I12", "hypothesis_set_status", {"hypothesis_id": b, "status": "accepted", "basis": ok}
    )
    assert a in err.diagnostic
    led.apply(
        "hypothesis_set_status",
        {"hypothesis_id": a, "status": "live", "basis": {"reason": "reopen"}},
        actor="lead",
    )
    led.apply(
        "hypothesis_set_status",
        {"hypothesis_id": b, "status": "accepted", "basis": ok},
        actor="lead",
    )


def test_I14_same_as(led):
    base(led)
    a = led.apply("hypothesis_create", {"statement": "A"}, actor="lead")["id"]
    b = led.apply("hypothesis_create_human", {"statement": "A again"}, actor="human")["id"]
    st = led.state()
    led.apply(
        "cluster_assign",
        {
            "hypothesis_id": a,
            "for_seq": st.hypotheses[led.q(a)]["content_seq"],
            "method": "no_candidates",
        },
        actor="scaffold",
    )
    cl = led.state().hypotheses[led.q(a)]["cluster_id"]
    basis = {"reason": "restates", "same_as": a}
    rejects(
        led,
        "I14",
        "hypothesis_set_status",
        {"hypothesis_id": b, "status": "unresolved", "basis": basis},
    )
    led.apply(
        "cluster_assign",
        {
            "hypothesis_id": b,
            "for_seq": led.state().hypotheses[led.q(b)]["content_seq"],
            "cluster_id": cl,
            "method": "llm_judge",
            "restates": a,
        },
        actor="scaffold",
    )
    rejects(
        led,
        "I14",
        "hypothesis_set_status",
        {"hypothesis_id": b, "status": "refuted", "basis": {"reason": "x", "same_as": a}},
    )
    led.apply(
        "hypothesis_set_status",
        {"hypothesis_id": b, "status": "unresolved", "basis": basis},
        actor="lead",
    )


# ---- I6, I7, I9 -----------------------------------------------------------------------------


def test_I6_I7_failures_questions_objections(led):
    e1 = base(led)
    f = led.apply(
        "failure_record",
        {
            "query_id": f"{led.run_id}/q000009",
            "tool": "sql",
            "error_type": "timeout",
            "diagnostic": "slow",
            "retried": False,
        },
        actor="scaffold",
    )["id"]
    add_query(led, "q000003", [], status="error")
    rejects(led, "I6", "failure_resolve", {"failure_id": f, "by_query_id": "q000003", "note": "n"})
    led.apply(
        "failure_resolve",
        {"failure_id": f, "by_query_id": "q000001", "note": "retried narrower"},
        actor="lead",
    )
    rejects(
        led, "I7", "question_answer", {"question_id": "qu009", "text": "t", "evidence_ids": [e1]}
    )
    h = led.apply("hypothesis_create", {"statement": "A"}, actor="lead")["id"]
    rejects(
        led,
        "I7",
        "objection_raise",
        {"text": "t", "target": {"hypothesis_id": h, "link_id": f"{h}.l9"}},
        actor="critic",
    )
    o = led.apply(
        "objection_raise",
        {"text": "no contrast", "target": {"hypothesis_id": h}},
        actor="critic",
        round=1,
    )["id"]
    led.apply(
        "objection_resolve",
        {"objection_id": o, "kind": "concession", "note": "conceded"},
        actor="lead",
    )
    assert led.state().objections[led.q(o)]["status"] == "conceded"


def test_I9_foreign_run_ids(led):
    base(led)
    rejects(
        led,
        "I9",
        "hypothesis_link_evidence",
        {"hypothesis_id": "rOTHER/hy001", "evidence_id": "ev0001", "stance": "supports"},
    )


# ---- standing hypothesis, >=2-live, dedup assignment --------------------------------------


def test_standing_excluded_from_two_live(led):
    led.apply("standing_seed", {}, actor="scaffold")
    rejects(led, "standing", "standing_seed", {}, actor="scaffold")
    a = led.apply("hypothesis_create", {"statement": "A"}, actor="lead")["id"]
    assert two_live_check(led.state())[0] is False  # standing + one lead hypothesis is not two
    st = led.state()
    sid = st.standing_id()
    # a lead restatement of the standing hypothesis joins its cluster and doesn't count either
    led.apply(
        "cluster_assign",
        {"hypothesis_id": sid, "for_seq": st.hypotheses[sid]["content_seq"], "method": "seed"},
        actor="scaffold",
    )
    cl = led.state().hypotheses[sid]["cluster_id"]
    b = led.apply("hypothesis_create", {"statement": "cause is outside telemetry"}, actor="lead")[
        "id"
    ]
    led.apply(
        "cluster_assign",
        {
            "hypothesis_id": b,
            "for_seq": led.state().hypotheses[led.q(b)]["content_seq"],
            "cluster_id": cl,
            "method": "llm_judge",
        },
        actor="scaffold",
    )
    assert two_live_check(led.state())[0] is False
    led.apply("hypothesis_create", {"statement": "C"}, actor="lead")
    assert two_live_check(led.state())[0] is True
    assert led.state().hypotheses[led.q(a)]["origin"] == "lead"


def test_cluster_assign_never_recomputed_and_revise_repends(led):
    e1 = base(led)
    a = led.apply("hypothesis_create", {"statement": "A"}, actor="lead")["id"]
    seq = led.state().hypotheses[led.q(a)]["content_seq"]
    led.apply(
        "cluster_assign",
        {"hypothesis_id": a, "for_seq": seq, "method": "no_candidates"},
        actor="scaffold",
    )
    rejects(
        led,
        "cluster",
        "cluster_assign",
        {"hypothesis_id": a, "for_seq": seq, "method": "llm_judge"},
        actor="scaffold",
    )
    led.apply("hypothesis_revise", {"hypothesis_id": a, "branches": [chain(e1)]}, actor="lead")
    h = led.state().hypotheses[led.q(a)]
    assert h["cluster_status"] == "pending" and h["content_seq"] > seq


# ---- conclusion ------------------------------------------------------------------------


def test_conclusion_structure(led):
    e1 = base(led)
    a = led.apply("hypothesis_create", {"statement": "A", "branches": [chain(e1)]}, actor="lead")[
        "id"
    ]
    concl = {
        "kind": "conclusive",
        "accepted_hypothesis_id": a,
        "confidence": "medium",
        "confidence_basis": {"strong": [1], "text": "t"},
        "sufficiency_note": "s",
        "symptoms": [{"text": "latency", "evidence_ids": [e1], "explained_by": f"{a}.l1"}],
    }
    rejects(led, "conclusion", "conclusion_propose", concl)  # not accepted yet
    led.apply(
        "hypothesis_set_status",
        {"hypothesis_id": a, "status": "accepted", "basis": {"reason": "r"}},
        actor="lead",
    )
    led.apply("conclusion_propose", concl, actor="lead")
    rejects(led, "schema", "conclusion_propose", dict(concl, confidence=None))
    inc = {
        "kind": "inconclusive",
        "symptoms": concl["symptoms"],
        "discriminators": [{"between": [a], "evidence_needed": "x"}],
    }
    rejects(led, "schema", "conclusion_propose", inc)


# ---- fold determinism, render, amendment -----------------------------------------------------


def test_fold_is_deterministic(led):
    e1 = base(led)
    for i in range(4):
        led.apply(
            "hypothesis_create", {"statement": f"H{i}", "branches": [chain(e1)]}, actor="lead"
        )
    events = led.events()
    shuffled = events[:]
    random.Random(7).shuffle(shuffled)
    shuffled.sort(key=lambda e: e.seq)
    a, b = fold(events, led.run_id), fold(shuffled, led.run_id)
    assert render(a, led.short) == render(b, led.short)


def test_render_hides_origin_and_is_bounded(led):
    base(led)
    led.apply("hypothesis_create_human", {"statement": "hunch"}, actor="human")
    st = led.state()
    assert "origin=human" in render(st, led.short)
    assert "origin" not in render(st, led.short, hide_origin=True)
    for i in range(60):
        led.apply("hypothesis_create", {"statement": "x" * 200 + str(i)}, actor="lead")
    text = render(led.state(), led.short, max_chars=3000)
    assert len(text) <= 3100 and "more lines" in text


def test_amendment_leaves_parent_byte_identical(tmp_path):
    parent = Ledger.create(tmp_path / "runs", result_lookup=lookup)
    add_query(parent, "q000001", [{"v": 1.0}])
    e1 = ev(parent)
    h = parent.apply("hypothesis_create", {"statement": "A"}, actor="lead")["id"]
    parent.close()
    pfile = tmp_path / "runs" / parent.run_id / "run.duckdb"
    digest = hashlib.sha256(pfile.read_bytes()).hexdigest()

    child = Ledger.create(tmp_path / "runs", parent_run_dir=pfile.parent, result_lookup=lookup)
    ph, pe = f"{parent.run_id}/{h}", f"{parent.run_id}/{e1}"
    child.apply(
        "hypothesis_link_evidence",
        {"hypothesis_id": ph, "evidence_id": pe, "stance": "contradicts"},
        actor="lead",
    )
    child.apply(
        "hypothesis_set_status",
        {"hypothesis_id": ph, "status": "refuted", "basis": {"evidence_ids": [pe], "reason": "r"}},
        actor="lead",
    )
    assert child.state().hypotheses[ph]["status"] == "refuted"
    assert child.run_row()["depth"] == 1
    child.close()
    assert hashlib.sha256(pfile.read_bytes()).hexdigest() == digest
    reopened = Ledger.open(pfile.parent, read_only=True)
    assert reopened.state().hypotheses[ph]["status"] == "live"
    reopened.close()
