"""coverage-index + registry acceptance (PLAN.md #2), on real data."""

from __future__ import annotations

import json

import duckdb
import pytest
from pydantic import Field
from tests.conftest import BASELINE, INCIDENT

from agentic_rca.tools._common import Scope, ToolArgs, Window
from agentic_rca.tools.registry import TOOLS, ToolResult, tool


class HostArgs(ToolArgs):
    hosts: list[str] | None = None
    window: Window
    min_value: float | None = None


def _hosts_scope(a, source):
    return Scope(
        source=source, requested={"host": a.hosts or "*"}, window=(a.window.start, a.window.end)
    )


@tool(
    "t_container",
    HostArgs,
    description="test tool",
    order_by=["timestamp_s", "cmdb_id", "kpi_name"],
    sources=["container_metrics"],
    attempt_scopes=lambda a: [_hosts_scope(a, "container_metrics")],
)
def t_container(a, ctx):
    if a.hosts:
        ctx.index.check_known("host", a.hosts)
    where, params = ["timestamp_s >= ? AND timestamp_s < ?"], [a.window.start, a.window.end]
    if a.hosts:
        where.append(f"cmdb_id IN ({','.join('?' * len(a.hosts))})")
        params += a.hosts
    if a.min_value is not None:
        where.append("value >= ?")
        params.append(a.min_value)
    sql = f"SELECT timestamp_s, cmdb_id, kpi_name, value FROM container_metrics WHERE {' AND '.join(where)}"
    return ToolResult(ctx.fetch(sql, params), [_hosts_scope(a, "container_metrics")], [sql])


@tool("t_span_count", HostArgs, description="test aggregate", order_by=[], sources=["spans"])
def t_span_count(a, ctx):
    if a.hosts:
        ctx.index.check_known("host", a.hosts)
    sql = f"SELECT count(*) AS n FROM spans WHERE timestamp_s >= ? AND timestamp_s < ? AND cmdb_id IN ({','.join('?' * len(a.hosts))})"
    rows = ctx.fetch(sql, [a.window.start, a.window.end, *a.hosts])
    return ToolResult(rows, [_hosts_scope(a, "spans")], [sql])


@tool(
    "t_spans",
    HostArgs,
    description="test spans",
    order_by=["timestamp_s", "cmdb_id", "span_id"],
    sources=["spans"],
)
def t_spans(a, ctx):
    if a.hosts:
        ctx.index.check_known("host", a.hosts)
    q = "SELECT timestamp_s, cmdb_id, span_id, duration_ms FROM spans WHERE timestamp_s >= ? AND timestamp_s < ?"
    params = [a.window.start, a.window.end]
    if a.hosts:
        q += f" AND cmdb_id IN ({','.join('?' * len(a.hosts))})"
        params += a.hosts
    return ToolResult(ctx.fetch(q, params), [_hosts_scope(a, "spans")], [q])


class Nothing(ToolArgs):
    mode: str = Field(default="x")


@tool("t_unverified", Nothing, description="unverified scope", order_by=[], sources=["spans"])
def t_unverified(a, ctx):
    return ToolResult([], [Scope(source="spans", verified=False)])


@tool(
    "t_join", Nothing, description="join over logs.trace_id", order_by=[], sources=["spans", "logs"]
)
def t_join(a, ctx):
    return ToolResult(
        [],
        [
            Scope(
                source="logs",
                join_keys=[("spans", "trace_id"), ("logs", "trace_id")],
                window=INCIDENT,
            )
        ],
    )


@tool(
    "t_multi",
    HostArgs,
    description="multi-source",
    order_by=["timestamp_s", "source"],
    sources=["spans", "logs"],
)
def t_multi(a, ctx):
    rows = [{"timestamp_s": 1.0, "source": "spans", "cmdb_id": "Tomcat01"}]
    return ToolResult(
        rows,
        [
            Scope("spans", {"host": ["Tomcat01"]}, (a.window.start, a.window.end), row_count=1),
            Scope("logs", {"host": ["Tomcat01"]}, BASELINE, row_count=0),
        ],
    )


@tool("t_ties", Nothing, description="non-total order", order_by=["k"])
def t_ties(a, ctx):
    return ToolResult([{"k": 1, "v": "a"}, {"k": 1, "v": "b"}], [Scope("spans", verified=True)])


W = {"start": INCIDENT[0], "end": INCIDENT[0] + 120}


def qrow(rt, short):
    return rt.ledger.query_row(rt.ledger.q(short))


def test_unregistered_tool_cannot_be_dispatched(runtime):
    with pytest.raises(KeyError):
        runtime.call("not_a_tool", {})


def test_one_call_writes_query_coverage_inspection(runtime):
    env = runtime.call("t_container", {"hosts": ["Tomcat01"], "window": W})
    assert env.ok and env.row_count > 20
    q = qrow(runtime, env.query_id)
    assert q["status"] == "ok" and q["tool"] == "t_container" and q["result_digest"]
    cov = runtime.ledger.con.execute("SELECT kind, requested, returned FROM coverage").fetchall()
    assert cov[0][0] == "queried" and json.loads(cov[0][1]) == {"host": ["Tomcat01"]}
    assert json.loads(cov[0][2])["host"] == ["Tomcat01"]
    insp = runtime.ledger.con.execute("SELECT via, row_ranges FROM inspections").fetchall()
    assert insp[0][0] == "preview" and json.loads(insp[0][1])


def test_preview_mode_not_part_of_identity(runtime):
    a = runtime.call("t_container", {"hosts": ["Tomcat01"], "window": W})
    b = runtime.call("t_container", {"hosts": ["Tomcat01"], "window": W, "preview_mode": "head"})
    assert qrow(runtime, a.query_id)["args_sha256"] == qrow(runtime, b.query_id)["args_sha256"]
    assert b.preview_label.startswith("first 20")


def test_rows_ordered_total_and_row_ordinals(runtime):
    env = runtime.call("t_container", {"window": W})
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    assert [r["_row"] for r in rows] == list(range(len(rows)))
    assert env.order == {"by": ["timestamp_s", "cmdb_id", "kpi_name"], "total": True}
    assert runtime.call("t_ties", {}).order["total"] is False


def test_stratified_preview_deterministic_and_labelled(runtime):
    a = runtime.call("t_container", {"window": {"start": INCIDENT[0], "end": INCIDENT[1]}})
    b = runtime.call("t_container", {"window": {"start": INCIDENT[0], "end": INCIDENT[1]}})
    assert [r["_row"] for r in a.preview] == [r["_row"] for r in b.preview]
    hosts = {r["cmdb_id"] for r in a.preview}
    assert len(hosts) > 5  # spread across entities, not one host's head
    ts = sorted(r["timestamp_s"] for r in a.preview)
    assert ts[-1] - ts[0] > 600  # spread across time, not the first moment
    assert (
        "sample of" in a.preview_label and "rows" in a.preview_label and "host" in a.preview_label
    )
    assert a.preview_meta["n_shown"] == len(a.preview)
    assert "top" not in json.dumps(a.summary_stats)  # no top-K


def test_empty_because_values(runtime):
    # entity present in container metrics but never in spans
    env = runtime.call("t_spans", {"hosts": ["Mysql01"], "window": W})
    assert env.row_count == 0 and env.empty_because == "entity_absent_from_source"
    assert env.unobserved_entities == {"host": ["Mysql01"]}
    # spans hold nothing in the baseline window
    env = runtime.call(
        "t_spans", {"hosts": ["Tomcat01"], "window": {"start": BASELINE[0], "end": BASELINE[1]}}
    )
    assert env.empty_because == "no_data_in_window"
    # valid scope, predicate matched nothing
    env = runtime.call("t_container", {"hosts": ["Tomcat01"], "window": W, "min_value": 1e30})
    assert env.empty_because == "no_matches"
    assert runtime.call("t_unverified", {}).empty_because == "scope_unverified"
    assert runtime.call("t_join", {}).empty_because == "join_key_unpopulated"
    kinds = runtime.ledger.con.execute(
        "SELECT kind, attempt_reason FROM coverage ORDER BY coverage_id"
    ).fetchall()
    assert ("attempt", "entity_absent_from_source") in kinds and ("queried", None) in kinds
    assert ("attempt", "scope_unverified") in kinds


def test_aggregate_zero_over_unobserved_entity(runtime):
    env = runtime.call("t_span_count", {"hosts": ["Mysql01"], "window": W})
    assert env.row_count == 1 and env.preview[0]["n"] == 0
    assert env.unobserved_entities == {"host": ["Mysql01"]}
    assert json.loads(qrow(runtime, env.query_id)["unobserved_entities"]) == {"host": ["Mysql01"]}


def test_unknown_entity_is_invalid_query_with_suggestions(runtime):
    env = runtime.call("t_container", {"hosts": ["tomcat01"], "window": W})
    assert env.error.type == "invalid_query" and "Tomcat01" in env.error.suggestions
    st = runtime.ledger.state()
    assert len(st.failures) == 1
    cov = runtime.ledger.con.execute(
        "SELECT kind, attempt_reason, requested FROM coverage"
    ).fetchall()
    assert cov == [("attempt", "error", json.dumps({"host": ["tomcat01"]}))]
    assert runtime.call("t_container", {"window": W, "bogus": 1}).error.type == "invalid_query"


def test_too_large(runtime):
    runtime.row_cap = 50
    env = runtime.call("t_container", {"window": {"start": INCIDENT[0], "end": INCIDENT[1]}})
    assert env.error.type == "too_large" and "rows" in env.error.diagnostic


def test_inspect_pages_without_new_query(runtime):
    env = runtime.call("t_container", {"window": W})
    n_q = runtime.ledger.con.execute("SELECT count(*) FROM queries").fetchone()[0]
    page = runtime.inspect(env.query_id, offset=5, limit=3)
    assert [r["_row"] for r in page.preview] == [5, 6, 7] and page.query_id == env.query_id
    srt = runtime.inspect(env.query_id, sort="-value", limit=2)
    assert all("_row" in r for r in srt.preview)
    assert runtime.inspect(env.query_id, offset=10**6).error.type == "invalid_query"
    assert runtime.inspect(env.query_id, offset=-1).error.type == "invalid_query"
    assert runtime.inspect(env.query_id, preview_mode="tail").preview_label.startswith("last")
    assert runtime.ledger.con.execute("SELECT count(*) FROM queries").fetchone()[0] == n_q
    vias = [r[0] for r in runtime.ledger.con.execute("SELECT via FROM inspections").fetchall()]
    assert vias.count("inspect_result") == 2


def test_reexecute_reproduces_digest(runtime):
    env = runtime.call("t_container", {"hosts": ["Tomcat01", "Mysql01"], "window": W})
    q = qrow(runtime, env.query_id)
    again = runtime.reexecute(q)
    assert again["digest"] == q["result_digest"] and len(again["rows"]) == env.row_count
    assert (
        runtime.ledger.con.execute("SELECT count(*) FROM queries").fetchone()[0] == 1
    )  # not logged


def test_multi_scope_coverage(runtime):
    env = runtime.call("t_multi", {"window": W})
    cov = runtime.ledger.con.execute(
        "SELECT sources, kind, attempt_reason FROM coverage ORDER BY coverage_id"
    ).fetchall()
    assert cov == [('["spans"]', "queried", None), ('["logs"]', "attempt", "no_data_in_window")]
    assert env.notes["per_source"][1]["empty_because"] == "no_data_in_window"


def test_coverage_view_raw_lists(runtime):
    runtime.call("t_container", {"hosts": ["Tomcat01"], "window": W})
    runtime.call("t_spans", {"hosts": ["Mysql01"], "window": W})
    runtime.call("t_container", {"hosts": ["nope"], "window": W})
    view = runtime.coverage_view()
    assert len(view["queried"]) == 1 and view["queried"][0]["previewed"]
    assert view["unobserved_entities"][0]["requested"] == {"host": ["Mysql01"]}
    assert view["attempts"][0]["attempt_reason"] == "error"
    assert runtime.coverage_view(entity="Tomcat01")["queried"]


def test_store_connection_is_locked(runtime):
    for sql in [
        "SELECT * FROM read_csv('/etc/hostname')",
        "COPY spans TO 'x.csv'",
        "SET enable_external_access=true",
    ]:
        with pytest.raises(duckdb.Error):
            runtime.store.execute(sql)


def test_registry_is_the_only_write_path_to_coverage():
    from agentic_rca.ledger.models import OPS

    assert not any("coverage" in op or "quer" in op or "inspect" in op for op in OPS)
    assert "t_container" in TOOLS
