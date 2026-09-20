"""tool-generic-query (PLAN.md #6): sql scope extraction and certification."""
import agentic_rca.tools.catalog  # noqa: F401
from tests.conftest import INCIDENT

T0, T1 = INCIDENT


def q(rt, sql):
    return rt.call("sql", {"query": sql})


def test_verified_scope_certifies_no_matches(runtime):
    env = q(runtime, f"SELECT * FROM container_metrics WHERE cmdb_id = 'Tomcat01' AND timestamp_s >= {T0} AND timestamp_s < {T1} AND value > 1e30")
    assert env.ok and env.row_count == 0 and env.empty_because == "no_matches" and env.scope_verified


def test_unrecognised_shapes_are_unverified(runtime):
    for where in ("upper(cmdb_id) = 'TOMCAT01'", "(cmdb_id = 'Tomcat01' OR cmdb_id = 'Tomcat02')",
                  "cmdb_id LIKE 'zz%'", "cmdb_id IN (SELECT cmdb_id FROM logs WHERE 1=0)"):
        env = q(runtime, f"SELECT * FROM spans WHERE {where} AND duration_ms < -1")
        assert env.ok and env.row_count == 0, where
        assert env.empty_because == "scope_unverified" and not env.scope_verified, where


def test_aggregate_over_unobserved_entity_flagged(runtime):
    env = q(runtime, "SELECT count(*) AS n FROM spans WHERE cmdb_id = 'Mysql01'")
    assert env.row_count == 1 and env.unobserved_entities == {"host": ["Mysql01"]}


def test_decimal_window_constant(runtime):
    env = q(runtime, "SELECT * FROM spans WHERE cmdb_id = 'Tomcat01' AND timestamp_s >= 1614852000.5 AND timestamp_s < 1614852001.5 AND duration_ms < -1")
    assert env.empty_because == "no_matches"


def test_baseline_spans_no_data_in_window(runtime):
    env = q(runtime, "SELECT * FROM spans WHERE timestamp_s >= 1614848400 AND timestamp_s < 1614850200")
    assert env.empty_because == "no_data_in_window"


def test_rejections(runtime):
    assert q(runtime, "DELETE FROM spans").error.type == "invalid_query"
    assert q(runtime, "SELECT * FROM spans LIMIT 5").error.type == "invalid_query"
    assert "Tomcat01" in q(runtime, "SELECT * FROM spans WHERE cmdb_id = 'tomcat01'").error.suggestions
    assert q(runtime, "SELECT * FROM read_csv('/etc/hostname')").error.type == "invalid_query"
    assert q(runtime, "SELECT * FROM spns").error.type == "invalid_query"


def test_default_total_order_and_reexecute(runtime):
    env = q(runtime, "SELECT DISTINCT cmdb_id FROM spans")
    assert env.order == {"by": ["ALL"], "total": True}
    row = runtime.ledger.query_row(runtime.ledger.q(env.query_id))
    assert runtime.reexecute(row)["digest"] == row["result_digest"]


def test_discovery_tools(runtime):
    env = runtime.call("list_entities", {"kind": "host", "filter": "tomcat"})
    assert env.ok and {r["source"] for r in runtime.handles[runtime.ledger.q(env.query_id)]} >= {"spans", "container_metrics"}
    assert runtime.call("system_facts_lookup", {"query": "x"}).empty_because == "no_matches"
    assert runtime.call("describe_dataset", {}).row_count > 5
