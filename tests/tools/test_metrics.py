"""tool-metrics (B2.2): query_metrics, compare_windows_metrics, cross_correlate."""
import agentic_rca.tools.catalog  # noqa: F401
from tests.conftest import BASELINE, INCIDENT

T0, T1 = INCIDENT
B0, B1 = BASELINE


def _qm(rt, **kwargs):
    return rt.call("query_metrics", kwargs)


def _cw(rt, **kwargs):
    return rt.call("compare_windows_metrics", kwargs)


def _cc(rt, **kwargs):
    return rt.call("cross_correlate", kwargs)


# AC1: query_metrics returns rows for a valid in-window request
def test_query_metrics_returns_rows(runtime):
    env = _qm(runtime, source="app_metrics", window={"start": T0, "end": T1})
    assert env.ok and env.row_count > 0


# AC1 neutral order: order_by is total (timestamp, entity, metric)
def test_query_metrics_order_total(runtime):
    env = _qm(runtime, source="app_metrics", window={"start": T0, "end": T1})
    assert env.order["total"] is True


# AC1 empty: out-of-window → no_data_in_window
def test_query_metrics_no_data_in_window(runtime):
    # Far-future window with nothing
    env = _qm(runtime, source="app_metrics",
               window={"start": 2_000_000_000.0, "end": 2_000_003_600.0})
    assert env.row_count == 0 and env.empty_because == "no_data_in_window"


# AC1 aggregation: mean aggregation returns one row per (entity, metric)
def test_query_metrics_aggregation(runtime):
    env = _qm(runtime, source="app_metrics", window={"start": T0, "end": T1},
               aggregation="mean")
    assert env.ok and env.row_count > 0
    # No timestamp column — it's aggregated
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    assert all("bucket_start_s" not in r for r in rows)


# AC2: compare_windows diffs by identity key, not time alignment
def test_compare_windows_by_identity(runtime):
    env = _cw(runtime, source="app_metrics",
               window_a={"start": B0, "end": B1},
               window_b={"start": T0, "end": T1})
    assert env.ok and env.row_count > 0
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    # Every row has identity keys (entity + metric); delta is numeric or null
    for r in rows:
        assert "entity" in r and "metric" in r
        # delta may be null if a series is in only one window
        assert "delta" in r


# AC3: no ranking columns in compare_windows output
def test_compare_windows_no_ranking(runtime):
    env = _cw(runtime, source="app_metrics",
               window_a={"start": B0, "end": B1},
               window_b={"start": T0, "end": T1})
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    forbidden = {"rank", "top_k", "score", "anomaly", "flag", "severity"}
    for r in rows:
        assert not forbidden & set(r.keys()), f"Ranking column found: {set(r.keys()) & forbidden}"


# AC4: cross_correlate returns rows for every lag; no causal verdict column
def test_cross_correlate_all_lags(runtime):
    # Need two actual series; query what exists to pick entity/metric values
    app_rows = runtime.call("list_entities", {"kind": "service"})
    services = list({r["entity"] for r in runtime.handles[runtime.ledger.q(app_rows.query_id)]})
    if len(services) < 1:
        import pytest
        pytest.skip("no services in store")

    metric_rows = runtime.call("list_entities", {"kind": "metric"})
    metrics = list({r["entity"] for r in runtime.handles[runtime.ledger.q(metric_rows.query_id)]
                    if r["source"] == "app_metrics"})
    if len(metrics) < 2:
        import pytest
        pytest.skip("fewer than 2 metrics in app_metrics")

    svc = services[0]
    m1, m2 = metrics[0], metrics[1]
    env = _cc(runtime, source="app_metrics",
               series_a={"entity": svc, "metric": m1},
               series_b={"entity": svc, "metric": m2},
               window={"start": T0, "end": T1},
               max_lag_s=120.0)
    assert env.ok
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    assert len(rows) > 0
    # All lags are sortable (numeric)
    for r in rows:
        assert isinstance(r["lag_s"], (int, float))
    # No causal label
    forbidden = {"cause", "effect", "causal", "verdict"}
    for r in rows:
        assert not forbidden & set(r.keys())


# AC5: unknown entity → invalid_query with suggestions
def test_unknown_entity_invalid_query(runtime):
    env = _qm(runtime, source="app_metrics",
               entities=["no_such_service_xyz"],
               window={"start": T0, "end": T1})
    assert env.error is not None and env.error.type == "invalid_query"
    # suggestions are best-effort (difflib); the diagnostic must name the unknown value
    assert "no_such_service_xyz" in env.error.diagnostic


# AC7: entity_absent_from_source (entity exists globally but not in this source)
# Hosts in container_metrics include DB/cache tiers that never appear in spans.
# query that source=container_metrics with an entity known globally but absent from spans.
# We use the sql tool to confirm a host absent from spans but present elsewhere, then
# verify query_metrics emits entity_absent_from_source when asked for that host in spans.
# Since query_metrics only covers app/container metrics, we test via the sql escape hatch:
# any host in container_metrics that has no rows in spans triggers entity_absent_from_source.
def test_entity_absent_from_source(runtime):
    # Get hosts from spans (the subset that appear there)
    span_hosts_env = runtime.call("sql", {"query": "SELECT DISTINCT cmdb_id FROM spans"})
    span_host_rows = runtime.handles[runtime.ledger.q(span_hosts_env.query_id)]
    span_hosts = {r["cmdb_id"] for r in span_host_rows}

    # Get all hosts from container_metrics
    cm_hosts_env = runtime.call("list_entities", {"kind": "host", "filter": ""})
    cm_host_rows = runtime.handles[runtime.ledger.q(cm_hosts_env.query_id)]
    cm_hosts = {r["entity"] for r in cm_host_rows if r["source"] == "container_metrics"}

    # Hosts in container_metrics but not in spans (DB/cache tier, e.g. Mysql*, Redis*, apache*)
    absent_from_spans = sorted(cm_hosts - span_hosts)
    if not absent_from_spans:
        import pytest
        pytest.skip("all container_metrics hosts also appear in spans")

    host = absent_from_spans[0]
    # query_metrics on container_metrics with that host should succeed (host is in that source)
    env = _qm(runtime, source="container_metrics",
               entities=[host], window={"start": T0, "end": T1})
    # The host IS in container_metrics so this should return rows (or no_matches if kpi absent)
    assert env.error is None  # no invalid_query — entity is known in this source

    # Now verify entity_absent_from_source via sql: a query to spans for this host
    env2 = runtime.call("sql", {
        "query": f"SELECT * FROM spans WHERE cmdb_id = '{host}' AND timestamp_s >= {T0} AND timestamp_s < {T1} AND duration_ms < -1"
    })
    assert env2.empty_because == "entity_absent_from_source", \
        f"Expected entity_absent_from_source for {host} in spans, got {env2.empty_because}"
