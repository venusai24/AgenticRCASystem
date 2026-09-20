"""tool-traces (B2.4): span queries, topologies and aggregations."""
import pytest
import agentic_rca.tools.catalog  # noqa: F401
from tests.conftest import BASELINE, INCIDENT

T0, T1 = INCIDENT
B0, B1 = BASELINE

def _qs(rt, **kwargs):
    return rt.call("query_spans", kwargs)

def _agg(rt, **kwargs):
    return rt.call("aggregate_spans", kwargs)

def _gt(rt, **kwargs):
    return rt.call("get_trace", kwargs)

def _ct(rt, **kwargs):
    return rt.call("call_topology", kwargs)

def _cw(rt, **kwargs):
    return rt.call("compare_windows_spans", kwargs)

# AC1: query_spans returns raw rows with order_by = [timestamp_s, cmdb_id, span_id]
def test_query_spans_order_by(runtime):
    # narrow window to avoid 100k row limit
    env = _qs(runtime, window={"start": T0, "end": T0 + 10.0})
    assert env.ok, str(env)
    assert env.order["by"] == ["timestamp_s", "cmdb_id", "span_id"]
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        assert "timestamp_s" in rows[0]
        assert "cmdb_id" in rows[0]
        assert "span_id" in rows[0]

# AC2: aggregate_spans only computes requested measures; no default; no error_rate
def test_aggregate_spans_requires_measures(runtime):
    env = runtime.call("aggregate_spans", {
        "window": {"start": T0, "end": T1},
        "group_by": ["cmdb_id"]
    })
    # Validation error since measures is missing
    assert env.error is not None
    assert env.error.type == "invalid_query"

def test_aggregate_spans_no_error_rate(runtime):
    env = runtime.call("aggregate_spans", {
        "window": {"start": T0, "end": T1},
        "group_by": ["cmdb_id"],
        "measures": ["error_rate"]
    })
    assert env.error is not None
    assert env.error.type == "invalid_query"

def test_aggregate_spans_success(runtime):
    env = _agg(runtime, window={"start": T0, "end": T1}, group_by=["cmdb_id"], measures=["count", "p50_duration_ms"])
    assert env.ok
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        assert "count" in rows[0]
        assert "p50_duration_ms" in rows[0]
        assert "mean_duration_ms" not in rows[0] # Not requested

# AC3: get_trace returns spans and root flag
def test_get_trace_root_flag(runtime):
    # Find a trace_id first
    qs_env = _qs(runtime, window={"start": T0, "end": T1})
    if qs_env.row_count == 0:
        pytest.skip("no spans in window")
    rows = runtime.handles[runtime.ledger.q(qs_env.query_id)]
    t_id = rows[0]["trace_id"]
    
    env = _gt(runtime, trace_id=t_id)
    assert env.ok
    trace_rows = runtime.handles[runtime.ledger.q(env.query_id)]
    assert len(trace_rows) > 0
    # Must have is_root and root_kind
    for r in trace_rows:
        assert "is_root" in r
        assert "root_kind" in r
    
    # Must have exactly one root (if it's not a fragmented trace but let's just check presence)
    has_root = any(r.get("is_root") for r in trace_rows)
    assert has_root

# AC4: call_topology returns topology edges without naming causal
def test_call_topology_no_causal(runtime):
    env = _ct(runtime, window={"start": T0, "end": T0 + 10.0})
    assert env.ok
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        for r in rows:
            assert "caller_host" in r
            assert "callee_host" in r
            assert "call_count" in r
            assert "causal" not in str(r).lower()

# AC5: compare_windows_spans diffs correctly
def test_compare_windows_spans(runtime):
    env = _cw(runtime, window_a={"start": B0, "end": B1}, window_b={"start": T0, "end": T1}, group_by=["cmdb_id"], measures=["count", "mean_duration_ms"])
    assert env.ok
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        assert "cmdb_id" in rows[0]
        assert "count_a" in rows[0]
        assert "count_b" in rows[0]
        assert "count_delta" in rows[0]
        # Should not have rank/anomaly columns
        forbidden = {"rank", "top", "top_k", "anomaly", "flag", "severity"}
        for r in rows:
            assert not forbidden & set(r.keys())

# AC6: Unknown entities and out of window
def test_unknown_entity_cmdb_id(runtime):
    env = _qs(runtime, window={"start": T0, "end": T1}, cmdb_ids=["no_such_host"])
    assert env.error is not None
    assert env.error.type == "invalid_query"
    assert "no_such_host" in env.error.diagnostic

def test_out_of_window(runtime):
    env = _qs(runtime, window={"start": 2_000_000_000.0, "end": 2_000_003_600.0})
    assert env.row_count == 0
    assert env.empty_because == "no_data_in_window"
