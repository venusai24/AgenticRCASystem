"""tool-crosssource (B2.6): tests covering all acceptance criteria."""
import pytest
from pydantic import ValidationError
import agentic_rca.tools.catalog  # noqa: F401
from agentic_rca.tools._common import ToolInputError
from tests.conftest import BASELINE, INCIDENT

T0, T1 = INCIDENT
B0, B1 = BASELINE

def _tl(rt, **kw):
    return rt.call("timeline", kw)


# ---- AC1: join_by_trace scopes ----
def test_join_by_trace_scopes(runtime):
    # First get a valid trace_id from spans
    env_spans = runtime.call("query_spans", {"window": {"start": T0, "end": T1}})
    if env_spans.row_count == 0:
        pytest.skip("no spans")
    tid = runtime.handles[runtime.ledger.q(env_spans.query_id)][0]["trace_id"]

    env = runtime.call("join_by_trace", {"trace_id": tid})
    assert env.ok, str(env)
    
    # We must have two scopes recorded in the envelope coverage
    scopes = env.extra_stats.get("per_source", [])
    if not scopes:
        # Check if the registry dumped scopes into the envelope directly
        # The tool result itself has 2 scopes. The registry writes them.
        pass
    
    # Check what registry populated in envelope
    assert len(env.per_source) == 2
    logs_scope = next(s for s in env.per_source if s["source"] == "logs")
    spans_scope = next(s for s in env.per_source if s["source"] == "spans")
    
    assert logs_scope["empty_because"] == "join_key_unpopulated"
    assert spans_scope["empty_because"] is None

# ---- AC2: timeline schema & AC9/4 extra_stats ----
def test_timeline_schema_and_extra_stats(runtime):
    env = _tl(runtime, window={"start": T0, "end": T0 + 5.0}, sources=["spans", "logs"], series={})
    assert env.ok, str(env)
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        r = rows[0]
        expected_cols = {
            "_row", "ts_s", "ts_granularity_s", "source", "cmdb_id", "tc",
            "trace_id", "span_id", "parent_id", "duration_ms",
            "log_id", "log_name", "template_id", "text",
            "metric_name", "kpi_name", "value"
        }
        assert set(r.keys()) == expected_cols

    assert "clock_offsets" in env.summary_stats
    assert "ordering_note" in env.summary_stats
    assert "filter_notes" in env.summary_stats
    assert "per_source" in env.notes


# ---- AC3: total order ----
def test_timeline_order(runtime):
    env = _tl(runtime, window={"start": T0, "end": T0 + 5.0}, sources=["spans", "logs"], series={})
    assert env.order["by"] == ["ts_s", "source", "cmdb_id", "tc", "kpi_name", "metric_name", "span_id", "log_id"]


# ---- AC5: strict entity applicability ----
def test_timeline_strict_applicability(runtime):
    # host filter with app_metrics (which lacks host)
    env = _tl(runtime, window={"start": T0, "end": T0 + 1.0}, sources=["app_metrics"], entities={"host": "something"}, series={"app_metrics": "*"})
    assert env.error is not None
    assert env.error.type == "invalid_query"
    assert "app_metrics" in str(env.error.diagnostic)


# ---- AC6: series bidirectional validation ----
def test_timeline_series_validation(runtime):
    # missing series key for requested source
    env = _tl(runtime, window={"start": T0, "end": T0 + 1.0}, sources=["app_metrics"], series={})
    assert env.error is not None
    assert env.error.type == "invalid_query"
    assert "app_metrics" in str(env.error.diagnostic)

    # series key provided for source NOT in sources
    env2 = _tl(runtime, window={"start": T0, "end": T0 + 1.0}, sources=["spans"], series={"app_metrics": "*"})
    assert env2.error is not None
    assert env2.error.type == "invalid_query"
    assert "app_metrics" in str(env2.error.diagnostic)


# ---- AC7: too_large ----
def test_timeline_too_large(runtime):
    # Set the registry cap artificially low for the test
    original_cap = runtime.row_cap
    runtime.row_cap = 10
    try:
        env = _tl(runtime, window={"start": T0, "end": T1}, sources=["spans", "logs"], series={})
        assert env.error is not None
        assert env.error.type == "too_large"
        assert "spans" in str(env.error.diagnostic)
        assert "logs" in str(env.error.diagnostic)
    finally:
        runtime.row_cap = original_cap


# ---- AC8: zero-row source ----
def test_timeline_zero_row_source(runtime):
    # B0 baseline window has NO logs or spans, only metrics.
    env = _tl(runtime, window={"start": B0, "end": B1}, sources=["logs", "container_metrics"], series={"container_metrics": "*"})
    assert env.ok, str(env)
    
    logs_scope = next(s for s in env.notes["per_source"] if s["source"] == "logs")
    metrics_scope = next(s for s in env.notes["per_source"] if s["source"] == "container_metrics")
    
    assert logs_scope["empty_because"] == "no_data_in_window"
    assert metrics_scope["empty_because"] is None
