"""tool-logs (B2.5): tests covering all acceptance criteria."""
import pytest
import agentic_rca.tools.catalog  # noqa: F401
from tests.conftest import BASELINE, INCIDENT

T0, T1 = INCIDENT
B0, B1 = BASELINE


def _sl(rt, **kw):
    return rt.call("search_logs", kw)

def _lt(rt, **kw):
    return rt.call("log_templates", kw)

def _ts(rt, **kw):
    return rt.call("template_timeseries", kw)

def _cw(rt, **kw):
    return rt.call("compare_windows_logs", kw)

def _gc(rt, **kw):
    return rt.call("get_log_context", kw)


# ---- AC1: search_logs returns value_raw, no is_error/severity ----

def test_search_logs_returns_value_raw(runtime):
    env = _sl(runtime, window={"start": T0, "end": T0 + 60.0})
    assert env.ok, str(env)
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        assert "value_raw" in rows[0]
        assert "is_error" not in rows[0]
        assert "severity" not in rows[0]

def test_search_logs_order_by(runtime):
    env = _sl(runtime, window={"start": T0, "end": T0 + 60.0})
    assert env.order["by"] == ["timestamp_s", "cmdb_id", "log_id"]

def test_search_logs_keyword_filter(runtime):
    env = _sl(runtime, window={"start": T0, "end": T1}, keyword="POST")
    assert env.ok, str(env)

def test_search_logs_log_name_filter(runtime):
    env = _sl(runtime, window={"start": T0, "end": T0 + 60.0}, log_names=["apache_access_log"])
    assert env.ok, str(env)


# ---- AC2: log_templates returns template registry ----

def test_log_templates_all(runtime):
    env = _lt(runtime)
    assert env.ok, str(env)
    assert env.row_count > 0
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    assert "template_id" in rows[0]
    assert "pattern" in rows[0]
    assert "log_name" in rows[0]
    # Must not have data columns
    assert "value_raw" not in rows[0]

def test_log_templates_filtered(runtime):
    env = _lt(runtime, log_names=["gc"])
    assert env.ok, str(env)
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        assert all(r["log_name"] == "gc" for r in rows)


# ---- AC3: template_timeseries requires resolution_s and template_ids ----

def test_template_timeseries_requires_resolution(runtime):
    # get a valid template_id first
    env = _lt(runtime)
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    tid = rows[0]["template_id"]

    result = runtime.call("template_timeseries", {
        "window": {"start": T0, "end": T1},
        "template_ids": [tid],
        # resolution_s deliberately omitted
    })
    # pydantic validation should reject
    assert result.error is not None
    assert result.error.type == "invalid_query"

def test_template_timeseries_requires_template_ids(runtime):
    result = runtime.call("template_timeseries", {
        "window": {"start": T0, "end": T1},
        "resolution_s": 60.0,
        # template_ids deliberately omitted
    })
    assert result.error is not None
    assert result.error.type == "invalid_query"

def test_template_timeseries_success(runtime):
    env = _lt(runtime)
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    tid = rows[0]["template_id"]

    env2 = _ts(runtime, window={"start": T0, "end": T1}, resolution_s=300.0, template_ids=[tid])
    assert env2.ok, str(env2)
    assert env2.order["by"] == ["bucket_start_s", "template_id"]


# ---- AC4: compare_windows_logs FULL OUTER JOIN ----

def test_compare_windows_logs_diff(runtime):
    env = _cw(runtime, window_a={"start": B0, "end": B1}, window_b={"start": T0, "end": T1}, group_by=["cmdb_id", "log_name"])
    assert env.ok, str(env)
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        assert "cmdb_id" in rows[0]
        assert "log_name" in rows[0]
        assert "count_a" in rows[0]
        assert "count_b" in rows[0]
        assert "count_delta" in rows[0]
        # must not have ranking columns
        forbidden = {"rank", "top", "anomaly", "flag", "severity", "is_error"}
        assert not forbidden & set(rows[0].keys())

def test_compare_windows_logs_invalid_group_by(runtime):
    env = runtime.call("compare_windows_logs", {
        "window_a": {"start": B0, "end": B1},
        "window_b": {"start": T0, "end": T1},
        "group_by": ["nonexistent_field"]
    })
    assert env.error is not None
    assert env.error.type == "invalid_query"


# ---- AC5: get_log_context returns ordered context around anchor ----

def test_get_log_context(runtime):
    # find a real log_id
    env = _sl(runtime, window={"start": T0, "end": T0 + 60.0})
    if env.row_count == 0:
        pytest.skip("no logs in window")
    rows = runtime.handles[runtime.ledger.q(env.query_id)]
    target_id = rows[0]["log_id"]

    env2 = _gc(runtime, log_id=target_id, n_before=3, n_after=3)
    assert env2.ok, str(env2)
    ctx_rows = runtime.handles[runtime.ledger.q(env2.query_id)]
    assert any(r["log_id"] == target_id for r in ctx_rows)
    # chronological order
    ts_seq = [r["timestamp_s"] for r in ctx_rows]
    assert ts_seq == sorted(ts_seq)


# ---- AC6: unknown entity / out-of-window ----

def test_unknown_cmdb_id(runtime):
    env = _sl(runtime, window={"start": T0, "end": T1}, cmdb_ids=["no_such_host"])
    assert env.error is not None
    assert env.error.type == "invalid_query"
    assert "no_such_host" in env.error.diagnostic

def test_unknown_log_name(runtime):
    env = _sl(runtime, window={"start": T0, "end": T1}, log_names=["no_such_log"])
    assert env.error is not None
    assert env.error.type == "invalid_query"

def test_unknown_template_id(runtime):
    env = _sl(runtime, window={"start": T0, "end": T1}, template_ids=["T9999"])
    assert env.error is not None
    assert env.error.type == "invalid_query"

def test_out_of_window(runtime):
    env = _sl(runtime, window={"start": 2_000_000_000.0, "end": 2_000_003_600.0})
    assert env.row_count == 0
    assert env.empty_because == "no_data_in_window"

def test_get_log_context_bad_id(runtime):
    env = _gc(runtime, log_id="NO_SUCH_LOG_ID", n_before=2, n_after=2)
    assert env.error is not None
    assert env.error.type == "invalid_query"
