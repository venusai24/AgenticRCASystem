"""tool-changepoints (B2.3): detect_changepoints."""
import pytest
import agentic_rca.tools.catalog  # noqa: F401
from tests.conftest import INCIDENT

T0, T1 = INCIDENT


def _cp(rt, **kwargs):
    return rt.call("detect_changepoints", kwargs)


# AC1: returns rows with required columns for valid in-window request
def test_detect_returns_expected_columns(runtime):
    env = _cp(runtime, source="app_metrics",
               window={"start": T0, "end": T1},
               method="Pelt", sensitivity=10.0)
    assert env.ok
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        required = {"entity", "metric", "changepoint_ts_s", "magnitude", "score",
                    "n_points", "method", "sensitivity"}
        assert required <= set(rows[0].keys()), f"Missing columns: {required - set(rows[0].keys())}"


# AC2: method is required — invalid value raises invalid_query
def test_method_required(runtime):
    # omit method entirely: should fail pydantic validation → invalid_query
    env = runtime.call("detect_changepoints", {
        "source": "app_metrics",
        "window": {"start": T0, "end": T1},
        "sensitivity": 10.0,
        # no "method"
    })
    assert env.error is not None and env.error.type == "invalid_query"


def test_invalid_method_name(runtime):
    env = _cp(runtime, source="app_metrics",
               window={"start": T0, "end": T1},
               method="NonExistentModel", sensitivity=10.0)
    assert env.error is not None and env.error.type == "invalid_query"


# AC2: sensitivity is required — invalid value raises invalid_query
def test_sensitivity_required(runtime):
    env = runtime.call("detect_changepoints", {
        "source": "app_metrics",
        "window": {"start": T0, "end": T1},
        "method": "Pelt",
        # no "sensitivity"
    })
    assert env.error is not None and env.error.type == "invalid_query"


# AC3: no ranking columns in output
def test_no_ranking_columns(runtime):
    env = _cp(runtime, source="app_metrics",
               window={"start": T0, "end": T1},
               method="Pelt", sensitivity=10.0)
    assert env.ok
    if env.row_count > 0:
        rows = runtime.handles[runtime.ledger.q(env.query_id)]
        forbidden = {"rank", "top", "top_k", "anomaly", "flag", "severity"}
        for r in rows:
            assert not forbidden & set(r.keys())


# AC5: unknown entity → invalid_query
def test_unknown_entity(runtime):
    env = _cp(runtime, source="app_metrics",
               entities=["no_such_service_zzz"],
               window={"start": T0, "end": T1},
               method="Pelt", sensitivity=10.0)
    assert env.error is not None and env.error.type == "invalid_query"
    assert "no_such_service_zzz" in env.error.diagnostic


# AC6: out-of-window → no_data_in_window
def test_no_data_in_window(runtime):
    env = _cp(runtime, source="app_metrics",
               window={"start": 2_000_000_000.0, "end": 2_000_003_600.0},
               method="Pelt", sensitivity=10.0)
    assert env.row_count == 0 and env.empty_because == "no_data_in_window"


# AC7: order_by is neutral (entity, metric, time) — not score or magnitude
def test_order_by_is_neutral(runtime):
    env = _cp(runtime, source="app_metrics",
               window={"start": T0, "end": T1},
               method="Pelt", sensitivity=10.0)
    assert env.ok
    order = env.order or {}
    cols = order.get("by", [])
    assert "score" not in cols
    assert "magnitude" not in cols


# AC1 alternative method: Binseg also accepted
def test_binseg_method_accepted(runtime):
    env = _cp(runtime, source="app_metrics",
               window={"start": T0, "end": T1},
               method="Binseg", sensitivity=5.0)
    assert env.ok  # may return 0 rows if no changepoints found — that's fine
