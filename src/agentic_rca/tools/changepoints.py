"""`tool-changepoints` (B2.3): detect_changepoints.

Runs a ruptures change-point detector over a metric series. The caller
must supply method and sensitivity — no defaults are provided here because
a hidden default would be a silent prior about which detector fits this
incident's metrics (contracts-b1 §5 prior 2, encoding-risk logged).

Output: one row per detected breakpoint per (entity, metric). All fields
are sortable; no ranking, no top-K cut, no anomaly flag. If the model
wants results ordered by score or magnitude, it uses inspect_result(sort=).
"""

from __future__ import annotations

from typing import Any

import ruptures
from pydantic import Field, field_validator

from agentic_rca.tools._common import Scope, ToolArgs, ToolInputError, Window
from agentic_rca.tools.registry import ToolResult, tool

# Methods ruptures exposes as string names (subset we accept)
_RUPTURES_MODELS: dict[str, type] = {
    "Pelt": ruptures.Pelt,
    "Binseg": ruptures.Binseg,
    "BottomUp": ruptures.BottomUp,
    "Window": ruptures.Window,
}

_SOURCE_COLS: dict[str, tuple[str, str]] = {
    "app_metrics": ("tc", "metric_name"),
    "container_metrics": ("cmdb_id", "kpi_name"),
}

_MIN_POINTS = 3  # ruptures minimum


class DetectChangepointsArgs(ToolArgs):
    source: str = Field(description="'app_metrics' or 'container_metrics'")
    entities: list[str] | None = Field(
        default=None,
        description="entity values to include (tc for app_metrics; cmdb_id for container_metrics); omit for all",
    )
    metrics: list[str] | None = Field(
        default=None,
        description="metric names to include; omit for all",
    )
    window: Window
    method: str = Field(
        description=(
            "ruptures model name: Pelt, Binseg, BottomUp, or Window. "
            "Required; no default is provided — the caller must choose."
        )
    )
    sensitivity: float = Field(
        gt=0,
        description=(
            "Detector sensitivity parameter. For Pelt/Binseg/BottomUp this is the penalty "
            "(lower = more breakpoints). Required; no default is provided."
        ),
    )

    @field_validator("source")
    @classmethod
    def _check_source(cls, v: str) -> str:
        if v not in _SOURCE_COLS:
            raise ValueError(f"source must be one of {list(_SOURCE_COLS)}, got {v!r}")
        return v

    @field_validator("method")
    @classmethod
    def _check_method(cls, v: str) -> str:
        if v not in _RUPTURES_MODELS:
            raise ValueError(
                f"method must be one of {list(_RUPTURES_MODELS)}, got {v!r}"
            )
        return v


def _detect_series(
    values: list[float], timestamps: list[float], method: str, sensitivity: float
) -> list[tuple[float, float, float]]:
    """Run ruptures on one series. Returns list of (changepoint_ts_s, magnitude, score)."""
    if len(values) < _MIN_POINTS:
        return []
    try:
        model_cls = _RUPTURES_MODELS[method]
        algo = model_cls(model="rbf").fit(
            [[v] for v in values]
        )
        # breakpoints: list of end-indices (1-indexed), last one = n
        bkps = algo.predict(pen=sensitivity)
    except Exception:
        return []

    results = []
    prev = 0
    for bkp in bkps[:-1]:  # skip the final n (end of series)
        before = values[prev:bkp]
        after = values[bkp:]
        if not before or not after:
            prev = bkp
            continue
        ts = timestamps[bkp] if bkp < len(timestamps) else timestamps[-1]
        mean_before = sum(before) / len(before)
        mean_after = sum(after) / len(after)
        magnitude = mean_after - mean_before
        # score = cost reduction at this breakpoint (sum of squared residuals)
        def sse(seg: list[float]) -> float:
            m = sum(seg) / len(seg)
            return sum((x - m) ** 2 for x in seg)

        score = sse(values[prev:]) - sse(before) - sse(after)
        results.append((ts, magnitude, score))
        prev = bkp
    return results


@tool(
    "detect_changepoints",
    DetectChangepointsArgs,
    description=(
        "Detect statistical changepoints in metric series using a ruptures model. "
        "method and sensitivity are required — no defaults are provided. "
        "Output: one row per detected breakpoint per (entity, metric). "
        "All fields are sortable; no ranking, top-K cut, or anomaly flag. "
        "Use inspect_result(sort=...) to reorder by score or magnitude."
    ),
    order_by=["entity", "metric", "changepoint_ts_s"],
    sources=["app_metrics", "container_metrics"],
)
def detect_changepoints(a: DetectChangepointsArgs, ctx) -> ToolResult:
    ec, mc = _SOURCE_COLS[a.source]
    entity_kind = "service" if ec == "tc" else "host"

    # Validate entity and metric values
    for kind, vals in [(entity_kind, a.entities or []), ("metric", a.metrics or [])]:
        if vals:
            ctx.index.check_known(kind, vals)

    # Build the filter
    clauses: list[str] = [
        "timestamp_s >= ? AND timestamp_s < ?",
    ]
    params: list[Any] = [a.window.start, a.window.end]
    if a.entities:
        ph = ",".join("?" for _ in a.entities)
        clauses.append(f'"{ec}" IN ({ph})')
        params += a.entities
    if a.metrics:
        ph = ",".join("?" for _ in a.metrics)
        clauses.append(f'"{mc}" IN ({ph})')
        params += a.metrics

    where = "WHERE " + " AND ".join(clauses)
    sql = (
        f'SELECT "{ec}" AS entity, "{mc}" AS metric, timestamp_s, value '
        f'FROM "{a.source}" {where} ORDER BY entity, metric, timestamp_s'
    )
    raw = ctx.con.execute(sql, params).fetchall()

    if not raw:
        # No rows — determine empty_because from scope
        req: dict[str, Any] = {}
        if a.entities:
            req[entity_kind] = a.entities
        if a.metrics:
            req["metric"] = a.metrics
        scope = Scope(
            source=a.source,
            requested=req,
            window=(a.window.start, a.window.end),
            row_count=0,
        )
        return ToolResult([], [scope], [sql])

    # Group by (entity, metric)
    series: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for entity, metric, ts, val in raw:
        key = (str(entity), str(metric))
        if key not in series:
            series[key] = ([], [])
        series[key][0].append(float(ts))
        series[key][1].append(float(val))

    rows = []
    for (entity, metric), (timestamps, values) in sorted(series.items()):
        bkps = _detect_series(values, timestamps, a.method, a.sensitivity)
        for (ts, magnitude, score) in bkps:
            rows.append({
                "entity": entity,
                "metric": metric,
                "changepoint_ts_s": ts,
                "magnitude": magnitude,
                "score": score,
                "n_points": len(values),
                "method": a.method,
                "sensitivity": a.sensitivity,
            })

    req_scope: dict[str, Any] = {}
    if a.entities:
        req_scope[entity_kind] = a.entities
    if a.metrics:
        req_scope["metric"] = a.metrics
    scope = Scope(
        source=a.source,
        requested=req_scope,
        window=(a.window.start, a.window.end),
        row_count=len(rows),
    )
    return ToolResult(rows, [scope], [sql])
