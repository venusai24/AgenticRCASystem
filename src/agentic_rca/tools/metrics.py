"""`tool-metrics` (B2.2): query_metrics, compare_windows_metrics, cross_correlate.

Three neutral reads over app_metrics and container_metrics. No anomaly
scoring, no ranking, no top-K cuts, no hardcoded entity names or windows.
The model picks what to query, what windows to compare, and how much lag
to explore. Each tool outputs raw facts; interpretation is the model's job.

Cross-correlate is labelled "correlation, not causation" in its description.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from agentic_rca.tools._common import Scope, ToolArgs, Window
from agentic_rca.tools.registry import ToolResult, tool

# ---- shared helpers ----------------------------------------------------------

_SOURCE_ENTITY: dict[str, tuple[str, str]] = {
    "app_metrics": ("tc", "metric_name"),
    "container_metrics": ("cmdb_id", "kpi_name"),
}


def _entity_col(source: str) -> str:
    return _SOURCE_ENTITY[source][0]


def _metric_col(source: str) -> str:
    return _SOURCE_ENTITY[source][1]


def _where_params(
    source: str,
    entities: list[str] | None,
    metrics: list[str] | None,
    window: Window | None,
) -> tuple[str, list[Any]]:
    ec, mc = _entity_col(source), _metric_col(source)
    clauses, params = [], []
    if window:
        clauses.append("timestamp_s >= ? AND timestamp_s < ?")
        params += [window.start, window.end]
    if entities:
        placeholders = ",".join("?" for _ in entities)
        clauses.append(f'"{ec}" IN ({placeholders})')
        params += entities
    if metrics:
        placeholders = ",".join("?" for _ in metrics)
        clauses.append(f'"{mc}" IN ({placeholders})')
        params += metrics
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# ---- query_metrics -----------------------------------------------------------


class QueryMetricsArgs(ToolArgs):
    source: Literal["app_metrics", "container_metrics"]
    entities: list[str] | None = Field(
        default=None,
        description="entity values to include (tc for app_metrics; cmdb_id for container_metrics); omit for all",
    )
    metrics: list[str] | None = Field(
        default=None,
        description="metric names to include (metric_name / kpi_name); omit for all",
    )
    window: Window | None = None
    aggregation: Literal["none", "mean", "min", "max", "sum"] = Field(
        default="none",
        description="time aggregation over the window; 'none' returns raw rows",
    )
    resolution_s: float | None = Field(
        default=None,
        description="bucket width in seconds for time-bucketed output; requires aggregation != none",
    )

    @model_validator(mode="after")
    def _check_resolution(self) -> "QueryMetricsArgs":
        if self.resolution_s is not None and self.aggregation == "none":
            raise ValueError("resolution_s requires aggregation != 'none'")
        return self


@tool(
    "query_metrics",
    QueryMetricsArgs,
    description=(
        "Raw or time-bucketed metric values from app_metrics (service KPIs) or "
        "container_metrics (host KPIs). Returns rows in neutral order; no ranking or anomaly flag. "
        "Specify entities and metrics to narrow scope; omit to see all."
    ),
    order_by=["timestamp_s", "entity", "metric"],
    sources=["app_metrics", "container_metrics"],
)
def query_metrics(a: QueryMetricsArgs, ctx) -> ToolResult:
    ec, mc = _entity_col(a.source), _metric_col(a.source)
    for kind, vals in [("service" if ec == "tc" else "host", a.entities or []),
                       ("metric", a.metrics or [])]:
        if vals:
            ctx.index.check_known(kind, vals)
    where, params = _where_params(a.source, a.entities, a.metrics, a.window)
    if a.aggregation == "none":
        sql = f'SELECT timestamp_s, "{ec}" AS entity, "{mc}" AS metric, value FROM "{a.source}" {where} ORDER BY timestamp_s, entity, metric'
        rows = ctx.fetch(sql, params)
        order_by = ["timestamp_s", "entity", "metric"]
    else:
        agg_fn = a.aggregation.upper()
        if a.resolution_s:
            bucket = f"floor(timestamp_s / {a.resolution_s}) * {a.resolution_s}"
            sql = (
                f'SELECT {bucket} AS bucket_start_s, "{ec}" AS entity, "{mc}" AS metric, '
                f'{agg_fn}(value) AS value FROM "{a.source}" {where} '
                f'GROUP BY 1, 2, 3 ORDER BY bucket_start_s, entity, metric'
            )
            order_by = ["bucket_start_s", "entity", "metric"]
        else:
            sql = (
                f'SELECT "{ec}" AS entity, "{mc}" AS metric, {agg_fn}(value) AS value, '
                f'count(*) AS n_rows FROM "{a.source}" {where} '
                f'GROUP BY 1, 2 ORDER BY entity, metric'
            )
            order_by = ["entity", "metric"]
        rows = ctx.fetch(sql, params)

    w = (a.window.start, a.window.end) if a.window else None
    req: dict[str, Any] = {}
    if a.entities:
        kind = "service" if ec == "tc" else "host"
        req[kind] = a.entities
    if a.metrics:
        req["metric"] = a.metrics
    scope = Scope(source=a.source, requested=req, window=w, row_count=len(rows))
    return ToolResult(rows, [scope], [sql], order_by=order_by)


# ---- compare_windows_metrics -------------------------------------------------


class CompareWindowsMetricsArgs(ToolArgs):
    source: Literal["app_metrics", "container_metrics"]
    window_a: Window = Field(description="first window (e.g. baseline)")
    window_b: Window = Field(description="second window (e.g. incident)")
    entities: list[str] | None = Field(default=None, description="restrict to these entity values")
    metrics: list[str] | None = Field(default=None, description="restrict to these metric names")


@tool(
    "compare_windows_metrics",
    CompareWindowsMetricsArgs,
    description=(
        "Per-(entity, metric) distribution statistics for window_a vs window_b: "
        "mean, min, max, count, delta and pct_change. Diff is keyed by identity "
        "(tc+metric_name for app, cmdb_id+kpi_name for container), never by time alignment. "
        "Series present in only one window appear with null stats for the missing window. "
        "Output is in neutral (entity, metric) order — no ranking by magnitude."
    ),
    order_by=["entity", "metric"],
    sources=["app_metrics", "container_metrics"],
)
def compare_windows_metrics(a: CompareWindowsMetricsArgs, ctx) -> ToolResult:
    ec, mc = _entity_col(a.source), _metric_col(a.source)
    for kind, vals in [("service" if ec == "tc" else "host", a.entities or []),
                       ("metric", a.metrics or [])]:
        if vals:
            ctx.index.check_known(kind, vals)

    extra_where = []
    extra_params: list[Any] = []
    if a.entities:
        ph = ",".join("?" for _ in a.entities)
        extra_where.append(f'"{ec}" IN ({ph})')
        extra_params += a.entities
    if a.metrics:
        ph = ",".join("?" for _ in a.metrics)
        extra_where.append(f'"{mc}" IN ({ph})')
        extra_params += a.metrics
    ew = (" AND " + " AND ".join(extra_where)) if extra_where else ""

    sql = f"""
WITH a AS (
    SELECT "{ec}" AS entity, "{mc}" AS metric,
           AVG(value) AS mean_a, MIN(value) AS min_a, MAX(value) AS max_a, COUNT(*) AS count_a
    FROM "{a.source}"
    WHERE timestamp_s >= ? AND timestamp_s < ?{ew}
    GROUP BY 1, 2
),
b AS (
    SELECT "{ec}" AS entity, "{mc}" AS metric,
           AVG(value) AS mean_b, MIN(value) AS min_b, MAX(value) AS max_b, COUNT(*) AS count_b
    FROM "{a.source}"
    WHERE timestamp_s >= ? AND timestamp_s < ?{ew}
    GROUP BY 1, 2
)
SELECT
    COALESCE(a.entity, b.entity) AS entity,
    COALESCE(a.metric, b.metric) AS metric,
    a.mean_a, a.min_a, a.max_a, a.count_a,
    b.mean_b, b.min_b, b.max_b, b.count_b,
    (b.mean_b - a.mean_a) AS delta,
    CASE WHEN a.mean_a IS NOT NULL AND a.mean_a != 0
         THEN (b.mean_b - a.mean_a) / ABS(a.mean_a) * 100.0
         ELSE NULL END AS pct_change
FROM a
FULL OUTER JOIN b ON a.entity = b.entity AND a.metric = b.metric
ORDER BY entity, metric
"""
    params = [a.window_a.start, a.window_a.end] + extra_params + \
             [a.window_b.start, a.window_b.end] + extra_params
    rows = ctx.fetch(sql, params)
    req: dict[str, Any] = {}
    if a.entities:
        req["service" if ec == "tc" else "host"] = a.entities
    if a.metrics:
        req["metric"] = a.metrics
    # Two scopes: one per window
    scope_a = Scope(source=a.source, requested=req,
                    window=(a.window_a.start, a.window_a.end))
    scope_b = Scope(source=a.source, requested=req,
                    window=(a.window_b.start, a.window_b.end))
    scope_a.row_count = len(rows)
    scope_b.row_count = len(rows)
    return ToolResult(rows, [scope_a, scope_b], [sql])


# ---- cross_correlate ---------------------------------------------------------


class SeriesRef(ToolArgs):
    model_config = {"extra": "forbid"}
    entity: str = Field(description="entity value (tc or cmdb_id)")
    metric: str = Field(description="metric name (metric_name or kpi_name)")


class CrossCorrelateArgs(ToolArgs):
    source: Literal["app_metrics", "container_metrics"]
    series_a: SeriesRef
    series_b: SeriesRef
    window: Window
    max_lag_s: float = Field(
        description="maximum lag in seconds (required; the model must specify the range to explore)",
        gt=0,
    )


@tool(
    "cross_correlate",
    CrossCorrelateArgs,
    description=(
        "Pearson correlation between two metric series at lags from -max_lag_s to +max_lag_s, "
        "at the series' natural cadence. Positive lag means series_a leads series_b. "
        "Output is correlation, not causation — a nonzero correlation at a lag is consistent "
        "with both A causing B, B causing A, and a common driver; further evidence is needed. "
        "All rows are sortable; no causal verdict is produced."
    ),
    order_by=["lag_s"],
    sources=["app_metrics", "container_metrics"],
)
def cross_correlate(a: CrossCorrelateArgs, ctx) -> ToolResult:
    ec, mc = _entity_col(a.source), _metric_col(a.source)
    entity_kind = "service" if ec == "tc" else "host"
    ctx.index.check_known(entity_kind, [a.series_a.entity, a.series_b.entity])
    ctx.index.check_known("metric", [a.series_a.metric, a.series_b.metric])

    # Pull both series in the window
    sql_a = (
        f'SELECT timestamp_s, value FROM "{a.source}" '
        f'WHERE "{ec}" = ? AND "{mc}" = ? AND timestamp_s >= ? AND timestamp_s < ? '
        f'ORDER BY timestamp_s'
    )
    rows_a = ctx.con.execute(sql_a, [
        a.series_a.entity, a.series_a.metric, a.window.start, a.window.end
    ]).fetchall()
    rows_b = ctx.con.execute(sql_a, [
        a.series_b.entity, a.series_b.metric, a.window.start, a.window.end
    ]).fetchall()

    if not rows_a or not rows_b:
        scope = Scope(
            source=a.source,
            requested={entity_kind: [a.series_a.entity, a.series_b.entity],
                       "metric": [a.series_a.metric, a.series_b.metric]},
            window=(a.window.start, a.window.end),
            row_count=0,
        )
        return ToolResult([], [scope], [sql_a])

    # Build timestamp-keyed dicts
    dict_a = {ts: v for ts, v in rows_a}
    dict_b = {ts: v for ts, v in rows_b}

    # Determine cadence: median delta from series_a timestamps
    ts_a = sorted(dict_a.keys())
    if len(ts_a) >= 2:
        deltas = [ts_a[i + 1] - ts_a[i] for i in range(len(ts_a) - 1)]
        deltas.sort()
        cadence = deltas[len(deltas) // 2]
    else:
        cadence = 1.0

    # Compute correlation at each lag
    max_lag = a.max_lag_s
    lags = []
    lag = -max_lag
    while lag <= max_lag + 1e-9:
        lags.append(round(lag, 6))
        lag += cadence

    result_rows = []
    for lag_s in lags:
        pairs = [
            (dict_a[ts], dict_b.get(ts + lag_s) or dict_b.get(
                min(dict_b.keys(), key=lambda t: abs(t - (ts + lag_s)), default=None)
            ))
            for ts in ts_a
            if (ts + lag_s) in dict_b or dict_b
        ]
        # strict: only exact timestamp matches count
        pairs = [(va, dict_b[ts + lag_s])
                 for ts, va in dict_a.items()
                 if (ts + lag_s) in dict_b]
        n = len(pairs)
        if n < 2:
            result_rows.append({"lag_s": lag_s, "correlation": None, "n_pairs": n})
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        corr = (num / (sx * sy)) if sx > 0 and sy > 0 else None
        result_rows.append({"lag_s": lag_s, "correlation": corr, "n_pairs": n})

    scope = Scope(
        source=a.source,
        requested={entity_kind: [a.series_a.entity, a.series_b.entity],
                   "metric": [a.series_a.metric, a.series_b.metric]},
        window=(a.window.start, a.window.end),
        row_count=len(result_rows),
    )
    return ToolResult(result_rows, [scope], [sql_a], order_by=["lag_s"])
