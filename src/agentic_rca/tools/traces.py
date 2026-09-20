"""tool-traces (B2.4): trace querying, aggregation, and topology."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from agentic_rca.tools._common import Scope, ToolArgs, ToolInputError, Window
from agentic_rca.tools.registry import ToolResult, tool

_SPAN_MEASURES = {
    "count": "COUNT(*)",
    "mean_duration_ms": "AVG(duration_ms)",
    "p50_duration_ms": "MEDIAN(duration_ms)",
    "p95_duration_ms": "QUANTILE_CONT(duration_ms, 0.95)",
    "max_duration_ms": "MAX(duration_ms)",
    "sum_duration_ms": "SUM(duration_ms)",
}


class QuerySpansArgs(ToolArgs):
    window: Window
    cmdb_ids: list[str] | None = Field(None, description="hosts to include; omit for all")
    trace_ids: list[str] | None = Field(None, description="trace ids to include; omit for all")
    span_ids: list[str] | None = Field(None, description="span ids to include; omit for all")
    fields: list[str] | None = Field(
        None,
        description="columns to return (e.g. ['timestamp_s', 'cmdb_id', 'duration_ms']). Omit for all.",
    )


@tool(
    "query_spans",
    QuerySpansArgs,
    description="Fetch raw spans in the given window, optionally filtered by host or trace id.",
    order_by=["timestamp_s", "cmdb_id", "span_id"],
    sources=["spans"],
)
def query_spans(a: QuerySpansArgs, ctx) -> ToolResult:
    # validate entities
    if a.cmdb_ids:
        ctx.index.check_known("host", a.cmdb_ids)
    if a.trace_ids:
        ctx.index.check_known("trace", a.trace_ids)
    if a.span_ids:
        ctx.index.check_known("span", a.span_ids)

    clauses = ["timestamp_s >= ?", "timestamp_s < ?"]
    params: list[Any] = [a.window.start, a.window.end]

    req_scope: dict[str, Any] = {}

    if a.cmdb_ids:
        ph = ",".join("?" for _ in a.cmdb_ids)
        clauses.append(f"cmdb_id IN ({ph})")
        params += a.cmdb_ids
        req_scope["host"] = a.cmdb_ids

    if a.trace_ids:
        ph = ",".join("?" for _ in a.trace_ids)
        clauses.append(f"trace_id IN ({ph})")
        params += a.trace_ids
        req_scope["trace"] = a.trace_ids

    if a.span_ids:
        ph = ",".join("?" for _ in a.span_ids)
        clauses.append(f"span_id IN ({ph})")
        params += a.span_ids
        req_scope["span"] = a.span_ids

    select = "*" if not a.fields else ", ".join(f'"{f}"' for f in a.fields)
    where = " AND ".join(clauses)
    sql = f"SELECT {select} FROM spans WHERE {where} ORDER BY timestamp_s, cmdb_id, span_id"
    rows = ctx.fetch(sql, params)

    scope = Scope(source="spans", requested=req_scope, window=(a.window.start, a.window.end), row_count=len(rows))
    return ToolResult(rows, [scope], [sql])


class AggregateSpansArgs(ToolArgs):
    window: Window
    resolution_s: float | None = Field(
        None, description="bucket size in seconds; omit to aggregate the whole window into one bucket per group"
    )
    group_by: list[str] | None = Field(
        None, description="fields to group by (e.g. ['cmdb_id']). Omit for global aggregation."
    )
    measures: list[str] = Field(
        description="measures to compute: count, mean_duration_ms, p50_duration_ms, p95_duration_ms, max_duration_ms, sum_duration_ms. Required; no defaults are provided."
    )

    @field_validator("measures")
    @classmethod
    def _check_measures(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("measures list cannot be empty")
        for m in v:
            if m not in _SPAN_MEASURES:
                raise ValueError(f"unknown measure {m!r}. Allowed: {list(_SPAN_MEASURES)}")
        return v


@tool(
    "aggregate_spans",
    AggregateSpansArgs,
    description="Compute aggregates over spans (e.g., duration quantiles or counts).",
    order_by=["bucket_start_s", "cmdb_id", "trace_id"],  # will dynamically use whatever group_by is
    sources=["spans"],
)
def aggregate_spans(a: AggregateSpansArgs, ctx) -> ToolResult:
    clauses = ["timestamp_s >= ?", "timestamp_s < ?"]
    params: list[Any] = [a.window.start, a.window.end]

    selects = []
    groups = []
    order_by_cols = []

    if a.resolution_s:
        # bucket: floor(ts / res) * res
        expr = f"FLOOR(timestamp_s / {a.resolution_s}) * {a.resolution_s}"
        selects.append(f"{expr} AS bucket_start_s")
        groups.append("bucket_start_s")
        order_by_cols.append("bucket_start_s")

    if a.group_by:
        for g in a.group_by:
            selects.append(f'"{g}"')
            groups.append(f'"{g}"')
            order_by_cols.append(f'"{g}"')

    for m in a.measures:
        selects.append(f"{_SPAN_MEASURES[m]} AS {m}")

    sel_str = ", ".join(selects) if selects else "*"
    where_str = " AND ".join(clauses)
    group_str = " GROUP BY " + ", ".join(groups) if groups else ""
    order_str = " ORDER BY " + ", ".join(order_by_cols) if order_by_cols else ""

    sql = f"SELECT {sel_str} FROM spans WHERE {where_str}{group_str}{order_str}"
    rows = ctx.fetch(sql, params)

    scope = Scope(source="spans", requested={}, window=(a.window.start, a.window.end), row_count=len(rows))
    return ToolResult(rows, [scope], [sql], order_by=order_by_cols)


class GetTraceArgs(ToolArgs):
    trace_id: str


@tool(
    "get_trace",
    GetTraceArgs,
    description="Fetch all spans for a given trace_id. Output includes root_kind to distinguish true roots from fallback roots.",
    order_by=["timestamp_s", "span_id"],
    sources=["spans", "span_roots"],
)
def get_trace(a: GetTraceArgs, ctx) -> ToolResult:
    ctx.index.check_known("trace", [a.trace_id])

    # First get the root info to join it efficiently
    sql = """
    SELECT
        s.*,
        CASE WHEN s.span_id = r.root_span_id THEN true ELSE false END AS is_root,
        r.root_kind
    FROM spans s
    LEFT JOIN span_roots r ON s.trace_id = r.trace_id
    WHERE s.trace_id = ?
    ORDER BY s.timestamp_s, s.span_id
    """
    rows = ctx.fetch(sql, [a.trace_id])
    
    # We don't have a specific window for this query, use an infinite window
    # since we are fetching by ID.
    scope = Scope(source="spans", requested={"trace": [a.trace_id]}, window=(0.0, 2e9), row_count=len(rows))
    return ToolResult(rows, [scope], [sql])


class CallTopologyArgs(ToolArgs):
    window: Window
    cmdb_ids: list[str] | None = Field(None, description="hosts to include as caller; omit for all")


@tool(
    "call_topology",
    CallTopologyArgs,
    description="Aggregate caller→callee relationships. Describes observed call structure, not causal structure.",
    order_by=["caller_host", "callee_host"],
    sources=["spans"],  # It's derived from spans, though we compute it dynamically here
)
def call_topology(a: CallTopologyArgs, ctx) -> ToolResult:
    if a.cmdb_ids:
        ctx.index.check_known("host", a.cmdb_ids)

    clauses = ["caller.timestamp_s >= ?", "caller.timestamp_s < ?"]
    params: list[Any] = [a.window.start, a.window.end]

    req_scope: dict[str, Any] = {}

    if a.cmdb_ids:
        ph = ",".join("?" for _ in a.cmdb_ids)
        clauses.append(f"caller.cmdb_id IN ({ph})")
        params += a.cmdb_ids
        req_scope["host"] = a.cmdb_ids

    # exclude self-referential spans from being counted as calls
    where_str = " AND ".join(clauses)
    
    # Join spans to itself where child.parent_id = caller.span_id
    sql = f"""
    SELECT 
        caller.cmdb_id AS caller_host,
        callee.cmdb_id AS callee_host,
        COUNT(*) AS call_count,
        MEDIAN(callee.duration_ms) AS duration_ms_p50,
        QUANTILE_CONT(callee.duration_ms, 0.95) AS duration_ms_p95,
        MAX(callee.duration_ms) AS duration_ms_max
    FROM spans caller
    JOIN spans callee ON callee.parent_id = caller.span_id AND callee.span_id != callee.parent_id
    WHERE {where_str}
    GROUP BY caller.cmdb_id, callee.cmdb_id
    ORDER BY caller.cmdb_id, callee.cmdb_id
    """
    
    rows = ctx.fetch(sql, params)
    
    scope = Scope(source="spans", requested=req_scope, window=(a.window.start, a.window.end), row_count=len(rows))
    return ToolResult(rows, [scope], [sql])


class CompareWindowsSpansArgs(ToolArgs):
    window_a: Window
    window_b: Window
    group_by: list[str] = Field(description="fields to group by for comparison (e.g. ['cmdb_id'])")
    measures: list[str] = Field(description="measures to compare")

    @field_validator("measures")
    @classmethod
    def _check_measures(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("measures list cannot be empty")
        for m in v:
            if m not in _SPAN_MEASURES:
                raise ValueError(f"unknown measure {m!r}. Allowed: {list(_SPAN_MEASURES)}")
        return v
        
    @field_validator("group_by")
    @classmethod
    def _check_group_by(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("group_by list cannot be empty for compare_windows_spans")
        return v


@tool(
    "compare_windows_spans",
    CompareWindowsSpansArgs,
    description="Compare span aggregates between two time windows. Uses FULL OUTER JOIN to include keys present in only one window.",
    order_by=["_merged"],  # placeholder, dynamically ordered by group_by
    sources=["spans"],
)
def compare_windows_spans(a: CompareWindowsSpansArgs, ctx) -> ToolResult:
    # Build aggregate CTE for a window
    def _cte(win: Window, alias: str) -> tuple[str, list[Any]]:
        params = [win.start, win.end]
        selects = []
        for g in a.group_by:
            selects.append(f'"{g}"')
        for m in a.measures:
            selects.append(f"{_SPAN_MEASURES[m]} AS {m}")
        
        sel_str = ", ".join(selects)
        group_str = " GROUP BY " + ", ".join(f'"{g}"' for g in a.group_by)
        sql = f"SELECT {sel_str} FROM spans WHERE timestamp_s >= ? AND timestamp_s < ? {group_str}"
        return sql, params
        
    sql_a, params_a = _cte(a.window_a, "a")
    sql_b, params_b = _cte(a.window_b, "b")
    
    # Outer join them
    join_conds = " AND ".join(f"a.\"{g}\" = b.\"{g}\"" for g in a.group_by)
    coalesce_groups = ", ".join(f"COALESCE(a.\"{g}\", b.\"{g}\") AS \"{g}\"" for g in a.group_by)
    
    selects = [coalesce_groups]
    for m in a.measures:
        selects.append(f"a.{m} AS {m}_a")
        selects.append(f"b.{m} AS {m}_b")
        selects.append(f"(b.{m} - a.{m}) AS {m}_delta")
        
    order_by_str = " ORDER BY " + ", ".join(f'"{g}"' for g in a.group_by)
    
    sql = f"""
    WITH a AS ({sql_a}),
         b AS ({sql_b})
    SELECT {", ".join(selects)}
    FROM a FULL OUTER JOIN b ON {join_conds}
    {order_by_str}
    """
    
    rows = ctx.fetch(sql, params_a + params_b)
    
    scope_a = Scope(source="spans", requested={}, window=(a.window_a.start, a.window_a.end), row_count=len(rows))
    scope_b = Scope(source="spans", requested={}, window=(a.window_b.start, a.window_b.end), row_count=len(rows))
    return ToolResult(rows, [scope_a, scope_b], [sql], order_by=a.group_by)
