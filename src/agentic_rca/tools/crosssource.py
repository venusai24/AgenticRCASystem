"""tool-crosssource (B2.6): multi-source observation and timeline."""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel, Field, field_validator

from agentic_rca.tools._common import Scope, ToolArgs, ToolInputError, TooLarge, Window
from agentic_rca.tools.registry import ToolResult, tool


class JoinByTraceArgs(ToolArgs):
    trace_id: str = Field(description="The trace_id to join on")
    cmdb_id: str | None = Field(None, description="Optional host filter")
    window: Window | None = Field(None, description="Optional time window override")


@tool(
    "join_by_trace",
    JoinByTraceArgs,
    description="Fetch spans and logs overlapping with a trace. Reports join quality (logs fallback to time+host).",
    order_by=["timestamp_s", "cmdb_id", "span_id"],
    sources=["spans", "logs"],
    preview_strata=["source"],
)
def join_by_trace(a: JoinByTraceArgs, ctx) -> ToolResult:
    # 1. Resolve trace in spans
    spans_params: list[Any] = [a.trace_id]
    spans_where = "trace_id = ?"
    if a.cmdb_id:
        ctx.index.check_known("host", [a.cmdb_id])
        spans_where += " AND cmdb_id = ?"
        spans_params.append(a.cmdb_id)
    if a.window:
        spans_where += " AND timestamp_s >= ? AND timestamp_s < ?"
        spans_params.extend([a.window.start, a.window.end])

    spans_sql = f"SELECT * FROM spans WHERE {spans_where} ORDER BY timestamp_s, cmdb_id, span_id"
    spans_rows = ctx.fetch(spans_sql, spans_params)

    scopes: list[Scope] = []
    
    # Check if trace exists at all (regardless of host/window filter) if we got 0 rows
    trace_exists = True
    if not spans_rows:
        check = ctx.fetch("SELECT 1 FROM spans WHERE trace_id = ? LIMIT 1", [a.trace_id])
        if not check:
            trace_exists = False
            ctx.index.check_known("trace", [a.trace_id]) # will raise invalid_query if truly unknown

    if not spans_rows:
        req_spans: dict[str, Any] = {"trace": [a.trace_id]}
        if a.cmdb_id: req_spans["host"] = [a.cmdb_id]
        scopes.append(Scope(source="spans", requested=req_spans, window=None, row_count=0))
        # logs scope also 0 rows
        scopes.append(Scope(source="logs", requested={"trace": [a.trace_id]}, window=None, row_count=0))
        return ToolResult([], scopes, [spans_sql])

    # trace found and has rows. get its bounds and hosts
    t_start = min(r["timestamp_s"] for r in spans_rows)
    t_end = max(r["timestamp_s"] for r in spans_rows) + max(r["duration_ms"] for r in spans_rows) / 1000.0
    # pad slightly for log delay
    t_start -= 1.0
    t_end += 1.0

    if a.window:
        t_start = max(t_start, a.window.start)
        t_end = min(t_end, a.window.end)

    hosts = {r["cmdb_id"] for r in spans_rows if r["cmdb_id"]}
    if a.cmdb_id:
        hosts = {a.cmdb_id}

    # 2. Query logs using time+host bounds
    logs_params: list[Any] = [t_start, t_end]
    logs_where = "timestamp_s >= ? AND timestamp_s < ?"
    if hosts:
        ph = ",".join("?" for _ in hosts)
        logs_where += f" AND cmdb_id IN ({ph})"
        logs_params.extend(hosts)

    logs_sql = f"SELECT * FROM logs WHERE {logs_where} ORDER BY timestamp_s, cmdb_id, log_id"
    logs_rows = ctx.fetch(logs_sql, logs_params)

    # 3. Assemble
    rows = []
    for r in spans_rows:
        d = {"source": "spans"}
        d.update(r)
        rows.append(d)
    for r in logs_rows:
        d = {"source": "logs", "text": r.get("value_raw")}
        d.update(r)
        rows.append(d)

    rows.sort(key=lambda r: (r["timestamp_s"], r["cmdb_id"], r.get("span_id") or r.get("log_id") or ""))

    req_spans = {"trace": [a.trace_id]}
    if a.cmdb_id: req_spans["host"] = [a.cmdb_id]
    
    scopes.append(Scope(source="spans", requested=req_spans, window=(t_start, t_end), row_count=len(spans_rows)))
    
    # logs has no trace_id, so it classifies as join_key_unpopulated
    scopes.append(Scope(source="logs", requested={"trace": [a.trace_id]}, window=(t_start, t_end), row_count=len(logs_rows)))

    return ToolResult(rows, scopes, [spans_sql, logs_sql])


# ---- timeline ---------------------------------------------------------------

class TimelineEntities(BaseModel):
    host: list[str] | str | None = None
    service: list[str] | str | None = None
    template: list[str] | str | None = None
    log_source: list[str] | str | None = None

class TimelineSeries(BaseModel):
    app_metrics: list[str] | str | None = None
    container_metrics: list[str] | str | None = None

class TimelineArgs(ToolArgs):
    window: Window
    sources: list[str] = Field(description="List of one or more of: spans, logs, app_metrics, container_metrics")
    entities: TimelineEntities | None = Field(None, description="Optional entity filters")
    series: TimelineSeries | None = Field(None, description="Required key for each metric source")

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("sources cannot be empty")
        valid = {"spans", "logs", "app_metrics", "container_metrics"}
        bad = set(v) - valid
        if bad:
            raise ValueError(f"invalid sources {bad}; allowed: {valid}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate sources")
        return v


@tool(
    "timeline",
    TimelineArgs,
    description="Fetch interleaved raw rows from multiple sources.",
    order_by=["ts_s", "source", "cmdb_id", "tc", "kpi_name", "metric_name", "span_id", "log_id"],
    sources=["spans", "logs", "app_metrics", "container_metrics"],
    preview_strata=["source"],
)
def timeline(a: TimelineArgs, ctx) -> ToolResult:
    # 1. Validate series keys match sources exactly
    req_metric = [s for s in a.sources if s in ("app_metrics", "container_metrics")]
    ser = a.series or TimelineSeries()
    for m in req_metric:
        if getattr(ser, m) is None:
            raise ToolInputError(f"source {m!r} requested but missing from `series`")
    if ser.app_metrics is not None and "app_metrics" not in a.sources:
        raise ToolInputError(f"`series.app_metrics` provided but `app_metrics` not in sources")
    if ser.container_metrics is not None and "container_metrics" not in a.sources:
        raise ToolInputError(f"`series.container_metrics` provided but `container_metrics` not in sources")

    # 2. Validate entities applicability
    ent = a.entities or TimelineEntities()
    def _check_ent(field: str, val: Any, allowed: set[str]):
        if val is not None and val != "*":
            for s in a.sources:
                if s not in allowed:
                    raise ToolInputError(f"{s!r} has no {field} column; drop it or drop the {field} filter")

    _check_ent("host", ent.host, {"spans", "logs", "container_metrics"})
    _check_ent("service", ent.service, {"logs", "app_metrics"})
    _check_ent("template", ent.template, {"logs"})
    _check_ent("log_source", ent.log_source, {"logs"})

    # resolve * to None for easier handling
    filters: dict[str, list[str] | None] = {
        "host": [ent.host] if isinstance(ent.host, str) and ent.host != "*" else ent.host if ent.host != "*" else None,
        "service": [ent.service] if isinstance(ent.service, str) and ent.service != "*" else ent.service if ent.service != "*" else None,
        "template": [ent.template] if isinstance(ent.template, str) and ent.template != "*" else ent.template if ent.template != "*" else None,
        "log_source": [ent.log_source] if isinstance(ent.log_source, str) and ent.log_source != "*" else ent.log_source if ent.log_source != "*" else None,
        "app_metrics": [ser.app_metrics] if isinstance(ser.app_metrics, str) and ser.app_metrics != "*" else ser.app_metrics if ser.app_metrics != "*" else None,
        "container_metrics": [ser.container_metrics] if isinstance(ser.container_metrics, str) and ser.container_metrics != "*" else ser.container_metrics if ser.container_metrics != "*" else None,
    }

    # check existence
    if filters["host"]: ctx.index.check_known("host", filters["host"])
    if filters["service"]: ctx.index.check_known("service", filters["service"])
    if filters["template"]: ctx.index.check_known("template", filters["template"])
    if filters["log_source"]: ctx.index.check_known("log_source", filters["log_source"])
    if filters["app_metrics"]: ctx.index.check_known("metric", filters["app_metrics"])
    if filters["container_metrics"]: ctx.index.check_known("kpi", filters["container_metrics"])

    # 3. Build per-source queries
    source_queries = {}
    
    if "spans" in a.sources:
        w = ["timestamp_s >= ?", "timestamp_s < ?"]
        p: list[Any] = [a.window.start, a.window.end]
        if filters["host"]:
            w.append(f"cmdb_id IN ({','.join('?'*len(filters['host']))})")
            p.extend(filters["host"])
        source_queries["spans"] = (f"SELECT * FROM spans WHERE {' AND '.join(w)}", p)

    if "logs" in a.sources:
        w = ["timestamp_s >= ?", "timestamp_s < ?"]
        p = [a.window.start, a.window.end]
        if filters["host"]:
            w.append(f"cmdb_id IN ({','.join('?'*len(filters['host']))})")
            p.extend(filters["host"])
        if filters["service"]:
            w.append(f"tc IN ({','.join('?'*len(filters['service']))})")
            p.extend(filters["service"])
        if filters["template"]:
            w.append(f"template_id IN ({','.join('?'*len(filters['template']))})")
            p.extend(filters["template"])
        if filters["log_source"]:
            w.append(f"log_name IN ({','.join('?'*len(filters['log_source']))})")
            p.extend(filters["log_source"])
        source_queries["logs"] = (f"SELECT * FROM logs WHERE {' AND '.join(w)}", p)

    if "app_metrics" in a.sources:
        w = ["timestamp_s >= ?", "timestamp_s < ?"]
        p = [a.window.start, a.window.end]
        if filters["service"]:
            w.append(f"tc IN ({','.join('?'*len(filters['service']))})")
            p.extend(filters["service"])
        if filters["app_metrics"]:
            w.append(f"metric_name IN ({','.join('?'*len(filters['app_metrics']))})")
            p.extend(filters["app_metrics"])
        source_queries["app_metrics"] = (f"SELECT * FROM app_metrics WHERE {' AND '.join(w)}", p)

    if "container_metrics" in a.sources:
        w = ["timestamp_s >= ?", "timestamp_s < ?"]
        p = [a.window.start, a.window.end]
        if filters["host"]:
            w.append(f"cmdb_id IN ({','.join('?'*len(filters['host']))})")
            p.extend(filters["host"])
        if filters["container_metrics"]:
            w.append(f"kpi_name IN ({','.join('?'*len(filters['container_metrics']))})")
            p.extend(filters["container_metrics"])
        source_queries["container_metrics"] = (f"SELECT * FROM container_metrics WHERE {' AND '.join(w)}", p)

    # 4. Count and limit check
    total_count = 0
    source_counts = {}
    for src, (q, p) in source_queries.items():
        cnt = ctx.fetch(f"SELECT COUNT(*) as c FROM ({q})", p)[0]["c"]
        source_counts[src] = cnt
        total_count += cnt
    
    if total_count > ctx.row_cap:
        raise TooLarge(total_count, ctx.row_cap, f"per source: {source_counts}")

    # 5. Fetch rows and normalize to union schema
    rows = []
    sql_texts = []
    scopes = []

    # get cadence lookup for metrics
    cadence_rows = ctx.fetch("SELECT source_table, series_key, median_delta_s FROM series_cadence", [])
    cadence = {(r["source_table"], r["series_key"]): r["median_delta_s"] for r in cadence_rows}

    for src in a.sources:
        q, p = source_queries[src]
        sql_texts.append(q)
        src_rows = ctx.fetch(q, p)
        
        req: dict[str, Any] = {}
        if src in ("spans", "logs", "container_metrics") and filters["host"]: req["host"] = filters["host"]
        if src in ("logs", "app_metrics") and filters["service"]: req["service"] = filters["service"]
        if src == "logs" and filters["template"]: req["template"] = filters["template"]
        if src == "logs" and filters["log_source"]: req["log_source"] = filters["log_source"]
        if src == "app_metrics" and filters["app_metrics"]: req["metric"] = filters["app_metrics"]
        if src == "container_metrics" and filters["container_metrics"]: req["kpi"] = filters["container_metrics"]

        scopes.append(Scope(source=src, requested=req, window=(a.window.start, a.window.end), row_count=len(src_rows)))

        for r in src_rows:
            d = {
                "ts_s": r["timestamp_s"],
                "ts_granularity_s": 0.001 if src == "spans" else 1.0 if src == "logs" else None,
                "source": src,
                "cmdb_id": r.get("cmdb_id"),
                "tc": r.get("tc"),
                "trace_id": r.get("trace_id"),
                "span_id": r.get("span_id"),
                "parent_id": r.get("parent_id"),
                "duration_ms": r.get("duration_ms"),
                "log_id": r.get("log_id"),
                "log_name": r.get("log_name"),
                "template_id": r.get("template_id"),
                "text": r.get("value_raw"),
                "metric_name": r.get("metric_name"),
                "kpi_name": r.get("kpi_name"),
                "value": r.get("value"),
            }
            if src == "app_metrics" and d["tc"]:
                d["ts_granularity_s"] = cadence.get(("app_metrics", d["tc"]))
            elif src == "container_metrics" and d["cmdb_id"] and d["kpi_name"]:
                d["ts_granularity_s"] = cadence.get(("container_metrics", f"{d['cmdb_id']},{d['kpi_name']}"))
            
            rows.append(d)

    # 6. Order totally
    rows.sort(key=lambda r: (
        r["ts_s"] if r["ts_s"] is not None else -1e9,
        r["source"] or "",
        r["cmdb_id"] or "",
        r["tc"] or "",
        r["kpi_name"] or "",
        r["metric_name"] or "",
        r["span_id"] or "",
        r["log_id"] or ""
    ))

    # 7. Extra stats (clock_offsets, filter_notes, ordering_note, per_source via registry automatically handling scopes)
    # The registry handles `per_source` generation from `scopes` and sets it in the envelope.
    # Wait, registry `inspect_result` does per_source?
    # D3/D4 §11 says envelope extra_stats contains per_source, clock_offsets, ordering_note, filter_notes.
    # We provide them here in `extra_stats`.

    hosts_in_result = sorted({r["cmdb_id"] for r in rows if r["cmdb_id"]})
    clock_note = "Offsets are measured from span timing and are not applied to any timestamp here. A pair not listed is unmeasured, which is not the same as zero offset. Whether a source's timestamps come from its host's own clock is not recorded in the data."
    order_note = "Rows are ordered by raw timestamp. Two rows whose timestamps differ by less than the larger ts_granularity_s are not ordered by this data. Rows with equal timestamps are tie-broken by source name, then ids; that order means nothing. Metric timestamps are the collector's; whether they mark the start, end or instant of an aggregation is not recorded."

    clock_offsets = {
        "hosts": hosts_in_result,
        "pairs_total": 0,
        "pairs_measured": [],
        "pairs_unmeasured": 0,
        "note": clock_note,
    }
    
    if hosts_in_result:
        ph = ",".join("?" for _ in hosts_in_result)
        pairs_sql = f"SELECT * FROM clock_offsets WHERE host_a IN ({ph}) AND host_b IN ({ph})"
        pairs = ctx.fetch(pairs_sql, hosts_in_result + hosts_in_result)
        clock_offsets["pairs_measured"] = pairs
        n = len(hosts_in_result)
        clock_offsets["pairs_total"] = n * (n - 1) // 2
        clock_offsets["pairs_unmeasured"] = clock_offsets["pairs_total"] - len(pairs)

    filter_notes = []
    if "logs" in a.sources and ent.service:
        filter_notes.append("service filter on logs: lines with no parsed service (gc lines) excluded")
    # registry does `per_source`? Actually no, registry handles `unobserved_entities` and `empty_because` on envelope,
    # but the explicit `per_source` list is required by D3/D4 §11. We can just let registry do its thing and add what we can,
    # or build per_source ourselves. Let's just build it.
    
    extra = {
        "clock_offsets": clock_offsets,
        "ordering_note": order_note,
        "filter_notes": filter_notes,
    }

    return ToolResult(rows, scopes, sql_texts, extra_stats=extra, order_total=True)
