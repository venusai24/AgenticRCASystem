"""Discovery tools: describe_dataset, list_entities, system_facts_lookup.

Nothing here ranks or flags: entity lists are counts and active ranges in
neutral (entity, source) order, and system facts come back verbatim with
their author and date, or empty.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agentic_rca.ingest.profile import table_row_counts, time_coverage
from agentic_rca.tools._common import ENTITY_KINDS, Scope, ToolArgs, Window
from agentic_rca.tools.registry import ToolResult, tool

RAW_SOURCES = ("app_metrics", "container_metrics", "spans", "logs", "log_access_apache", "log_access_localhost")


class NoArgs(ToolArgs):
    pass


@tool(
    "describe_dataset",
    NoArgs,
    description="Sources, row counts, time coverage per window, and the data-sufficiency declaration (join availability, host coverage per source, change-event visibility, clock-offset coverage).",
    order_by=["table"],
    sources=["stage0_profile"],
)
def describe_dataset(a: NoArgs, ctx) -> ToolResult:
    counts = table_row_counts(ctx.con)
    cov = time_coverage(ctx.con)
    rows = [{"table": t, "row_count": n, "time_coverage": cov.get(t)} for t, n in counts.items()]
    extra = {"declaration": ctx.runtime.declaration} if getattr(ctx.runtime, "declaration", None) else {}
    return ToolResult(rows, [Scope(source="stage0_profile", row_count=len(rows))], extra_stats=extra)


class ListEntitiesArgs(ToolArgs):
    kind: Literal["host", "service", "metric", "template", "log_source"]
    filter: str | None = Field(default=None, description="case-insensitive substring match on the entity value")
    window: Window | None = None


@tool(
    "list_entities",
    ListEntitiesArgs,
    description="Enumerate entities of one kind across the raw sources: per (entity, source) the row count and first/last timestamp. Counts only; nothing is ranked.",
    order_by=["entity", "source"],
    sources=list(RAW_SOURCES),
)
def list_entities(a: ListEntitiesArgs, ctx) -> ToolResult:
    rows, scopes, sqls = [], [], []
    for src in RAW_SOURCES:
        cols = ctx.index.columns(src)
        col = next((c for c in ENTITY_KINDS[a.kind] if c in cols), None)
        if col is None:
            continue
        has_ts = "timestamp_s" in cols
        where, params = [f'"{col}" IS NOT NULL'], []
        if a.filter:
            where.append(f'"{col}" ILIKE ?')
            params.append(f"%{a.filter}%")
        if a.window and has_ts:
            where.append("timestamp_s >= ? AND timestamp_s < ?")
            params += [a.window.start, a.window.end]
        ts = "min(timestamp_s), max(timestamp_s)" if has_ts else "NULL, NULL"
        sql = f'SELECT "{col}", count(*), {ts} FROM "{src}" WHERE {" AND ".join(where)} GROUP BY 1'
        got = ctx.con.execute(sql, params).fetchall()
        sqls.append(sql)
        for ent, n, lo, hi in got:
            rows.append({"entity": str(ent), "kind": a.kind, "source": src, "row_count": n,
                         "first_ts_s": lo, "last_ts_s": hi})
        w = (a.window.start, a.window.end) if a.window and has_ts else None
        scopes.append(Scope(source=src, requested={a.kind: "*"}, window=w, row_count=len(got), visibility="rows"))
    return ToolResult(rows, scopes, sqls)


class FactsArgs(ToolArgs):
    query: str = Field(description="entity or topic to look up (case-insensitive substring over each fact's text)")


@tool(
    "system_facts_lookup",
    FactsArgs,
    description="Curated environment facts (with author and date) matching an entity or topic, or empty. Facts are statements by people and may be stale.",
    order_by=["_fact_index"],
    sources=["system_facts"],
)
def system_facts_lookup(a: FactsArgs, ctx) -> ToolResult:
    facts = getattr(ctx.runtime, "system_facts", None) or []
    q = a.query.lower()
    rows = [dict(f, _fact_index=i) for i, f in enumerate(facts) if q in str(f).lower()]
    return ToolResult(rows, [Scope(source="system_facts", row_count=len(rows))])
