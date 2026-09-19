"""Common response envelope for every data tool (Target_Design.MD, Tool inventory).

Tools never return raw rows to a model context; they return this envelope,
which carries a handle plus enough summary to decide what to do next.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

EmptyReason = Literal["no_matches", "no_data_in_window", "unknown_field_or_entity"]
ErrorType = Literal["transient", "invalid_query", "too_large", "timeout"]

_query_id_counter = itertools.count(1)


def next_query_id() -> str:
    """Monotonic, process-local id. Uniqueness is enough for this stage; it is
    not re-derived across runs, so it carries no meaning beyond this session."""
    return f"q{next(_query_id_counter):06d}"


# Backing store for ResultHandle ids: minted in make_envelope, resolved by
# inspect_result. In-memory and process-local, same lifetime as query_id --
# no ledger/persistence layer exists yet for it to live in instead.
# ponytail: unbounded dict, never evicted. Fine for a single investigation
# session over one incident; add TTL/size-based eviction if a long-running
# session accumulates enough large handles for this to matter.
_RESULT_STORE: dict[str, tuple[list[dict[str, Any]], Provenance]] = {}


@dataclass(frozen=True)
class ToolError:
    type: ErrorType
    diagnostic: str
    suggestions: list[str] = field(default_factory=list)
    retried: bool = False


@dataclass(frozen=True)
class Provenance:
    source: str
    window: str | None = None


@dataclass(frozen=True)
class ResultHandle:
    """A pointer to a result set held server-side, not the rows themselves."""

    id: str
    row_count: int


@dataclass(frozen=True)
class ToolEnvelope:
    query_id: str
    result_handle: ResultHandle | None
    row_count: int
    preview: list[dict[str, Any]]
    summary_stats: dict[str, Any]
    truncated: bool
    coverage: dict[str, Any]
    provenance: Provenance
    empty_because: EmptyReason | None = None
    error: ToolError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def make_envelope(
    *,
    rows: list[dict[str, Any]],
    provenance: Provenance,
    coverage: dict[str, Any],
    preview_limit: int = 20,
    summary_stats: dict[str, Any] | None = None,
    empty_because: EmptyReason | None = None,
) -> ToolEnvelope:
    """Build a passing envelope from a materialized row list.

    A handle is only created when there is something to page through later;
    an empty result carries `result_handle=None` rather than a handle to
    nothing, so callers cannot mistake "no result" for "unfetched result."
    """
    row_count = len(rows)
    handle = ResultHandle(id=next_query_id(), row_count=row_count) if row_count else None
    if handle:
        _RESULT_STORE[handle.id] = (rows, provenance)
    return ToolEnvelope(
        query_id=handle.id if handle else next_query_id(),
        result_handle=handle,
        row_count=row_count,
        preview=rows[:preview_limit],
        summary_stats=summary_stats or {},
        truncated=row_count > preview_limit,
        coverage=coverage,
        provenance=provenance,
        empty_because=empty_because if row_count == 0 else None,
    )


def make_error_envelope(
    *,
    error: ToolError,
    provenance: Provenance,
    coverage: dict[str, Any] | None = None,
) -> ToolEnvelope:
    return ToolEnvelope(
        query_id=next_query_id(),
        result_handle=None,
        row_count=0,
        preview=[],
        summary_stats={},
        truncated=False,
        coverage=coverage or {},
        provenance=provenance,
        error=error,
    )


def inspect_result(
    handle_id: str,
    *,
    offset: int = 0,
    limit: int = 20,
    sort: str | None = None,
    columns: list[str] | None = None,
) -> ToolEnvelope:
    """Tool inventory's `inspect_result`: page through a result handle minted
    by make_envelope. `sort` is a column name, `-column` for descending."""
    entry = _RESULT_STORE.get(handle_id)
    if entry is None:
        return make_error_envelope(
            error=ToolError(
                type="invalid_query",
                diagnostic=f"unknown or expired result handle: {handle_id!r}",
            ),
            provenance=Provenance(source="inspect_result"),
        )
    rows, provenance = entry

    try:
        if sort:
            key = sort[1:] if sort.startswith("-") else sort
            rows = sorted(rows, key=lambda r: r.get(key), reverse=sort.startswith("-"))
        page = rows[offset : offset + limit]
        if columns:
            page = [{c: r.get(c) for c in columns} for r in page]
    except TypeError as exc:
        return make_error_envelope(
            error=ToolError(type="invalid_query", diagnostic=f"cannot sort by {sort!r}: {exc}"),
            provenance=provenance,
        )

    total = len(rows)
    return ToolEnvelope(
        query_id=next_query_id(),
        result_handle=ResultHandle(id=handle_id, row_count=total),
        row_count=total,
        preview=page,
        summary_stats={},
        truncated=offset + limit < total,
        coverage={"offset": offset, "limit": limit},
        provenance=provenance,
    )
