"""Common response envelope for every data tool (Target_Design.MD, Tool inventory).

Tools never return raw rows to a model context; they return this envelope,
which carries a handle plus enough summary to decide what to do next.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

# user patch 31 + the fifth value (DECISIONS 2026-09-19). Only no_matches
# licenses a negative claim; an unknown identifier is an invalid_query error,
# never an empty result.
EmptyReason = Literal[
    "no_matches",
    "no_data_in_window",
    "entity_absent_from_source",
    "join_key_unpopulated",
    "scope_unverified",
]
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

def get_result(query_id: str) -> ToolEnvelope | None:
    """Read-only accessor for tests and ledger to verify values in _RESULT_STORE."""
    entry = _RESULT_STORE.get(query_id)
    if not entry:
        return None
    rows, provenance = entry
    # Return a basic envelope with the cached rows.
    # We construct a fake envelope just to carry the row_count and preview for I2 checking.
    from agentic_rca.tools.envelope import ResultHandle, ToolEnvelope
    return ToolEnvelope(
        query_id=query_id,
        result_handle=ResultHandle(id=query_id, row_count=len(rows)),
        row_count=len(rows),
        preview=rows,
        summary_stats={},
        truncated=False,
        coverage={},
        provenance=provenance,
    )

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
    # added by coverage-index (contracts-b1 §8, §11.6); defaults keep B0 callers working
    order: dict[str, Any] | None = None
    preview_label: str | None = None
    preview_meta: dict[str, Any] | None = None
    unobserved_entities: dict[str, list[str]] | None = None
    scope_verified: bool = True
    notes: dict[str, Any] = field(default_factory=dict)

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
    if limit < 1:
        return make_error_envelope(
            error=ToolError(
                type="invalid_query",
                diagnostic=f"limit must be >= 1 (got limit={limit})",
            ),
            provenance=provenance,
        )
    if offset < 0:
        offset = max(0, len(rows) + offset)

    if offset >= len(rows):
        return make_error_envelope(
            error=ToolError(
                type="invalid_query",
                diagnostic=f"offset {offset} is past the end of the result ({len(rows)} rows)",
                suggestions=[f"offset <= {max(0, len(rows) - 1)}"],
            ),
            provenance=provenance,
        )

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
