"""Span-tree root detection and host-to-host call topology.

Target_Design.MD / SchemaOfCSVs.MD: a span is a root iff its parent_id does
not appear anywhere in the file's span_id column -- id string SHAPE (the
dash/@ format) carries no meaning; that was a verified-wrong assumption in
an earlier draft. 85% of traces have no true root row in-window at all, so
every trace without one falls back to its earliest-timestamp span. Edges are
emitted as OBSERVED call structure ("a fact about calls"), never as a causal
claim -- callers should not read more into an edge than "these two hosts
were seen calling each other."
"""

from __future__ import annotations

import duckdb
import pandas as pd


def assign_span_roots(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Populate span_roots: one row per trace_id, root_kind in
    {'true_root', 'earliest_fallback'}. Returns counts for verification."""
    spans = con.execute("SELECT span_id, parent_id, trace_id, timestamp_ms FROM spans").fetchdf()

    span_id_set = set(spans["span_id"])
    is_root = ~spans["parent_id"].isin(span_id_set)
    true_roots = spans[is_root]

    # A trace can have zero, one, or (rarely) several true-root rows
    # (SchemaOfCSVs.MD: up to 19 in this dataset). Pick the earliest by
    # timestamp as *the* representative root for that trace -- an arbitrary
    # but stated tie-break, not a claim that the others aren't also roots.
    true_root_per_trace = true_roots.sort_values("timestamp_ms").drop_duplicates(
        subset="trace_id", keep="first"
    )
    true_root_per_trace = true_root_per_trace.assign(root_kind="true_root")[
        ["trace_id", "span_id", "root_kind"]
    ].rename(columns={"span_id": "root_span_id"})

    traces_with_true_root = set(true_root_per_trace["trace_id"])
    remaining = spans[~spans["trace_id"].isin(traces_with_true_root)]
    fallback_per_trace = remaining.sort_values("timestamp_ms").drop_duplicates(
        subset="trace_id", keep="first"
    )
    fallback_per_trace = fallback_per_trace.assign(root_kind="earliest_fallback")[
        ["trace_id", "span_id", "root_kind"]
    ].rename(columns={"span_id": "root_span_id"})

    span_roots_df = pd.concat([true_root_per_trace, fallback_per_trace], ignore_index=True)  # noqa: F841
    con.execute("INSERT INTO span_roots SELECT * FROM span_roots_df")

    return {
        "true_root_rows": int(is_root.sum()),
        "traces_with_true_root": len(traces_with_true_root),
        "traces_with_fallback": len(fallback_per_trace),
        "total_traces": spans["trace_id"].nunique(),
    }


def build_topology_edges(con: duckdb.DuckDBPyConnection) -> int:
    """Emit observed caller-host -> callee-host edges with call counts and
    duration distributions, per window. A row means only "a child span on
    callee_host was observed under a parent span on caller_host" -- described
    as call structure, never as causal structure (Target_Design.MD).

    Excludes span_id == parent_id rows (cycle-8 VERIFY_FAIL): a span that is
    its own parent is not a distinct child observed under a parent, so
    counting it as a call doubled the IG01/IG02 self-edge count exactly 2x
    (16,987 -> 8,493 measured on this dataset). profile.py's
    trace_root_coverage already treats this same row shape as a structural
    artifact, not a call/root; this join now agrees with that.
    """
    edges_df = con.execute(
        """
        SELECT
            child.window_label AS window_label,
            parent.cmdb_id AS caller_host,
            child.cmdb_id AS callee_host,
            COUNT(*) AS call_count,
            QUANTILE_CONT(child.duration_ms, 0.5) AS duration_ms_p50,
            QUANTILE_CONT(child.duration_ms, 0.95) AS duration_ms_p95,
            MAX(child.duration_ms) AS duration_ms_max
        FROM spans child
        JOIN spans parent ON child.parent_id = parent.span_id
        WHERE child.span_id != child.parent_id
        GROUP BY child.window_label, parent.cmdb_id, child.cmdb_id
        """
    ).fetchdf()
    con.execute("INSERT INTO topology_edges SELECT * FROM edges_df")
    return len(edges_df)
