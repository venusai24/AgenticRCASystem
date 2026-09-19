"""Mechanical schema/coverage profiling. No interpretation, no scoring, no
flagging anything as suspicious (Target_Design.MD Stage 0) -- this module
only counts and describes what is present.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from agentic_rca.ingest.store import TABLE_NAMES

_PROFILED_TABLES = TABLE_NAMES

_JOIN_KEY_COLUMNS = (
    "tc",
    "cmdb_id",
    "kpi_name",
    "log_name",
    "trace_id",
    "span_id",
    "log_id",
    "template_id",
    "parent_id",
    "root_span_id",
    "gateway",
)


def _distinct_non_null(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[list, int]:
    """Run a single-column `sql` query; return (sorted distinct non-NULL
    values, count of NULL rows) instead of a bare `sorted()` that crashes on
    a mixed str/None set. Fifth recurrence of that crash class (cycles 3, 9,
    13, 14) -- cycle 15's VERIFY_FAIL reproduced it end-to-end via a single
    blank `cmdb_id` cell in `container_metrics.csv`, in functions cycle 14
    itself added without applying its own "fix at the source" rule to every
    site of the same shape. Every DISTINCT/EXCEPT identity-key query in this
    module now funnels through here."""
    rows = [r[0] for r in con.execute(sql).fetchall()]
    non_null = sorted({v for v in rows if v is not None})
    null_count = sum(1 for v in rows if v is None)
    return non_null, null_count


def table_row_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _PROFILED_TABLES}


def time_coverage(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, float | None]]:
    """Per (table, window_label) min/max timestamp_s -- the raw material for
    a "known gaps" statement, computed here without deciding what a gap means."""
    coverage: dict[str, dict[str, float | None]] = {}
    for table in ("app_metrics", "container_metrics", "spans", "logs"):
        rows = con.execute(
            f"SELECT window_label, MIN(timestamp_s), MAX(timestamp_s), COUNT(*) "
            f"FROM {table} GROUP BY window_label"
        ).fetchall()
        coverage[table] = {
            window_label: {"min_s": lo, "max_s": hi, "row_count": n}
            for window_label, lo, hi, n in rows
        }
    return coverage


def host_inventory_per_source(con: duckdb.DuckDBPyConnection) -> dict[str, set[str]]:
    """The verified fact that no single id column spans the full
    infrastructure: each source has its own host set.

    Cross-window union, not per-window -- spans and logs have no baseline
    file at all, so a window split is meaningless for them. container_metrics
    DOES have both windows and they differ (SchemaOfCSVs.MD: dockerB2 is
    incident-only); that specific gap is reported by
    container_metrics_window_diff below, not folded in here, since mixing a
    cross-window union with a per-window diff in one dict invites exactly
    the "18 of 18 reads as full coverage" misreading that was cycle-11's
    VERIFY_FAIL.
    """
    spans_hosts, _ = _distinct_non_null(con, "SELECT DISTINCT cmdb_id FROM spans")
    container_hosts, _ = _distinct_non_null(con, "SELECT DISTINCT cmdb_id FROM container_metrics")
    logs_hosts, _ = _distinct_non_null(con, "SELECT DISTINCT cmdb_id FROM logs")
    return {
        "spans": set(spans_hosts),
        "container_metrics": set(container_hosts),
        "logs": set(logs_hosts),
    }


def host_inventory_null_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """How many rows per source had a NULL cmdb_id, dropped from
    host_inventory_per_source's sets rather than crashing sorted() on them or
    vanishing with no trace (cycle-15 VERIFY_FAIL)."""
    return {
        "spans": con.execute("SELECT COUNT(*) FROM spans WHERE cmdb_id IS NULL").fetchone()[0],
        "container_metrics": con.execute(
            "SELECT COUNT(*) FROM container_metrics WHERE cmdb_id IS NULL"
        ).fetchone()[0],
        "logs": con.execute("SELECT COUNT(*) FROM logs WHERE cmdb_id IS NULL").fetchone()[0],
    }


def app_metrics_window_diff(con: duckdb.DuckDBPyConnection) -> dict:
    """app_metrics' baseline-vs-incident diff by its identity key, `tc`
    (CLAUDE.md). Cycle-14 VERIFY_FAIL: container_metrics got this treatment
    in cycle 11 after "18 of 18" was found to hide dockerB2 being
    incident-only; app_metrics never got the equivalent, and the omission
    was itself unlogged -- an absent check reads the same as "checked,
    identical" whether or not anyone actually looked. On this dataset the
    two windows' `tc` sets are in fact identical (verified: 11/11, no
    incident-only or baseline-only values), but that is a fact this function
    now states, not an assumption the declaration's silence left standing.
    """
    tc_counts = dict(
        con.execute(
            "SELECT window_label, COUNT(DISTINCT tc) FROM app_metrics GROUP BY window_label"
        ).fetchall()
    )
    tc_incident_only, incident_only_null = _distinct_non_null(
        con,
        "SELECT DISTINCT tc FROM app_metrics WHERE window_label = 'incident' "
        "EXCEPT SELECT DISTINCT tc FROM app_metrics WHERE window_label = 'baseline'",
    )
    tc_baseline_only, baseline_only_null = _distinct_non_null(
        con,
        "SELECT DISTINCT tc FROM app_metrics WHERE window_label = 'baseline' "
        "EXCEPT SELECT DISTINCT tc FROM app_metrics WHERE window_label = 'incident'",
    )
    return {
        "tc_count_in_baseline_window": tc_counts.get("baseline", 0),
        "tc_count_in_incident_window": tc_counts.get("incident", 0),
        "tc_incident_only": tc_incident_only,
        "tc_baseline_only": tc_baseline_only,
        "tc_null_rows": incident_only_null + baseline_only_null,
        "note": "identity key is tc, diffed directly (EXCEPT), never by "
        "aligning the two windows positionally. COUNT(DISTINCT tc) above "
        "and the incident_only/baseline_only lists both exclude NULL tc "
        "rows; tc_null_rows counts them separately rather than silently "
        "dropping or crashing on them.",
    }


def container_metrics_window_diff(con: duckdb.DuckDBPyConnection) -> dict:
    """container_metrics is the only source with both a baseline and an
    incident window, and the two are NOT the same population: cycle-11
    VERIFY_FAIL found the declaration's "18 of 18" host phrasing was a
    cross-window union that made dockerB2 (incident-only, SchemaOfCSVs.MD)
    invisible, and that kpi_name's vocabulary differs per window (336
    baseline / 341 incident) with no per-entity denominator anywhere.

    Diffs by identity key (cmdb_id, kpi_name), never by concatenating or
    positionally aligning the two windows (CLAUDE.md's non-negotiable rule
    for this dataset) -- an "incident-only key" here means that exact
    (cmdb_id, kpi_name) pair has incident rows and no baseline rows, not
    that the raw row counts differ.
    """
    host_counts = dict(
        con.execute(
            "SELECT window_label, COUNT(DISTINCT cmdb_id) FROM container_metrics "
            "GROUP BY window_label"
        ).fetchall()
    )
    hosts_incident_only, host_incident_only_null = _distinct_non_null(
        con,
        "SELECT DISTINCT cmdb_id FROM container_metrics WHERE window_label = 'incident' "
        "EXCEPT SELECT DISTINCT cmdb_id FROM container_metrics WHERE window_label = 'baseline'",
    )
    hosts_baseline_only, host_baseline_only_null = _distinct_non_null(
        con,
        "SELECT DISTINCT cmdb_id FROM container_metrics WHERE window_label = 'baseline' "
        "EXCEPT SELECT DISTINCT cmdb_id FROM container_metrics WHERE window_label = 'incident'",
    )
    kpi_name_counts = dict(
        con.execute(
            "SELECT window_label, COUNT(DISTINCT kpi_name) FROM container_metrics "
            "GROUP BY window_label"
        ).fetchall()
    )
    key_counts = con.execute(
        """
        WITH baseline_keys AS (
            SELECT DISTINCT cmdb_id, kpi_name FROM container_metrics WHERE window_label = 'baseline'
        ),
        incident_keys AS (
            SELECT DISTINCT cmdb_id, kpi_name FROM container_metrics WHERE window_label = 'incident'
        )
        SELECT
            (SELECT COUNT(*) FROM baseline_keys),
            (SELECT COUNT(*) FROM incident_keys),
            (SELECT COUNT(*) FROM incident_keys i
             WHERE NOT EXISTS (SELECT 1 FROM baseline_keys b
                                WHERE b.cmdb_id = i.cmdb_id AND b.kpi_name = i.kpi_name)),
            (SELECT COUNT(*) FROM baseline_keys b
             WHERE NOT EXISTS (SELECT 1 FROM incident_keys i
                                WHERE i.cmdb_id = b.cmdb_id AND i.kpi_name = b.kpi_name))
        """
    ).fetchone()
    baseline_key_count, incident_key_count, incident_only_key_count, baseline_only_key_count = (
        key_counts
    )
    return {
        "hosts_in_baseline_window": host_counts.get("baseline", 0),
        "hosts_in_incident_window": host_counts.get("incident", 0),
        "hosts_incident_only": hosts_incident_only,
        "hosts_baseline_only": hosts_baseline_only,
        "kpi_name_count_in_baseline_window": kpi_name_counts.get("baseline", 0),
        "kpi_name_count_in_incident_window": kpi_name_counts.get("incident", 0),
        "identity_key_count_in_baseline_window": baseline_key_count,
        "identity_key_count_in_incident_window": incident_key_count,
        "identity_keys_incident_only": incident_only_key_count,
        "identity_keys_baseline_only": baseline_only_key_count,
        "cmdb_id_null_rows": host_incident_only_null + host_baseline_only_null,
        "note": "identity key is (cmdb_id, kpi_name), diffed directly, never "
        "by aligning the two windows positionally. kpi_name vocabulary "
        "differs per window -- don't assume a fixed KPI vocabulary across "
        "files (SchemaOfCSVs.MD). hosts_incident_only/hosts_baseline_only "
        "exclude NULL cmdb_id rows; cmdb_id_null_rows counts them "
        "separately rather than silently dropping or crashing on them.",
    }


def column_cardinality(con: duckdb.DuckDBPyConnection, table: str, column: str) -> int:
    return con.execute(f"SELECT COUNT(DISTINCT {column}) FROM {table}").fetchone()[0]


def metric_value_variance(con: duckdb.DuckDBPyConnection, metric_name: str) -> float | None:
    """Whether a given app_metrics metric ever varies at all -- the
    mechanical fact behind statements like "sr never dips"."""
    row = con.execute(
        "SELECT STDDEV(value) FROM app_metrics WHERE metric_name = ?", [metric_name]
    ).fetchone()
    return row[0] if row else None


def table_schemas(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    """Per profiled table, its column names and DuckDB-reported types --
    mechanical schema discovery, not a hand-maintained copy of the DDL."""
    schemas: dict[str, list[dict]] = {}
    for table in _PROFILED_TABLES:
        rows = con.execute(f"DESCRIBE {table}").fetchall()
        schemas[table] = [{"column": r[0], "type": r[1]} for r in rows]
    return schemas


def populate_series_cadence(con: duckdb.DuckDBPyConnection) -> int:
    """Per-series median and max inter-sample delta (seconds), one row per
    grouping key actually available in that table -- no threshold or
    "typical cadence" judgment applied here; that belongs to a later stage,
    if anywhere.

    Written to the series_cadence table rather than returned in-memory:
    container_metrics alone produces ~3,259 per-series rows (verified cycle-4
    VERIFY_FAIL -- 471 KB of a 478 KB declaration when this was inlined,
    directly violating Target_Design.MD:239's "tools never dump large results
    into context"). A DuckDB table is itself the queryable handle; the
    declaration references it by name and row count instead of embedding
    its contents (same treatment span_roots/topology_edges already get).

    Grouping keys differ per table because no finer shared series identifier
    exists uniformly across sources (agentic_rca.ingest.sources.SourceFile
    identity_key is a row-identity key for spans/logs, not a series key):
      - app_metrics: rr/sr/cnt/mrt are melted onto a shared timestamp, so
        rows are first deduped to one row per (window_label, tc, timestamp_s)
        before computing deltas, else the melt would manufacture zero-delta
        duplicate timestamps.
      - container_metrics: (window_label, cmdb_id, kpi_name) -- exactly
        sources.py's identity_key for that source.
      - spans: (window_label, cmdb_id) -- computed on timestamp_ms, reported
        in seconds.
      - logs: (window_label, cmdb_id) -- on timestamp_s (no ms column here).
    """

    def _cadence_from_deltas(rows: list[tuple]) -> list[dict]:
        # rows: (key..., delta_s) with one row per (key, ordered sample);
        # first sample per key has delta_s NULL from LAG and is dropped.
        by_key: dict[tuple, list[float]] = {}
        for *key, delta in rows:
            if delta is None:
                continue
            by_key.setdefault(tuple(key), []).append(delta)
        out = []
        for key, deltas in by_key.items():
            deltas_sorted = sorted(deltas)
            n = len(deltas_sorted)
            median = (
                deltas_sorted[n // 2]
                if n % 2
                else (deltas_sorted[n // 2 - 1] + deltas_sorted[n // 2]) / 2
            )
            out.append(
                {
                    "key": key,
                    "n_deltas": n,
                    "median_delta_s": median,
                    "max_delta_s": max(deltas_sorted),
                }
            )
        return out

    cadence: dict[str, list[dict]] = {}

    app_rows = con.execute(
        """
        WITH deduped AS (
            SELECT DISTINCT window_label, tc, timestamp_s
            FROM app_metrics
        )
        SELECT window_label, tc,
               timestamp_s - LAG(timestamp_s) OVER (
                   PARTITION BY window_label, tc ORDER BY timestamp_s
               ) AS delta_s
        FROM deduped
        """
    ).fetchall()
    cadence["app_metrics"] = _cadence_from_deltas(app_rows)

    container_rows = con.execute(
        """
        SELECT window_label, cmdb_id, kpi_name,
               timestamp_s - LAG(timestamp_s) OVER (
                   PARTITION BY window_label, cmdb_id, kpi_name ORDER BY timestamp_s
               ) AS delta_s
        FROM container_metrics
        """
    ).fetchall()
    cadence["container_metrics"] = _cadence_from_deltas(container_rows)

    span_rows = con.execute(
        """
        SELECT window_label, cmdb_id,
               (timestamp_ms - LAG(timestamp_ms) OVER (
                   PARTITION BY window_label, cmdb_id ORDER BY timestamp_ms
               )) / 1000.0 AS delta_s
        FROM spans
        """
    ).fetchall()
    cadence["spans"] = _cadence_from_deltas(span_rows)

    log_rows = con.execute(
        """
        SELECT window_label, cmdb_id,
               timestamp_s - LAG(timestamp_s) OVER (
                   PARTITION BY window_label, cmdb_id ORDER BY timestamp_s
               ) AS delta_s
        FROM logs
        """
    ).fetchall()
    cadence["logs"] = _cadence_from_deltas(log_rows)

    rows = [
        {
            "source_table": source_table,
            "window_label": entry["key"][0],
            "series_key": ",".join(str(k) for k in entry["key"][1:]),
            "n_deltas": entry["n_deltas"],
            "median_delta_s": entry["median_delta_s"],
            "max_delta_s": entry["max_delta_s"],
        }
        for source_table, entries in cadence.items()
        for entry in entries
    ]
    if not rows:
        return 0
    cadence_df = pd.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO series_cadence SELECT * FROM cadence_df")
    return len(rows)


def key_cardinalities(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """COUNT(DISTINCT column) for every (table, column) pair where column is
    one of the known join keys and actually exists on that table --
    discovered via information_schema rather than a hardcoded table list, so
    it also picks up tc on the access-log tables and trace_id/span_id on
    span_roots without needing to be told about them by name."""
    pairs = con.execute(
        "SELECT table_name, column_name FROM information_schema.columns WHERE column_name = ANY(?)",
        [list(_JOIN_KEY_COLUMNS)],
    ).fetchall()
    result: dict[str, int] = {}
    for table, column in pairs:
        n = con.execute(f"SELECT COUNT(DISTINCT {column}) FROM {table}").fetchone()[0]
        result[f"{table}.{column}"] = n
    return result


def clock_offset_coverage(con: duckdb.DuckDBPyConnection) -> dict:
    """How many cross-host span-linked pairs could and couldn't be given a
    clock offset. skew.py's own rule is that a missing pair means "couldn't
    check," never "offset is zero" -- but that guarantee previously lived
    only in a docstring no model reads (cycle-5 VERIFY_FAIL: all 16 skipped
    pairs were the Tomcat tier, invisible next to the 8 reported offsets)."""
    observed = con.execute(
        """
        SELECT DISTINCT LEAST(parent.cmdb_id, child.cmdb_id) AS host_a,
               GREATEST(parent.cmdb_id, child.cmdb_id) AS host_b
        FROM spans child
        JOIN spans parent ON child.parent_id = parent.span_id
        WHERE child.cmdb_id != parent.cmdb_id
        """
    ).fetchall()
    observed_pairs = {(a, b) for a, b in observed}
    estimated = con.execute("SELECT host_a, host_b FROM clock_offsets").fetchall()
    estimated_pairs = {(a, b) for a, b in estimated}
    not_estimable = sorted(observed_pairs - estimated_pairs)
    return {
        "unordered_pairs_observed": len(observed_pairs),
        "pairs_estimated": len(estimated_pairs),
        "pairs_not_estimable": [{"host_a": a, "host_b": b} for a, b in not_estimable],
        "note": "a pair absent from clock_offsets means the offset could not "
        "be estimated (only one call direction observed), not that it is zero",
        "method_assumption": "clock_offsets.method="
        "'ntp_style_parent_child_reciprocal' assumes a child span cannot "
        "begin before its parent, plus symmetric network delay in both call "
        "directions (Target_Design.MD's amended §4.1) -- a premise "
        "about call semantics, not a claim about what caused this incident. "
        "The offset self-validates on this dataset (implied one-way delay "
        "~2ms matches the independently-measured same-host parent-child "
        "delay) but a model should be able to reject an estimate on this "
        "assumption's grounds, which is why it is stated here rather than "
        "only in skew.py's docstring.",
    }


def trace_root_coverage(con: duckdb.DuckDBPyConnection) -> dict:
    """true_root vs. earliest_fallback counts from span_roots, plus two
    coverage caveats cycle-6 VERIFY_FAIL found missing from this same table:

    1. A self-referential span (span_id == parent_id) exists for effectively
       every trace but is invisible to the parent_id-not-in-span_id root rule
       (its own parent_id is trivially a member of span_id) -- surfaced here
       as a count, not as a root_kind change: whether that marker should ever
       count as a root is a rule question this function does not decide.
    2. earliest_fallback picks the globally-earliest span by raw timestamp_ms,
       comparing across hosts whose clocks skew.py measures as up to
       hundreds of seconds apart for some pairs. This is a caveat about the
       comparison, not a correction to it -- topology.py's fallback rule is
       unchanged; a fix that corrected the selected root using clock_offsets
       would apply skew.py's own logged causal assumption to what
       SchemaOfCSVs.MD documents as a purely mechanical rule, which is a
       design question left to rca-planner, not a bug fix.
    """
    counts = dict(
        con.execute("SELECT root_kind, COUNT(*) FROM span_roots GROUP BY root_kind").fetchall()
    )
    self_referential_span_count = con.execute(
        "SELECT COUNT(*) FROM spans WHERE span_id = parent_id"
    ).fetchone()[0]
    fallback_traces_with_self_referential_span = con.execute(
        """
        SELECT COUNT(DISTINCT s.trace_id)
        FROM spans s
        JOIN span_roots sr ON s.trace_id = sr.trace_id AND sr.root_kind = 'earliest_fallback'
        WHERE s.span_id = s.parent_id
        """
    ).fetchone()[0]
    multi_host_fallback = con.execute(
        """
        WITH trace_hosts AS (
            SELECT trace_id, COUNT(DISTINCT cmdb_id) AS n_hosts FROM spans GROUP BY trace_id
        )
        SELECT sr.trace_id
        FROM span_roots sr
        JOIN trace_hosts th ON sr.trace_id = th.trace_id
        WHERE sr.root_kind = 'earliest_fallback' AND th.n_hosts > 1
        """
    ).fetchall()
    fallback_traces_spanning_multiple_hosts = len(multi_host_fallback)
    # Cycle-8 VERIFY_FAIL (on top of cycle-7's own fix): "has >=1 measured
    # pair" is not "bounded" -- every one of those traces ALSO spans an
    # unmeasured pair on this dataset (verified: 0 traces have every pair
    # measured), so pairing that bucket with max_measured_offset_ms as if it
    # were an applicable bound was the same absence-as-measurement error
    # again, one level deeper. Report all_pairs_measured explicitly (0 here)
    # so "bounded" is never implied for a bucket that isn't.
    pair_measurement = con.execute(
        """
        WITH multi_host_fallback AS (
            SELECT unnest(?::VARCHAR[]) AS trace_id
        ),
        trace_pairs AS (
            SELECT DISTINCT mhf.trace_id,
                   LEAST(s1.cmdb_id, s2.cmdb_id) AS host_a,
                   GREATEST(s1.cmdb_id, s2.cmdb_id) AS host_b
            FROM multi_host_fallback mhf
            JOIN spans s1 ON s1.trace_id = mhf.trace_id
            JOIN spans s2 ON s2.trace_id = mhf.trace_id AND s1.cmdb_id < s2.cmdb_id
        )
        SELECT
            SUM(CASE WHEN co.host_a IS NOT NULL THEN 1 ELSE 0 END) > 0 AS any_measured,
            SUM(CASE WHEN co.host_a IS NULL THEN 1 ELSE 0 END) = 0 AS all_measured
        FROM trace_pairs tp
        LEFT JOIN clock_offsets co ON tp.host_a = co.host_a AND tp.host_b = co.host_b
        GROUP BY tp.trace_id
        """,
        [[t for (t,) in multi_host_fallback]],
    ).fetchdf()
    fallback_traces_with_all_pairs_measured = (
        int(pair_measurement["all_measured"].sum()) if len(pair_measurement) else 0
    )
    fallback_traces_with_some_pair_measured = (
        int((pair_measurement["any_measured"] & ~pair_measurement["all_measured"]).sum())
        if len(pair_measurement)
        else 0
    )
    fallback_traces_with_no_pair_measured = (
        int((~pair_measurement["any_measured"]).sum()) if len(pair_measurement) else 0
    )
    max_measured_offset_ms = con.execute(
        "SELECT MAX(ABS(offset_ms)) FROM clock_offsets"
    ).fetchone()[0]
    return {
        "total_traces": sum(counts.values()),
        "true_root": counts.get("true_root", 0),
        "earliest_fallback": counts.get("earliest_fallback", 0),
        "note": "earliest_fallback means every span in this trace has a "
        "parent_id present in the span_id set (no row lacks a parent within "
        "the file), so no true root was discoverable; the fallback is the "
        "earliest-timestamp span in that trace, not a discovered call root",
        "self_referential_span_count": self_referential_span_count,
        "fallback_traces_with_self_referential_span": fallback_traces_with_self_referential_span,
        "self_referential_span_note": "span_id == parent_id rows exist for "
        "effectively every trace but are structurally invisible to the "
        "parent_id-not-in-span_id root rule (trivially a member of span_id); "
        "not treated as a root here, reported as a coverage fact only",
        "fallback_traces_spanning_multiple_hosts": fallback_traces_spanning_multiple_hosts,
        "fallback_traces_with_all_pairs_measured": fallback_traces_with_all_pairs_measured,
        "fallback_traces_with_some_pair_measured": fallback_traces_with_some_pair_measured,
        "fallback_traces_with_no_pair_measured": fallback_traces_with_no_pair_measured,
        "max_measured_cross_host_clock_offset_ms": max_measured_offset_ms,
        "cross_host_fallback_caveat": "earliest_fallback compares raw "
        "timestamp_ms across hosts. fallback_traces_with_all_pairs_measured "
        "is the count of traces where EVERY host pair the trace spans has a "
        "measured clock offset -- only for those is the skew fully known "
        "(up to max_measured_cross_host_clock_offset_ms), and this dataset "
        "has zero such traces. fallback_traces_with_some_pair_measured have "
        "a measured offset for at least one of their host pairs but at "
        "least one other pair unmeasured, so the trace's true skew is not "
        "bounded even though partial information exists. "
        "fallback_traces_with_no_pair_measured have no measured information "
        "at all. In no case is 'earliest' corrected for skew here.",
    }


def clock_offset_facts(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Every measured cross-host clock offset as a raw, attributable record
    (host_a, host_b, offset_ms, n_pairs, implied_one_way_delay_ms,
    residual_std_ms, method) -- not just the row count. A row count alone
    tells the model clock_offsets is non-empty; this surfaces the actual
    offset values (e.g. the ~321,682 ms MG01/dockerA1 offset) so the lead
    investigator can read them directly instead of inferring them exist."""
    cols = (
        "host_a",
        "host_b",
        "offset_ms",
        "n_pairs",
        "implied_one_way_delay_ms",
        "residual_std_ms",
        "method",
    )
    rows = con.execute(f"SELECT {', '.join(cols)} FROM clock_offsets").fetchall()
    return [dict(zip(cols, row, strict=True)) for row in rows]
