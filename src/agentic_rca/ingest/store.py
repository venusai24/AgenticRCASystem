"""DuckDB connection and DDL.

Persisted to a file rather than in-memory (plan underspecified-point #7):
re-runs are cheap and a verifier can re-execute cited queries against a
stable store after the ingest process that built it has exited.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

DDL = """
CREATE TABLE IF NOT EXISTS app_metrics (
    timestamp_s DOUBLE,
    window_label VARCHAR,
    tc VARCHAR,
    metric_name VARCHAR,
    value DOUBLE
);

CREATE TABLE IF NOT EXISTS container_metrics (
    timestamp_s DOUBLE,
    window_label VARCHAR,
    cmdb_id VARCHAR,
    kpi_name VARCHAR,
    value DOUBLE
);

CREATE TABLE IF NOT EXISTS spans (
    timestamp_ms BIGINT,
    timestamp_s DOUBLE,
    window_label VARCHAR,
    cmdb_id VARCHAR,
    parent_id VARCHAR,
    span_id VARCHAR,
    trace_id VARCHAR,
    duration_ms BIGINT
);

CREATE TABLE IF NOT EXISTS logs (
    log_id VARCHAR,
    timestamp_s DOUBLE,
    window_label VARCHAR,
    cmdb_id VARCHAR,
    log_name VARCHAR,
    value_raw VARCHAR,
    template_id VARCHAR
);

CREATE TABLE IF NOT EXISTS log_access_apache (
    log_id VARCHAR,
    method VARCHAR,
    path VARCHAR,
    tc VARCHAR,
    protocol VARCHAR,
    status VARCHAR,
    bytes VARCHAR,
    agent VARCHAR,
    tail_raw VARCHAR
);

CREATE TABLE IF NOT EXISTS log_access_localhost (
    log_id VARCHAR,
    gateway VARCHAR,
    method VARCHAR,
    path VARCHAR,
    tc VARCHAR,
    protocol VARCHAR,
    status VARCHAR,
    bytes VARCHAR,
    agent VARCHAR,
    self_reported_duration_s DOUBLE,
    tail_raw VARCHAR
);

CREATE TABLE IF NOT EXISTS log_templates (
    template_id VARCHAR,
    log_name VARCHAR,
    pattern VARCHAR,
    line_count BIGINT,
    example_log_id VARCHAR
);

CREATE TABLE IF NOT EXISTS span_roots (
    trace_id VARCHAR,
    root_span_id VARCHAR,
    root_kind VARCHAR
);

CREATE TABLE IF NOT EXISTS topology_edges (
    window_label VARCHAR,
    caller_host VARCHAR,
    callee_host VARCHAR,
    call_count BIGINT,
    duration_ms_p50 DOUBLE,
    duration_ms_p95 DOUBLE,
    duration_ms_max DOUBLE
);

CREATE TABLE IF NOT EXISTS clock_offsets (
    host_a VARCHAR,
    host_b VARCHAR,
    offset_ms DOUBLE,
    n_pairs BIGINT,
    implied_one_way_delay_ms DOUBLE,
    residual_std_ms DOUBLE,
    method VARCHAR
);

CREATE TABLE IF NOT EXISTS series_cadence (
    source_table VARCHAR,
    window_label VARCHAR,
    series_key VARCHAR,
    n_deltas BIGINT,
    median_delta_s DOUBLE,
    max_delta_s DOUBLE
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_spans_span_id ON spans(span_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent_id ON spans(parent_id);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_logs_log_id ON logs(log_id);
"""

# Single source of truth for table names, derived from DDL itself rather than
# hand-maintained twice (here and in profile.py's _PROFILED_TABLES) -- the
# fix-cycle-4 root cause of that duplication drifting apart otherwise.
TABLE_NAMES: tuple[str, ...] = tuple(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", DDL))


def connect(db_path: Path | str) -> duckdb.DuckDBPyConnection:
    """Open (creating if absent) the store and ensure schema exists."""
    con = duckdb.connect(str(db_path))
    con.execute(DDL)
    con.execute(INDEXES)
    return con


def reset(con: duckdb.DuckDBPyConnection) -> None:
    """Truncate all tables so a re-run starts clean without dropping schema."""
    for t in TABLE_NAMES:
        con.execute(f"DELETE FROM {t}")
