"""The per-run DuckDB file (contracts-b1 §1): `runs/<run_id>/run.duckdb`.

The ingest store is never attached here, and tools never receive this
connection -- that separation is what keeps `sql` from reading the ledger
(and the critic from reading `origin` through it).

All run-DB DDL lives in this one module. contracts-b1 §13 splits DDL
ownership across ledger-core / coverage-index / llm-client for planning;
in code one schema module is simpler and avoids three partial migrations
(DECISIONS: autonomous-build).
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from pathlib import Path

import duckdb

RUN_DDL = """
CREATE TABLE IF NOT EXISTS run (
    run_id VARCHAR PRIMARY KEY,
    parent_run_id VARCHAR,
    parent_run_path VARCHAR,
    root_run_id VARCHAR,
    depth INTEGER,
    created_at VARCHAR,
    store_path VARCHAR,
    dataset_digest VARCHAR,     -- JSON {csv_name: sha256}
    code_version VARCHAR,
    incident_ts DOUBLE,
    params VARCHAR              -- JSON
);

CREATE TABLE IF NOT EXISTS ledger_events (
    run_id VARCHAR,
    seq BIGINT,
    event_id VARCHAR,
    created_at VARCHAR,
    actor VARCHAR,
    step INTEGER,
    op VARCHAR,
    record_type VARCHAR,
    record_id VARCHAR,
    payload VARCHAR             -- JSON
);

CREATE TABLE IF NOT EXISTS queries (
    run_id VARCHAR,
    query_id VARCHAR,           -- qualified <run_id>/q000001
    seq BIGINT,
    actor VARCHAR,
    step INTEGER,
    tool VARCHAR,
    args_json VARCHAR,
    args_sha256 VARCHAR,
    input_query_ids VARCHAR,    -- JSON list
    sql_texts VARCHAR,          -- JSON list, audit only
    status VARCHAR,             -- ok | error
    error_type VARCHAR,
    diagnostic VARCHAR,
    retried BOOLEAN,
    row_count BIGINT,
    empty_because VARCHAR,
    unobserved_entities VARCHAR,  -- JSON {kind: [values]} | null
    scope_verified BOOLEAN,
    order_by VARCHAR,           -- JSON list
    order_total BOOLEAN,
    result_digest VARCHAR,
    preview_mode VARCHAR,
    preview_label VARCHAR,
    preview_json VARCHAR,
    summary_stats_json VARCHAR,
    started_at VARCHAR,
    elapsed_ms DOUBLE
);

CREATE TABLE IF NOT EXISTS coverage (
    run_id VARCHAR,
    coverage_id VARCHAR,
    query_id VARCHAR,
    actor VARCHAR,
    kind VARCHAR,               -- queried | attempt
    attempt_reason VARCHAR,
    unobserved_entities VARCHAR,
    scope_verified BOOLEAN,
    sources VARCHAR,            -- JSON list
    requested VARCHAR,          -- JSON {kind: [values] | "*"}
    requested_window VARCHAR,   -- JSON {start_s, end_s} | null
    resolution DOUBLE,
    returned VARCHAR,           -- JSON {kind: [values]}
    entity_visibility VARCHAR,  -- rows | aggregated | unknown
    returned_window VARCHAR,
    row_count BIGINT
);

CREATE TABLE IF NOT EXISTS inspections (
    run_id VARCHAR,
    query_id VARCHAR,
    actor VARCHAR,
    via VARCHAR,                -- preview | inspect_result | reader | sandbox
    row_ranges VARCHAR,         -- JSON [[a, b), ...]
    created_at VARCHAR
);

CREATE TABLE IF NOT EXISTS llm_calls (
    run_id VARCHAR,
    call_id VARCHAR,
    role VARCHAR,
    provider VARCHAR,
    model VARCHAR,
    prompt_id VARCHAR,
    prompt_sha VARCHAR,
    request VARCHAR,            -- JSON (never contains credentials)
    response VARCHAR,           -- JSON
    input_tokens BIGINT,
    output_tokens BIGINT,
    latency_ms DOUBLE,
    error VARCHAR,
    created_at VARCHAR
);

CREATE TABLE IF NOT EXISTS trajectory (
    run_id VARCHAR,
    step INTEGER,
    actor VARCHAR,
    mode VARCHAR,
    action VARCHAR,
    args_sha256 VARCHAR,
    query_id VARCHAR,
    row_count BIGINT,
    error_type VARCHAR,
    ledger_seq_before BIGINT,
    ledger_seq_after BIGINT,
    tokens BIGINT,
    note VARCHAR,
    created_at VARCHAR
);

CREATE TABLE IF NOT EXISTS run_inputs (
    run_id VARCHAR,
    name VARCHAR,
    payload VARCHAR             -- JSON snapshot (clock_offsets, declaration, source units, ...)
);
"""


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def mint_run_id() -> str:
    """`r<UTC YYYYMMDDTHHMMSSZ>-<6 hex>` (contracts-b1 §2): sorts by time,
    unique without coordination."""
    return "r" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def connect_run(path: Path | str, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(RUN_DDL)
    return con
