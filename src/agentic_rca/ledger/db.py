import os
import duckdb
from pathlib import Path


def init_db(run_dir: str, run_id: str, is_amendment: bool = False, parent_db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Initialize the run DB in `runs/<run_id>/run.duckdb`.
    If is_amendment is True, we attach the parent DB read-only and create a new local file.
    """
    os.makedirs(run_dir, exist_ok=True)
    db_path = os.path.join(run_dir, "run.duckdb")
    
    con = duckdb.connect(db_path)
    
    # In case of amendment, attach the parent database read-only
    if is_amendment and parent_db_path:
        con.execute(f"ATTACH '{parent_db_path}' AS parent_run (READ_ONLY)")
        
    con.execute("""
        CREATE TABLE IF NOT EXISTS run (
            run_id VARCHAR PRIMARY KEY,
            parent_run_id VARCHAR,
            root_run_id VARCHAR,
            depth INTEGER,
            created_at TIMESTAMP,
            store_path VARCHAR,
            dataset_digest JSON,
            code_version VARCHAR,
            incident_ts TIMESTAMP,
            params JSON
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS ledger_events (
            run_id VARCHAR,
            seq BIGINT,
            event_id VARCHAR PRIMARY KEY,
            created_at TIMESTAMP,
            actor VARCHAR,
            step VARCHAR,
            op VARCHAR,
            record_type VARCHAR,
            record_id VARCHAR,
            payload JSON,
            UNIQUE(run_id, seq)
        );
    """)

    # DDL for queries table (written by coverage-index)
    con.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            run_id VARCHAR,
            query_id VARCHAR PRIMARY KEY,
            seq BIGINT,
            actor VARCHAR,
            step VARCHAR,
            tool VARCHAR,
            args_json JSON,
            args_sha256 VARCHAR,
            input_query_ids VARCHAR[],
            sql_texts VARCHAR[],
            status VARCHAR,
            error_type VARCHAR,
            diagnostic VARCHAR,
            retried BOOLEAN,
            row_count BIGINT,
            empty_because VARCHAR,
            order_by VARCHAR[],
            order_total BOOLEAN,
            result_digest VARCHAR,
            preview_mode VARCHAR,
            preview_label VARCHAR,
            preview_json JSON,
            summary_stats_json JSON,
            started_at TIMESTAMP,
            elapsed_ms DOUBLE
        );
    """)
    
    return con


def mint_run_id(timestamp_str: str, hex_random: str) -> str:
    """Mint a run ID like r<UTC YYYYMMDDTHHMMSSZ>-<6 hex random>"""
    return f"r{timestamp_str}-{hex_random}"

