"""CSV -> DuckDB tables. Read-only against incident_data/: never writes back
to the source CSVs (the protect-data hook blocks Edit/Write there, but that
guards this session's tools, not a stray pandas.to_csv, so the loader takes
paths and only ever reads them).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from agentic_rca.ingest.sources import SOURCES_BY_KEY, resolve_path
from agentic_rca.ingest.units import to_seconds

# App-metrics wide format (rr, sr, cnt, mrt) pivoted to long (metric_name, value).
# Target_Design.MD's schema-assumptions bracket sanctions this explicitly:
# "If it is wide format, ingest pivots it. Nothing downstream changes."
_APP_METRIC_COLUMNS = ("rr", "sr", "cnt", "mrt")


def load_app_metrics(con: duckdb.DuckDBPyConnection, data_dir: Path, key: str) -> int:
    source = SOURCES_BY_KEY[key]
    df = pd.read_csv(resolve_path(source, data_dir))
    long_df = df.melt(
        id_vars=["timestamp", "tc"],
        value_vars=list(_APP_METRIC_COLUMNS),
        var_name="metric_name",
        value_name="value",
    )
    long_df["timestamp_s"] = long_df["timestamp"].astype(float)
    long_df["window_label"] = source.window_label
    out = long_df[["timestamp_s", "window_label", "tc", "metric_name", "value"]]
    con.execute("INSERT INTO app_metrics SELECT * FROM out")
    return len(out)


def load_container_metrics(con: duckdb.DuckDBPyConnection, data_dir: Path, key: str) -> int:
    source = SOURCES_BY_KEY[key]
    df = pd.read_csv(resolve_path(source, data_dir))
    df["timestamp_s"] = df["timestamp"].astype(float)
    df["window_label"] = source.window_label
    out = df[["timestamp_s", "window_label", "cmdb_id", "kpi_name", "value"]]
    con.execute("INSERT INTO container_metrics SELECT * FROM out")
    return len(out)


def load_spans(con: duckdb.DuckDBPyConnection, data_dir: Path, key: str = "incident_traces") -> int:
    source = SOURCES_BY_KEY[key]
    df = pd.read_csv(resolve_path(source, data_dir))
    df["timestamp_ms"] = df["timestamp"].astype("int64")
    df["timestamp_s"] = df["timestamp_ms"].map(to_seconds)
    df["window_label"] = source.window_label
    df["duration_ms"] = df["duration"].astype("int64")
    out = df[
        [
            "timestamp_ms",
            "timestamp_s",
            "window_label",
            "cmdb_id",
            "parent_id",
            "span_id",
            "trace_id",
            "duration_ms",
        ]
    ]
    con.execute("INSERT INTO spans SELECT * FROM out")
    return len(out)


def load_logs(
    con: duckdb.DuckDBPyConnection, data_dir: Path, key: str = "cluster_incident_logs"
) -> int:
    source = SOURCES_BY_KEY[key]
    df = pd.read_csv(resolve_path(source, data_dir))
    df["timestamp_s"] = df["timestamp"].astype(float)
    df["window_label"] = source.window_label
    df["value_raw"] = df["value"]
    df["template_id"] = None  # filled in later by logparse.assign_templates
    out = df[
        ["log_id", "timestamp_s", "window_label", "cmdb_id", "log_name", "value_raw", "template_id"]
    ]
    con.execute("INSERT INTO logs SELECT * FROM out")
    return len(out)


def load_all(con: duckdb.DuckDBPyConnection, data_dir: Path) -> dict[str, int]:
    """Load every source file. Returns row counts per table for the caller to
    verify against SchemaOfCSVs.MD's known ground truth."""
    counts = {
        "baseline_app_metrics": load_app_metrics(con, data_dir, "baseline_app_metrics"),
        "cluster_app_metrics": load_app_metrics(con, data_dir, "cluster_app_metrics"),
        "baseline_container_metrics": load_container_metrics(
            con, data_dir, "baseline_container_metrics"
        ),
        "container_metrics": load_container_metrics(con, data_dir, "container_metrics"),
        "incident_traces": load_spans(con, data_dir),
        "cluster_incident_logs": load_logs(con, data_dir),
    }
    return counts
