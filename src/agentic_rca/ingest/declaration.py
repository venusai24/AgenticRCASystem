"""The data-sufficiency declaration (Target_Design.MD Stage 0): a mechanical
list of what joins exist, what coverage gaps exist, and what was measured --
each entry an observation with its supporting count. No entry may say
"suspicious," "anomalous," or "likely"; that judgment belongs to the
investigation stages, not to ingest.
"""

from __future__ import annotations

import duckdb

from agentic_rca.ingest.profile import (
    app_metrics_window_diff,
    clock_offset_coverage,
    clock_offset_facts,
    container_metrics_window_diff,
    host_inventory_null_counts,
    host_inventory_per_source,
    key_cardinalities,
    metric_value_variance,
    table_row_counts,
    table_schemas,
    time_coverage,
    trace_root_coverage,
)

# The full host inventory as verified in SchemaOfCSVs.MD -- container_metrics
# is the only source that sees it, and only in the incident window (dockerB2
# is incident-only). Used here purely to phrase coverage as "N of 18", not as
# an assumption baked into any query.
_FULL_HOST_INVENTORY_INCIDENT = {
    "IG01",
    "IG02",
    "MG01",
    "MG02",
    "Mysql01",
    "Mysql02",
    "Redis01",
    "Redis02",
    "Tomcat01",
    "Tomcat02",
    "Tomcat03",
    "Tomcat04",
    "apache01",
    "apache02",
    "dockerA1",
    "dockerA2",
    "dockerB1",
    "dockerB2",
}


def _distinct_non_null(
    con: duckdb.DuckDBPyConnection, column: str, *tables: str
) -> tuple[list, int]:
    """DISTINCT values of `column` across `tables`, sorted, with NULLs
    dropped rather than left to break sorted() -- and counted rather than
    silently discarded. `column` here is a regex-extracted logparse output
    (gateway, status), which is None on a parse-regex miss by design
    (logparse.py: "raw text is still retained... nothing is lost"); that
    None reaching a bare `sorted()` crashed all of Stage 0 over one bad log
    line (cycle-13 VERIFY_FAIL, reproduced against the real store)."""
    union_sql = " UNION ALL ".join(f"SELECT {column} FROM {t}" for t in tables)
    rows = [r[0] for r in con.execute(union_sql).fetchall()]
    values = sorted({v for v in rows if v is not None})
    null_count = sum(1 for v in rows if v is None)
    return values, null_count


def build_declaration(con: duckdb.DuckDBPyConnection) -> dict:
    hosts = host_inventory_per_source(con)
    coverage = time_coverage(con)
    row_counts = table_row_counts(con)

    trace_hosts = hosts["spans"]
    container_hosts = hosts["container_metrics"]
    log_hosts = hosts["logs"]

    gateway_hosts, gateway_parse_failures = _distinct_non_null(
        con, "gateway", "log_access_localhost"
    )
    sr_variance = metric_value_variance(con, "sr")
    status_codes, status_parse_failures = _distinct_non_null(
        con, "status", "log_access_apache", "log_access_localhost"
    )

    declaration = {
        "joins": {
            "tc_to_log_path": {
                "available": True,
                "note": "ServiceTestN path segment inside log value text, structurally extracted",
            },
            "cmdb_id_across_sources": {
                "available": "partial",
                "trace_hosts": sorted(trace_hosts),
                "container_metric_hosts": sorted(container_hosts),
                "log_hosts": sorted(log_hosts),
                "note": "no file's cmdb_id spans the full infrastructure; "
                "Mysql*/Redis*/apache* never appear in spans",
            },
            "trace_to_log": {
                "available": False,
                "note": "no trace_id column in logs; would require time+host+path fallback",
            },
            "log_gateway_to_spans_and_container_metrics": {
                "available": True,
                "gateway_hosts": gateway_hosts,
                "gateway_parse_failures": gateway_parse_failures,
                "note": "log_access_localhost.gateway (IG01/IG02) is the only "
                "structural link from the log tier into the gateway tier -- "
                "logs.cmdb_id itself never leaves apache01/02 + Tomcat01-04; "
                "both gateway hosts also appear in spans.cmdb_id and "
                "container_metrics.cmdb_id. gateway_parse_failures counts "
                "rows where this regex-extracted field failed to parse "
                "(None), rather than crashing or silently dropping them.",
            },
        },
        "baseline_availability": {
            "app_metrics": "baseline_app_metrics.csv present",
            "container_metrics": "baseline_container_metrics.csv present",
            "spans": "no baseline traces file exists",
            "logs": "no baseline logs file exists",
        },
        "host_coverage": {
            "spans": f"{len(trace_hosts)} hosts (app/gateway tier only, verified never touches "
            "Mysql*/Redis*/apache*)",
            "container_metrics": f"{len(container_hosts)} of "
            f"{len(_FULL_HOST_INVENTORY_INCIDENT)} known incident-window hosts, cross-window "
            "union -- baseline and incident host counts differ; see "
            "container_metrics_window_diff, not this figure, for per-window coverage",
            "logs": f"{len(log_hosts)} hosts (apache01/02 + Tomcat01-04 only)",
        },
        "host_inventory_null_cmdb_id_rows": host_inventory_null_counts(con),
        "app_metrics_window_diff": app_metrics_window_diff(con),
        "container_metrics_window_diff": container_metrics_window_diff(con),
        "metric_facts": {
            "sr_has_zero_variance": sr_variance == 0.0,
            "sr_stddev": sr_variance,
        },
        "log_status_codes_observed": status_codes,
        "log_status_parse_failures": status_parse_failures,
        "row_counts": row_counts,
        "time_coverage": coverage,
        "trace_root_coverage": trace_root_coverage(con),
        "schemas": table_schemas(con),
        "series_cadence_note": "see series_cadence table: per-series "
        "(source_table, window_label, series_key) median/max inter-sample "
        "delta in seconds; row count already reported in row_counts above "
        "-- not inlined here, ~3,300 rows on this dataset (Target_Design.MD's "
        '"never dump large results into context" rule)',
        "change_events": {
            "available": False,
            "note": "no source file in incident_data/ carries deploy/change-event records",
        },
        "key_cardinalities": key_cardinalities(con),
        "clock_offsets": clock_offset_facts(con),
        "clock_offsets_coverage": clock_offset_coverage(con),
    }
    return declaration
