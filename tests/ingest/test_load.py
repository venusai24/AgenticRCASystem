from agentic_rca.ingest.load import (
    load_all,
    load_app_metrics,
    load_container_metrics,
    load_logs,
    load_spans,
)

# Raw CSV row counts, verified directly against the source files (ground
# truth also recorded in SchemaOfCSVs.MD / plans/stage0-ingest.md's order of
# work: "341/341/27,173/26,990/324,321/34,047").
RAW_ROW_COUNTS = {
    "baseline_app_metrics": 341,
    "cluster_app_metrics": 341,
    "baseline_container_metrics": 27173,
    "container_metrics": 26990,
    "incident_traces": 324321,
    "cluster_incident_logs": 34047,
}

# What load_all() actually inserts per table -- 1:1 with raw rows, except the
# two app-metrics sources, which are pivoted wide-to-long at a fixed 4
# metrics per row (rr, sr, cnt, mrt), so inserted count is 4x raw count.
LOADED_ROW_COUNTS = {
    **RAW_ROW_COUNTS,
    "baseline_app_metrics": RAW_ROW_COUNTS["baseline_app_metrics"] * 4,
    "cluster_app_metrics": RAW_ROW_COUNTS["cluster_app_metrics"] * 4,
}


def test_load_all_matches_known_row_counts(db_con, data_dir):
    counts = load_all(db_con, data_dir)
    assert counts == LOADED_ROW_COUNTS


def test_app_metrics_pivot_yields_four_rows_per_source_row(db_con, data_dir):
    n = load_app_metrics(db_con, data_dir, "cluster_app_metrics")
    assert n == RAW_ROW_COUNTS["cluster_app_metrics"] * 4
    stored = db_con.execute(
        "SELECT COUNT(*) FROM app_metrics WHERE window_label = 'incident'"
    ).fetchone()[0]
    assert stored == n


def test_app_metrics_pivot_produces_exactly_four_metric_names(db_con, data_dir):
    load_app_metrics(db_con, data_dir, "cluster_app_metrics")
    names = {
        r[0] for r in db_con.execute("SELECT DISTINCT metric_name FROM app_metrics").fetchall()
    }
    assert names == {"rr", "sr", "cnt", "mrt"}


def test_sr_never_dips_matches_schema_doc_finding(db_con, data_dir):
    # SchemaOfCSVs.MD: "sr is 100.0 in both files, always" -- verify ingest
    # doesn't silently corrupt or drop this fact during the pivot.
    load_app_metrics(db_con, data_dir, "cluster_app_metrics")
    load_app_metrics(db_con, data_dir, "baseline_app_metrics")
    distinct_sr = db_con.execute(
        "SELECT DISTINCT value FROM app_metrics WHERE metric_name = 'sr'"
    ).fetchall()
    assert distinct_sr == [(100.0,)]


def test_container_metrics_row_count(db_con, data_dir):
    n = load_container_metrics(db_con, data_dir, "container_metrics")
    assert n == RAW_ROW_COUNTS["container_metrics"]


def test_spans_timestamp_normalized_from_ms_to_s(db_con, data_dir):
    load_spans(db_con, data_dir)
    row = db_con.execute("SELECT timestamp_ms, timestamp_s FROM spans LIMIT 1").fetchone()
    ts_ms, ts_s = row
    assert ts_s == ts_ms / 1000.0
    # Verified incident window per SchemaOfCSVs.MD: 2021-03-04 10:00-10:30 UTC.
    assert 1614852000 <= ts_s <= 1614853800


def test_spans_span_id_has_no_duplicates(db_con, data_dir):
    load_spans(db_con, data_dir)
    total, distinct = db_con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT span_id) FROM spans"
    ).fetchone()
    assert total == distinct


def test_logs_row_count_and_three_log_names(db_con, data_dir):
    n = load_logs(db_con, data_dir)
    assert n == RAW_ROW_COUNTS["cluster_incident_logs"]
    names = {r[0] for r in db_con.execute("SELECT DISTINCT log_name FROM logs").fetchall()}
    assert names == {"apache_access_log", "localhost_access_log", "gc"}
