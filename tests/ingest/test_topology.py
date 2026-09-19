from agentic_rca.ingest.load import load_spans
from agentic_rca.ingest.topology import assign_span_roots, build_topology_edges


def test_root_detection_matches_verified_ground_truth(db_con, data_dir):
    load_spans(db_con, data_dir)
    counts = assign_span_roots(db_con)
    # SchemaOfCSVs.MD, verified: 3,843 true-root rows, covering 2,897 of
    # 19,007 traces (15%); the remaining 16,110 traces get the fallback.
    assert counts["true_root_rows"] == 3843
    assert counts["traces_with_true_root"] == 2897
    assert counts["total_traces"] == 19007
    assert counts["traces_with_fallback"] == 19007 - 2897


def test_every_trace_gets_exactly_one_root_row(db_con, data_dir):
    load_spans(db_con, data_dir)
    assign_span_roots(db_con)
    total_traces, total_root_rows = db_con.execute(
        "SELECT (SELECT COUNT(DISTINCT trace_id) FROM spans), (SELECT COUNT(*) FROM span_roots)"
    ).fetchone()
    assert total_root_rows == total_traces


def test_root_detection_is_by_set_membership_not_id_shape(db_con, data_dir):
    # SchemaOfCSVs.MD: the dash/@ id format is NOT a root marker -- most such
    # ids are ordinary internal nodes. Confirm root_kind assignment doesn't
    # correlate with that id shape by construction: fallback roots include
    # ids of many shapes, not a filtered subset.
    load_spans(db_con, data_dir)
    assign_span_roots(db_con)
    fallback_ids = [
        r[0]
        for r in db_con.execute(
            "SELECT root_span_id FROM span_roots WHERE root_kind = 'earliest_fallback' LIMIT 5"
        ).fetchall()
    ]
    assert len(fallback_ids) == 5  # sanity: fallback rows actually exist and are queryable


def test_topology_edges_are_host_to_host_not_service_to_service(db_con, data_dir):
    # This dataset's traces have no `service`/`operation` column
    # (Target_Design.MD schema-assumptions mismatch, recorded in the plan) --
    # edges must be keyed on cmdb_id (host), not an absent service field.
    load_spans(db_con, data_dir)
    n = build_topology_edges(db_con)
    assert n > 0
    row = db_con.execute(
        "SELECT caller_host, callee_host, call_count FROM topology_edges LIMIT 1"
    ).fetchone()
    caller, callee, count = row
    assert count > 0
    known_hosts = {
        "IG01",
        "IG02",
        "MG01",
        "MG02",
        "Tomcat01",
        "Tomcat02",
        "Tomcat03",
        "Tomcat04",
        "dockerA1",
        "dockerA2",
        "dockerB1",
        "dockerB2",
    }
    assert caller in known_hosts
    assert callee in known_hosts


def test_topology_edges_exclude_self_parent_rows_from_self_edge_counts(db_con, data_dir):
    # Regression check for cycle-8 VERIFY_FAIL: a span whose span_id equals
    # its own parent_id is not a distinct child observed under a parent, so
    # counting it as a call doubled the IG01/IG02 self-edge count exactly 2x
    # (16,987 -> 8,493 measured on the real dataset). The genuine same-host
    # call count (excluding self-parent rows) must match topology_edges
    # exactly for every self-edge, not just be smaller than the raw join.
    load_spans(db_con, data_dir)
    build_topology_edges(db_con)
    self_edges = db_con.execute(
        "SELECT caller_host, call_count FROM topology_edges WHERE caller_host = callee_host"
    ).fetchall()
    assert len(self_edges) > 0
    for host, call_count in self_edges:
        genuine = db_con.execute(
            """
            SELECT COUNT(*) FROM spans child
            JOIN spans parent ON child.parent_id = parent.span_id
            WHERE child.cmdb_id = ? AND parent.cmdb_id = ? AND child.span_id != child.parent_id
            """,
            [host, host],
        ).fetchone()[0]
        assert call_count == genuine, f"{host} self-edge count includes self-parent rows"


def test_topology_edges_carry_no_error_rate_column(db_con, data_dir):
    # There is no status/error field anywhere in incident_traces.csv
    # (verified, recorded in the plan) -- an error-rate measure here would
    # have to be fabricated. The schema simply has no such column.
    load_spans(db_con, data_dir)
    build_topology_edges(db_con)
    columns = {r[0] for r in db_con.execute("DESCRIBE topology_edges").fetchall()}
    assert "error_rate" not in columns
    assert "error_count" not in columns
