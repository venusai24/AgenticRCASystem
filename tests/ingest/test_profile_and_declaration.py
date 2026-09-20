import pytest

from agentic_rca.ingest.declaration import build_declaration
from agentic_rca.ingest.load import load_all
from agentic_rca.ingest.logparse import assign_templates, load_access_log_tables
from agentic_rca.ingest.profile import (
    _PROFILED_TABLES,
    app_metrics_window_diff,
    container_metrics_window_diff,
    host_inventory_per_source,
    key_cardinalities,
    metric_value_variance,
    populate_series_cadence,
    table_row_counts,
    table_schemas,
    time_coverage,
)
from agentic_rca.ingest.skew import estimate_clock_offsets
from agentic_rca.ingest.store import connect
from agentic_rca.ingest.topology import assign_span_roots, build_topology_edges


@pytest.fixture(scope="module")
def db_con(data_dir, tmp_path_factory):
    # Module-scoped, not function-scoped: every test in this file is a
    # read-only check against the same ingested state (row counts, coverage,
    # declaration facts). Re-running the full pipeline -- including two
    # self-joins over 324k spans -- once per test rather than once per module
    # was the difference between ~20s and several minutes for this file, with
    # no gain in test isolation since nothing here mutates the store.
    db_path = tmp_path_factory.mktemp("profile_decl") / "test_store.duckdb"
    con = connect(db_path)
    load_all(con, data_dir)
    load_access_log_tables(con)
    assign_templates(con)
    assign_span_roots(con)
    build_topology_edges(con)
    estimate_clock_offsets(con)
    populate_series_cadence(con)
    yield con
    con.close()


def test_table_row_counts_are_nonzero_after_load(db_con):
    counts = table_row_counts(db_con)
    assert all(n > 0 for n in counts.values())


def test_table_row_counts_include_the_four_previously_unprofiled_tables(db_con):
    # Regression check: log_templates, span_roots, topology_edges, and
    # clock_offsets are all populated by the pipeline but were absent from
    # _PROFILED_TABLES, making them invisible to describe_dataset/declaration.
    counts = table_row_counts(db_con)
    for table in ("span_roots", "topology_edges", "log_templates", "clock_offsets"):
        assert counts[table] > 0, f"{table} should be profiled and nonzero"


def test_series_cadence_is_a_table_not_an_inlined_declaration_payload(db_con):
    # Regression check for the cycle-4 VERIFY_FAIL: series_cadence must be
    # queryable via table_row_counts (a real table, ~3,300 rows here) and
    # must NOT appear as a key holding the raw per-series list in the
    # declaration -- that's what made the declaration 478 KB.
    assert table_row_counts(db_con)["series_cadence"] > 1000
    decl = build_declaration(db_con)
    assert "series_cadence" not in decl
    assert "series_cadence_note" in decl


def test_time_coverage_reflects_disjoint_windows(db_con):
    coverage = time_coverage(db_con)
    app = coverage["app_metrics"]
    assert "baseline" in app and "incident" in app
    # Verified disjoint 30-min windows, 1 hour apart (SchemaOfCSVs.MD).
    assert app["baseline"]["max_s"] < app["incident"]["min_s"]


def test_host_inventory_differs_per_source(db_con):
    hosts = host_inventory_per_source(db_con)
    # The core verified claim: no file's host set equals another's, and
    # spans never touches the DB/cache/web tier.
    assert hosts["spans"] != hosts["container_metrics"]
    assert not (
        {"Mysql01", "Mysql02", "Redis01", "Redis02", "apache01", "apache02"} & hosts["spans"]
    )


def test_sr_has_zero_variance(db_con):
    assert metric_value_variance(db_con, "sr") == 0.0


def test_app_metrics_window_diff_confirms_the_tc_sets_match(db_con):
    # Regression check for cycle-14 VERIFY_FAIL: container_metrics got a
    # window diff in cycle 11, app_metrics never did, and the omission was
    # itself unlogged -- "checked, identical" must be distinguishable from
    # "never checked" even when (as verified here) the two windows' tc sets
    # are in fact identical on this dataset.
    diff = app_metrics_window_diff(db_con)
    assert diff["tc_count_in_baseline_window"] == diff["tc_count_in_incident_window"] == 11
    assert diff["tc_incident_only"] == []
    assert diff["tc_baseline_only"] == []


def test_container_metrics_window_diff_surfaces_dockerb2_as_incident_only(db_con):
    # Regression check for cycle-11 VERIFY_FAIL: host_inventory_per_source's
    # cross-window union made dockerB2 (incident-only, SchemaOfCSVs.MD)
    # invisible behind a "18 of 18" phrasing that read as full baseline
    # coverage. Diffed by identity key (cmdb_id, kpi_name), never
    # positionally, per CLAUDE.md's non-negotiable rule for this dataset.
    diff = container_metrics_window_diff(db_con)
    assert diff["hosts_in_incident_window"] > diff["hosts_in_baseline_window"]
    assert diff["hosts_incident_only"] == ["dockerB2"]
    assert diff["hosts_baseline_only"] == []
    assert diff["identity_keys_incident_only"] > 0
    assert (
        diff["identity_key_count_in_incident_window"]
        - diff["identity_keys_incident_only"]
        + diff["identity_keys_baseline_only"]
        <= diff["identity_key_count_in_baseline_window"]
    )


def test_declaration_container_metrics_host_coverage_points_to_the_window_diff(db_con):
    decl = build_declaration(db_con)
    assert "container_metrics_window_diff" in decl
    assert "cross-window union" in decl["host_coverage"]["container_metrics"]
    assert "dockerB2" in decl["container_metrics_window_diff"]["hosts_incident_only"]


def test_declaration_states_no_trace_to_log_join(db_con):
    decl = build_declaration(db_con)
    assert decl["joins"]["trace_to_log"]["available"] is False


def test_declaration_states_the_log_gateway_join(db_con):
    # Regression check for cycle-12 VERIFY_FAIL: log_access_localhost.gateway
    # (IG01/IG02) is the only structural link from the log tier into the
    # gateway tier -- logs.cmdb_id itself never leaves apache01/02 +
    # Tomcat01-04 -- and it was absent from the declaration's join list.
    decl = build_declaration(db_con)
    gateway_join = decl["joins"]["log_gateway_to_spans_and_container_metrics"]
    assert gateway_join["available"] is True
    assert gateway_join["gateway_hosts"] == ["IG01", "IG02"]


def test_declaration_states_no_baseline_for_spans_and_logs(db_con):
    decl = build_declaration(db_con)
    assert "no baseline" in decl["baseline_availability"]["spans"]
    assert "no baseline" in decl["baseline_availability"]["logs"]


def test_declaration_language_avoids_judgment_words(db_con):
    # Target_Design.MD: "no entry says 'suspicious,' 'anomalous,' or
    # 'likely.'" A crude but real check: none of those words appear anywhere
    # in the rendered declaration text.
    decl = build_declaration(db_con)
    text = str(decl).lower()
    for banned in ("suspicious", "anomalous", "likely"):
        assert banned not in text


def test_declaration_reports_sr_zero_variance_as_a_fact(db_con):
    decl = build_declaration(db_con)
    assert decl["metric_facts"]["sr_has_zero_variance"] is True


def test_declaration_status_codes_are_uniformly_200(db_con):
    decl = build_declaration(db_con)
    assert decl["log_status_codes_observed"] == ["200"]
    assert decl["log_status_parse_failures"] == 0


def test_declaration_survives_a_parse_failure_in_gateway_or_status(db_con):
    # Regression check for cycle-13 VERIFY_FAIL: a regex-extracted logparse
    # column (gateway, status) is None on a parse miss by design; a bare
    # sorted() over that column crashed all of Stage 0 on a single bad log
    # line (reproduced against the real store before this fix). Corrupts one
    # row of each table directly via SQL rather than mocking, then confirms
    # build_declaration doesn't raise and counts the failure instead of
    # silently dropping or crashing on it.
    localhost_id, orig_gateway, orig_localhost_status = db_con.execute(
        "SELECT log_id, gateway, status FROM log_access_localhost LIMIT 1"
    ).fetchone()
    apache_id, orig_apache_status = db_con.execute(
        "SELECT log_id, status FROM log_access_apache LIMIT 1"
    ).fetchone()
    db_con.execute(
        "UPDATE log_access_localhost SET gateway = NULL, status = NULL WHERE log_id = ?",
        [localhost_id],
    )
    db_con.execute("UPDATE log_access_apache SET status = NULL WHERE log_id = ?", [apache_id])
    try:
        decl = build_declaration(db_con)
        assert (
            decl["joins"]["log_gateway_to_spans_and_container_metrics"]["gateway_parse_failures"]
            == 1
        )
        assert decl["log_status_parse_failures"] == 2
        assert (
            None not in decl["joins"]["log_gateway_to_spans_and_container_metrics"]["gateway_hosts"]
        )
        assert None not in decl["log_status_codes_observed"]
    finally:
        # Restore the ORIGINAL values (not a hardcoded guess) so later tests
        # in this module-scoped fixture see the real data, unmodified.
        db_con.execute(
            "UPDATE log_access_localhost SET gateway = ?, status = ? WHERE log_id = ?",
            [orig_gateway, orig_localhost_status, localhost_id],
        )
        db_con.execute(
            "UPDATE log_access_apache SET status = ? WHERE log_id = ?",
            [orig_apache_status, apache_id],
        )


def test_declaration_survives_null_identity_keys(db_con):
    # Regression check for cycle-15 VERIFY_FAIL, the fifth recurrence of "one
    # bad cell kills all of Stage 0": a NULL cmdb_id / tc reached a bare
    # sorted() over an identity-key set. It only crashes when the NULL lands
    # in a set that also holds a str -- here container_metrics' incident-only
    # hosts already hold dockerB2 -- so the NULL is placed in the incident
    # window. Original values are captured and restored (not hardcoded)
    # because db_con is module-scoped and shared with every other test here.
    clean = build_declaration(db_con)
    assert clean["host_inventory_null_cmdb_id_rows"] == {
        "spans": 0,
        "container_metrics": 0,
        "logs": 0,
    }

    cm_rowid, cm_orig = db_con.execute(
        "SELECT rowid, cmdb_id FROM container_metrics WHERE window_label = 'incident' LIMIT 1"
    ).fetchone()
    ap_rowid, ap_orig = db_con.execute(
        "SELECT rowid, tc FROM app_metrics WHERE window_label = 'incident' LIMIT 1"
    ).fetchone()
    # A span that is the child end of a real parent/child link, so the
    # clock-offset coverage counter has something to count.
    sp_rowid, sp_orig = db_con.execute(
        "SELECT c.rowid, c.cmdb_id FROM spans c JOIN spans p ON c.parent_id = p.span_id "
        "WHERE c.span_id != c.parent_id LIMIT 1"
    ).fetchone()

    db_con.execute("UPDATE container_metrics SET cmdb_id = NULL WHERE rowid = ?", [cm_rowid])
    db_con.execute("UPDATE app_metrics SET tc = NULL WHERE rowid = ?", [ap_rowid])
    db_con.execute("UPDATE spans SET cmdb_id = NULL WHERE rowid = ?", [sp_rowid])
    try:
        decl = build_declaration(db_con)  # must not raise

        container = decl["container_metrics_window_diff"]
        assert container["cmdb_id_null_rows"] == 1
        assert None not in container["hosts_incident_only"]
        assert "dockerB2" in container["hosts_incident_only"]

        app = decl["app_metrics_window_diff"]
        assert app["tc_null_rows"] == 1
        assert None not in app["tc_incident_only"]

        assert decl["host_inventory_null_cmdb_id_rows"] == {
            "spans": 1,
            "container_metrics": 1,
            "logs": 0,
        }
        hosts = decl["joins"]["cmdb_id_across_sources"]
        assert None not in hosts["trace_hosts"]
        assert None not in hosts["container_metric_hosts"]

        assert decl["clock_offsets_coverage"]["linked_span_pairs_with_null_host"] >= 1
    finally:
        db_con.execute(
            "UPDATE container_metrics SET cmdb_id = ? WHERE rowid = ?", [cm_orig, cm_rowid]
        )
        db_con.execute("UPDATE app_metrics SET tc = ? WHERE rowid = ?", [ap_orig, ap_rowid])
        db_con.execute("UPDATE spans SET cmdb_id = ? WHERE rowid = ?", [sp_orig, sp_rowid])


def test_declaration_clock_offsets_are_inline_with_host_and_offset_fields(db_con):
    decl = build_declaration(db_con)
    offsets = decl["clock_offsets"]
    assert isinstance(offsets, list)
    assert len(offsets) > 0
    for entry in offsets:
        assert "host_a" in entry
        assert "host_b" in entry
        assert "offset_ms" in entry


def test_table_schemas_cover_every_profiled_table(db_con):
    schemas = table_schemas(db_con)
    assert set(schemas.keys()) == set(_PROFILED_TABLES)
    for table, columns in schemas.items():
        assert len(columns) > 0, f"{table} should report at least one column"
        for col in columns:
            assert "column" in col and "type" in col


def test_series_cadence_covers_all_four_tables_with_sane_deltas(db_con):
    # populate_series_cadence writes to the table rather than returning a
    # dict -- a container_metrics-sized payload (~3,259 series) was the
    # cycle-4 VERIFY_FAIL, dumped whole into describe_dataset's summary_stats.
    rows = db_con.execute(
        "SELECT source_table, series_key, n_deltas, median_delta_s, max_delta_s FROM series_cadence"
    ).fetchall()
    tables_present = {r[0] for r in rows}
    assert tables_present == {"app_metrics", "container_metrics", "spans", "logs"}
    for _source_table, _series_key, n_deltas, median_delta_s, max_delta_s in rows:
        # Structural check only -- no claim about what a "typical" cadence
        # is, just that max can never be smaller than median.
        assert max_delta_s >= median_delta_s
        assert n_deltas > 0


def test_series_cadence_app_metrics_known_value_spot_check(db_con):
    # app_metrics is melted (rr/sr/cnt/mrt share a timestamp) and sampled
    # every 60s per SchemaOfCSVs.MD; after deduping to one row per
    # (window_label, tc, timestamp_s) every series should show that cadence.
    row = db_con.execute(
        "SELECT median_delta_s FROM series_cadence "
        "WHERE source_table = 'app_metrics' AND window_label = 'incident' "
        "AND series_key = 'ServiceTest4'"
    ).fetchone()
    assert row is not None
    assert row[0] == 60.0


def test_change_events_declared_unavailable(db_con):
    decl = build_declaration(db_con)
    assert decl["change_events"]["available"] is False


def test_clock_offsets_coverage_reports_a_denominator_for_skipped_pairs(db_con):
    # Regression check for cycle-5 VERIFY_FAIL: 24 unordered host pairs have
    # cross-host linked spans on this dataset but only 8 get an offset (the
    # entire Tomcat tier is skipped, one direction only) -- this must be a
    # visible fact, not silently absent next to the 8 that ARE reported.
    decl = build_declaration(db_con)
    coverage = decl["clock_offsets_coverage"]
    assert coverage["unordered_pairs_observed"] > coverage["pairs_estimated"] > 0
    assert coverage["unordered_pairs_observed"] == coverage["pairs_estimated"] + len(
        coverage["pairs_not_estimable"]
    )
    for pair in coverage["pairs_not_estimable"]:
        assert "host_a" in pair and "host_b" in pair


def test_clock_offsets_coverage_states_the_estimator_assumption(db_con):
    # Regression check for cycle-10 VERIFY_FAIL: Target_Design.MD's third
    # required clock-offset constraint ("state the assumption") was
    # implemented only in skew.py's docstring, with zero model-visible text
    # in the declaration -- a model can't reject an estimate on an
    # assumption it was never told about.
    decl = build_declaration(db_con)
    assumption = decl["clock_offsets_coverage"]["method_assumption"]
    assert "child" in assumption and "parent" in assumption
    assert "symmetric" in assumption


def test_trace_root_coverage_reports_the_fallback_majority(db_con):
    # Regression check for cycle-5 VERIFY_FAIL: row_counts["span_roots"]
    # alone reads as "every trace has a root"; most traces on this dataset
    # actually have none (SchemaOfCSVs.MD: ~85% earliest_fallback).
    decl = build_declaration(db_con)
    coverage = decl["trace_root_coverage"]
    assert coverage["total_traces"] == coverage["true_root"] + coverage["earliest_fallback"]
    assert coverage["earliest_fallback"] > coverage["true_root"]


def test_trace_root_coverage_surfaces_self_referential_and_cross_host_caveats(db_con):
    # Regression check for cycle-6 VERIFY_FAIL: every trace has a
    # span_id == parent_id row invisible to the root rule, and most fallback
    # roots are chosen by comparing raw timestamps across hosts with measured
    # clock skew -- both must be visible facts, not silently absent.
    decl = build_declaration(db_con)
    coverage = decl["trace_root_coverage"]
    assert coverage["self_referential_span_count"] > 0
    assert coverage["fallback_traces_with_self_referential_span"] > 0
    assert coverage["fallback_traces_with_self_referential_span"] <= coverage["earliest_fallback"]
    assert coverage["fallback_traces_spanning_multiple_hosts"] > 0
    assert coverage["max_measured_cross_host_clock_offset_ms"] > 0
    # Regression check for cycle-9 VERIFY_FAIL: cycle-7's "has >=1 measured
    # pair" bucket is NOT the same as "bounded" -- on this dataset every
    # trace with a measured pair also has an unmeasured one (0 traces have
    # ALL pairs measured), so the three-way split must make that explicit
    # rather than implying bucket 1 is safely bounded by the max offset.
    assert (
        coverage["fallback_traces_with_all_pairs_measured"]
        + coverage["fallback_traces_with_some_pair_measured"]
        + coverage["fallback_traces_with_no_pair_measured"]
        == coverage["fallback_traces_spanning_multiple_hosts"]
    )
    assert coverage["fallback_traces_with_all_pairs_measured"] == 0
    assert coverage["fallback_traces_with_some_pair_measured"] > 0
    assert coverage["fallback_traces_with_no_pair_measured"] > 0


def test_key_cardinalities_cover_the_named_join_key_columns(db_con):
    kc = key_cardinalities(db_con)
    expected_present = {
        "app_metrics.tc",
        "container_metrics.cmdb_id",
        "container_metrics.kpi_name",
        "logs.cmdb_id",
        "logs.log_name",
        "log_access_apache.tc",
        "log_access_localhost.tc",
        "spans.cmdb_id",
        "spans.span_id",
        "spans.trace_id",
        "span_roots.trace_id",
        # cycle-5 additions: these are the joins actually needed to use the
        # tables this component built (logs<->log_access_*, logs<->
        # log_templates, the span parent/child self-join, span_roots<->spans)
        # -- flagged VERIFY_FAIL in cycle 4 as missing from "join-key
        # availability" despite the declaration claiming to state it.
        "log_access_apache.log_id",
        "log_access_localhost.log_id",
        "log_templates.template_id",
        "spans.parent_id",
        "span_roots.root_span_id",
        # cycle-12 addition: the only structural link from the log tier into
        # the gateway tier -- flagged non-blocking cycles 5/7/10, VERIFY_FAIL
        # in cycle 12 for carrying 7 cycles with no DECISIONS.md row.
        "log_access_localhost.gateway",
    }
    assert expected_present <= set(kc.keys())
    for key in expected_present:
        assert kc[key] > 0
