import pytest

from agentic_rca.ingest.pipeline import build_index
from agentic_rca.ingest.profile import table_row_counts
from agentic_rca.tools.describe_dataset import describe_dataset


@pytest.fixture(scope="module")
def ingest_result(data_dir, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("pipeline") / "store.duckdb"
    result = build_index(data_dir, db_path)
    yield result
    result.con.close()


def test_build_index_runs_every_step_without_error(ingest_result):
    assert ingest_result.row_counts["incident_traces"] == 324321
    assert ingest_result.template_count > 0
    assert ingest_result.root_counts["total_traces"] == 19007
    assert ingest_result.edge_count > 0
    assert ingest_result.offset_pair_count > 0
    assert ingest_result.cadence_row_count > 0
    assert ingest_result.declaration["joins"]["trace_to_log"]["available"] is False


def test_describe_dataset_returns_a_valid_envelope(ingest_result):
    env = describe_dataset(ingest_result)
    assert env.ok
    assert env.row_count > 0
    assert env.result_handle is not None
    assert env.result_handle.row_count == env.row_count
    assert not env.truncated  # small fixed table count, fits in one preview
    table_names = {row["table"] for row in env.preview}
    assert "spans" in table_names
    assert "logs" in table_names


def test_describe_dataset_summary_stats_carry_the_declaration(ingest_result):
    env = describe_dataset(ingest_result)
    assert "joins" in env.summary_stats
    assert "baseline_availability" in env.summary_stats
    assert env.summary_stats["metric_facts"]["sr_has_zero_variance"] is True


def test_describe_dataset_provenance_is_stated(ingest_result):
    env = describe_dataset(ingest_result)
    assert env.provenance.source == "stage0_ingest"


def test_build_index_is_idempotent_against_the_same_db_path(data_dir, tmp_path):
    # Regression test for the cycle-1 VERIFY_FAIL bug: connect() uses CREATE
    # TABLE IF NOT EXISTS against a persisted file, so without pipeline.py's
    # reset() call, a second build_index() against the same db_path appended
    # on top of the first run instead of replacing it (spans 324,321 ->
    # 648,642). Every other fixture in this file builds into a fresh
    # tmp_path_factory dir per module, which would never have caught this.
    db_path = tmp_path / "store.duckdb"

    first = build_index(data_dir, db_path)
    first_counts = dict(first.row_counts)
    first_table_counts = table_row_counts(first.con)
    first.con.close()

    second = build_index(data_dir, db_path)
    try:
        assert second.row_counts == first_counts
        second_table_counts = table_row_counts(second.con)
        # Full per-table comparison, not just spans: a table-specific
        # regression (e.g. only log_templates or clock_offsets doubling)
        # would have passed the old single-table assertion.
        assert second_table_counts == first_table_counts
        assert second_table_counts["spans"] == first_counts["incident_traces"]
    finally:
        second.con.close()
