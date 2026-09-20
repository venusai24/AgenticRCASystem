from pathlib import Path

import pytest
import duckdb

from agentic_rca.ingest.pipeline import IngestResult
from agentic_rca.tools.describe_dataset import describe_dataset

def test_describe_dataset_error_path():
    # Calling with kwargs should return an error envelope
    dummy_result = IngestResult(con=None, declaration={}, row_counts={}, template_count=0, root_counts={}, edge_count=0, offset_pair_count=0, cadence_row_count=0, system_facts=[])
    res = describe_dataset(dummy_result, foo="bar")
    assert not res.ok
    assert res.error.type == "invalid_query"

def test_describe_dataset_evidence_kind(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "test.db"))
    # create dummy tables
    from agentic_rca.ingest.profile import _PROFILED_TABLES
    for t in _PROFILED_TABLES:
        con.execute(f"CREATE TABLE {t} (id INT, window_label VARCHAR, timestamp_s DOUBLE)")
    
    con.execute("CREATE TABLE IF NOT EXISTS spans (id INT, window_label VARCHAR, timestamp_s DOUBLE)")
    con.execute("CREATE TABLE IF NOT EXISTS topology_edges (id INT, window_label VARCHAR, timestamp_s DOUBLE)")
    
    dummy_result = IngestResult(con=con, declaration={}, row_counts={}, template_count=0, root_counts={}, edge_count=0, offset_pair_count=0, cadence_row_count=0, system_facts=[])
    res = describe_dataset(dummy_result)
    assert res.ok
    
    # Check that spans gets occurrence, topology_edges gets derived
    for row in res.preview:
        if row["table"] == "spans":
            assert row["evidence_kind"] == "occurrence"
        elif row["table"] == "topology_edges":
            assert row["evidence_kind"] == "derived"
