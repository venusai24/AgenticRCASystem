import pytest
from agentic_rca.tools.envelope import make_envelope, inspect_result, Provenance, ToolError

def test_inspect_result_paging_past_end():
    rows = [{"id": 1}, {"id": 2}]
    env = make_envelope(
        rows=rows, 
        provenance=Provenance(source="test"),
        coverage={"test": True}
    )
    handle_id = env.result_handle.id
    
    # Paging past the end should return an invalid_query error
    res = inspect_result(handle_id, offset=5, limit=10)
    assert not res.ok
    assert res.error.type == "invalid_query"
    assert "past the end of the result" in res.error.diagnostic

def test_inspect_result_negative_offset():
    rows = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    env = make_envelope(
        rows=rows, 
        provenance=Provenance(source="test"),
        coverage={"test": True}
    )
    handle_id = env.result_handle.id
    
    # Negative offset should slice from the end
    res = inspect_result(handle_id, offset=-2, limit=10)
    assert res.ok
    assert len(res.preview) == 2
    assert res.preview[0]["id"] == 3
    assert res.preview[1]["id"] == 4

def test_inspect_result_negative_limit():
    rows = [{"id": 1}, {"id": 2}]
    env = make_envelope(
        rows=rows, 
        provenance=Provenance(source="test"),
        coverage={"test": True}
    )
    handle_id = env.result_handle.id
    
    # Negative limit should return an error
    res = inspect_result(handle_id, offset=0, limit=-1)
    assert not res.ok
    assert res.error.type == "invalid_query"
    assert "limit must be >= 1" in res.error.diagnostic
