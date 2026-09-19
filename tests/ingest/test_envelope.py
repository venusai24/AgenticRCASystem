from agentic_rca.tools.envelope import (
    Provenance,
    ToolError,
    inspect_result,
    make_envelope,
    make_error_envelope,
)


def test_envelope_carries_the_common_fields():
    env = make_envelope(
        rows=[{"a": 1}, {"a": 2}],
        provenance=Provenance(source="test"),
        coverage={"window": "incident"},
    )
    assert env.row_count == 2
    assert env.result_handle is not None
    assert env.result_handle.row_count == 2
    assert env.ok
    assert env.empty_because is None


def test_empty_result_has_no_handle_and_states_why():
    env = make_envelope(
        rows=[],
        provenance=Provenance(source="test"),
        coverage={},
        empty_because="no_data_in_window",
    )
    assert env.row_count == 0
    assert env.result_handle is None
    assert env.empty_because == "no_data_in_window"


def test_truncation_flag_set_when_rows_exceed_preview_limit():
    rows = [{"a": i} for i in range(25)]
    env = make_envelope(
        rows=rows, provenance=Provenance(source="test"), coverage={}, preview_limit=20
    )
    assert env.truncated is True
    assert len(env.preview) == 20
    assert env.row_count == 25


def test_error_envelope_has_no_result_and_is_not_ok():
    env = make_error_envelope(
        error=ToolError(type="invalid_query", diagnostic="unknown column"),
        provenance=Provenance(source="test"),
    )
    assert not env.ok
    assert env.result_handle is None
    assert env.error.type == "invalid_query"


def test_inspect_result_pages_through_a_handle():
    rows = [{"a": i} for i in range(25)]
    env = make_envelope(rows=rows, provenance=Provenance(source="test"), coverage={})

    page = inspect_result(env.result_handle.id, offset=20, limit=20)
    assert page.ok
    assert [r["a"] for r in page.preview] == [20, 21, 22, 23, 24]
    assert page.row_count == 25
    assert page.truncated is False


def test_inspect_result_sorts_and_projects_columns():
    rows = [{"a": 2, "b": "y"}, {"a": 1, "b": "x"}, {"a": 3, "b": "z"}]
    env = make_envelope(rows=rows, provenance=Provenance(source="test"), coverage={})

    page = inspect_result(env.result_handle.id, sort="-a", columns=["a"])
    assert page.preview == [{"a": 3}, {"a": 2}, {"a": 1}]


def test_inspect_result_unknown_handle_is_an_error():
    page = inspect_result("q999999")
    assert not page.ok
    assert page.error.type == "invalid_query"


def test_inspect_result_truncated_flag_when_more_rows_remain():
    rows = [{"a": i} for i in range(25)]
    env = make_envelope(rows=rows, provenance=Provenance(source="test"), coverage={})

    page = inspect_result(env.result_handle.id, offset=0, limit=20)
    assert page.truncated is True
    assert len(page.preview) == 20
