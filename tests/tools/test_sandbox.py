"""Tests for the python_sandbox tool (#9).

Verifies the bwrap security model, parquet I/O, and tool integration.
"""

from __future__ import annotations

import pytest
import shutil
from pathlib import Path
from agentic_rca.tools.sandbox import python_sandbox, SandboxArgs
from agentic_rca.tools._common import ToolInputError

# We need a mock context for the tool.
class MockContext:
    def __init__(self, data: dict[str, list[dict]]):
        self._data = data

    def result_lookup(self, query_id: str) -> list[dict] | None:
        return self._data.get(query_id)


def test_sandbox_success_parquet_io():
    """Basic parquet I/O test: read the pre-loaded df, write JSON results."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not found on PATH")

    ctx = MockContext({"q1": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]})
    code = """
import pandas as pd
# df is pre-loaded
df["z"] = df["x"] * 10
save_result(df.to_dict(orient="records"))
"""
    arg = SandboxArgs(query_id="q1", code=code, description="test")
    res = python_sandbox(arg, ctx)
    assert len(res.rows) == 2
    assert res.rows[0] == {"x": 1, "y": "a", "z": 10}
    assert res.rows[1] == {"x": 2, "y": "b", "z": 20}
    assert res.scopes[0].row_count == 2
    assert res.scopes[0].source == "derived"
    assert res.input_query_ids == ["q1"]


def test_sandbox_empty_input():
    """Handles an empty input list correctly."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not found on PATH")

    ctx = MockContext({"q1": []})
    code = """
save_result([{"len": len(df)}])
"""
    arg = SandboxArgs(query_id="q1", code=code, description="test")
    res = python_sandbox(arg, ctx)
    assert len(res.rows) == 1
    assert res.rows[0]["len"] == 0


def test_sandbox_invalid_query_id():
    """Raises ToolInputError for missing query_id."""
    ctx = MockContext({})
    arg = SandboxArgs(query_id="missing", code="pass", description="test")
    with pytest.raises(ToolInputError, match="No result handle found"):
        python_sandbox(arg, ctx)


def test_sandbox_runtime_error():
    """A python error in the sandbox is returned in the notes, producing 0 rows."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not found on PATH")
    
    ctx = MockContext({"q1": [{"x": 1}]})
    arg = SandboxArgs(query_id="q1", code="1 / 0", description="fail")
    res = python_sandbox(arg, ctx)
    assert len(res.rows) == 0
    assert "sandbox_exit_code" in res.notes
    assert res.notes["sandbox_exit_code"] == 1
    assert "ZeroDivisionError" in res.notes["stderr"]


def test_sandbox_no_save_result():
    """If the script doesn't call save_result, returns 0 rows with explanation."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not found on PATH")

    ctx = MockContext({"q1": [{"x": 1}]})
    arg = SandboxArgs(query_id="q1", code="pass", description="fail")
    res = python_sandbox(arg, ctx)
    assert len(res.rows) == 0
    assert "No result.json written" in res.notes["stderr"]


# ── security boundaries ───────────────────────────────────────────────────────

def test_sandbox_security_no_network():
    """The sandbox must not have network access."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not found on PATH")

    ctx = MockContext({"q1": []})
    code = """
import socket, sys
try:
    s = socket.socket()
    s.settimeout(1.0)
    s.connect(("8.8.8.8", 53))
except OSError as e:
    print(f"blocked: {e}")
    sys.exit(2)
sys.exit(0)
"""
    arg = SandboxArgs(query_id="q1", code=code, description="network check")
    res = python_sandbox(arg, ctx)
    assert len(res.rows) == 0
    assert res.notes["sandbox_exit_code"] == 2
    assert "blocked: " in res.notes["stdout"]


def test_sandbox_security_no_home_access():
    """The sandbox must not be able to read ~/.ssh or the repo."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not found on PATH")

    ctx = MockContext({"q1": []})
    code = """
import os, sys
if os.path.exists("/home/VenuSai/.ssh") or os.path.exists("/home/VenuSai/AgenticRCA"):
    sys.exit(2)  # Failed to block
else:
    sys.exit(0)  # Expected
"""
    arg = SandboxArgs(query_id="q1", code=code, description="home check")
    res = python_sandbox(arg, ctx)
    assert len(res.rows) == 0
    assert res.notes.get("sandbox_exit_code", 0) == 0

