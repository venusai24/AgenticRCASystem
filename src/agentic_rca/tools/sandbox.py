"""tool-python-sandbox (#9): custom computation over result handles under bwrap.

Security model (verified on this host, bwrap 0.11.0):
  --unshare-all     : new user, mount, network, pid, IPC, UTS namespaces
  --die-with-parent : sandbox exits when the host process exits
  ro-bind /usr, <interpreter prefix>, /lib, /lib64 : readonly; no /home, no repo
  --tmpfs /tmp      : writable scratch; cleared at sandbox exit
  /in (ro-bind)     : input handle written as parquet; script at /in/script.py
  /out (bind)       : writable; sandbox writes results here as JSON

PYTHONHASHSEED=0 ensures reproducible output across re-executions (X3, contracts-b1 §7).

The tool writes one input handle to /in as parquet (via duckdb COPY TO), runs the
user script under bwrap, reads /out/result.json back.  The result is a derived
ToolResult whose reexecute goes through this same path (PYTHONHASHSEED=0 ensures
determinism).

Invariants asserted by tests:
  - Repo, ~/.ssh, network are all unreachable from inside the sandbox.
  - Re-execution on the same input reproduces the same output (digest match).
  - stdout is capped at 8 KiB; wall-clock timeout is 60 s.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import Field

from agentic_rca.tools._common import Scope, ToolArgs
from agentic_rca.tools.registry import ToolResult, tool

_PYTHON = sys.executable
_INTERPRETER_PREFIX = sys.prefix  # e.g. /home/user/miniforge3
_WALL_TIMEOUT_S = 60
_STDOUT_CAP = 8192  # bytes

# Libraries available inside the sandbox (read from the interpreter prefix):
# numpy, pandas, scipy, duckdb, pyarrow are all present in the miniforge env.

_PREAMBLE = """\
# Sandbox preamble: load the /in/handle.parquet into `df` (a pandas DataFrame).
import pathlib, json, pandas as pd
_in = pathlib.Path('/in')
_out = pathlib.Path('/out')
handle_path = _in / 'handle.parquet'
df = pd.read_parquet(handle_path) if handle_path.exists() else pd.DataFrame()

# Write result: call save_result(rows) with a list of dicts, or assign to `result`.
def save_result(rows):
    with open(_out / 'result.json', 'w') as _f:
        json.dump({'rows': rows}, _f)
"""


# ── args ──────────────────────────────────────────────────────────────────────


class SandboxArgs(ToolArgs):
    query_id: str = Field(
        description="The query_id of the result handle whose rows are loaded into `df` as a pandas DataFrame inside the sandbox."
    )
    code: str = Field(
        description=(
            "Python code to run. `df` is pre-loaded from the handle. "
            "Call save_result(rows) with a list of dicts to return output. "
            "Only stdlib + numpy + pandas + scipy + duckdb + pyarrow are available. "
            "No network. No filesystem access outside /in and /out."
        )
    )
    description: str = Field(
        description="One-sentence description of what this computation produces (stored in the result for provenance)."
    )


# ── bwrap runner ──────────────────────────────────────────────────────────────


def _bwrap_cmd(in_dir: Path, out_dir: Path) -> list[str]:
    """Build the bwrap command line.  The script is at /in/script.py."""
    cmd = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        # Read-only system bindings
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", _INTERPRETER_PREFIX, _INTERPRETER_PREFIX,
    ]
    # /lib and /lib64 may not exist on all systems; bind only if present
    for lib in ("/lib", "/lib64", "/lib32"):
        if Path(lib).exists():
            cmd += ["--ro-bind", lib, lib]
    cmd += [
        "--tmpfs", "/tmp",
        "--dev", "/dev",
        "--proc", "/proc",
        # Data directories
        "--ro-bind", str(in_dir), "/in",
        "--bind", str(out_dir), "/out",
        # Environment
        "--setenv", "PYTHONHASHSEED", "0",
        "--setenv", "HOME", "/tmp",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        # Python interpreter + entry point
        _PYTHON, "/in/script.py",
    ]
    return cmd


def _run_sandbox(in_dir: Path, out_dir: Path) -> tuple[int, str, str]:
    """Execute the sandbox; return (returncode, stdout[:cap], stderr[:cap])."""
    try:
        r = subprocess.run(
            _bwrap_cmd(in_dir, out_dir),
            capture_output=True,
            timeout=_WALL_TIMEOUT_S,
        )
        stdout = r.stdout[:_STDOUT_CAP].decode("utf-8", errors="replace")
        stderr = r.stderr[:_STDOUT_CAP].decode("utf-8", errors="replace")
        return r.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {_WALL_TIMEOUT_S} s"
    except FileNotFoundError:
        return -1, "", "bwrap not found on PATH"


# ── tool ──────────────────────────────────────────────────────────────────────


@tool(
    "python_sandbox",
    SandboxArgs,
    description=(
        "Run custom Python code over a result handle in an isolated bwrap sandbox. "
        "The handle's rows are available as a pandas DataFrame `df`. "
        "Call save_result(rows) to return a list-of-dicts result. "
        "Only numpy, pandas, scipy, duckdb and pyarrow are available; no network."
    ),
    order_by=["_row"],
    sources=["derived"],
)
def python_sandbox(a: SandboxArgs, ctx) -> ToolResult:
    # Resolve the handle rows
    rows = ctx.result_lookup(a.query_id)
    if rows is None:
        from agentic_rca.tools._common import ToolInputError  # noqa: PLC0415

        raise ToolInputError(
            f"No result handle found for query_id={a.query_id!r}. "
            "Pass a query_id returned by a previous tool call."
        )

    tmpdir = Path(tempfile.mkdtemp(prefix="sandbox_"))
    try:
        in_dir = tmpdir / "in"
        out_dir = tmpdir / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        # Materialise the handle rows as parquet via duckdb
        import duckdb  # noqa: PLC0415

        if rows:
            con = duckdb.connect()
            # duckdb.execute("CREATE TABLE _handle AS SELECT * FROM ?", [rows]) failed when rows is a list of dicts directly in this syntax, but DuckDB can read list of dicts? Wait, duckdb can query a list of dicts via pandas or arrow, or duckdb.sql.
            # Actually, the simplest is inserting if it's a pandas df, but it's list of dicts.
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Since duckdb might not auto-infer list of dicts via ?, use pyarrow.
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, str(in_dir / "handle.parquet"))
            con.close()

        # Write the script: preamble + user code
        script = in_dir / "script.py"
        script.write_text(_PREAMBLE + "\n" + a.code)

        # Run
        rc, stdout, stderr = _run_sandbox(in_dir, out_dir)

        if rc != 0:
            notes = {
                "sandbox_exit_code": rc,
                "stdout": stdout,
                "stderr": stderr[:1000],
                "description": a.description,
            }
            return ToolResult(
                [],
                [Scope(source="derived", verified=True, row_count=0)],
                notes=notes,
            )

        # Read output
        result_file = out_dir / "result.json"
        if not result_file.exists():
            notes = {
                "sandbox_exit_code": rc,
                "stdout": stdout,
                "stderr": "No result.json written — call save_result(rows) in your code.",
                "description": a.description,
            }
            return ToolResult(
                [],
                [Scope(source="derived", verified=True, row_count=0)],
                notes=notes,
            )

        result_data = json.loads(result_file.read_text())
        out_rows = result_data.get("rows", [])

        return ToolResult(
            out_rows,
            [Scope(source="derived", verified=True, row_count=len(out_rows))],
            notes={
                "description": a.description,
                "input_query_id": a.query_id,
                "stdout": stdout,
            },
            input_query_ids=[a.query_id],
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
