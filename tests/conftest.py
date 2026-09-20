"""Shared fixtures for B1+ tests: one real-data store per test session
(~9 s), opened read-only by every tool, plus a fresh run ledger per test.
The ingest tests keep their own per-test stores (tests/ingest/conftest.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "incident_data"

INCIDENT = (1614852000.0, 1614853800.0)  # 10:00-10:30 UTC
BASELINE = (1614848400.0, 1614850200.0)  # 09:00-09:30 UTC


@pytest.fixture(scope="session")
def store_path(tmp_path_factory) -> Path:
    if not (DATA_DIR / "incident_traces.csv").exists():
        pytest.skip(f"incident_data/ not present at {DATA_DIR}")
    from agentic_rca.ingest.pipeline import build_index

    path = tmp_path_factory.mktemp("store") / "store.duckdb"
    result = build_index(DATA_DIR, path)
    result.con.close()
    return path


@pytest.fixture
def runtime(tmp_path, store_path):
    from agentic_rca.ledger import Ledger
    from agentic_rca.tools.registry import ToolRuntime

    ledger = Ledger.create(tmp_path / "runs", store_path=str(store_path))
    rt = ToolRuntime(store_path, ledger)
    yield rt
    rt.close()
    ledger.close()
