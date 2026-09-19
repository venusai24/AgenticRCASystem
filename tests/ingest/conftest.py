from pathlib import Path

import pytest

from agentic_rca.ingest.store import connect

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "incident_data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA_DIR / "incident_traces.csv").exists():
        pytest.skip(f"incident_data/ not present at {DATA_DIR} in this checkout")
    return DATA_DIR


@pytest.fixture
def db_con(tmp_path):
    con = connect(tmp_path / "test_store.duckdb")
    yield con
    con.close()
