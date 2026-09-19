"""Orchestrates Stage 0: build_index() runs every ingest step in dependency
order and returns a single result the lead investigator's first tool call
(describe_dataset) reads from. Nothing here scores, ranks, or interprets;
it only sequences already-tested steps (Target_Design.MD Stage 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from agentic_rca.ingest.declaration import build_declaration
from agentic_rca.ingest.load import load_all
from agentic_rca.ingest.logparse import assign_templates, load_access_log_tables
from agentic_rca.ingest.profile import populate_series_cadence
from agentic_rca.ingest.skew import estimate_clock_offsets
from agentic_rca.ingest.store import connect, reset
from agentic_rca.ingest.system_facts import load_system_facts
from agentic_rca.ingest.topology import assign_span_roots, build_topology_edges


@dataclass(frozen=True)
class IngestResult:
    con: duckdb.DuckDBPyConnection
    row_counts: dict[str, int]
    template_count: int
    root_counts: dict[str, int]
    edge_count: int
    offset_pair_count: int
    cadence_row_count: int
    declaration: dict
    system_facts: list[dict]


def build_index(data_dir: Path, db_path: Path) -> IngestResult:
    """Run every Stage 0 step in dependency order against data_dir, persisting
    to db_path. Each step was independently tested; this function's own job
    is only sequencing, so it is not itself the place to add new logic.

    Idempotent by construction: connect() opens the store with CREATE TABLE
    IF NOT EXISTS against a persisted file (store.py's own docstring justifies
    persistence as "re-runs are cheap"), so a second call against the same
    db_path without a reset would append on top of the first run's rows
    rather than replace them -- caught by rca-critic running this function
    twice and diffing table counts (row counts doubled, e.g. spans
    324,321 -> 648,642), which no test caught because every test fixture
    used a fresh tmp_path. reset() truncates before loading so re-running
    against an existing store always reflects exactly one ingest, not N.
    """
    con = connect(db_path)
    reset(con)

    row_counts = load_all(con, data_dir)
    load_access_log_tables(con)
    template_count = assign_templates(con)
    root_counts = assign_span_roots(con)
    edge_count = build_topology_edges(con)
    offset_pair_count = estimate_clock_offsets(con)
    cadence_row_count = populate_series_cadence(con)
    declaration = build_declaration(con)
    system_facts = load_system_facts(data_dir)

    return IngestResult(
        con=con,
        row_counts=row_counts,
        template_count=template_count,
        root_counts=root_counts,
        edge_count=edge_count,
        offset_pair_count=offset_pair_count,
        cadence_row_count=cadence_row_count,
        declaration=declaration,
        system_facts=system_facts,
    )
