"""describe_dataset -- the Discovery tool from Target_Design.MD's tool
inventory: "Sources, schemas, time coverage, gaps, cardinalities, join-key
availability." No inputs. The first real consumer of the common envelope
(tools/envelope.py), exercising it end-to-end rather than only in unit tests.
"""

from __future__ import annotations

from agentic_rca.ingest.pipeline import IngestResult
from agentic_rca.ingest.profile import table_row_counts, time_coverage
from agentic_rca.tools.envelope import Provenance, ToolEnvelope, make_envelope


def describe_dataset(result: IngestResult) -> ToolEnvelope:
    """One row per source table: its row count and, where applicable, its
    per-window time coverage. summary_stats carries the full data-sufficiency
    declaration -- joins, baseline availability, host coverage per source --
    so the model can decide what it can and cannot ask for next."""
    counts = table_row_counts(result.con)
    coverage = time_coverage(result.con)

    rows = [
        {
            "table": table,
            "row_count": row_count,
            "time_coverage": coverage.get(table),
        }
        for table, row_count in counts.items()
    ]

    return make_envelope(
        rows=rows,
        provenance=Provenance(source="stage0_ingest", window="baseline+incident"),
        coverage={"tables_profiled": list(counts.keys())},
        summary_stats=result.declaration,
        preview_limit=len(rows),  # small, fixed set of tables -- no truncation needed
    )
