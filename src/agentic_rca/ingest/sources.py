"""Declarative registry of the six incident_data source files.

SchemaOfCSVs.MD is the ground truth this registry encodes. It is the single
place that knows each file's window label, timestamp unit, and grain, so
that fact is not re-derived (or mis-derived) at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TimestampUnit = Literal["s", "ms"]
WindowLabel = Literal["baseline", "incident"]


@dataclass(frozen=True)
class SourceFile:
    key: str
    filename: str
    window_label: WindowLabel
    timestamp_unit: TimestampUnit
    grain: str
    identity_key: tuple[str, ...]


SOURCES: tuple[SourceFile, ...] = (
    SourceFile(
        key="baseline_app_metrics",
        filename="baseline_app_metrics.csv",
        window_label="baseline",
        timestamp_unit="s",
        grain="(timestamp, tc)",
        identity_key=("tc",),
    ),
    SourceFile(
        key="cluster_app_metrics",
        filename="cluster_app_metrics.csv",
        window_label="incident",
        timestamp_unit="s",
        grain="(timestamp, tc)",
        identity_key=("tc",),
    ),
    SourceFile(
        key="baseline_container_metrics",
        filename="baseline_container_metrics.csv",
        window_label="baseline",
        timestamp_unit="s",
        grain="(timestamp, cmdb_id, kpi_name)",
        identity_key=("cmdb_id", "kpi_name"),
    ),
    SourceFile(
        key="container_metrics",
        filename="container_metrics.csv",
        window_label="incident",
        timestamp_unit="s",
        grain="(timestamp, cmdb_id, kpi_name)",
        identity_key=("cmdb_id", "kpi_name"),
    ),
    SourceFile(
        key="incident_traces",
        filename="incident_traces.csv",
        window_label="incident",
        timestamp_unit="ms",
        grain="one row per span",
        identity_key=("span_id",),
    ),
    SourceFile(
        key="cluster_incident_logs",
        filename="cluster_incident_logs.csv",
        window_label="incident",
        timestamp_unit="s",
        grain="one row per log line",
        identity_key=("log_id",),
    ),
)

SOURCES_BY_KEY: dict[str, SourceFile] = {s.key: s for s in SOURCES}


def resolve_path(source: SourceFile, data_dir: Path) -> Path:
    return data_dir / source.filename
