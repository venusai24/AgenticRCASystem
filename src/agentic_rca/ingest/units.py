"""The ONLY place millisecond-vs-second timestamp units are decided.

SchemaOfCSVs.MD: incident_traces.csv carries 13-digit millisecond epochs;
every other source file carries 10-digit second epochs. Every other module
imports these two functions rather than doing the *1000/1000 conversion
inline, so the unit boundary is enforced in one place.
"""

from __future__ import annotations


def to_seconds(timestamp_ms: int) -> float:
    """Convert a millisecond epoch (traces) to a second epoch (everything else)."""
    return timestamp_ms / 1000.0


def to_ms(timestamp_s: float) -> int:
    """Convert a second epoch to a millisecond epoch, for round-tripping."""
    return round(timestamp_s * 1000)
