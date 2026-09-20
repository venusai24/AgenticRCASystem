"""Shared pieces every data tool uses, so the tools can't disagree:
the entity map (contracts-b1 §11.3), per-source time columns and units,
the Window input type, Scope, the one `empty_because` classifier (§11.6),
`unobserved_entities`, and the stratified preview sampler (§8).
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import duckdb
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---- entity map (contracts-b1 §11.3): the single place ----------------------------

ENTITY_KINDS: dict[str, tuple[str, ...]] = {
    "host": ("cmdb_id", "caller_host", "callee_host", "host_a", "host_b"),
    "service": ("tc",),
    "metric": ("kpi_name", "metric_name"),
    "template": ("template_id",),
    "log_source": ("log_name",),
}
COLUMN_KIND: dict[str, str] = {c: k for k, cols in ENTITY_KINDS.items() for c in cols}

# Time columns a result may carry, in preference order for the time stratum.
TIME_COLUMNS = ("timestamp_s", "ts_s", "bucket_start_s", "window_start_s", "timestamp_ms")

# Raw store sources: their entity columns per kind, and timestamp unit (granularity, s).
SOURCES: dict[str, dict[str, Any]] = {
    "app_metrics": {"entities": {"service": "tc", "metric": "metric_name"}, "unit_s": 1.0},
    "container_metrics": {"entities": {"host": "cmdb_id", "metric": "kpi_name"}, "unit_s": 1.0},
    "spans": {"entities": {"host": "cmdb_id"}, "unit_s": 0.001},
    "logs": {
        "entities": {"host": "cmdb_id", "template": "template_id", "log_source": "log_name"},
        "unit_s": 1.0,
    },
}


class ToolInputError(Exception):
    """An invalid identifier or argument -> error.type invalid_query, with
    near-miss suggestions. Never an empty result (user patch 31)."""

    def __init__(self, diagnostic: str, suggestions: list[str] | None = None):
        super().__init__(diagnostic)
        self.diagnostic = diagnostic
        self.suggestions = suggestions or []


class TooLarge(Exception):
    def __init__(self, count: int, cap: int, hint: str = ""):
        super().__init__(f"{count} rows exceeds the row cap of {cap}")
        self.count, self.cap, self.hint = count, cap, hint


def nearest(value: str, candidates: list[str], n: int = 5) -> list[str]:
    return difflib.get_close_matches(value, candidates, n=n, cutoff=0.5)


# ---- inputs -----------------------------------------------------------------------


def _to_epoch_s(v: Any) -> float:
    if isinstance(v, (int, float)):
        x = float(v)
        return x / 1000.0 if x > 1e11 else x  # accept ms epochs, normalise to s
    s = str(v).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


class Window(BaseModel):
    """A half-open time window [start, end). Epoch seconds, epoch
    milliseconds, or ISO-8601 (UTC if no zone) are all accepted and
    normalised to epoch seconds."""

    model_config = ConfigDict(extra="forbid")
    start: float = Field(description="window start: epoch seconds or ISO-8601 (UTC)")
    end: float = Field(description="window end (exclusive): epoch seconds or ISO-8601 (UTC)")

    @field_validator("start", "end", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> float:
        return _to_epoch_s(v)

    @model_validator(mode="after")
    def _ordered(self) -> Window:
        if not self.end > self.start:
            raise ValueError("window end must be after start")
        return self


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def iso(ts: float | None) -> str | None:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- scope ------------------------------------------------------------------------


@dataclass
class Scope:
    """What one source was asked for. `requested` maps entity kind to a list
    of values, or "*" when that kind wasn't filtered. `verified=False` means
    the scope couldn't be established (free-form sql)."""

    source: str
    requested: dict[str, Any] = field(default_factory=dict)
    window: tuple[float, float] | None = None
    join_keys: list[tuple[str, str]] | None = None  # [(table, column)] both sides
    verified: bool = True
    row_count: int | None = None  # per-scope rows for multi-source tools
    visibility: str | None = None  # rows | aggregated | unknown; derived if None
    resolution_s: float | None = None


class StoreIndex:
    """Cached distinct entity values: globally per kind, and per source."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self._global: dict[str, set[str]] = {}
        self._per_source: dict[tuple[str, str], set[str]] = {}
        self._tables: dict[str, set[str]] = {}

    def columns(self, table: str) -> set[str]:
        if table not in self._tables:
            rows = self.con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]
            ).fetchall()
            self._tables[table] = {r[0] for r in rows}
        return self._tables[table]

    def tables(self) -> list[str]:
        return [
            r[0]
            for r in self.con.execute(
                "SELECT table_name FROM information_schema.tables ORDER BY table_name"
            ).fetchall()
        ]

    def source_values(self, source: str, kind: str) -> set[str]:
        key = (source, kind)
        if key not in self._per_source:
            cols = [c for c in ENTITY_KINDS[kind] if c in self.columns(source)]
            vals: set[str] = set()
            for c in cols:
                vals |= {
                    str(r[0])
                    for r in self.con.execute(
                        f'SELECT DISTINCT "{c}" FROM "{source}" WHERE "{c}" IS NOT NULL'
                    ).fetchall()
                }
            self._per_source[key] = vals
        return self._per_source[key]

    def global_values(self, kind: str) -> set[str]:
        if kind not in self._global:
            vals: set[str] = set()
            for t in self.tables():
                if any(c in self.columns(t) for c in ENTITY_KINDS[kind]):
                    vals |= self.source_values(t, kind)
            self._global[kind] = vals
        return self._global[kind]

    def check_known(self, kind: str, values: list[str]) -> None:
        """A requested entity found nowhere in the store is invalid_query with
        suggestions, whatever the row count (contracts-b1 §11.6)."""
        known = self.global_values(kind)
        unknown = [v for v in values if str(v) not in known]
        if unknown:
            sugg = sorted({s for v in unknown for s in nearest(str(v), sorted(known))})
            raise ToolInputError(
                f"unknown {kind} value(s) {unknown}: not present anywhere in the dataset",
                sugg,
            )

    def unobserved(self, scope: Scope) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for kind, vals in scope.requested.items():
            if vals == "*" or not vals or kind not in ENTITY_KINDS:
                continue
            present = self.source_values(scope.source, kind)
            missing = sorted(str(v) for v in vals if str(v) not in present)
            if missing:
                out[kind] = missing
        return out

    def rows_in_window(self, source: str, window: tuple[float, float]) -> int:
        if "timestamp_s" not in self.columns(source):
            return -1
        return self.con.execute(
            f'SELECT count(*) FROM "{source}" WHERE timestamp_s >= ? AND timestamp_s < ?',
            list(window),
        ).fetchone()[0]

    def join_key_unpopulated(self, keys: list[tuple[str, str]], window) -> bool:
        for table, col in keys:
            if col not in self.columns(table):
                return True
            where = f'"{col}" IS NOT NULL'
            params: list = []
            if window and "timestamp_s" in self.columns(table):
                where += " AND timestamp_s >= ? AND timestamp_s < ?"
                params = list(window)
            n = self.con.execute(
                f'SELECT count(*) FROM "{table}" WHERE {where}', params
            ).fetchone()[0]
            if n == 0:
                return True
        return False


def classify_empty(index: StoreIndex, scope: Scope) -> str:
    """contracts-b1 §11.6, for a scope that returned zero rows. Precedence:
    entity_absent_from_source > no_data_in_window > join_key_unpopulated >
    scope_unverified > no_matches. Never defaults to no_matches: it's
    returned only when the scope is verified and checks 1-3 all ran."""
    if index.unobserved(scope):
        return "entity_absent_from_source"
    if scope.window is not None and index.rows_in_window(scope.source, scope.window) == 0:
        return "no_data_in_window"
    if scope.join_keys and index.join_key_unpopulated(scope.join_keys, scope.window):
        return "join_key_unpopulated"
    if not scope.verified:
        return "scope_unverified"
    return "no_matches"


EMPTY_PRECEDENCE = [
    "entity_absent_from_source",
    "no_data_in_window",
    "join_key_unpopulated",
    "scope_unverified",
    "no_matches",
]


# ---- preview sampling (contracts-b1 §8) -----------------------------------------------


def _first(order_by: list[str], cols: set[str], pred) -> str | None:
    for c in order_by:
        if c in cols and pred(c):
            return c
    return None


def sample_preview(
    rows: list[dict],
    *,
    mode: str,
    n: int,
    order_by: list[str],
    strata: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """Deterministic preview. Returns (rows, preview_meta). Rows keep `_row`."""
    total = len(rows)
    cols = set(rows[0].keys()) if rows else set()
    time_col = _first(order_by, cols, lambda c: c in TIME_COLUMNS) or next(
        (c for c in TIME_COLUMNS if c in cols), None
    )
    if strata:
        ent_col = next((c for c in strata if c in cols), None)
    else:
        ent_col = _first(order_by, cols, lambda c: c in COLUMN_KIND)
    meta: dict[str, Any] = {
        "mode": mode,
        "row_count": total,
        "time_col": time_col,
        "entity_col": ent_col,
        "entities_total": None,
        "entities_shown": None,
        "time_range": None,
        "n": n,
    }
    if time_col and rows:
        ts = [r[time_col] for r in rows if isinstance(r.get(time_col), (int, float))]
        if ts:
            meta["time_range"] = [min(ts), max(ts)]

    if total <= n:
        chosen = list(range(total))
        meta["mode"] = "all"
    elif mode == "head":
        chosen = list(range(n))
    elif mode == "tail":
        chosen = list(range(total - n, total))
    else:
        groups: dict[Any, list[int]] = {}
        order: list[Any] = []
        if ent_col:
            for i, r in enumerate(rows):
                k = r.get(ent_col)
                if k not in groups:
                    groups[k] = []
                    order.append(k)
                groups[k].append(i)
        else:
            groups = {None: list(range(total))}
            order = [None]
        e = len(order)
        meta["entities_total"] = e if ent_col else None
        if e > n:
            keys_sorted = sorted(order, key=lambda k: (k is None, str(k)))
            picks = [keys_sorted[int(i * e / n)] for i in range(n)]
            slots = {k: 1 for k in picks}
        else:
            base, rem = divmod(n, e)
            slots = {k: base + (1 if i < rem else 0) for i, k in enumerate(order)}
        meta["entities_shown"] = len([k for k, s in slots.items() if s > 0]) if ent_col else None
        chosen = []
        for k, s in slots.items():
            idx = groups[k]
            if s <= 0:
                continue
            if mode == "by_entity":
                chosen += idx[:s]
            else:
                chosen += _spread(rows, idx, s, time_col)
        chosen = sorted(set(chosen))
    meta["n_shown"] = len(chosen)
    meta["label"] = preview_label(meta, order_by)
    return [rows[i] for i in chosen], meta


def _spread(rows: list[dict], idx: list[int], k: int, time_col: str | None) -> list[int]:
    """Pick k rows of idx nearest k evenly spaced points (bucket midpoints)
    across the group's time range; ties to the lower _row."""
    if k >= len(idx):
        return list(idx)
    if time_col is None:
        return [idx[int((j + 0.5) * len(idx) / k)] for j in range(k)]
    ts = [(rows[i].get(time_col), i) for i in idx]
    ts = [(t, i) for t, i in ts if isinstance(t, (int, float))]
    if not ts:
        return [idx[int((j + 0.5) * len(idx) / k)] for j in range(k)]
    lo, hi = min(t for t, _ in ts), max(t for t, _ in ts)
    out = []
    for j in range(k):
        target = lo + (j + 0.5) * (hi - lo) / k
        best = min(ts, key=lambda ti: (abs(ti[0] - target), ti[1]))
        out.append(best[1])
    return out


def preview_label(meta: dict, order_by: list[str]) -> str:
    total = meta["row_count"]
    shown = meta.get("n_shown", 0)
    if meta["mode"] == "all":
        return f"all {total} rows"
    if meta["mode"] in ("head", "tail"):
        which = "first" if meta["mode"] == "head" else "last"
        return f"{which} {shown} of {total:,} rows (ordered by {', '.join(order_by) or 'result order'})"
    parts = [f"{'sample' if meta['mode'] == 'sample' else 'per-entity head'} of {shown}"]
    if meta["entity_col"]:
        et, es = meta["entities_total"], meta["entities_shown"]
        kind = COLUMN_KIND.get(meta["entity_col"], meta["entity_col"])
        across = f"across {es} {kind}s ({meta['entity_col']})"
        if et and es and et > es:
            across += f", {es} of {et} {kind}s shown"
        parts.append(across)
    if meta["time_range"]:
        a, b = meta["time_range"]
        if meta["time_col"] == "timestamp_ms":
            a, b = a / 1000.0, b / 1000.0
        parts.append(f"{iso(a)}–{iso(b)}")
    parts.append(f"of {total:,} rows")
    label = ", ".join(parts[:-1]) + " " + parts[-1]
    if shown < min(total, meta.get("n", 20)) and meta["mode"] == "sample":
        label += " (fewer than requested: rows coincided, not back-filled)"
    return label
