"""The tool registry: the only way a data tool executes (contracts-b1 §7-§12).

Because every call goes through `ToolRuntime.call`, a registered tool gets
coverage, the executed-query log, `_row` ordinals, a labelled stratified
preview, standard summary stats, `empty_because` and `unobserved_entities`
without implementing any of them -- and can't skip them. An unregistered
function can't be dispatched at all: the loop's action table is built from
`TOOLS`.

Tools receive a read-only store connection with external access disabled
and configuration locked, so no SQL can read or write files outside the
store (X7), and they never see the run DB.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, ValidationError

from agentic_rca.ledger import Ledger
from agentic_rca.ledger.rundb import now_iso
from agentic_rca.tools._common import (
    COLUMN_KIND,
    EMPTY_PRECEDENCE,
    TIME_COLUMNS,
    Scope,
    StoreIndex,
    TooLarge,
    ToolInputError,
    classify_empty,
    iso,
    nearest,
    sample_preview,
)
from agentic_rca.tools.envelope import Provenance, ResultHandle, ToolEnvelope, ToolError

PREVIEW_MODES = ("sample", "head", "tail", "by_entity")
PREVIEW_N = 20


@dataclass
class ToolResult:
    rows: list[dict]
    scopes: list[Scope]
    sql_texts: list[str] = field(default_factory=list)
    order_by: list[str] | None = None  # overrides the spec's (e.g. sql)
    order_total: bool | None = None
    presorted: bool = False  # rows already in their final order (sql)
    extra_stats: dict[str, Any] = field(default_factory=dict)
    input_query_ids: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    fn: Callable[[BaseModel, ToolContext], ToolResult]
    input_model: type[BaseModel]
    description: str
    order_by: list[str]
    sources: list[str]
    preview_strata: list[str] | None = None
    attempt_scopes: Callable[[BaseModel], list[Scope]] | None = None


TOOLS: dict[str, ToolSpec] = {}


def tool(
    name: str,
    input_model: type[BaseModel],
    *,
    description: str,
    order_by: list[str] | None = None,
    sources: list[str] | None = None,
    preview_strata: list[str] | None = None,
    attempt_scopes: Callable[[BaseModel], list[Scope]] | None = None,
):
    """Register a data tool. `order_by` is the neutral total order its rows
    are sorted into (time, then entity, then a unique tiebreaker)."""

    def deco(fn):
        TOOLS[name] = ToolSpec(
            name,
            fn,
            input_model,
            description,
            list(order_by or []),
            list(sources or []),
            preview_strata,
            attempt_scopes,
        )
        return fn

    return deco


def tool_schemas(names: list[str] | None = None) -> list[dict]:
    """JSON-schema tool definitions for the model. `preview_mode` is added
    to every tool: it changes what's shown, never what's computed."""
    out = []
    for n in sorted(names or TOOLS):
        spec = TOOLS[n]
        schema = spec.input_model.model_json_schema()
        schema.setdefault("properties", {})["preview_mode"] = {
            "type": "string",
            "enum": list(PREVIEW_MODES),
            "default": "sample",
            "description": "sample (default): stratified across time and entities; head/tail: "
            "first/last rows of the base order; by_entity: earliest rows of each entity",
        }
        out.append({"name": n, "description": spec.description, "parameters": schema})
    return out


# ---- canonical form, digest, stats ---------------------------------------------------


def canonical_args(args: BaseModel) -> str:
    return json.dumps(args.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _canon_value(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return format(v, ".12g")
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)


def result_digest(rows: list[dict]) -> str:
    """contracts-b1 §9: sha256 over sorted canonical rows; order-independent,
    and informational only (the verdict compares values)."""
    lines = sorted(
        json.dumps({k: _canon_value(v) for k, v in r.items() if k != "_row"}, sort_keys=True)
        for r in rows
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def summary_stats(rows: list[dict]) -> dict:
    """Standard, registry-computed, descriptive only: no top-K, no 'most
    frequent' (those are rankings)."""
    stats: dict[str, Any] = {"row_count": len(rows), "columns": {}}
    if not rows:
        return stats
    cols = [c for c in rows[0] if c != "_row"]
    for c in cols:
        vals = [r.get(c) for r in rows]
        nn = [v for v in vals if v is not None]
        s: dict[str, Any] = {"non_null": len(nn)}
        nums = [
            v
            for v in nn
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and not (isinstance(v, float) and math.isnan(v))
        ]
        if nums and len(nums) == len(nn):
            s["min"], s["max"] = min(nums), max(nums)
            s["mean"] = sum(nums) / len(nums)
            if c in TIME_COLUMNS and c != "timestamp_ms":
                s["min_iso"], s["max_iso"] = iso(float(s["min"])), iso(float(s["max"]))
        if c in COLUMN_KIND:
            s["distinct"] = len({str(v) for v in nn})
        stats["columns"][c] = s
    return stats


def _sort_key(order_by: list[str]):
    def k(r: dict):
        out = []
        for c in order_by:
            v = r.get(c)
            if v is None:
                out.append((1, 0, ""))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((0, 0, v))
            else:
                out.append((0, 1, str(v)))
        return tuple(out)

    return k


def _ranges(rows: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for r in sorted(set(rows)):
        if out and out[-1][1] == r:
            out[-1][1] = r + 1
        else:
            out.append([r, r + 1])
    return out


# ---- runtime -------------------------------------------------------------------------


def open_store(store_path: Path | str) -> duckdb.DuckDBPyConnection:
    """Read-only store connection with external access off and config locked
    (X7; verified on duckdb 1.5.4: read_csv, COPY TO and re-enabling all fail)."""
    con = duckdb.connect(str(store_path), read_only=True)
    try:
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
    except duckdb.InvalidInputException:
        # Same-process connections share one database instance: a second open
        # finds the config already locked. Accept that only if it is locked
        # *in the safe state*; otherwise fail closed.
        off = con.execute("SELECT current_setting('enable_external_access')").fetchone()[0]
        if off not in (False, "false"):
            con.close()
            raise
    return con


class ToolContext:
    def __init__(self, runtime: ToolRuntime, overrides: dict[str, list[dict]] | None = None):
        self.runtime = runtime
        self.con = runtime.store
        self.index = runtime.index
        self.run_id = runtime.ledger.run_id
        self.row_cap = runtime.row_cap
        self._overrides = overrides or {}

    def q(self, qid: str) -> str:
        return self.runtime.ledger.q(qid)

    def get_handle(self, qid: str) -> list[dict]:
        full = self.q(qid)
        if full in self._overrides:
            return self._overrides[full]
        rows = self.runtime.handles.get(full)
        if rows is None:
            raise ToolInputError(f"unknown or expired result handle {qid!r}")
        return rows

    def fetch(self, sql: str, params: list | None = None, *, hint: str = "") -> list[dict]:
        """Run SQL, materialise at most row_cap rows; over the cap -> TooLarge
        with the true count (never a silent head)."""
        cur = self.con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        batch = cur.fetchmany(self.row_cap + 1)
        if len(batch) > self.row_cap:
            n = self.con.execute(f"SELECT count(*) FROM ({sql})", params or []).fetchone()[0]
            raise TooLarge(n, self.row_cap, hint)
        return [dict(zip(cols, r, strict=True)) for r in batch]


class ToolRuntime:
    def __init__(
        self,
        store_path: Path | str,
        ledger: Ledger,
        *,
        row_cap: int = 100_000,
        timeout_s: float = 30.0,
    ):
        self.store_path = str(store_path)
        self.store = open_store(store_path)
        self.index = StoreIndex(self.store)
        self.ledger = ledger
        self.handles: dict[str, list[dict]] = {}
        self.row_cap = row_cap
        self.timeout_s = timeout_s
        self.declaration: dict | None = None  # set by the orchestrator (describe_dataset)
        self.system_facts: list[dict] = []  # set by the orchestrator (system_facts_lookup)
        ledger.result_lookup = self.lookup

    def close(self) -> None:
        self.store.close()

    def lookup(self, qid: str) -> list[dict] | None:
        return self.handles.get(qid)

    # ---- execution ---------------------------------------------------------------
    def _execute(self, spec: ToolSpec, args: BaseModel, overrides=None):
        ctx = ToolContext(self, overrides)
        timer = threading.Timer(self.timeout_s, self.store.interrupt)
        timer.start()
        try:
            result = spec.fn(args, ctx)
        finally:
            timer.cancel()
        rows = result.rows
        if result.presorted:
            order_by = list(result.order_by or [])
            total = bool(result.order_total) if result.order_total is not None else False
        else:
            order_by = list(result.order_by or spec.order_by)
            if order_by:
                rows.sort(key=_sort_key(order_by))
            key = _sort_key(order_by)
            total = bool(order_by) and all(
                key(rows[i]) != key(rows[i + 1]) for i in range(len(rows) - 1)
            )
            if len(rows) <= 1:
                total = True
        for i, r in enumerate(rows):
            r["_row"] = i
        return result, rows, order_by, total

    def _map_error(self, exc: Exception) -> ToolError:
        if isinstance(exc, ToolInputError):
            return ToolError("invalid_query", exc.diagnostic, exc.suggestions)
        if isinstance(exc, TooLarge):
            return ToolError(
                "too_large",
                f"result has {exc.count:,} rows, over the cap of {exc.cap:,}. {exc.hint}".strip(),
                ["narrow the window or entity filter", "aggregate (resolution / group-by)"],
            )
        if isinstance(exc, duckdb.InterruptException):
            return ToolError(
                "timeout",
                f"query exceeded {self.timeout_s:.0f}s",
                ["narrow the window", "filter on fewer entities"],
            )
        if isinstance(
            exc, (duckdb.BinderException, duckdb.CatalogException, duckdb.ParserException)
        ):
            msg = str(exc).splitlines()[0]
            cands = sorted(
                {c for t in self.index.tables() for c in self.index.columns(t)}
                | set(self.index.tables())
            )
            words = [
                w.strip("\"'")
                for w in msg.replace(":", " ").split()
                if w.isidentifier() or "_" in w
            ]
            sugg = sorted({s for w in words for s in nearest(w, cands, 3)})[:8]
            return ToolError("invalid_query", msg, sugg)
        if isinstance(exc, duckdb.Error):
            return ToolError("invalid_query", str(exc).splitlines()[0])
        return ToolError("transient", f"{type(exc).__name__}: {exc}")

    def call(
        self, name: str, raw_args: dict | None = None, *, actor: str = "lead", step: int = 0
    ) -> ToolEnvelope:
        if name not in TOOLS:
            raise KeyError(f"{name!r} is not a registered tool")
        spec = TOOLS[name]
        raw = dict(raw_args or {})
        pmode = raw.pop("preview_mode", "sample") or "sample"
        started, t0 = now_iso(), time.perf_counter()
        args = None
        try:
            if pmode not in PREVIEW_MODES:
                raise ToolInputError(
                    f"preview_mode must be one of {PREVIEW_MODES}", list(PREVIEW_MODES)
                )
            try:
                args = spec.input_model.model_validate(raw)
            except ValidationError as ve:
                errs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in ve.errors())
                fields = list(spec.input_model.model_fields)
                raise ToolInputError(
                    f"invalid arguments: {errs}", [f"valid fields: {fields}"]
                ) from ve
            result, rows, order_by, total = self._execute(spec, args)
        except Exception as exc:  # noqa: BLE001 - every failure becomes an error envelope
            err = self._map_error(exc)
            canon = (
                canonical_args(args)
                if args is not None
                else json.dumps(raw, sort_keys=True, default=str)
            )
            return self._record_error(spec, canon, err, actor, step, started, t0, args)
        canon = canonical_args(args)
        return self._record_ok(
            spec, canon, result, rows, order_by, total, pmode, actor, step, started, t0
        )

    def _next_qid(self) -> str:
        n = self.ledger.con.execute(
            "SELECT count(*) FROM queries WHERE run_id = ?", [self.ledger.run_id]
        ).fetchone()[0]
        return f"{self.ledger.run_id}/q{n + 1:06d}"

    def _insert_query(self, **kw) -> None:
        cols = list(kw)
        self.ledger.con.execute(
            f"INSERT INTO queries ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [kw[c] for c in cols],
        )

    def _insert_coverage(self, qid: str, actor: str, n: int, **kw) -> str:
        cid = f"{qid}.c{n}"
        kw.update(run_id=self.ledger.run_id, coverage_id=cid, query_id=qid, actor=actor)
        cols = list(kw)
        self.ledger.con.execute(
            f"INSERT INTO coverage ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [kw[c] for c in cols],
        )
        return cid

    def _insert_inspection(self, qid: str, actor: str, via: str, rows: list[int]) -> None:
        if not rows:
            return
        self.ledger.con.execute(
            "INSERT INTO inspections VALUES (?,?,?,?,?,?)",
            [self.ledger.run_id, qid, actor, via, json.dumps(_ranges(rows)), now_iso()],
        )

    def _record_error(
        self, spec, canon, err: ToolError, actor, step, started, t0, args
    ) -> ToolEnvelope:
        con = self.ledger.con
        scopes = []
        if args is not None and spec.attempt_scopes:
            try:
                scopes = spec.attempt_scopes(args)
            except Exception:  # noqa: BLE001
                scopes = []
        if not scopes:
            scopes = [Scope(source=s) for s in (spec.sources or ["unknown"])]
        con.execute("BEGIN TRANSACTION")
        try:
            qid = self._next_qid()
            self._insert_query(
                run_id=self.ledger.run_id,
                query_id=qid,
                seq=self.ledger.last_seq(),
                actor=actor,
                step=step,
                tool=spec.name,
                args_json=canon,
                args_sha256=hashlib.sha256(canon.encode()).hexdigest(),
                input_query_ids="[]",
                sql_texts="[]",
                status="error",
                error_type=err.type,
                diagnostic=err.diagnostic,
                retried=False,
                row_count=0,
                started_at=started,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
            for i, sc in enumerate(scopes, 1):
                self._insert_coverage(
                    qid,
                    actor,
                    i,
                    kind="attempt",
                    attempt_reason="error",
                    sources=json.dumps([sc.source]),
                    requested=json.dumps(sc.requested),
                    scope_verified=sc.verified,
                    requested_window=json.dumps({"start_s": sc.window[0], "end_s": sc.window[1]})
                    if sc.window
                    else None,
                    row_count=0,
                )
            self.ledger.apply(
                "failure_record",
                {
                    "query_id": qid,
                    "tool": spec.name,
                    "error_type": err.type,
                    "diagnostic": err.diagnostic,
                    "retried": False,
                },
                actor="scaffold",
                step=step,
                in_transaction=True,
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        return ToolEnvelope(
            query_id=self.ledger.short(qid),
            result_handle=None,
            row_count=0,
            preview=[],
            summary_stats={},
            truncated=False,
            coverage={"kind": "attempt"},
            provenance=Provenance(source=",".join(spec.sources) or spec.name),
            error=err,
        )

    def _record_ok(
        self,
        spec,
        canon,
        result: ToolResult,
        rows,
        order_by,
        total,
        pmode,
        actor,
        step,
        started,
        t0,
    ):
        con = self.ledger.con
        n = len(rows)
        scopes = result.scopes or [
            Scope(source=s, verified=False) for s in (spec.sources or ["unknown"])
        ]
        if len(scopes) == 1 and scopes[0].row_count is None:
            scopes[0].row_count = n
        unobserved: dict[str, list[str]] = {}
        per_scope = []
        for sc in scopes:
            un = self.index.unobserved(sc) if sc.verified or sc.requested else {}
            for k, v in un.items():
                unobserved[k] = sorted(set(unobserved.get(k, [])) | set(v))
            reason = classify_empty(self.index, sc) if (sc.row_count or 0) == 0 else None
            per_scope.append((sc, un, reason))
        empty = None
        if n == 0:
            reasons = [r for _, _, r in per_scope if r]
            empty = min(reasons, key=EMPTY_PRECEDENCE.index) if reasons else "scope_unverified"
        scope_verified = all(sc.verified for sc in scopes)
        preview, meta = sample_preview(
            rows, mode=pmode, n=PREVIEW_N, order_by=order_by, strata=spec.preview_strata
        )
        stats = summary_stats(rows)
        stats.update(result.extra_stats)
        time_col = meta.get("time_col")
        ent_cols = [c for c in (rows[0] if rows else {}) if c in COLUMN_KIND]
        con.execute("BEGIN TRANSACTION")
        try:
            qid = self._next_qid()
            self._insert_query(
                run_id=self.ledger.run_id,
                query_id=qid,
                seq=self.ledger.last_seq(),
                actor=actor,
                step=step,
                tool=spec.name,
                args_json=canon,
                args_sha256=hashlib.sha256(canon.encode()).hexdigest(),
                input_query_ids=json.dumps([self.ledger.q(x) for x in result.input_query_ids]),
                sql_texts=json.dumps(result.sql_texts),
                status="ok",
                retried=False,
                row_count=n,
                empty_because=empty,
                unobserved_entities=json.dumps(unobserved) if unobserved else None,
                scope_verified=scope_verified,
                order_by=json.dumps(order_by),
                order_total=total,
                result_digest=result_digest(rows),
                preview_mode=pmode,
                preview_label=meta["label"],
                preview_json=json.dumps(preview, default=str),
                summary_stats_json=json.dumps(stats, default=str),
                started_at=started,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
            cov_ids = []
            for i, (sc, un, reason) in enumerate(per_scope, 1):
                srows = (
                    [r for r in rows if r.get("source") == sc.source]
                    if len(scopes) > 1 and rows and "source" in rows[0]
                    else rows
                )
                returned: dict[str, list[str]] = {}
                for c in ent_cols:
                    returned.setdefault(COLUMN_KIND[c], [])
                    returned[COLUMN_KIND[c]] = sorted(
                        set(returned[COLUMN_KIND[c]])
                        | {str(r[c]) for r in srows if r.get(c) is not None}
                    )
                vis = sc.visibility or (
                    "rows" if ent_cols else ("aggregated" if srows else "unknown")
                )
                rw = None
                if time_col and srows:
                    ts = [r[time_col] for r in srows if isinstance(r.get(time_col), (int, float))]
                    if ts:
                        div = 1000.0 if time_col == "timestamp_ms" else 1.0
                        rw = {"min_s": min(ts) / div, "max_s": max(ts) / div}
                queried = (sc.row_count or 0) > 0 or reason == "no_matches"
                cov_ids.append(
                    self._insert_coverage(
                        qid,
                        actor,
                        i,
                        kind="queried" if queried else "attempt",
                        attempt_reason=None if queried else reason,
                        unobserved_entities=json.dumps(un) if un else None,
                        scope_verified=sc.verified,
                        sources=json.dumps([sc.source]),
                        requested=json.dumps(sc.requested),
                        requested_window=json.dumps(
                            {"start_s": sc.window[0], "end_s": sc.window[1]}
                        )
                        if sc.window
                        else None,
                        resolution=sc.resolution_s,
                        returned=json.dumps(returned),
                        entity_visibility=vis,
                        returned_window=json.dumps(rw) if rw else None,
                        row_count=sc.row_count or 0,
                    )
                )
            self._insert_inspection(qid, actor, "preview", [r["_row"] for r in preview])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        self.handles[qid] = rows
        notes = dict(result.notes)
        if len(scopes) > 1:
            notes["per_source"] = [
                {"source": sc.source, "row_count": sc.row_count or 0, "empty_because": reason}
                for sc, _, reason in per_scope
            ]
        if not scope_verified:
            notes["scope"] = (
                "this query's entity/time predicates couldn't be fully interpreted, so an empty "
                "or zero result here can't be certified as absence"
            )
        short = self.ledger.short(qid)
        return ToolEnvelope(
            query_id=short,
            result_handle=ResultHandle(short, n) if n else None,
            row_count=n,
            preview=preview,
            summary_stats=stats,
            truncated=n > len(preview),
            coverage={"coverage_ids": [self.ledger.short(c) for c in cov_ids]},
            provenance=Provenance(
                source=",".join(sc.source for sc in scopes),
                window=", ".join(
                    f"{iso(sc.window[0])}..{iso(sc.window[1])}" for sc in scopes if sc.window
                )
                or None,
            ),
            empty_because=empty,
            order={"by": order_by, "total": total},
            preview_label=meta["label"],
            preview_meta=meta,
            unobserved_entities=unobserved or None,
            scope_verified=scope_verified,
            notes=notes,
        )

    # ---- paging / re-preview (no new query id) ---------------------------------------
    def inspect(
        self,
        handle: str,
        *,
        offset: int = 0,
        limit: int = 20,
        sort: str | None = None,
        columns: list[str] | None = None,
        preview_mode: str | None = None,
        actor: str = "lead",
        via: str = "inspect_result",
    ) -> ToolEnvelope:
        def err(msg, sugg=None):
            return ToolEnvelope(
                query_id=handle,
                result_handle=None,
                row_count=0,
                preview=[],
                summary_stats={},
                truncated=False,
                coverage={},
                provenance=Provenance(source="inspect_result"),
                error=ToolError("invalid_query", msg, sugg or []),
            )

        try:
            qid = self.ledger.q(handle)
        except Exception as exc:  # noqa: BLE001
            return err(str(exc))
        rows = self.handles.get(qid)
        qrow = self.ledger.query_row(qid)
        if rows is None or qrow is None:
            return err(f"unknown or expired result handle {handle!r}")
        n = len(rows)
        order_by = json.loads(qrow["order_by"] or "[]")
        if preview_mode:
            if preview_mode not in PREVIEW_MODES:
                return err(f"preview_mode must be one of {PREVIEW_MODES}")
            spec = TOOLS.get(qrow["tool"])
            page, meta = sample_preview(
                rows,
                mode=preview_mode,
                n=min(max(limit, 1), 200),
                order_by=order_by,
                strata=spec.preview_strata if spec else None,
            )
            label = meta["label"]
            self._insert_inspection(qid, actor, "preview", [r["_row"] for r in page])
        else:
            if offset < 0 or limit < 1:
                return err(f"offset must be >= 0 and limit >= 1 (got {offset}, {limit})")
            if n and offset >= n:
                return err(
                    f"offset {offset} is past the end of the result ({n} rows)",
                    [f"offset <= {n - 1}"],
                )
            view = rows
            if sort:
                col = sort.lstrip("-")
                if rows and col not in rows[0]:
                    return err(
                        f"cannot sort by unknown column {col!r}", nearest(col, list(rows[0]))
                    )
                view = sorted(rows, key=_sort_key([col]), reverse=sort.startswith("-"))
            page = view[offset : offset + max(1, min(limit, 500))]
            if columns:
                missing = [c for c in columns if rows and c not in rows[0]]
                if missing:
                    return err(f"unknown columns {missing}", sorted(rows[0]) if rows else [])
                page = [{**{c: r.get(c) for c in columns}, "_row": r["_row"]} for r in page]
            label = f"rows {offset}-{offset + len(page) - 1} of {n:,}" + (
                f" sorted by {sort}" if sort else ""
            )
            meta = {"mode": "page", "offset": offset, "limit": limit, "sort": sort}
            self._insert_inspection(qid, actor, via, [r["_row"] for r in page])
        return ToolEnvelope(
            query_id=self.ledger.short(qid),
            result_handle=ResultHandle(self.ledger.short(qid), n),
            row_count=n,
            preview=page,
            summary_stats={},
            truncated=len(page) < n,
            coverage={},
            provenance=Provenance(source=qrow["tool"]),
            empty_because=qrow["empty_because"],
            order={"by": order_by, "total": bool(qrow["order_total"])},
            preview_label=label,
            preview_meta=meta,
        )

    # ---- verifier entry point -------------------------------------------------------
    def reexecute(self, query_row: dict, _depth: int = 0) -> dict:
        """Re-run a logged query from its canonical args through the same code
        path, recursing through input handles, without logging anything."""
        if _depth > 8:
            raise RecursionError("input_query_ids chain too deep")
        spec = TOOLS[query_row["tool"]]
        args = spec.input_model.model_validate(json.loads(query_row["args_json"]))
        overrides = {}
        for iq in json.loads(query_row["input_query_ids"] or "[]"):
            parent = self.ledger.query_row(iq)
            overrides[iq] = self.reexecute(parent, _depth + 1)["rows"]
        _result, rows, order_by, total = self._execute(spec, args, overrides)
        return {
            "rows": rows,
            "order_total": total,
            "order_by": order_by,
            "digest": result_digest(rows),
        }

    # ---- coverage_view (raw, unranked) -------------------------------------------------
    def coverage_view(self, source: str | None = None, entity: str | None = None) -> dict:
        con = self.ledger.con
        cur = con.execute(
            "SELECT c.*, q.tool FROM coverage c JOIN queries q USING (query_id) ORDER BY c.coverage_id"
        )
        cols = [d[0] for d in cur.description]
        cov_rows = cur.fetchall()  # fetch before the next execute on this connection
        insp: dict[str, dict[str, list]] = {}
        for qid, via, ranges in con.execute(
            "SELECT query_id, via, row_ranges FROM inspections"
        ).fetchall():
            key = "previewed" if via == "preview" else "inspected"
            insp.setdefault(qid, {"previewed": [], "inspected": []})[key] += json.loads(ranges)
        queried, attempts, unobserved = [], [], []
        for row in cov_rows:
            c = dict(zip(cols, row, strict=True))
            srcs = json.loads(c["sources"])
            req = json.loads(c["requested"] or "{}")
            ret = json.loads(c["returned"] or "{}")
            un = json.loads(c["unobserved_entities"] or "null") or {}
            if source and not any(source in s for s in srcs):
                continue
            if entity:
                vals = {
                    str(v)
                    for d in (req, ret, un)
                    for vs in d.values()
                    if isinstance(vs, list)
                    for v in vs
                }
                if entity not in vals:
                    continue
            item = {
                "query_id": self.ledger.short(c["query_id"]),
                "tool": c["tool"],
                "sources": srcs,
                "requested": req,
                "window": json.loads(c["requested_window"] or "null"),
                "row_count": c["row_count"],
            }
            if c["kind"] == "queried":
                item.update(
                    returned=ret,
                    entity_visibility=c["entity_visibility"],
                    **insp.get(c["query_id"], {"previewed": [], "inspected": []}),
                )
                if not c["scope_verified"]:
                    item["scope_verified"] = False
                queried.append(item)
            else:
                item["attempt_reason"] = c["attempt_reason"]
                (
                    unobserved if c["attempt_reason"] == "entity_absent_from_source" else attempts
                ).append(item)
            if un and c["kind"] == "queried":
                unobserved.append(
                    {"query_id": item["query_id"], "sources": srcs, "unobserved_entities": un}
                )

        def neutral(i):
            first = next(
                (str(v[0]) for v in i.get("requested", {}).values() if isinstance(v, list) and v),
                "",
            )
            w = i.get("window") or {}
            return (",".join(i["sources"]), first, w.get("start_s") or 0, i["query_id"])

        return {
            "queried": sorted(queried, key=neutral),
            "attempts": sorted(attempts, key=neutral),
            "unobserved_entities": sorted(
                unobserved, key=lambda i: (",".join(i["sources"]), i["query_id"])
            ),
            "note": "raw coverage facts in neutral order; nothing here is ranked",
        }


def envelope_for_model(env: ToolEnvelope) -> dict:
    """The dict a model sees (the loop wraps it with provenance framing)."""
    out: dict[str, Any] = {"query_id": env.query_id, "row_count": env.row_count}
    if env.error:
        out["error"] = {
            "type": env.error.type,
            "diagnostic": env.error.diagnostic,
            "suggestions": env.error.suggestions,
        }
        return out
    if env.result_handle:
        out["handle"] = env.result_handle.id
    if env.preview_label:
        out["preview_label"] = env.preview_label
    out["preview"] = env.preview
    if env.empty_because:
        out["empty_because"] = env.empty_because
    if env.unobserved_entities:
        out["unobserved_entities"] = env.unobserved_entities
    if not env.scope_verified:
        out["scope_verified"] = False
    if env.order:
        out["order"] = env.order
    if env.summary_stats:
        out["summary_stats"] = env.summary_stats
    if env.notes:
        out["notes"] = env.notes
    return out
