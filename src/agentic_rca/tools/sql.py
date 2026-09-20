"""`sql`: read-only SQL over the indexed store -- "any view the designer
didn't anticipate" (§3 calls it the most important anti-encoding choice).

Read-only-ness is the engine's (read-only connection, external access off,
config locked: X7), not this module's. What this module adds is the scope
extractor (contracts-b1 §11.2): it reads DuckDB's own parse tree, never a
regex, and recognises only
  * a top-level AND of `entity_col = const` / `entity_col IN (consts)`
  * comparisons / BETWEEN of a time column against constants
  * join ON clauses that are column = column equalities
Anything else touching an entity or time column makes the scope unverified,
so an empty or zero result from it can never certify absence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from pydantic import Field

from agentic_rca.tools._common import (
    COLUMN_KIND,
    Scope,
    StoreIndex,
    ToolArgs,
    ToolInputError,
)
from agentic_rca.tools.registry import ToolResult, tool

TIME_COLS = {"timestamp_s": 1.0, "timestamp_ms": 0.001}
_CMP = {
    "COMPARE_GREATERTHAN": ">",
    "COMPARE_GREATERTHANOREQUALTO": ">=",
    "COMPARE_LESSTHAN": "<",
    "COMPARE_LESSTHANOREQUALTO": "<=",
    "COMPARE_EQUAL": "=",
}
_FLIP = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "="}


class SqlArgs(ToolArgs):
    query: str = Field(description="one read-only SELECT over the store's tables")


@dataclass
class _Rel:
    alias: str
    table: str
    requested: dict = field(default_factory=dict)
    lo: float | None = None
    hi: float | None = None
    join_keys: list = field(default_factory=list)
    verified: bool = True


@dataclass
class Extracted:
    scopes: list[Scope]
    verified: bool
    has_order: bool
    has_limit: bool


def _value(v: dict):
    """A serialised constant's value. DECIMAL literals (e.g. 1614852000.5) are
    serialised as scaled integers: undo the scale, or a time window is off by
    10**scale and an empty result gets the wrong empty_because."""
    t = v.get("type") or {}
    if t.get("id") == "DECIMAL" and isinstance(v.get("value"), int):
        return v["value"] / 10 ** (t.get("type_info") or {}).get("scale", 0)
    return v["value"]


def _const(n: dict):
    if n.get("class") == "CONSTANT" and not n["value"].get("is_null"):
        return _value(n["value"])
    if n.get("class") == "CAST" and n.get("child", {}).get("class") == "CONSTANT":
        return _value(n["child"]["value"])
    return None


def _colref(n: dict) -> list[str] | None:
    return n["column_names"] if n.get("class") == "COLUMN_REF" else None


def _all_colrefs(n) -> list[list[str]]:
    out = []
    if isinstance(n, dict):
        if n.get("class") == "COLUMN_REF":
            out.append(n["column_names"])
        for v in n.values():
            out += _all_colrefs(v)
    elif isinstance(n, list):
        for v in n:
            out += _all_colrefs(v)
    return out


def _all_tables(n) -> set[str]:
    out = set()
    if isinstance(n, dict):
        if n.get("type") == "BASE_TABLE" and "table_name" in n:
            out.add(n["table_name"])
        for v in n.values():
            out |= _all_tables(v)
    elif isinstance(n, list):
        for v in n:
            out |= _all_tables(v)
    return out


class _Extractor:
    def __init__(self, index: StoreIndex):
        self.index = index
        self.store_tables = set(index.tables())
        self.rels: list[_Rel] = []
        self.verified = True
        self.handled: set[str] = set()

    def _targets(self, names: list[str], rels: list[_Rel]) -> list[_Rel]:
        if len(names) >= 2:
            return [r for r in rels if r.alias == names[-2] or r.table == names[-2]]
        return [r for r in rels if names[-1] in self.index.columns(r.table)]

    def _sensitive(self, names: list[str]) -> bool:
        return names[-1] in COLUMN_KIND or names[-1] in TIME_COLS

    def _term(self, t: dict, rels: list[_Rel]) -> None:
        cls, typ = t.get("class"), t.get("type")
        if cls == "COMPARISON" and typ in _CMP:
            left, right, op = t["left"], t["right"], _CMP[typ]
            if _colref(right) and _const(left) is not None:
                left, right, op = right, left, _FLIP[op]
            names, val = _colref(left), _const(right)
            if names and val is not None:
                col = names[-1]
                for r in self._targets(names, rels):
                    if col in COLUMN_KIND and op == "=":
                        r.requested.setdefault(COLUMN_KIND[col], []).append(str(val))
                    elif col in TIME_COLS:
                        v = float(val) * TIME_COLS[col]
                        if op in (">", ">="):
                            r.lo = v if r.lo is None else max(r.lo, v)
                        elif op in ("<", "<="):
                            r.hi = v if r.hi is None else min(r.hi, v)
                        else:
                            r.lo, r.hi = v, v + TIME_COLS[col]
                    elif col in COLUMN_KIND:
                        r.verified = False
                return
        if cls == "OPERATOR" and typ == "COMPARE_IN":
            names = _colref(t["children"][0])
            vals = [_const(c) for c in t["children"][1:]]
            if names and all(v is not None for v in vals):
                col = names[-1]
                for r in self._targets(names, rels):
                    if col in COLUMN_KIND:
                        r.requested.setdefault(COLUMN_KIND[col], []).extend(map(str, vals))
                    elif col in TIME_COLS:
                        r.verified = False
                return
        if cls == "BETWEEN":
            names, lo, hi = _colref(t["input"]), _const(t["lower"]), _const(t["upper"])
            if names and lo is not None and hi is not None and names[-1] in TIME_COLS:
                u = TIME_COLS[names[-1]]
                for r in self._targets(names, rels):
                    r.lo = float(lo) * u if r.lo is None else max(r.lo, float(lo) * u)
                    r.hi = float(hi) * u + u if r.hi is None else min(r.hi, float(hi) * u + u)
                return
        # any other shape: irrelevant unless it touches an entity or time column
        for names in _all_colrefs(t):
            if self._sensitive(names):
                for r in self._targets(names, rels) or rels:
                    r.verified = False
        if cls == "SUBQUERY":
            for r in rels:
                r.verified = False

    def _from(self, f: dict, ctes: dict, rels: list[_Rel]) -> None:
        typ = f.get("type")
        if typ == "BASE_TABLE":
            name = f["table_name"]
            if name in ctes:
                self._select(ctes[name])
                return
            rel = _Rel(f.get("alias") or name, name)
            rels.append(rel)
            self.handled.add(name)
        elif typ == "JOIN":
            self._from(f["left"], ctes, rels)
            self._from(f["right"], ctes, rels)
            cond = f.get("condition")
            if f.get("using_columns"):
                for c in f["using_columns"]:
                    for r in rels:
                        r.join_keys.append((r.table, c))
            elif cond:
                terms = cond["children"] if cond.get("type") == "CONJUNCTION_AND" else [cond]
                for t in terms:
                    if (
                        t.get("type") == "COMPARE_EQUAL"
                        and _colref(t["left"])
                        and _colref(t["right"])
                    ):
                        for side in (t["left"], t["right"]):
                            names = _colref(side)
                            for r in self._targets(names, rels):
                                r.join_keys.append((r.table, names[-1]))
                    else:
                        self._term(t, rels)
        elif typ == "SUBQUERY":
            self._node(f["subquery"]["node"])
        elif typ in ("EMPTY", None):
            return
        else:  # table functions, pivots, ...
            self.verified = False

    def _select(self, q: dict) -> None:
        self._node(q["node"] if "node" in q else q)

    def _node(self, node: dict) -> None:
        if node.get("type") == "SET_OPERATION_NODE":
            self.verified = False
            self._node(node["left"])
            self._node(node["right"])
            return
        if node.get("type") != "SELECT_NODE":
            self.verified = False
            return
        ctes = {m["key"]: m["value"]["query"] for m in node.get("cte_map", {}).get("map", [])}
        rels: list[_Rel] = []
        self._from(node.get("from_table") or {}, ctes, rels)
        w = node.get("where_clause")
        if w:
            terms = w["children"] if w.get("type") == "CONJUNCTION_AND" else [w]
            for t in terms:
                self._term(t, rels)
        self.rels += rels

    def run(self, node: dict) -> list[Scope]:
        self._node(node)
        scopes = []
        for r in self.rels:
            if r.table not in self.store_tables:
                continue
            ok = self.verified and r.verified
            req = {k: sorted(set(v)) for k, v in r.requested.items()}
            window = None
            if r.lo is not None or r.hi is not None:
                window = (
                    r.lo if r.lo is not None else -math.inf,
                    r.hi if r.hi is not None else math.inf,
                )
            scopes.append(
                Scope(
                    source=r.table,
                    requested=req,
                    window=window,
                    join_keys=r.join_keys or None,
                    verified=ok,
                )
            )
        for t in sorted(_all_tables(node) - self.handled):
            if t in self.store_tables:
                scopes.append(Scope(source=t, verified=False))
        return scopes


def extract(sql: str, index: StoreIndex) -> Extracted:
    con = index.con
    raw = con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
    parsed = json.loads(raw)
    if parsed.get("error"):
        raise ToolInputError(
            f"not a single read-only SELECT: {parsed.get('error_message', 'parse error')}",
            ["write one SELECT statement over the store's tables"],
        )
    stmts = parsed["statements"]
    if len(stmts) != 1:
        raise ToolInputError("exactly one statement is allowed")
    node = stmts[0]["node"]
    mods = {m["type"] for m in node.get("modifiers", [])}
    ex = _Extractor(index)
    scopes = ex.run(node)
    return Extracted(
        scopes,
        all(s.verified for s in scopes),
        "ORDER_MODIFIER" in mods,
        bool(mods & {"LIMIT_MODIFIER", "LIMIT_PERCENT_MODIFIER"}),
    )


@tool(
    "sql",
    SqlArgs,
    description=(
        "Run one read-only SELECT over the indexed tables (app_metrics, container_metrics, spans, logs, "
        "log_access_apache, log_access_localhost, log_templates, span_roots, topology_edges, clock_offsets, "
        "series_cadence). Without ORDER BY the result is ordered by all columns; LIMIT requires ORDER BY. "
        "Filter entities with `col = 'x'` or `col IN (...)` and time with comparisons on timestamp_s, so an "
        "empty result can be certified as absence; other filter shapes still run but mark the scope unverified."
    ),
    sources=["unknown"],
)
def run_sql(a: SqlArgs, ctx) -> ToolResult:
    q = a.query.strip().rstrip(";")
    ex = extract(q, ctx.index)
    for sc in ex.scopes:
        for kind, vals in sc.requested.items():
            ctx.index.check_known(kind, vals)
    if ex.has_limit and not ex.has_order:
        raise ToolInputError(
            "LIMIT/OFFSET without ORDER BY selects arbitrary rows that no re-execution could reproduce",
            ["add ORDER BY"],
        )
    if ex.has_order:
        final, total = q, False
    else:
        final = f"SELECT * FROM ({q}) AS _q ORDER BY ALL"
    rows = ctx.fetch(final)
    if not ex.has_order:
        key = [tuple(json.dumps(v, default=str) for v in r.values()) for r in rows]
        total = all(key[i] != key[i + 1] for i in range(len(key) - 1))
    ent = bool(rows) and any(c in COLUMN_KIND for c in rows[0])
    for sc in ex.scopes:
        if not ent:
            sc.visibility = "unknown"
        sc.row_count = len(rows) if len(ex.scopes) == 1 else (len(rows) or 0)
    scopes = ex.scopes or [Scope(source="unknown", verified=False, row_count=len(rows))]
    notes = {} if ex.has_order else {"order": "no ORDER BY given: rows are ordered by all columns"}
    return ToolResult(
        rows,
        scopes,
        [final],
        order_by=[] if ex.has_order else ["ALL"],
        order_total=total,
        presorted=True,
        notes=notes,
    )
