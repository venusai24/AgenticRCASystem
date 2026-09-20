"""Ledger ops: validate -> integrity-check -> mint ids -> append one event,
all or nothing (contracts-b1 §3, §6).

A rejected op raises LedgerError and appends nothing. The loop hands the
error back to the model as a correctable diagnostic; it is never a crash.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agentic_rca.ledger import models as M
from agentic_rca.ledger.fold import STANDING_STATEMENT, Event, LedgerState, fold
from agentic_rca.ledger.rundb import connect_run, mint_run_id, now_iso

ResultLookup = Callable[[str], "list[dict] | None"]


class LedgerError(Exception):
    def __init__(self, rule: str, diagnostic: str, field_errors: list | None = None):
        super().__init__(f"[{rule}] {diagnostic}")
        self.rule = rule
        self.diagnostic = diagnostic
        self.field_errors = field_errors or []

    def as_dict(self) -> dict:
        return {
            "error": {
                "type": "invalid_op",
                "rule": self.rule,
                "diagnostic": self.diagnostic,
                "field_errors": self.field_errors,
            }
        }


def values_match(a: Any, b: Any) -> bool:
    """contracts-b1 §9: floats by isclose(rel 1e-9, abs 1e-12), else equality.
    NaN equals NaN (it serialises to the same string)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return str(a) == str(b)


def _no_lookup(_qid: str) -> list[dict] | None:
    return None


class Ledger:
    """One run's ledger over `runs/<run_id>/run.duckdb`, with its lineage
    (parent runs, oldest first) folded in read-only."""

    def __init__(
        self,
        con,
        run_id: str,
        run_dir: Path,
        lineage: list[tuple[str, Path]],
        result_lookup: ResultLookup | None = None,
    ):
        self.con = con
        self.run_id = run_id
        self.run_dir = run_dir
        self.lineage = lineage  # [(run_id, run.duckdb path)] ancestors, oldest first
        self.result_lookup = result_lookup or _no_lookup
        self._parent_events: list[Event] | None = None
        self._parent_queries: dict[str, dict] | None = None
        self._state: LedgerState | None = None
        self._state_seq = -1

    # ---- lifecycle -----------------------------------------------------------
    @classmethod
    def create(
        cls,
        runs_root: Path,
        *,
        store_path: str = "",
        dataset_digest: dict | None = None,
        code_version: str = "",
        incident_ts: float | None = None,
        params: dict | None = None,
        parent_run_dir: Path | None = None,
        run_id: str | None = None,
        result_lookup: ResultLookup | None = None,
    ) -> Ledger:
        run_id = run_id or mint_run_id()
        run_dir = Path(runs_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        lineage: list[tuple[str, Path]] = []
        parent_id = root_id = None
        depth = 0
        if parent_run_dir is not None:
            parent = cls.open(parent_run_dir, read_only=True)
            lineage = parent.lineage + [(parent.run_id, parent.run_dir / "run.duckdb")]
            prow = parent.run_row()
            parent.close()
            parent_id = prow["run_id"]
            root_id = prow["root_run_id"]
            depth = int(prow["depth"]) + 1
        con = connect_run(run_dir / "run.duckdb")
        con.execute(
            "INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                run_id,
                parent_id,
                str(parent_run_dir) if parent_run_dir else None,
                root_id or run_id,
                depth,
                now_iso(),
                str(store_path),
                json.dumps(dataset_digest or {}, sort_keys=True),
                code_version,
                incident_ts,
                json.dumps(params or {}, sort_keys=True),
            ],
        )
        return cls(con, run_id, run_dir, lineage, result_lookup)

    @classmethod
    def open(
        cls, run_dir: Path, *, read_only: bool = False, result_lookup: ResultLookup | None = None
    ) -> Ledger:
        run_dir = Path(run_dir)
        con = connect_run(run_dir / "run.duckdb", read_only=read_only)
        row = con.execute("SELECT run_id, parent_run_path FROM run").fetchone()
        run_id, parent_path = row
        lineage: list[tuple[str, Path]] = []
        if parent_path:
            parent = cls.open(Path(parent_path), read_only=True)
            lineage = parent.lineage + [(parent.run_id, parent.run_dir / "run.duckdb")]
            parent.close()
        return cls(con, run_id, run_dir, lineage, result_lookup)

    def close(self) -> None:
        self.con.close()

    def run_row(self) -> dict:
        cur = self.con.execute("SELECT * FROM run")
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, cur.fetchone(), strict=True))

    # ---- reading ---------------------------------------------------------------
    @staticmethod
    def _read_events(con, run_id: str | None = None) -> list[Event]:
        rows = con.execute(
            "SELECT run_id, seq, actor, step, op, record_type, record_id, payload "
            "FROM ledger_events ORDER BY seq"
        ).fetchall()
        return [Event(r[0], r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7])) for r in rows]

    def _load_parents(self) -> None:
        if self._parent_events is not None:
            return
        events: list[Event] = []
        queries: dict[str, dict] = {}
        for _rid, path in self.lineage:
            pcon = connect_run(path, read_only=True)
            events.extend(self._read_events(pcon))
            cur = pcon.execute("SELECT * FROM queries")
            cols = [d[0] for d in cur.description]
            for r in cur.fetchall():
                d = dict(zip(cols, r, strict=True))
                queries[d["query_id"]] = d
            pcon.close()
        self._parent_events = events
        self._parent_queries = queries

    def events(self) -> list[Event]:
        self._load_parents()
        return list(self._parent_events or []) + self._read_events(self.con)

    def last_seq(self) -> int:
        return self.con.execute("SELECT coalesce(max(seq), 0) FROM ledger_events").fetchone()[0]

    def state(self) -> LedgerState:
        seq = self.last_seq()
        if self._state is None or seq != self._state_seq:
            self._state = fold(self.events(), self.run_id)
            self._state_seq = seq
        return self._state

    def events_since(self, seq: int) -> list[Event]:
        return [e for e in self._read_events(self.con) if e.seq > seq]

    def query_row(self, qid: str) -> dict | None:
        cur = self.con.execute("SELECT * FROM queries WHERE query_id = ?", [qid])
        row = cur.fetchone()
        if row is not None:
            return dict(zip([d[0] for d in cur.description], row, strict=True))
        self._load_parents()
        return (self._parent_queries or {}).get(qid)

    # ---- ids -----------------------------------------------------------------
    def lineage_ids(self) -> set[str]:
        return {rid for rid, _ in self.lineage} | {self.run_id}

    def q(self, short_or_qualified: str) -> str:
        """Normalise an id to its qualified `<run_id>/<id>` form (I9)."""
        if "/" in short_or_qualified:
            rid = short_or_qualified.split("/", 1)[0]
            if rid not in self.lineage_ids():
                raise LedgerError(
                    "I9", f"{short_or_qualified!r} belongs to run {rid}, not this run's lineage"
                )
            return short_or_qualified
        return f"{self.run_id}/{short_or_qualified}"

    def short(self, qid: str) -> str:
        rid, _, rest = qid.partition("/")
        return rest if rid == self.run_id else qid

    def _mint(self, state: LedgerState, prefix: str, width: int) -> str:
        key = f"{self.run_id}|{prefix}"
        n = state.counters.get(key, 0) + 1
        state.counters[key] = n  # reserve within this op
        return f"{self.run_id}/{prefix}{n:0{width}d}"

    # ---- the one write path ----------------------------------------------------
    def apply(
        self,
        op: str,
        payload: dict | None = None,
        *,
        actor: str,
        step: int = 0,
        round: int | None = None,
        in_transaction: bool = False,
    ) -> dict:
        """Validate and append one op. Returns {"ok": True, "id": <short id>, ...}.
        Raises LedgerError, having appended nothing."""
        if op not in M.OPS:
            raise LedgerError("op", f"unknown ledger op {op!r}")
        model, actors = M.OPS[op]
        if actor not in actors:
            raise LedgerError("I8", f"actor {actor!r} may not perform {op!r}")
        payload = payload or {}
        if model is not None:
            try:
                payload = model.model_validate(payload).model_dump(mode="json")
            except ValidationError as exc:
                errs = [
                    {"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]}
                    for e in exc.errors()
                ]
                raise LedgerError("schema", f"{op}: payload failed validation", errs) from exc
        # fold a private copy so reservations made while minting never leak
        state = fold(self.events(), self.run_id)
        handler = getattr(self, f"_op_{op}")
        record_type, record_id, stored = handler(state, payload, actor=actor, round=round)
        seq = self._append(op, actor, step, record_type, record_id, stored, in_transaction)
        out = {"ok": True, "seq": seq}
        if record_id:
            out["id"] = self.short(record_id)
        return out

    def _append(self, op, actor, step, record_type, record_id, payload, in_transaction=False) -> int:
        """`in_transaction=True`: the caller (the tool registry) owns an open
        transaction and commits this event together with its own rows."""
        if not in_transaction:
            self.con.execute("BEGIN TRANSACTION")
        try:
            seq = self.last_seq() + 1
            self.con.execute(
                "INSERT INTO ledger_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    self.run_id,
                    seq,
                    f"{self.run_id}/e{seq:06d}",
                    now_iso(),
                    actor,
                    step,
                    op,
                    record_type,
                    record_id,
                    json.dumps(payload, sort_keys=True, default=str),
                ],
            )
            if not in_transaction:
                self.con.execute("COMMIT")
        except Exception:
            if not in_transaction:
                self.con.execute("ROLLBACK")
            raise
        return seq

    # ---- shared checks -----------------------------------------------------------
    def _ev(self, state: LedgerState, eid: str) -> str:
        qid = self.q(eid)
        if qid not in state.evidence:
            raise LedgerError("I4", f"evidence {eid!r} does not exist")
        return qid

    def _hyp(self, state: LedgerState, hid: str) -> dict:
        qid = self.q(hid)
        if qid not in state.hypotheses:
            raise LedgerError("I7", f"hypothesis {hid!r} does not exist")
        return state.hypotheses[qid]

    def _build_chain(self, state: LedgerState, hid: str, branches_in: list[dict]):
        """Mint link ids across all branches and resolve `joins` (I4, I11)."""
        if not branches_in:
            return [], []
        links: list[dict] = []
        branches: list[dict] = []
        link_ids: list[list[str]] = []
        for b in branches_in:
            ids = []
            for li in b["links"]:
                lid = self._mint(state, f"{hid.split('/', 1)[1]}.l", 1)
                ev = [
                    {"evidence_id": self._ev(state, r["evidence_id"]), "role": r["role"]}
                    for r in li["evidence"]
                ]
                ev_ids = {r["evidence_id"] for r in ev}
                anchors = {}
                for end in ("cause_at", "effect_at"):
                    a = li.get(end)
                    if a is None:
                        anchors[end] = None
                        continue
                    aid = self._ev(state, a["evidence_id"])
                    if aid not in ev_ids:
                        raise LedgerError(
                            "I4",
                            f"{end}.evidence_id {a['evidence_id']!r} is not in the link's evidence",
                        )
                    rows = {r["row"] for r in state.evidence[aid]["row_refs"]}
                    if a["row"] not in rows:
                        raise LedgerError(
                            "I4", f"{end}.row {a['row']} is not among {a['evidence_id']}'s row_refs"
                        )
                    anchors[end] = dict(a, evidence_id=aid)
                links.append(
                    {
                        "id": lid,
                        "cause": li["cause"],
                        "effect": li["effect"],
                        "evidence": ev,
                        **anchors,
                    }
                )
                ids.append(lid)
            link_ids.append(ids)
        for bi, b in enumerate(branches_in):
            j = b.get("joins")
            joins = None
            if j is not None:
                if j["branch"] == bi or j["branch"] >= len(branches_in):
                    raise LedgerError("I11", f"branch {bi} joins invalid branch {j['branch']}")
                if j["link"] >= len(link_ids[j["branch"]]):
                    raise LedgerError("I11", f"branch {bi} joins a link that doesn't exist")
                joins = link_ids[j["branch"]][j["link"]]
            branches.append(
                {
                    "id": f"{hid}.b{bi + 1}",
                    "links": link_ids[bi],
                    "joins": joins,
                    "joins_branch": None if j is None else j["branch"],
                }
            )
        terminal = [b for b in branches if b["joins"] is None]
        if len(terminal) != 1:
            raise LedgerError(
                "I11",
                f"exactly one branch must end at the symptoms (joins: null); got {len(terminal)}",
            )
        # cycle check over branch -> joined branch
        for start in range(len(branches)):
            seen, cur = set(), start
            while branches[cur]["joins_branch"] is not None:
                if cur in seen:
                    raise LedgerError("I11", "branch joins form a cycle")
                seen.add(cur)
                cur = branches[cur]["joins_branch"]
        for b in branches:
            b.pop("joins_branch")
        return branches, links

    def _build_factors(
        self, state: LedgerState, hid: str, factors_in: list[dict], link_ids: set[str]
    ):
        out = []
        for f in factors_in:
            fid = self._mint(state, f"{hid.split('/', 1)[1]}.c", 1)
            acts = None
            if f.get("acts_on"):
                acts = self.q(f["acts_on"])
                if acts not in link_ids:
                    raise LedgerError(
                        "I4", f"acts_on {f['acts_on']!r} is not a link of this hypothesis"
                    )
            out.append(
                {
                    "id": fid,
                    "statement": f["statement"],
                    "evidence": [
                        {"evidence_id": self._ev(state, r["evidence_id"]), "role": r["role"]}
                        for r in f["evidence"]
                    ],
                    "acts_on": acts,
                }
            )
        return out

    # ---- op handlers: return (record_type, record_id, stored_payload) ------------------
    def _op_evidence_record(self, state, p, **_):
        qid = self.q(p["query_id"])
        qrow = self.query_row(qid)
        if qrow is None or qrow["status"] != "ok":
            raise LedgerError(
                "I1",
                f"query {p['query_id']!r} does not exist or did not succeed; evidence can't cite it",
            )
        row_count = int(qrow["row_count"] or 0)
        refs = p["row_refs"]
        if row_count == 0 and refs:
            raise LedgerError("I3", "the query returned 0 rows, so row_refs must be empty")
        if row_count > 0 and not refs:
            raise LedgerError(
                "I3", "the query returned rows; cite at least one (row_refs empty only for 0 rows)"
            )
        inputs = json.loads(qrow["input_query_ids"] or "[]")
        is_derived = qrow["tool"] == "python_sandbox" or bool(inputs)
        if (p["evidence_kind"] == "derived") != is_derived:
            raise LedgerError(
                "I3",
                "evidence_kind must be 'derived' exactly when the query is python_sandbox or reads other handles",
            )
        # I13 negative-evidence guard
        unobserved = json.loads(qrow["unobserved_entities"] or "null") or {}
        if (
            p["evidence_kind"] == "occurrence"
            and not refs
            and qrow["empty_because"] != "no_matches"
        ):
            raise LedgerError(
                "I13",
                f"a zero-row result with empty_because={qrow['empty_because']!r} can't support a negative "
                f"claim: {_EMPTY_MEANING.get(qrow['empty_because'], 'not a certified absence')}. "
                "Only empty_because='no_matches' licenses a negative claim.",
            )
        if p["evidence_kind"] == "occurrence" and any(unobserved.values()):
            raise LedgerError(
                "I13",
                f"the query names entities this source has never held: {unobserved}. That's an "
                "observability gap, not evidence. Re-run it on the observed entities only.",
            )
        # I2: rows in range, cited values equal the live handle's
        snapshot = []
        rows = self.result_lookup(qid) if qid.startswith(self.run_id + "/") else None
        for r in refs:
            if r["row"] >= row_count:
                raise LedgerError("I2", f"row {r['row']} is out of range (row_count={row_count})")
            if rows is not None:
                actual = rows[r["row"]]
                for col, val in r["values"].items():
                    if col not in actual:
                        raise LedgerError(
                            "I2", f"row {r['row']} has no column {col!r}; actual row: {actual}"
                        )
                    if not values_match(actual[col], val):
                        raise LedgerError(
                            "I2",
                            f"cited {col}={val!r} but row {r['row']} has {actual[col]!r}; actual row: {actual}",
                        )
                snapshot.append(actual)
        cov = self.con.execute(
            "SELECT coverage_id FROM coverage WHERE query_id = ? ORDER BY coverage_id", [qid]
        ).fetchall()
        eid = self._mint(state, "ev", 4)
        stored = dict(
            p,
            evidence_id=eid,
            query_id=qid,
            rows_snapshot=snapshot if rows is not None else None,
            coverage_ids=[c[0] for c in cov],
            guard="ok" if qrow["scope_verified"] in (True, None) else "unverified_scope",
            tool=qrow["tool"],
        )
        return "evidence", eid, stored

    def _create(self, state, p, origin: str):
        hid = self._mint(state, "hy", 3)
        branches, links = self._build_chain(state, hid, p.get("branches", []))
        factors = self._build_factors(
            state, hid, p.get("contributing_factors", []), {lk["id"] for lk in links}
        )
        stored = {
            "hypothesis_id": hid,
            "statement": p["statement"],
            "origin": origin,
            "branches": branches,
            "links": links,
            "factors": factors,
        }
        return "hypothesis", hid, stored

    def _op_hypothesis_create(self, state, p, actor, **_):
        return self._create(state, p, origin=actor)

    def _op_hypothesis_create_human(self, state, p, **_):
        return self._create(state, p, origin="human")

    def _op_standing_seed(self, state, p, **_):
        if state.standing_id():
            raise LedgerError("standing", "the standing hypothesis already exists")
        return self._create(state, {"statement": STANDING_STATEMENT}, origin="scaffold")

    def _op_hypothesis_revise(self, state, p, **_):
        h = self._hyp(state, p["hypothesis_id"])
        if h["origin"] == "scaffold":
            raise LedgerError(
                "standing", "the standing hypothesis's statement is fixed; set its status instead"
            )
        stored: dict[str, Any] = {"hypothesis_id": h["id"]}
        changed = False
        if p.get("statement") is not None and p["statement"] != h["statement"]:
            stored["statement"] = p["statement"]
            changed = True
        link_ids = {lid for b in h["branches"] for lid in b["links"]}
        if p.get("branches") is not None:
            branches, links = self._build_chain(state, h["id"], p["branches"])
            stored["branches"], stored["links"] = branches, links
            link_ids = {lk["id"] for lk in links}
            changed = True
        if p.get("contributing_factors") is not None:
            stored["factors"] = self._build_factors(
                state, h["id"], p["contributing_factors"], link_ids
            )
        stored["content_changed"] = changed
        return "hypothesis", h["id"], stored

    def _op_hypothesis_link_evidence(self, state, p, **_):
        h = self._hyp(state, p["hypothesis_id"])
        eid = self._ev(state, p["evidence_id"])
        for link in h["evidence"]:
            if (
                link["evidence_id"] == eid
                and link["withdrawn"] is None
                and link["stance"] == p["stance"]
            ):
                raise LedgerError("dup", f"{p['evidence_id']} is already linked as {p['stance']}")
        return (
            "hypothesis",
            h["id"],
            {"hypothesis_id": h["id"], "evidence_id": eid, "stance": p["stance"]},
        )

    def _op_hypothesis_withdraw_evidence(self, state, p, **_):
        h = self._hyp(state, p["hypothesis_id"])
        eid = self._ev(state, p["evidence_id"])
        if not any(lk["evidence_id"] == eid and lk["withdrawn"] is None for lk in h["evidence"]):
            raise LedgerError(
                "I10", f"{p['evidence_id']} is not an active evidence link of {p['hypothesis_id']}"
            )
        return (
            "hypothesis",
            h["id"],
            {"hypothesis_id": h["id"], "evidence_id": eid, "reason": p["reason"]},
        )

    def _op_hypothesis_set_status(self, state, p, **_):
        h = self._hyp(state, p["hypothesis_id"])
        new, basis = p["status"], dict(p["basis"])
        if new == h["status"]:
            raise LedgerError("status", f"{p['hypothesis_id']} is already {new}")
        if new != "live" and h["status"] != "live":
            raise LedgerError(
                "status", f"{p['hypothesis_id']} is {h['status']}; reopen it (status live) first"
            )
        basis["evidence_ids"] = [self._ev(state, e) for e in basis["evidence_ids"]]
        if basis.get("same_as"):
            if new != "unresolved":
                raise LedgerError("I14", "same_as is only legal with status 'unresolved'")
            target = self._hyp(state, basis["same_as"])
            if target["id"] == h["id"] or target["cluster_id"] != h["cluster_id"]:
                raise LedgerError(
                    "I14", "same_as must name a different hypothesis in the same cluster"
                )
            if h["cluster_method"] not in ("exact", "llm_judge"):
                raise LedgerError(
                    "I14", "same_as needs a successful dedup assignment (exact or llm_judge)"
                )
            basis["same_as"] = target["id"]
        if new == "refuted":
            contra = {
                lk["evidence_id"]
                for lk in h["evidence"]
                if lk["stance"] == "contradicts" and lk["withdrawn"] is None
            }
            if not set(basis["evidence_ids"]) & contra:
                raise LedgerError(
                    "I5",
                    "refuting needs >=1 basis evidence already linked to this hypothesis as 'contradicts'",
                )
        if new == "accepted":
            others = [a for a in state.accepted() if a != h["id"]]
            if others:
                raise LedgerError(
                    "I12",
                    f"{self.short(others[0])} is already accepted. At most one hypothesis is accepted: "
                    "reopen it first, or carry both as unresolved (an inconclusive result).",
                )
        return "hypothesis", h["id"], {"hypothesis_id": h["id"], "status": new, "basis": basis}

    def _op_prediction_add(self, state, p, **_):
        h = self._hyp(state, p["hypothesis_id"])
        pid = f"{h['id']}.p{len(h['predictions']) + 1}"
        return (
            "hypothesis",
            h["id"],
            {"hypothesis_id": h["id"], "prediction_id": pid, "text": p["text"]},
        )

    def _op_prediction_resolve(self, state, p, **_):
        h = self._hyp(state, p["hypothesis_id"])
        pid = self.q(p["prediction_id"])
        if not any(pr["prediction_id"] == pid for pr in h["predictions"]):
            raise LedgerError(
                "I7", f"prediction {p['prediction_id']!r} does not exist on {p['hypothesis_id']}"
            )
        ev = [self._ev(state, e) for e in p["outcome_evidence_ids"]]
        return (
            "hypothesis",
            h["id"],
            {
                "hypothesis_id": h["id"],
                "prediction_id": pid,
                "status": p["status"],
                "outcome_evidence_ids": ev,
            },
        )

    def _op_question_open(self, state, p, **_):
        qid = self._mint(state, "qu", 3)
        return "question", qid, {"question_id": qid, "text": p["text"]}

    def _op_question_answer(self, state, p, **_):
        qid = self.q(p["question_id"])
        if qid not in state.questions:
            raise LedgerError("I7", f"question {p['question_id']!r} does not exist")
        if state.questions[qid]["status"] != "open":
            raise LedgerError("I7", f"question {p['question_id']!r} is already answered")
        ev = [self._ev(state, e) for e in p["evidence_ids"]]
        return "question", qid, {"question_id": qid, "text": p["text"], "evidence_ids": ev}

    def _op_objection_raise(self, state, p, round=None, **_):
        h = self._hyp(state, p["target"]["hypothesis_id"])
        link = None
        if p["target"].get("link_id"):
            link = self.q(p["target"]["link_id"])
            if link not in state.links:
                raise LedgerError("I7", f"link {p['target']['link_id']!r} does not exist")
        oid = self._mint(state, "ob", 3)
        return (
            "objection",
            oid,
            {
                "objection_id": oid,
                "round": round or 1,
                "text": p["text"],
                "target": {"hypothesis_id": h["id"], "link_id": link},
                "evidence_ids": [self._ev(state, e) for e in p["evidence_ids"]],
            },
        )

    def _op_objection_reopen(self, state, p, round=None, **_):
        oid = self.q(p["objection_id"])
        if oid not in state.objections:
            raise LedgerError("I7", f"objection {p['objection_id']!r} does not exist")
        if state.objections[oid]["status"] == "open":
            raise LedgerError("I7", f"objection {p['objection_id']!r} is already open")
        return "objection", oid, {"objection_id": oid, "note": p["note"], "round": round}

    def _op_objection_resolve(self, state, p, round=None, **_):
        oid = self.q(p["objection_id"])
        if oid not in state.objections:
            raise LedgerError("I7", f"objection {p['objection_id']!r} does not exist")
        if state.objections[oid]["status"] != "open":
            raise LedgerError("I7", f"objection {p['objection_id']!r} is not open")
        return (
            "objection",
            oid,
            {
                "objection_id": oid,
                "kind": p["kind"],
                "evidence_ids": [self._ev(state, e) for e in p["evidence_ids"]],
                "note": p["note"],
                "round": round,
            },
        )

    def _op_focus_set(self, state, p, **_):
        fid = self._mint(state, "fo", 3)
        return (
            "focus",
            fid,
            dict(p, focus_id=fid, evidence_ids=[self._ev(state, e) for e in p["evidence_ids"]]),
        )

    def _op_focus_clear(self, state, p, **_):
        return "focus", None, {}

    def _op_failure_record(self, state, p, **_):
        fid = self._mint(state, "fa", 3)
        return "failure", fid, dict(p, failure_id=fid)

    def _op_failure_resolve(self, state, p, **_):
        fid = self.q(p["failure_id"])
        if fid not in state.failures:
            raise LedgerError("I7", f"failure {p['failure_id']!r} does not exist")
        if state.failures[fid]["resolution"] is not None:
            raise LedgerError("I7", f"failure {p['failure_id']!r} is already resolved")
        by = self.q(p["by_query_id"])
        row = self.query_row(by)
        if row is None or row["status"] != "ok":
            raise LedgerError("I6", f"by_query_id {p['by_query_id']!r} is not a successful query")
        return "failure", fid, {"failure_id": fid, "by_query_id": by, "note": p["note"]}

    def _op_cluster_assign(self, state, p, **_):
        h = self._hyp(state, p["hypothesis_id"])
        if p["for_seq"] != h["content_seq"]:
            raise LedgerError(
                "cluster", f"for_seq {p['for_seq']} is not {h['id']}'s current revision"
            )
        if h["cluster_status"] != "pending":
            raise LedgerError(
                "cluster", f"{h['id']} is already assigned for this revision (never recomputed)"
            )
        cid = p.get("cluster_id") or self._mint(state, "cl", 3)
        return "hypothesis", h["id"], dict(p, hypothesis_id=h["id"], cluster_id=self.q(cid))

    def _op_conclusion_propose(self, state, p, **_):
        stored = dict(p)
        if p["kind"] == "conclusive":
            aid = self._hyp(state, p["accepted_hypothesis_id"])["id"]
            if state.accepted() != [aid]:
                raise LedgerError(
                    "conclusion",
                    "accepted_hypothesis_id must be the (one) hypothesis whose status is accepted",
                )
            stored["accepted_hypothesis_id"] = aid
        for s in stored["symptoms"]:
            s["evidence_ids"] = [self._ev(state, e) for e in s["evidence_ids"]]
            if s.get("explained_by"):
                x = self.q(s["explained_by"])
                if x not in state.links and x not in state.factors:
                    raise LedgerError(
                        "conclusion", f"explained_by {s['explained_by']!r} is not a link or factor"
                    )
                s["explained_by"] = x
        for n in stored["ordering_notes"]:
            n["link_id"] = self.q(n["link_id"])
            if n["link_id"] not in state.links:
                raise LedgerError("conclusion", f"ordering note names unknown link {n['link_id']}")
            n["evidence_ids"] = [self._ev(state, e) for e in n["evidence_ids"]]
        for d in stored["discriminators"]:
            d["between"] = [self._hyp(state, x)["id"] for x in d["between"]]
        return "conclusion", None, stored

    def _passthrough(self, state, p, **_):
        return None, None, dict(p)

    _op_evidence_verdict = _passthrough
    _op_link_entailment = _passthrough
    _op_done_check_result = _passthrough
    _op_run_outcome = _passthrough


_EMPTY_MEANING = {
    "no_data_in_window": "the source holds no rows at all for this window (a coverage gap)",
    "entity_absent_from_source": "the entity was never observed in this source (an observability gap, not health)",
    "join_key_unpopulated": "the join key is null or missing, so the question was not answered",
    "scope_unverified": "the query's entity/time predicates couldn't be interpreted, so absence isn't certified",
}
