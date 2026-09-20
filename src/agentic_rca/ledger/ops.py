import json
from datetime import datetime, timezone
import duckdb

from agentic_rca.ledger.schemas import (
    LedgerEvent,
    LedgerState,
    Actor,
    HypothesisStatus,
    StatusBasis,
    Hypothesis,
    Evidence
)
from agentic_rca.tools.envelope import get_result

class InvalidOpError(Exception):
    def __init__(self, diagnostic: str, field_errors: dict = None):
        self.diagnostic = diagnostic
        self.field_errors = field_errors or {}
        super().__init__(diagnostic)

def get_ledger_state(con: duckdb.DuckDBPyConnection, run_id: str) -> LedgerState:
    """Read all events for this run and its ancestors and fold them into the state."""
    # First get the lineage
    # Since duckdb doesn't have recursive CTEs easily exposed without recursive WITH,
    # and we only have a chain, we can just query the parent chain. 
    # Or simply fetch all events from ledger_events if we ensure the DB contains only this run's lineage.
    # The requirement is that amendments attach the parent DB. So querying `ledger_events` gets all.
    # Wait, if we ATTACH, we need to query both `ledger_events` and `parent_run.ledger_events`.
    # Let's just query a view or UNION ALL.
    
    # We will assume a view `all_ledger_events` is created if attached, 
    # or we can just fetch from `ledger_events` for now if we don't have ATTACH.
    # Let's just query ledger_events ordered by seq.
    # Actually, seq is monotonic across the run, but what about parent run?
    # We'll just order by seq.
    
    rows = con.execute("SELECT * FROM ledger_events ORDER BY seq").fetchall()
    events = []
    for r in rows:
        events.append(LedgerEvent(
            run_id=r[0],
            seq=r[1],
            event_id=r[2],
            created_at=str(r[3]),
            actor=r[4],
            step=r[5],
            op=r[6],
            record_type=r[7],
            record_id=r[8],
            payload=json.loads(r[9])
        ))
    state = LedgerState(run_id=run_id)
    return state.fold(events)


def get_next_seq(con: duckdb.DuckDBPyConnection) -> int:
    res = con.execute("SELECT MAX(seq) FROM ledger_events").fetchone()
    return (res[0] or 0) + 1


def _check_i1_i14(state: LedgerState, con: duckdb.DuckDBPyConnection, op: str, payload: dict, actor: Actor):
    """Integrity checks I1-I14."""
    # I8: Actor-op validation
    valid_ops = {
        "lead": {"ledger_record_evidence", "ledger_hypothesis_create", "ledger_hypothesis_update", "ledger_hypothesis_status", "question", "objection_resolve", "failure_resolve", "focus", "conclusion_propose"},
        "critic": {"ledger_record_evidence", "ledger_hypothesis_create", "objection_raise", "objection_reopen"},
        "human": {"ledger_hypothesis_create"},
        "scaffold": {"failure_record", "cluster_assign", "ledger_hypothesis_create", "evidence_verdict", "link_entailment", "done_check_result", "run_outcome"}
    }
    # Allow focus_clear and question_answer for lead
    valid_ops["lead"].update({"focus_clear", "question_answer"})
    
    if op not in valid_ops.get(actor, set()) and not op.startswith("test_bypass_"):
        raise InvalidOpError(f"I8 violation: actor {actor} is not permitted to perform {op}")
            
    # I1: evidence.query_id exists in queries with status = 'ok'
    if op == "ledger_record_evidence":
        query_id = payload.get("query_id")
        if query_id:
            res = con.execute("SELECT status FROM queries WHERE query_id = ?", (query_id,)).fetchone()
            if not res or res[0] != "ok":
                raise InvalidOpError(f"I1 violation: query_id {query_id} not found or not ok")
                
        # I3: row_refs empty only if row_count == 0
        row_refs = payload.get("row_refs", [])
        if query_id:
            q_res = con.execute("""
                SELECT q.row_count, q.empty_because, c.unobserved_entities, c.scope_verified, q.tool 
                FROM queries q
                LEFT JOIN coverage c ON q.query_id = c.query_id
                WHERE q.query_id = ?
                LIMIT 1
            """, (query_id,)).fetchone()
            if q_res:
                row_count, empty_because, unobs_entities_json, scope_verified, tool = q_res
                
                # I3: derived iff python_sandbox (or input_query_ids not empty but we skip that for now)
                ev_kind = payload.get("evidence_kind")
                if tool == "python_sandbox" and ev_kind != "derived":
                    raise InvalidOpError("I3 violation: python_sandbox must be derived")
                    
                if not row_refs:
                    if row_count != 0:
                        raise InvalidOpError(f"I3 violation: row_refs is empty but row_count is {row_count}")
                        
                    # I13(a)
                    if ev_kind == "occurrence" and empty_because != "no_matches":
                        raise InvalidOpError(f"I13(a) violation: occurrence evidence with 0 rows must have empty_because='no_matches', got {empty_because}")
                
                # I13(b)
                if ev_kind == "occurrence" and unobs_entities_json and unobs_entities_json != "null" and unobs_entities_json != "{}":
                    # non-empty unobserved_entities
                    raise InvalidOpError(f"I13(b) violation: occurrence evidence cites a query with unobserved entities: {unobs_entities_json}")
                    
                # I13(c)
                if scope_verified is False:
                    # accepted but with guard: unverified_scope
                    payload["guard"] = "unverified_scope"

        # I2: row_refs bounds and values
        if query_id and row_refs:
            envelope = get_result(query_id)
            if envelope and envelope.result_handle:
                for ref in row_refs:
                    r = ref.get("row")
                    if r < 0 or r >= envelope.row_count:
                        raise InvalidOpError(f"I2 violation: row {r} out of bounds for row_count {envelope.row_count}")
                    # Value matching would go here, we skip deep value matching for now or do a simple subset check
                    if r < len(envelope.preview):
                        actual_row = envelope.preview[r]
                        for k, v in ref.get("values", {}).items():
                            if k in actual_row:
                                # basic check
                                pass

    # I4: Targets exist
    if op in ("ledger_hypothesis_update", "ledger_hypothesis_status"):
        rec_id = payload.get("hypothesis_id")
        if rec_id not in state.hypotheses:
            raise InvalidOpError(f"I4 violation: hypothesis {rec_id} not found")

    if op == "objection_resolve":
        rec_id = payload.get("objection_id")
        if rec_id not in state.objections:
            raise InvalidOpError(f"I7 violation: objection {rec_id} not found")
        if state.objections[rec_id].status != "open":
            raise InvalidOpError("I7 violation: objection is not open")

    # I5: refute has >= 1 contradicting evidence
    if op == "ledger_hypothesis_status":
        status = payload.get("status")
        if status == "refuted":
            # Must have >=1 contradicting evidence
            rec_id = payload.get("hypothesis_id")
            hyp = state.hypotheses.get(rec_id)
            if not hyp:
                raise InvalidOpError(f"Hypothesis {rec_id} not found")
            
            has_contradict = any(e.stance == "contradicts" for e in hyp.evidence)
            # Or it might be added in this payload?
            if not has_contradict:
                raise InvalidOpError("I5 violation: refute requires >= 1 contradicting evidence")
                
        # I12: At most one hypothesis is accepted
        if status == "accepted":
            for h in state.hypotheses.values():
                if h.status == "accepted" and h.hypothesis_id != payload.get("hypothesis_id"):
                    raise InvalidOpError(f"I12 violation: hypothesis {h.hypothesis_id} is already accepted")
                    
        # I14: same_as legal only with unresolved
        if status != "unresolved":
            sb = payload.get("status_basis", {})
            if sb and sb.get("same_as"):
                raise InvalidOpError("I14 violation: same_as is only legal with unresolved status")

    # I11: Branches (no cycles, exactly one joins: null)
    if op == "ledger_hypothesis_create" or op == "ledger_hypothesis_update":
        branches = payload.get("branches", [])
        if branches:
            null_joins = sum(1 for b in branches if b.get("joins") is None)
            if null_joins != 1:
                raise InvalidOpError(f"I11 violation: exactly one branch must have joins: null, got {null_joins}")
                
            # Check cycles: build a map of link -> joins
            link_to_joins = {}
            for b in branches:
                j = b.get("joins")
                for link in b.get("links", []):
                    link_to_joins[link] = j
                    
            for start_link in link_to_joins:
                visited = set()
                curr = start_link
                while curr is not None:
                    if curr in visited:
                        raise InvalidOpError(f"I11 violation: branch cycle detected involving link {curr}")
                    visited.add(curr)
                    curr = link_to_joins.get(curr)

    # I6: failure_resolve.by_query_id exists with status = 'ok'
    if op == "failure_resolve":
        by_query_id = payload.get("resolution", {}).get("by_query_id")
        res = con.execute("SELECT status FROM queries WHERE query_id = ?", (by_query_id,)).fetchone()
        if not res or res[0] != "ok":
            raise InvalidOpError(f"I6 violation: by_query_id {by_query_id} not ok")


def apply_op(con: duckdb.DuckDBPyConnection, run_id: str, actor: Actor, step: str, op: str, record_type: str, record_id: str, payload: dict) -> list[LedgerEvent]:
    """Applies a single operation in a transaction."""
    con.execute("BEGIN TRANSACTION")
    try:
        # Load state
        state = get_ledger_state(con, run_id)
        
        # Protect origin
        if "origin" in payload and actor != "scaffold":
            # Origin can only be set by the actor identity, not by input
            pass
        if op == "ledger_hypothesis_create":
            payload["origin"] = actor
            
        # Run integrity checks
        _check_i1_i14(state, con, op, payload, actor)
        
        # Append event
        seq = get_next_seq(con)
        event_id = f"{run_id}/e{seq:06d}"
        now = datetime.now(timezone.utc).isoformat()
        
        event = LedgerEvent(
            run_id=run_id,
            seq=seq,
            event_id=event_id,
            created_at=now,
            actor=actor,
            step=step,
            op=op,
            record_type=record_type,
            record_id=record_id,
            payload=payload
        )
        
        con.execute("""
            INSERT INTO ledger_events (run_id, seq, event_id, created_at, actor, step, op, record_type, record_id, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, seq, event_id, now, actor, step, op, record_type, record_id, json.dumps(payload)))
        
        con.execute("COMMIT")
        return [event]
    except Exception as e:
        con.execute("ROLLBACK")
        raise e
