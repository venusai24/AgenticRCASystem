"""Report and coverage rendering."""

import json
import duckdb
from dataclasses import dataclass
from agentic_rca.ledger.fold import Event, fold

def _get_state(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute("SELECT run_id, seq, actor, step, op, record_type, record_id, payload FROM ledger_events ORDER BY seq").fetchall()
    
    events = []
    for r in rows:
        events.append(
            Event(
                run_id=r[0],
                seq=r[1],
                actor=r[2],
                step=r[3],
                op=r[4],
                record_type=r[5],
                record_id=r[6],
                payload=json.loads(r[7]) if r[7] else {}
            )
        )
        
    res = con.execute("SELECT run_id FROM run").fetchone()
    if res:
        run_id = res[0]
    else:
        # Fallback for tests if run table is empty
        run_id = events[-1].run_id if events else "test"
        
    return fold(events, run_id)

def render_report(db_path: str) -> str:
    """Render report.md from the final run.duckdb state."""
    state = _get_state(db_path)
    
    lines = ["# AgenticRCA Report\n"]
    
    outcome = state.outcome
    if not outcome:
        lines.append("Run incomplete or outcome not recorded.\n")
        return "\n".join(lines)
        
    kind = outcome.get("kind", "inconclusive")
    reason = outcome.get("reason", "")
    
    lines.append(f"**Outcome**: {kind.upper()}")
    if reason:
        lines.append(f"**Reason**: {reason}")
    lines.append("")
    
    # Read the latest done_check_result
    dc = state.done_checks[-1] if state.done_checks else None
    
    # Render Accepted Hypothesis if conclusive
    if kind == "conclusive" and outcome.get("accepted_hypothesis_id"):
        hid = outcome["accepted_hypothesis_id"]
        h = state.hypotheses.get(hid)
        if h:
            lines.append(f"## Accepted Hypothesis: {hid}")
            lines.append(f"{h['statement']}\n")
            # Links
            if h["branches"]:
                lines.append("### Evidence Chain")
                for branch in h["branches"]:
                    for lid in branch.get("links", []):
                        link = state.links.get(lid)
                        if link:
                            lines.append(f"- **{link.get('cause', '')}** -> **{link.get('effect', '')}**")
                            # entailment flag
                            ent = state.entailments.get(lid)
                            if ent:
                                lines.append(f"  - Entailment: {ent.get('verdict')} ({ent.get('rationale', '')})")
            lines.append("")
            
    # Render Inconclusive Tiers if inconclusive
    if kind == "inconclusive" and dc and dc.get("candidates"):
        lines.append("## Inconclusive Candidates")
        for cand in dc["candidates"]:
            hid = cand.get("id", "")
            h = state.hypotheses.get(hid, {})
            tier = cand.get("tier", 0)
            lines.append(f"- **{hid}** (Tier {tier}): {h.get('statement', '')}")
            if cand.get("tied"):
                lines.append("  - *Tied*")
            if cand.get("flag"):
                lines.append(f"  - Flag: {cand['flag']}")
        lines.append("")
        
    # Render Refutations
    refuted = [h for h in state.hypotheses.values() if h.get("status") == "refuted"]
    if refuted:
        lines.append("## Refutations")
        for h in refuted:
            lines.append(f"- **{h['id']}**: {h['statement']}")
            basis = h.get("status_basis", {})
            if basis and basis.get("evidence_ids"):
                lines.append(f"  - Evidence: {', '.join(basis['evidence_ids'])}")
        lines.append("")

    # Render Observability Gaps
    if dc and dc.get("observability_gaps"):
        lines.append("## Observability Gaps")
        gaps = dc["observability_gaps"]
        
        on_path = gaps.get("on_path", [])
        if on_path:
            lines.append("### On Path")
            for gap in on_path:
                lines.append(f"- {gap.get('attempt_reason')} | {gap.get('source')} | {gap.get('entity_kind')} | {gap.get('values')}")
                
        other = gaps.get("other", [])
        if other:
            lines.append("### Other")
            for gap in other:
                lines.append(f"- {gap.get('attempt_reason')} | {gap.get('source')} | {gap.get('entity_kind')} | {gap.get('values')}")
        lines.append("")
        
    # Render Unmet Criteria
    if dc and dc.get("unmet"):
        lines.append("## Unmet Criteria")
        for unmet in dc["unmet"]:
            lines.append(f"- **Criterion {unmet.get('criterion')}** ({unmet.get('code')}): target {unmet.get('target')}")
            if unmet.get('detail'):
                lines.append(f"  - {unmet['detail']}")
        lines.append("")
        
    # Render Objections
    open_obj = [o for o in state.objections.values() if o.get("status") == "open"]
    if open_obj:
        lines.append("## Open Objections")
        for o in open_obj:
            lines.append(f"- **{o['id']}** (Target: {o['target']}): {o['text']}")
        lines.append("")
        
    resolved_obj = [o for o in state.objections.values() if o.get("status") in ("resolved", "conceded")]
    if resolved_obj:
        lines.append("## Resolved Objections")
        for o in resolved_obj:
            lines.append(f"- **{o['id']}** (Target: {o['target']}): {o['text']}")
            for res in o.get("resolutions", []):
                lines.append(f"  - {res.get('kind')}: {res.get('note')}")
        lines.append("")
        
    return "\n".join(lines)


def render_coverage_human(db_path: str) -> str:
    """Render coverage_human.md from the coverage index."""
    con = duckdb.connect(db_path, read_only=True)
    
    # A simple post-termination prominence view
    lines = ["# Coverage (Human-Facing)\n"]
    
    # We list coverage attempts and successes
    rows = con.execute("SELECT kind, attempt_reason, sources, requested, row_count FROM coverage ORDER BY kind, sources").fetchall()
    if not rows:
        lines.append("No coverage recorded.\n")
        return "\n".join(lines)
        
    for r in rows:
        kind = r[0]
        reason = r[1]
        sources = r[2]
        requested = r[3]
        row_count = r[4]
        
        if kind == "queried":
            lines.append(f"- **SUCCESS**: Sources: {sources} | Requested: {requested} | Rows: {row_count}")
        else:
            lines.append(f"- **ATTEMPT**: Sources: {sources} | Requested: {requested} | Reason: {reason}")
            
    return "\n".join(lines)
