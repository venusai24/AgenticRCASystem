"""hypothesis-dedup (#10): group hypotheses into clusters for the >=2-live rule.

Uses an LLM judge (`role="dedup"`) to compare a new/updated hypothesis against
all existing clusters.  A `cluster_assign` event is written to the ledger and
never recomputed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from agentic_rca.ledger import Ledger
from agentic_rca.llm.base import LLMClient, LLMError, load_prompt

_MAX_INPUT_TOKENS = 60000  # Leave room for output + safety margin

@dataclass
class DedupResult:
    cluster_id: str
    unchecked: bool
    rationale: str


def _normalize(text: str | None) -> str:
    """Normalised text for exact matching."""
    if not text:
        return ""
    return " ".join(text.lower().split())


def _format_branch(b: dict) -> str:
    """Format one branch of a hypothesis for comparison."""
    out = []
    if c := b.get("cause"):
        out.append(f"Cause: {c}")
    if e := b.get("effect"):
        out.append(f"Effect: {e}")
    return "\n".join(out)


def _format_hypothesis(h: dict) -> str:
    """Format a hypothesis for the judge, omitting origin/status/evidence."""
    out = [f"Statement: {h.get('statement', '')}"]
    
    # Branches compared as unordered set
    branches = h.get("branches", [])
    if branches:
        out.append("Branches:")
        # Sort branches by their formatted text for stable output
        formatted_branches = sorted(_format_branch(b) for b in branches)
        for b_text in formatted_branches:
            if b_text.strip():
                out.append(f"- {b_text.replace(chr(10), chr(10)+'  ')}")
                
    return "\n".join(out)


def _exact_match(target_h: dict, existing_clusters: dict[str, list[dict]]) -> str | None:
    """Pre-check: if target matches any existing hypothesis exactly, return its cluster."""
    target_norm_stmt = _normalize(target_h.get("statement", ""))
    target_norm_branches = sorted(_normalize(_format_branch(b)) for b in target_h.get("branches", []))
    
    for cid, members in existing_clusters.items():
        for h in members:
            # Different branch count = distinct (DECISIONS.md)
            branches = h.get("branches", [])
            if len(branches) != len(target_h.get("branches", [])):
                continue
                
            norm_stmt = _normalize(h.get("statement", ""))
            norm_branches = sorted(_normalize(_format_branch(b)) for b in branches)
            
            if norm_stmt == target_norm_stmt and norm_branches == target_norm_branches:
                return cid
    return None


def _next_cluster_id(existing_cids: set[str]) -> str:
    """Mint a new cl### cluster id."""
    if not existing_cids:
        return "cl001"
    
    nums = [int(cid[2:]) for cid in existing_cids if cid.startswith("cl") and cid[2:].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return f"cl{next_num:03d}"


def assign_cluster(ledger: Ledger, client: LLMClient, target_id: str, for_seq: int) -> DedupResult:
    """Assign target_id to a cluster, appending a `cluster_assign` event.
    
    Returns the assigned cluster info.
    """
    state = ledger.state()
    target_h = state.hypotheses.get(target_id)
    if not target_h:
        raise ValueError(f"Hypothesis {target_id} not found in state")
        
    if target_h.get("origin") == "scaffold":
        res = DedupResult(target_id, unchecked=False, rationale="seed")
        ledger.apply("cluster_assign", {
            "hypothesis_id": target_id,
            "for_seq": for_seq,
            "cluster_id": res.cluster_id,
            "method": "seed",
            "unchecked": res.unchecked
        }, actor="scaffold")
        return res
        
    # Group existing by cluster, excluding target
    clusters: dict[str, list[dict]] = defaultdict(list)
    existing_cids = set()
    
    for hid, h in state.hypotheses.items():
        if hid == target_id:
            continue
        # Only consider members that have a cluster assigned
        if cid := h.get("cluster_id"):
            short_cid = cid.split("/")[-1]
            clusters[short_cid].append(h)
            existing_cids.add(short_cid)
            
    # 1. Exact match pre-check
    if exact_cid := _exact_match(target_h, clusters):
        res = DedupResult(exact_cid, unchecked=False, rationale="exact match")
        ledger.apply("cluster_assign", {
            "hypothesis_id": target_id,
            "for_seq": for_seq,
            "cluster_id": res.cluster_id,
            "method": "exact",
            "unchecked": res.unchecked
        }, actor="scaffold")
        return res
        
    new_cid = _next_cluster_id(existing_cids)
    
    if not clusters:
        res = DedupResult(new_cid, unchecked=False, rationale="first hypothesis")
        ledger.apply("cluster_assign", {
            "hypothesis_id": target_id,
            "for_seq": for_seq,
            "cluster_id": res.cluster_id,
            "method": "no_candidates",
            "unchecked": res.unchecked
        }, actor="scaffold")
        return res
        
    # 2. Format input for judge
    target_text = _format_hypothesis(target_h)
    
    clusters_text = []
    # Creation order (sort by cid)
    for cid in sorted(clusters.keys()):
        members = clusters[cid]
        # Sort members by hypothesis id for stability
        members_text = "\n\n---\n\n".join(_format_hypothesis(h) for h in sorted(members, key=lambda x: x["id"]))
        clusters_text.append(f"CLUSTER {cid}:\n\n{members_text}\n")
        
    clusters_block = "\n================\n".join(clusters_text)
    
    # 3. Check for overflow
    # A rough estimate (4 chars / token)
    if len(clusters_block) + len(target_text) > _MAX_INPUT_TOKENS * 4:
        res = DedupResult(new_cid, unchecked=True, rationale="context overflow")
        ledger.apply("cluster_assign", {
            "hypothesis_id": target_id,
            "for_seq": for_seq,
            "cluster_id": res.cluster_id,
            "method": "judge_failed",
            "unchecked": res.unchecked
        }, actor="scaffold")
        return res

    # 4. LLM Judge
    sys_text, sys_sha = load_prompt("dedup_system")
    
    msg = f"EXISTING CLUSTERS:\n{clusters_block}\n\nTARGET HYPOTHESIS:\n{target_text}"
    
    schema = {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "decision": {
                "type": "string",
                "enum": ["new"] + list(clusters.keys()),
                "description": "The ID of the matching cluster, or 'new' if distinct/unsure/matches multiple"
            }
        },
        "required": ["rationale", "decision"],
        "additionalProperties": False
    }
    
    
    llm_call_ids = []
    
    try:
        resp, call_id = client.complete(
            "dedup",
            [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": msg}
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "dedup", "schema": schema, "strict": True}},
            prompt_id="dedup_system",
            prompt_sha=sys_sha,
        )
        if call_id:
            llm_call_ids.append(call_id)
        
        # Check invalid output, try once
        decision_data = None
        try:
            decision_data = json.loads(resp.text or "{}")
            decision = decision_data.get("decision")
            if decision not in ["new"] + list(clusters.keys()):
                raise ValueError("Invalid decision")
        except Exception:
            # Re-ask once
            resp, call_id2 = client.complete(
                "dedup",
                [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": resp.text or ""},
                    {"role": "user", "content": "Please reply with a valid JSON matching the schema."}
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "dedup", "schema": schema, "strict": True}},
                prompt_id="dedup_system",
                prompt_sha=sys_sha,
            )
            if call_id2:
                llm_call_ids.append(call_id2)
            decision_data = json.loads(resp.text or "{}")
            decision = decision_data.get("decision")
            
        if decision == "new" or decision not in clusters:
            res = DedupResult(new_cid, unchecked=False, rationale=decision_data.get("rationale", ""))
        else:
            res = DedupResult(decision, unchecked=False, rationale=decision_data.get("rationale", ""))
            
    except Exception as exc:
        # Provider error or invalid output after retry
        res = DedupResult(new_cid, unchecked=True, rationale=f"Judge failed: {exc}")

    # 5. Apply event
    ledger.apply("cluster_assign", {
        "hypothesis_id": target_id,
        "for_seq": for_seq,
        "cluster_id": res.cluster_id,
        "method": "judge_failed" if res.unchecked else "llm_judge",
        "unchecked": res.unchecked,
        **({"llm_call_ids": llm_call_ids} if llm_call_ids else {}),
        **({"rationale": res.rationale} if res.rationale else {})
    }, actor="scaffold")
    
    return res

def on_hypothesis_write(ledger: Ledger, client: LLMClient, event_seq: int) -> None:
    """Callback for dispatcher: run dedup if the event creates/edits a hypothesis."""
    # We need the event to get the target id.
    # The dispatcher passes the sequence number of the event that just committed.
    ev = ledger.con.execute("SELECT op, payload FROM ledger_events WHERE seq=?", [event_seq]).fetchone()
    if not ev:
        return
        
    ev_type, payload_str = ev
    payload = json.loads(payload_str)
    
    if ev_type == "hypothesis_create":
        assign_cluster(ledger, client, payload["hypothesis_id"], event_seq)
    elif ev_type == "hypothesis_revise":
        # Only re-judge if statement or branches changed.
        # Check if the edit payload includes them
        if "statement" in payload or "branches" in payload:
            assign_cluster(ledger, client, payload["hypothesis_id"], event_seq)
