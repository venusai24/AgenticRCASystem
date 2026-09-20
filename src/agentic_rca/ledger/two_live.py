from agentic_rca.ledger.schemas import LedgerState

def two_live_check(state: LedgerState) -> tuple[bool, dict]:
    """
    Evaluates the >=2 live hypotheses rule.
    - A cluster is eliminated if every member is refuted.
    - A cluster is standing if any member is live, accepted, or unresolved.
    - Returns ok=True if non-scaffold hypotheses span >= 2 distinct cluster_ids (standing + eliminated)
      excluding any cluster that contains the standing hypothesis (origin: scaffold).
    """
    clusters = {}
    scaffold_clusters = set()
    pending_unchecked = []
    
    for h in state.hypotheses.values():
        if h.origin == "scaffold":
            scaffold_clusters.add(h.cluster_id)
            
        if h.cluster_status in ("pending", "unchecked"):
            pending_unchecked.append(h.hypothesis_id)
            
        if h.cluster_id not in clusters:
            clusters[h.cluster_id] = []
        clusters[h.cluster_id].append(h)
        
    standing_clusters = 0
    eliminated_clusters = 0
    
    for cid, members in clusters.items():
        if cid in scaffold_clusters:
            continue
            
        is_eliminated = all(m.status == "refuted" for m in members)
        if is_eliminated:
            eliminated_clusters += 1
        else:
            standing_clusters += 1
            
    total_valid_clusters = standing_clusters + eliminated_clusters
    ok = total_valid_clusters >= 2
    
    detail = {
        "standing_clusters": standing_clusters,
        "eliminated_clusters": eliminated_clusters,
        "pending_or_unchecked_hypotheses": pending_unchecked
    }
    
    return ok, detail
