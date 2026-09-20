"""Stage 4, layer 2: the entailment check (§4.1).

A small-context LLM call per chain link receives only the claim and its cited data
(no narrative) and answers whether the data supports the claim. Isolating the link
from the story prevents the narrative from carrying the judgement.
"""

from __future__ import annotations

import json

from agentic_rca.ledger import Ledger
from agentic_rca.llm.base import LLMClient
from agentic_rca.tools.registry import ToolRuntime


def verify(ledger: Ledger, runtime: ToolRuntime, llm: LLMClient) -> dict:
    """Run the entailment check for every link in the active folded state
    whose evidence successfully passed the deterministic verifier.
    Writes one `link_entailment` event per evaluated link.
    Returns counts and the per-link verdicts.
    """
    st = ledger.state()
    verdicts = []

    for lid, link in st.links.items():
        if link.get("retired"):
            continue

        # Check that ALL evidence for this link passed deterministic verification.
        evidence_passed = True
        evidence_items = link.get("evidence", [])
        if not evidence_items:
            continue

        for e_item in evidence_items:
            eid = e_item["evidence_id"]
            v = st.verdicts.get(eid)
            if not v or v.get("verdict") != "pass":
                evidence_passed = False
                break
        
        if not evidence_passed:
            continue

        # Gather data
        data_blocks = []
        for e_item in evidence_items:
            eid = e_item["evidence_id"]
            e = st.evidence[eid]
            # If row_refs exist, use them. If empty, the evidence is an empty result set.
            if e.get("row_refs"):
                rows = [ref.get("values", {}) for ref in e["row_refs"]]
                data_blocks.append(json.dumps(rows))
            else:
                data_blocks.append("[]")

        claim = f"{link.get('cause', '')} -> {link.get('effect', '')}"
        data_str = "\n".join(data_blocks)

        prompt = f"""Claim: {claim}

Data:
{data_str}

Does the data support the claim? Answer exactly YES or NO on the first line, followed by a brief rationale."""

        messages = [
            {"role": "system", "content": "You evaluate if data supports a claim. Be strict."},
            {"role": "user", "content": prompt}
        ]
        
        response, _ = llm.complete("critic", messages)
        
        lines = response.text.strip().split("\n")
        first_line = lines[0].upper()
        verdict = "pass" if "YES" in first_line else "fail"
        rationale = "\n".join(lines[1:]).strip() if len(lines) > 1 else first_line

        payload = {
            "link_id": lid,
            "verdict": verdict,
            "rationale": rationale
        }
        ledger.apply("link_entailment", payload, actor="scaffold")
        verdicts.append(payload)

    counts = {
        k: sum(1 for v in verdicts if v["verdict"] == k) for k in ("pass", "fail")
    }
    return {"counts": counts, "verdicts": verdicts}
