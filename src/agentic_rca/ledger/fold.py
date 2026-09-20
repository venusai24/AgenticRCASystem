"""Current state = fold(events) (contracts-b1 §3).

Pure and deterministic: ids are minted by the op layer *before* the event
is written, so folding only applies payloads -- replay never needs the
dedup judge, the model, or the clock.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

STANDING_STATEMENT = (
    "The cause lies in a component or condition not observed in the available telemetry."
)


@dataclass
class Event:
    run_id: str
    seq: int
    actor: str
    step: int
    op: str
    record_type: str | None
    record_id: str | None
    payload: dict[str, Any]


@dataclass
class LedgerState:
    run_id: str
    evidence: dict[str, dict] = field(default_factory=dict)
    hypotheses: dict[str, dict] = field(default_factory=dict)
    links: dict[str, dict] = field(default_factory=dict)  # every link ever minted
    factors: dict[str, dict] = field(default_factory=dict)
    failures: dict[str, dict] = field(default_factory=dict)
    questions: dict[str, dict] = field(default_factory=dict)
    objections: dict[str, dict] = field(default_factory=dict)
    focus: dict | None = None
    verdicts: dict[str, dict] = field(default_factory=dict)  # evidence_id -> latest verdict
    entailments: dict[str, dict] = field(default_factory=dict)  # link_id -> latest
    conclusions: list[dict] = field(default_factory=list)
    done_checks: list[dict] = field(default_factory=list)
    outcome: dict | None = None
    counters: dict[str, int] = field(default_factory=dict)  # (run_id, prefix) -> max used
    last_seq: int = 0

    # ---- helpers -----------------------------------------------------------
    def standing_id(self) -> str | None:
        for hid, h in self.hypotheses.items():
            if h["origin"] == "scaffold":
                return hid
        return None

    def accepted(self) -> list[str]:
        return [hid for hid, h in self.hypotheses.items() if h["status"] == "accepted"]

    def latest_conclusion(self) -> dict | None:
        return self.conclusions[-1] if self.conclusions else None


_NUM = re.compile(r"^(?P<prefix>[a-z]+)(?P<n>\d+)$")


def _bump(state: LedgerState, qid: str) -> None:
    run_id, _, short = qid.partition("/")
    base = short.split(".")[-1] if "." in short else short
    m = _NUM.match(base)
    if not m:
        return
    prefix = m["prefix"] if "." not in short else short.split(".")[0] + "." + m["prefix"]
    key = f"{run_id}|{prefix}"
    state.counters[key] = max(state.counters.get(key, 0), int(m["n"]))


def _apply(state: LedgerState, e: Event) -> None:
    p = e.payload
    op = e.op
    if op == "evidence_record":
        state.evidence[p["evidence_id"]] = dict(p, actor=e.actor, seq=e.seq)
        _bump(state, p["evidence_id"])
    elif op in ("hypothesis_create", "hypothesis_create_human", "standing_seed"):
        hid = p["hypothesis_id"]
        h = {
            "id": hid,
            "statement": p["statement"],
            "origin": p["origin"],
            "status": "live",
            "branches": p.get("branches", []),
            "factor_ids": [f["id"] for f in p.get("factors", [])],
            "evidence": [],
            "predictions": [],
            "cluster_id": hid,
            "cluster_status": "pending",
            "cluster_method": None,
            "content_seq": e.seq,
            "status_basis": None,
            "status_history": [],
            "created_seq": e.seq,
            "revision": 0,
        }
        state.hypotheses[hid] = h
        _bump(state, hid)
        for link in p.get("links", []):
            state.links[link["id"]] = dict(link, hypothesis_id=hid, retired=False)
            _bump(state, link["id"])
        for f in p.get("factors", []):
            state.factors[f["id"]] = dict(f, hypothesis_id=hid, retired=False)
            _bump(state, f["id"])
    elif op == "hypothesis_revise":
        h = state.hypotheses[p["hypothesis_id"]]
        h["revision"] += 1
        if p.get("statement") is not None:
            h["statement"] = p["statement"]
        if "branches" in p:
            for lid in [lid for b in h["branches"] for lid in b["links"]]:
                state.links[lid]["retired"] = True
            h["branches"] = p["branches"]
            for link in p.get("links", []):
                state.links[link["id"]] = dict(link, hypothesis_id=h["id"], retired=False)
                _bump(state, link["id"])
        if "factors" in p:
            for fid in h["factor_ids"]:
                state.factors[fid]["retired"] = True
            h["factor_ids"] = [f["id"] for f in p["factors"]]
            for f in p["factors"]:
                state.factors[f["id"]] = dict(f, hypothesis_id=h["id"], retired=False)
                _bump(state, f["id"])
        if p.get("content_changed"):
            h["content_seq"] = e.seq
            h["cluster_status"] = "pending"
    elif op == "hypothesis_link_evidence":
        h = state.hypotheses[p["hypothesis_id"]]
        h["evidence"].append(
            {"evidence_id": p["evidence_id"], "stance": p["stance"], "withdrawn": None}
        )
    elif op == "hypothesis_withdraw_evidence":
        h = state.hypotheses[p["hypothesis_id"]]
        for link in h["evidence"]:
            if link["evidence_id"] == p["evidence_id"] and link["withdrawn"] is None:
                link["withdrawn"] = p["reason"]
    elif op == "hypothesis_set_status":
        h = state.hypotheses[p["hypothesis_id"]]
        h["status_history"].append({"from": h["status"], "to": p["status"], "seq": e.seq})
        h["status"] = p["status"]
        h["status_basis"] = None if p["status"] == "live" else p["basis"]
    elif op == "prediction_add":
        h = state.hypotheses[p["hypothesis_id"]]
        h["predictions"].append(
            {
                "prediction_id": p["prediction_id"],
                "text": p["text"],
                "status": "open",
                "outcome_evidence_ids": [],
            }
        )
    elif op == "prediction_resolve":
        h = state.hypotheses[p["hypothesis_id"]]
        for pr in h["predictions"]:
            if pr["prediction_id"] == p["prediction_id"]:
                pr["status"] = p["status"]
                pr["outcome_evidence_ids"] = p["outcome_evidence_ids"]
    elif op == "question_open":
        state.questions[p["question_id"]] = {
            "id": p["question_id"],
            "text": p["text"],
            "status": "open",
            "answer": None,
        }
        _bump(state, p["question_id"])
    elif op == "question_answer":
        q = state.questions[p["question_id"]]
        q["status"] = "answered"
        q["answer"] = {"text": p["text"], "evidence_ids": p["evidence_ids"]}
    elif op == "objection_raise":
        state.objections[p["objection_id"]] = {
            "id": p["objection_id"],
            "round": p["round"],
            "text": p["text"],
            "target": p["target"],
            "evidence_ids": p["evidence_ids"],
            "status": "open",
            "resolutions": [],
        }
        _bump(state, p["objection_id"])
    elif op == "objection_reopen":
        o = state.objections[p["objection_id"]]
        o["status"] = "open"
        o["resolutions"].append({"kind": "reopen", "note": p["note"], "round": p.get("round")})
    elif op == "objection_resolve":
        o = state.objections[p["objection_id"]]
        o["status"] = "conceded" if p["kind"] == "concession" else "resolved"
        o["resolutions"].append(
            {
                "round": p.get("round"),
                "kind": p["kind"],
                "evidence_ids": p["evidence_ids"],
                "note": p["note"],
            }
        )
    elif op == "focus_set":
        state.focus = dict(p)
        _bump(state, p["focus_id"])
    elif op == "focus_clear":
        state.focus = None
    elif op == "failure_record":
        state.failures[p["failure_id"]] = dict(p, resolution=None)
        _bump(state, p["failure_id"])
    elif op == "failure_resolve":
        state.failures[p["failure_id"]]["resolution"] = {
            "by_query_id": p["by_query_id"],
            "note": p["note"],
        }
    elif op == "cluster_assign":
        h = state.hypotheses[p["hypothesis_id"]]
        _bump(state, p["cluster_id"])
        if p["for_seq"] == h["content_seq"]:
            h["cluster_id"] = p["cluster_id"]
            h["cluster_method"] = p["method"]
            h["cluster_status"] = "unchecked" if p["method"] == "judge_failed" else "assigned"
    elif op == "evidence_verdict":
        state.verdicts[p["evidence_id"]] = dict(p)
    elif op == "link_entailment":
        state.entailments[p["link_id"]] = dict(p)
    elif op == "conclusion_propose":
        state.conclusions.append(dict(p, seq=e.seq))
    elif op == "done_check_result":
        state.done_checks.append(dict(p, seq=e.seq))
    elif op == "run_outcome":
        state.outcome = dict(p, seq=e.seq)
    else:  # pragma: no cover - ops are validated before they are written
        raise ValueError(f"unknown op in event log: {op}")


def fold(events: list[Event], run_id: str) -> LedgerState:
    """Apply events in (lineage order, seq) order. The caller passes parent
    events first (oldest ancestor first), then this run's."""
    state = LedgerState(run_id=run_id)
    for e in events:
        _apply(state, copy.deepcopy(e))
        if e.run_id == run_id:
            state.last_seq = max(state.last_seq, e.seq)
    return state


def two_live_check(state: LedgerState) -> tuple[bool, dict]:
    """contracts-b1 §5 as amended by design-d3-d4 §8: ok iff non-scaffold
    hypotheses span >= 2 distinct clusters in total (standing + eliminated),
    excluding any cluster that contains the standing hypothesis."""
    standing = state.standing_id()
    excluded = {state.hypotheses[standing]["cluster_id"]} if standing else set()
    clusters: dict[str, list[dict]] = {}
    for h in state.hypotheses.values():
        if h["origin"] == "scaffold" or h["cluster_id"] in excluded:
            continue
        clusters.setdefault(h["cluster_id"], []).append(h)
    eliminated = [c for c, hs in clusters.items() if all(h["status"] == "refuted" for h in hs)]
    detail = {
        "clusters_total": len(clusters),
        "clusters_standing": len(clusters) - len(eliminated),
        "clusters_eliminated": len(eliminated),
        "pending": sorted(
            h["id"] for h in state.hypotheses.values() if h["cluster_status"] == "pending"
        ),
        "unchecked": sorted(
            h["id"] for h in state.hypotheses.values() if h["cluster_status"] == "unchecked"
        ),
    }
    return len(clusters) >= 2, detail
