from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, List, Optional, Dict, Union

from pydantic import BaseModel, Field

OriginType = Literal["lead", "human", "critic", "scaffold"]
HypothesisStatus = Literal["live", "refuted", "accepted", "unresolved"]
Stance = Literal["supports", "contradicts"]
Role = Literal["observation", "contrast"]
EvidenceKind = Literal["occurrence", "claimed_content", "derived"]
Actor = Literal["lead", "critic", "reader", "human", "scaffold"]


class TimeRange(BaseModel):
    start_s: float
    end_s: float


class RowRef(BaseModel):
    row: int
    values: dict[str, Any]


class Evidence(BaseModel):
    evidence_id: str
    claim: str
    query_id: str
    row_refs: list[RowRef]
    time_range: TimeRange | None = None
    evidence_kind: EvidenceKind
    rows_snapshot: list[dict[str, Any]] = Field(default_factory=list)
    coverage_ids: list[str] = Field(default_factory=list)
    guard: Literal["ok", "unverified_scope"] = "ok"
    run_id: str
    seq: int
    actor: Actor
    step: str


class EvidenceLink(BaseModel):
    evidence_id: str
    stance: Stance


class Prediction(BaseModel):
    prediction_id: str
    text: str
    status: Literal["open", "confirmed", "disconfirmed"] = "open"
    outcome_evidence_ids: list[str] = Field(default_factory=list)


class StatusBasis(BaseModel):
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str
    same_as: str | None = None


class LinkEvidence(BaseModel):
    evidence_id: str
    role: Role


class CauseEffectAt(BaseModel):
    evidence_id: str
    row: int
    ts_s: float
    host: str | None = None


class ChainLink(BaseModel):
    link_id: str
    cause: str
    effect: str
    evidence: list[LinkEvidence] = Field(default_factory=list)
    cause_at: CauseEffectAt
    effect_at: CauseEffectAt


class Branch(BaseModel):
    links: list[str]
    joins: str | None = None


class ContributingFactor(BaseModel):
    statement: str
    evidence: list[LinkEvidence] = Field(default_factory=list)
    acts_on: str | None = None


class Hypothesis(BaseModel):
    hypothesis_id: str
    statement: str
    origin: OriginType
    status: HypothesisStatus = "live"
    branches: list[Branch] = Field(default_factory=list)
    contributing_factors: list[ContributingFactor] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)
    predictions: list[Prediction] = Field(default_factory=list)
    cluster_id: str
    cluster_status: Literal["assigned", "pending", "unchecked"] = "unchecked"
    status_basis: StatusBasis | None = None
    run_id: str
    seq: int
    actor: Actor
    step: str


class Resolution(BaseModel):
    by_query_id: str
    note: str


class Failure(BaseModel):
    failure_id: str
    query_id: str
    tool: str
    error_type: str
    diagnostic: str
    retried: bool = False
    resolution: Resolution | None = None
    run_id: str
    seq: int
    actor: Actor
    step: str


class Answer(BaseModel):
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class Question(BaseModel):
    question_id: str
    text: str
    status: Literal["open", "answered"] = "open"
    answer: Answer | None = None
    run_id: str
    seq: int
    actor: Actor
    step: str


class ObjectionResolution(BaseModel):
    round: int
    kind: Literal["evidence", "revision", "concession"]
    evidence_ids: list[str] = Field(default_factory=list)
    note: str


class ObjectionTarget(BaseModel):
    hypothesis_id: str
    link_id: str | None = None


class Objection(BaseModel):
    objection_id: str
    round: int
    text: str
    target: ObjectionTarget
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["open", "resolved", "conceded"] = "open"
    resolutions: list[ObjectionResolution] = Field(default_factory=list)
    run_id: str
    seq: int
    actor: Actor
    step: str


class Focus(BaseModel):
    focus_id: str
    window: str
    entities: list[str] | None = None
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    run_id: str
    seq: int
    actor: Actor
    step: str


class LedgerEvent(BaseModel):
    run_id: str
    seq: int
    event_id: str
    created_at: str
    actor: Actor
    step: str
    op: str
    record_type: str
    record_id: str
    payload: dict[str, Any]


class LedgerState(BaseModel):
    run_id: str
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    hypotheses: dict[str, Hypothesis] = Field(default_factory=dict)
    failures: dict[str, Failure] = Field(default_factory=dict)
    questions: dict[str, Question] = Field(default_factory=dict)
    objections: dict[str, Objection] = Field(default_factory=dict)
    focus: Focus | None = None

    def fold(self, events: list[LedgerEvent]) -> "LedgerState":
        for ev in events:
            payload = ev.payload.copy()
            payload["run_id"] = ev.run_id
            payload["seq"] = ev.seq
            payload["actor"] = ev.actor
            payload["step"] = ev.step
            
            op = ev.op
            rec_id = ev.record_id
            
            if op == "evidence" or op == "ledger_record_evidence":
                self.evidence[rec_id] = Evidence(**payload)
            
            elif op == "hypothesis_create" or op == "ledger_hypothesis_create":
                self.hypotheses[rec_id] = Hypothesis(**payload)
            
            elif op == "hypothesis_update" or op == "ledger_hypothesis_update":
                if rec_id in self.hypotheses:
                    hyp = self.hypotheses[rec_id]
                    # Update fields based on payload
                    update_data = payload.copy()
                    # Ensure origin is not overwritten
                    if "origin" in update_data:
                        del update_data["origin"]
                    dump = hyp.model_dump()
                    dump.update(update_data)
                    updated = Hypothesis.model_validate(dump)
                    self.hypotheses[rec_id] = updated
            
            elif op == "hypothesis_status" or op == "ledger_hypothesis_status":
                if rec_id in self.hypotheses:
                    hyp = self.hypotheses[rec_id]
                    update_data = payload.copy()
                    if "status_basis" in update_data and update_data["status_basis"]:
                        update_data["status_basis"] = StatusBasis(**update_data["status_basis"])
                    dump = hyp.model_dump()
                    dump.update(update_data)
                    updated = Hypothesis.model_validate(dump)
                    self.hypotheses[rec_id] = updated
                    
            elif op == "cluster_assign":
                if rec_id in self.hypotheses:
                    hyp = self.hypotheses[rec_id]
                    dump = hyp.model_dump()
                    dump.update({"cluster_id": payload["cluster_id"], "cluster_status": "assigned"})
                    updated = Hypothesis.model_validate(dump)
                    self.hypotheses[rec_id] = updated

            elif op == "failure_record":
                self.failures[rec_id] = Failure(**payload)

            elif op == "failure_resolve":
                if rec_id in self.failures:
                    fail = self.failures[rec_id]
                    dump = fail.model_dump()
                    dump.update({"resolution": Resolution(**payload["resolution"])})
                    updated = Failure.model_validate(dump)
                    self.failures[rec_id] = updated

            elif op == "question":
                self.questions[rec_id] = Question(**payload)
                
            elif op == "question_answer":
                if rec_id in self.questions:
                    q = self.questions[rec_id]
                    dump = q.model_dump()
                    dump.update({"status": "answered", "answer": Answer(**payload["answer"])})
                    updated = Question.model_validate(dump)
                    self.questions[rec_id] = updated

            elif op == "objection_raise":
                self.objections[rec_id] = Objection(**payload)

            elif op == "objection_resolve":
                if rec_id in self.objections:
                    obj = self.objections[rec_id]
                    resolutions = obj.resolutions.copy()
                    resolutions.append(ObjectionResolution(**payload["resolution"]))
                    status = payload.get("status", obj.status)
                    dump = obj.model_dump()
                    dump.update({"resolutions": [r.model_dump() for r in resolutions], "status": status})
                    updated = Objection.model_validate(dump)
                    self.objections[rec_id] = updated

            elif op == "focus":
                self.focus = Focus(**payload)

            elif op == "focus_clear":
                self.focus = None

        return self
