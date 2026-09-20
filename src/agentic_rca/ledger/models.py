"""Model-facing payload schemas for ledger ops (contracts-b1 §4).

Every model is `extra="forbid"`: a field the model isn't allowed to set
(`origin`, ids, `guard`, `rows_snapshot`, ...) is rejected rather than
silently dropped, so the error tells the model what it did wrong.
Fields filled by the scaffold are simply absent from these schemas.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- evidence ---------------------------------------------------------------


class RowRef(_Strict):
    row: int = Field(ge=0, description="`_row` ordinal of the cited row in the query's result")
    values: dict[str, Any] = Field(description="the cited subset of that row's columns")


class TimeRange(_Strict):
    start_s: float
    end_s: float


class EvidenceIn(_Strict):
    claim: str = Field(min_length=1)
    query_id: str
    row_refs: list[RowRef] = Field(default_factory=list)
    time_range: TimeRange | None = None
    evidence_kind: Literal["occurrence", "claimed_content", "derived"]


# ---- hypotheses -------------------------------------------------------------


class EvidenceRole(_Strict):
    evidence_id: str
    role: Literal["observation", "contrast"] = "observation"


class Anchor(_Strict):
    evidence_id: str
    row: int = Field(ge=0)
    ts_s: float
    host: str | None = None


class LinkIn(_Strict):
    cause: str = Field(min_length=1)
    effect: str = Field(min_length=1)
    evidence: list[EvidenceRole] = Field(min_length=1)
    cause_at: Anchor | None = None
    effect_at: Anchor | None = None


class JoinRef(_Strict):
    """Where a branch's last effect feeds in: link `link` (0-based) of branch
    `branch` (0-based) in the same submitted chain."""

    branch: int = Field(ge=0)
    link: int = Field(ge=0)


class BranchIn(_Strict):
    links: list[LinkIn] = Field(min_length=1)
    joins: JoinRef | None = None


class FactorIn(_Strict):
    statement: str = Field(min_length=1)
    evidence: list[EvidenceRole] = Field(min_length=1)
    acts_on: str | None = Field(default=None, description="a link id it worsened, if any")


class HypothesisCreateIn(_Strict):
    statement: str = Field(min_length=1)
    branches: list[BranchIn] = Field(default_factory=list)
    contributing_factors: list[FactorIn] = Field(default_factory=list)


class HumanHypothesisIn(_Strict):
    """A human hunch: a statement and nothing else (§4.1). It can't carry a
    window, entities, budget or tool -- the schema has no field for them."""

    statement: str = Field(min_length=1)


class HypothesisReviseIn(_Strict):
    hypothesis_id: str
    statement: str | None = None
    branches: list[BranchIn] | None = None
    contributing_factors: list[FactorIn] | None = None

    @model_validator(mode="after")
    def _something(self) -> HypothesisReviseIn:
        if self.statement is None and self.branches is None and self.contributing_factors is None:
            raise ValueError(
                "revise needs at least one of statement, branches, contributing_factors"
            )
        return self


class LinkEvidenceIn(_Strict):
    hypothesis_id: str
    evidence_id: str
    stance: Literal["supports", "contradicts"]


class WithdrawEvidenceIn(_Strict):
    hypothesis_id: str
    evidence_id: str
    reason: str = Field(min_length=1)


class StatusBasis(_Strict):
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    same_as: str | None = None


class SetStatusIn(_Strict):
    hypothesis_id: str
    status: Literal["live", "refuted", "accepted", "unresolved"]
    basis: StatusBasis


class PredictionAddIn(_Strict):
    hypothesis_id: str
    text: str = Field(min_length=1)


class PredictionResolveIn(_Strict):
    hypothesis_id: str
    prediction_id: str
    status: Literal["confirmed", "disconfirmed"]
    outcome_evidence_ids: list[str] = Field(min_length=1)


# ---- questions, objections, focus, failures -----------------------------------


class QuestionOpenIn(_Strict):
    text: str = Field(min_length=1)


class QuestionAnswerIn(_Strict):
    question_id: str
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class ObjectionTarget(_Strict):
    hypothesis_id: str
    link_id: str | None = None


class ObjectionRaiseIn(_Strict):
    text: str = Field(min_length=1)
    target: ObjectionTarget
    evidence_ids: list[str] = Field(default_factory=list)


class ObjectionReopenIn(_Strict):
    objection_id: str
    note: str = Field(min_length=1)


class ObjectionResolveIn(_Strict):
    objection_id: str
    kind: Literal["evidence", "revision", "concession"]
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = Field(min_length=1)


class FocusSetIn(_Strict):
    window: TimeRange
    entities: dict[str, list[str]] | None = None
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class FailureResolveIn(_Strict):
    failure_id: str
    by_query_id: str
    note: str = Field(min_length=1)


# ---- conclusion (design-d6-d7 §3) ---------------------------------------------


class Symptom(_Strict):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    explained_by: str | None = None


class OrderingNote(_Strict):
    link_id: str
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class ConfidenceBasis(_Strict):
    strong: list[int] = Field(default_factory=list)
    text: str = Field(min_length=1)


class Discriminator(_Strict):
    between: list[str] = Field(min_length=2)
    evidence_needed: str = Field(min_length=1)


class ConclusionIn(_Strict):
    kind: Literal["conclusive", "inconclusive"]
    accepted_hypothesis_id: str | None = None
    symptoms: list[Symptom] = Field(min_length=1)
    ordering_notes: list[OrderingNote] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] | None = None
    confidence_basis: ConfidenceBasis | None = None
    sufficiency_note: str | None = None
    discriminators: list[Discriminator] = Field(default_factory=list)

    @model_validator(mode="after")
    def _nullability(self) -> ConclusionIn:
        conclusive = self.kind == "conclusive"
        if conclusive != (self.accepted_hypothesis_id is not None):
            raise ValueError("accepted_hypothesis_id is required iff kind is conclusive")
        if conclusive != (self.confidence is not None):
            raise ValueError(
                "confidence is required iff kind is conclusive (null when inconclusive)"
            )
        if conclusive and (self.confidence_basis is None or not self.sufficiency_note):
            raise ValueError("a conclusive conclusion needs confidence_basis and sufficiency_note")
        if conclusive and self.discriminators:
            raise ValueError("discriminators are for inconclusive conclusions only")
        if self.confidence_basis and any(not 1 <= c <= 11 for c in self.confidence_basis.strong):
            raise ValueError("confidence_basis.strong lists criterion numbers 1-11")
        return self


# ---- op registry: which payload model and which actors (contracts-b1 I8) -------

LEAD, CRITIC, HUMAN, SCAFFOLD = "lead", "critic", "human", "scaffold"

OPS: dict[str, tuple[type[BaseModel] | None, frozenset[str]]] = {
    "evidence_record": (EvidenceIn, frozenset({LEAD, CRITIC})),
    "hypothesis_create": (HypothesisCreateIn, frozenset({LEAD, CRITIC})),
    "hypothesis_create_human": (HumanHypothesisIn, frozenset({HUMAN})),
    "hypothesis_revise": (HypothesisReviseIn, frozenset({LEAD})),
    "hypothesis_link_evidence": (LinkEvidenceIn, frozenset({LEAD})),
    "hypothesis_withdraw_evidence": (WithdrawEvidenceIn, frozenset({LEAD})),
    "hypothesis_set_status": (SetStatusIn, frozenset({LEAD})),
    "prediction_add": (PredictionAddIn, frozenset({LEAD})),
    "prediction_resolve": (PredictionResolveIn, frozenset({LEAD})),
    "question_open": (QuestionOpenIn, frozenset({LEAD})),
    "question_answer": (QuestionAnswerIn, frozenset({LEAD})),
    "objection_raise": (ObjectionRaiseIn, frozenset({CRITIC})),
    "objection_reopen": (ObjectionReopenIn, frozenset({CRITIC})),
    "objection_resolve": (ObjectionResolveIn, frozenset({LEAD})),
    "focus_set": (FocusSetIn, frozenset({LEAD})),
    "focus_clear": (None, frozenset({LEAD})),
    "failure_resolve": (FailureResolveIn, frozenset({LEAD})),
    "conclusion_propose": (ConclusionIn, frozenset({LEAD})),
    # scaffold-only ops take trusted dict payloads; no model schema
    "standing_seed": (None, frozenset({SCAFFOLD})),
    "failure_record": (None, frozenset({SCAFFOLD})),
    "cluster_assign": (None, frozenset({SCAFFOLD})),
    "evidence_verdict": (None, frozenset({SCAFFOLD})),
    "link_entailment": (None, frozenset({SCAFFOLD})),
    "done_check_result": (None, frozenset({SCAFFOLD})),
    "run_outcome": (None, frozenset({SCAFFOLD})),
}

# ops the lead-facing action table exposes (no scaffold op, no coverage write)
MODEL_OPS = [n for n, (m, actors) in OPS.items() if m is not None and actors & {LEAD, CRITIC}]
