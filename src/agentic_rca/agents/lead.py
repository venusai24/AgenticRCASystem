"""The lead investigator (design Stage 2): an AgentLoop whose action table
is the registered data tools + paging/delegation/coverage + the lead's
ledger ops + budget extension + termination.

Coverage is not among the actions: it is written only by the registry.
"""

from __future__ import annotations

from collections.abc import Callable

from agentic_rca.agents.loop import (
    Action,
    AgentLoop,
    Budget,
    LoopConfig,
    coverage_action,
    data_tool_action,
    inspect_action,
    ledger_action,
    ledger_view_action,
)
from agentic_rca.ledger import two_live_check
from agentic_rca.ledger.models import ConclusionIn
import agentic_rca.tools.catalog  # noqa: F401 - registers every data tool
from agentic_rca.tools.registry import TOOLS, tool_schemas

LEDGER_OP_DOCS = {
    "evidence_record": "Record evidence: a claim, the query_id that produced it, and the exact rows cited (row = the `_row` ordinal, values = the cited columns). evidence_kind: occurrence (a thing happened / was measured), claimed_content (what a log line says), derived (computed in python_sandbox).",
    "hypothesis_create": "Propose a hypothesis: a statement and, when you have evidence, its causal chain as branches of ordered links (cause -> effect, each citing evidence; anchor cause_at/effect_at to cited rows with their timestamp and host). Contributing factors worsened but did not initiate.",
    "hypothesis_revise": "Revise a hypothesis's statement, chain or contributing factors (a revised chain replaces the old one).",
    "hypothesis_link_evidence": "Attach evidence to a hypothesis as supporting or contradicting.",
    "hypothesis_withdraw_evidence": "Withdraw an evidence link from a hypothesis, with a reason.",
    "hypothesis_set_status": "Set status: refuted (needs linked contradicting evidence), accepted (at most one), unresolved (carried into the report), or live (reopen). A reason is always required.",
    "prediction_add": "Record a prediction a hypothesis makes, before testing it.",
    "prediction_resolve": "Resolve a prediction as confirmed or disconfirmed, citing evidence.",
    "question_open": "Record an open question.",
    "question_answer": "Answer an open question, citing evidence.",
    "objection_resolve": "Resolve a critic objection by evidence, revision, or concession.",
    "focus_set": "Note the window/entities you are focusing on and why (a note, not a filter).",
    "focus_clear": "Abandon the current focus.",
    "failure_resolve": "Mark a recorded tool failure as resolved by a later successful query.",
}


def default_precheck(loop: AgentLoop) -> Callable[[str], dict]:
    """Stand-in until the done-check is wired in: only the >=2-live gate."""

    def check(phase: str) -> dict:
        ok, detail = two_live_check(loop.ledger.state())
        unmet = [] if ok else [{"criterion": "2-live", "code": "fewer_than_two_clusters", "message": str(detail)}]
        return {"met": ok, "unmet": unmet, "phase": phase}

    return check


def build_lead(
    *,
    ledger,
    runtime,
    llm,
    header: Callable[[], str],
    precheck: Callable[[str], dict] | None = None,
    reader=None,
    on_hypothesis_write: Callable[[str, str], None] | None = None,
    budget: Budget | None = None,
    config: LoopConfig | None = None,
    overseer=None,
    tool_names: list[str] | None = None,
) -> AgentLoop:
    loop = AgentLoop(
        actor="lead", ledger=ledger, runtime=runtime, llm=llm, prompt_name="lead_system",
        header=header, budget=budget, config=config, overseer=overseer,
    )
    for spec in tool_schemas(tool_names or sorted(TOOLS)):
        loop.add(data_tool_action(loop, spec["name"], spec["description"], spec["parameters"]))
    loop.add(inspect_action(loop))
    loop.add(coverage_action(loop))
    loop.add(ledger_view_action(loop))
    for op, doc in LEDGER_OP_DOCS.items():
        loop.add(ledger_action(loop, op, doc, on_hypothesis_write))
    check = precheck or default_precheck(loop)

    def ask_reader(a: dict):
        if reader is None:
            return {"error": {"type": "unavailable", "diagnostic": "no reader is configured in this run"}}, None
        return reader.ask(a.get("handle", ""), a.get("question", ""), int(a.get("max_claims", 10)), step=loop.step_no), None

    loop.add(Action(
        "ask_reader",
        "Delegate a large result handle to a stateless reader: it answers one question, citing row ids that are checked before you see them.",
        {"type": "object", "properties": {"handle": {"type": "string"}, "question": {"type": "string"},
                                          "max_claims": {"type": "integer", "default": 10}},
         "required": ["handle", "question"]},
        ask_reader,
    ))

    def extend(a: dict):
        b = loop.budget
        hids = a.get("hypothesis_ids") or []
        st = loop.ledger.state()
        try:
            hyps = [st.hypotheses[loop.ledger.q(h)] for h in hids]
        except KeyError:
            return {"granted": False, "reason": "unknown hypothesis id"}, None
        live_clusters = {h["cluster_id"] for h in hyps if h["status"] == "live" and h["origin"] != "scaffold"}
        if b.extensions >= b.max_extensions:
            return {"granted": False, "reason": f"extension ceiling reached ({b.max_extensions})"}, None
        if len(live_clusters) < 2 or not a.get("proposed_test") or not a.get("reason"):
            return {"granted": False, "reason": "an extension must cite >=2 live, distinct (undiscriminated) hypotheses, a reason, and the test that would separate them"}, None
        b.extend()
        return {"granted": True, "new_limit": b.line()}, None

    loop.add(Action(
        "request_budget_extension",
        "Ask for more budget, citing >=2 live hypotheses that remain undiscriminated and the test that would separate them.",
        {"type": "object", "properties": {"reason": {"type": "string"}, "hypothesis_ids": {"type": "array", "items": {"type": "string"}},
                                          "proposed_test": {"type": "string"}}, "required": ["reason", "hypothesis_ids", "proposed_test"]},
        extend,
    ))

    def terminate(a: dict):
        loop.ledger.apply("conclusion_propose", a, actor="lead", step=loop.step_no)
        result = check(loop.mode.lower())
        if result.get("met"):
            return {"accepted_for_challenge": True, "done_check": result}, "terminated"
        return {"accepted_for_challenge": False, "unmet": result.get("unmet", []),
                "note": "address these, then request termination again"}, None

    loop.add(Action(
        "request_termination",
        "Propose your conclusion (conclusive with one accepted hypothesis, or inconclusive with discriminators). It is checked against the done criteria; unmet criteria come back to you.",
        ConclusionIn.model_json_schema(),
        terminate,
    ))
    return loop


LEDGER_ONLY = set(LEDGER_OP_DOCS) | {"ledger_view", "request_termination"}
