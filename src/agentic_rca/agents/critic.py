"""Stage 3: the adversarial critic (§4.1).

Each round is a brand-new AgentLoop: fresh message history, never the
lead's trajectory, and a ledger render with `origin` hidden (the critic
must not defer to where a hypothesis came from). It has the same data
tools, can record evidence and alternatives, and raises / reopens
objections. From round 2 it sees its own earlier objections and their
resolutions through the ledger render, which is exactly the memory the
design allows it.
"""

from __future__ import annotations

import json
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
import agentic_rca.tools.catalog  # noqa: F401 - registers every data tool
from agentic_rca.tools.registry import TOOLS, tool_schemas

CRITIC_OPS = {
    "evidence_record": "Record evidence you found (query id + exact cited rows).",
    "hypothesis_create": "Propose an alternative explanation of the same symptoms.",
    "objection_raise": "Raise a specific objection against a hypothesis or one of its links, citing evidence where possible.",
    "objection_reopen": "Reopen an earlier objection whose resolution argued it away rather than resolving it.",
}


def run_critic_round(
    *,
    ledger,
    runtime,
    llm,
    round_no: int,
    declaration: dict | None = None,
    budget: Budget | None = None,
    config: LoopConfig | None = None,
    on_hypothesis_write: Callable[[str, str], None] | None = None,
    tool_names: list[str] | None = None,
    max_steps: int = 25,
):
    """One critic round in a fresh context. Returns (LoopResult, objections raised this round)."""

    def header() -> str:
        concl = ledger.state().latest_conclusion() or {}
        shown = {k: v for k, v in concl.items() if k != "seq"}
        parts = [f"CRITIC ROUND {round_no}", "PROPOSED CONCLUSION:\n" + json.dumps(shown, default=str, indent=1)]
        if declaration:
            parts.append("DATA-SUFFICIENCY DECLARATION:\n" + json.dumps(declaration, default=str)[:4000])
        return "\n\n".join(parts)

    loop = AgentLoop(
        actor="critic", ledger=ledger, runtime=runtime, llm=llm, prompt_name="critic_system",
        header=header, budget=budget or Budget(max_steps=max_steps), config=config, hide_origin=True,
    )
    loop.round = round_no
    for spec in tool_schemas(tool_names or sorted(TOOLS)):
        loop.add(data_tool_action(loop, spec["name"], spec["description"], spec["parameters"]))
    loop.add(inspect_action(loop))
    loop.add(coverage_action(loop))
    loop.add(ledger_view_action(loop))
    for op, doc in CRITIC_OPS.items():
        loop.add(ledger_action(loop, op, doc, on_hypothesis_write))
    loop.add(Action("finish_round", "End this review round.", {"type": "object", "properties": {}},
                    lambda a: ({"ok": True}, "round_done")))
    before = set(ledger.state().objections)
    result = loop.run(max_steps=max_steps, use_budget=False)
    raised = sorted(set(ledger.state().objections) - before)
    return result, raised
