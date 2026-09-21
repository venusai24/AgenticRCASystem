"""Generic one-action-per-step agent loop (design Stage 2, §4.3).

Context is recomposed at every step:

    system prompt + [header: task, manifest, mode, budget, step summary,
    LEDGER render] + the last K raw steps

so the ledger, not a growing transcript, is the memory. Every action
result shown to the model is wrapped as data with its provenance, and log
text can't break out of the frame (`<` is escaped inside JSON).

The lead (agents/lead.py) and the critic (agents/critic.py) are both
AgentLoops with different action tables and prompts.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict
from langgraph.graph import StateGraph, START, END

from agentic_rca.ledger import Ledger, LedgerError, render
from agentic_rca.ledger.rundb import now_iso
from agentic_rca.llm import LLMClient, LLMError, load_prompt
from agentic_rca.tools._common import nearest
from agentic_rca.tools.registry import ToolRuntime, envelope_for_model

CONSOLIDATE = (
    "CONSOLIDATION STEP: before anything else, write to the ledger what your recent steps "
    "established (evidence, hypothesis links, predictions, open questions), then continue. "
    "Anything not in the ledger will not survive in your context."
)
NO_ACTION = "Respond with exactly one tool call. Plain text is not an action."


@dataclass
class Budget:
    """Process knobs, not RCA beliefs (design-d6-d7 §12, design-gap)."""

    max_steps: int = 60
    max_tokens: int = 6_000_000
    max_seconds: float = 3600.0
    ext_steps: int = 20
    ext_tokens: int = 1_500_000
    ext_seconds: float = 900.0
    max_extensions: int = 2
    steps: int = 0
    tokens: int = 0
    extensions: int = 0
    started: float = field(default_factory=time.monotonic)

    def exhausted(self) -> str | None:
        if self.steps >= self.max_steps:
            return f"step budget exhausted ({self.steps}/{self.max_steps})"
        if self.tokens >= self.max_tokens:
            return f"token budget exhausted ({self.tokens:,}/{self.max_tokens:,})"
        if time.monotonic() - self.started >= self.max_seconds:
            return f"wall-clock budget exhausted ({self.max_seconds:.0f}s)"
        return None

    def extend(self) -> None:
        self.extensions += 1
        self.max_steps += self.ext_steps
        self.max_tokens += self.ext_tokens
        self.max_seconds += self.ext_seconds

    def line(self) -> str:
        el = time.monotonic() - self.started
        return (
            f"steps {self.steps}/{self.max_steps}, tokens {self.tokens:,}/{self.max_tokens:,}, "
            f"elapsed {el:.0f}/{self.max_seconds:.0f}s, extensions {self.extensions}/{self.max_extensions}"
        )


@dataclass
class LoopConfig:
    k_recent: int = 6
    consolidate_every: int = 8
    summary_every: int = 0  # 0 disables the model-written step summary (tests); runs use 6
    ledger_max_chars: int = 12000


@dataclass
class Action:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], tuple[dict, str | None]]  # -> (result, control)
    kind: str = "control"  # data | ledger | control


@dataclass
class StepRecord:
    step: int
    name: str | None
    args: dict | None
    call_id: str
    result_text: str
    text: str | None = None


@dataclass
class LoopResult:
    status: (
        str  # terminated | budget_exhausted | overseer_stop | step_limit | llm_error | round_done
    )
    reason: str = ""
    steps: int = 0
    detail: dict = field(default_factory=dict)


class AgentState(TypedDict):
    taken: int
    max_steps: int | None
    use_budget: bool
    loop_result: LoopResult | None
    resp: Any | None


def frame(obj: Any, **attrs: Any) -> str:
    """Wrap a result as data with provenance. `<` inside the JSON is escaped
    so no string in the data can close the frame."""
    head = " ".join(f'{k}="{v}"' for k, v in attrs.items() if v not in (None, ""))
    body = json.dumps(obj, default=str, ensure_ascii=False).replace("<", "\\u003c")
    return f"<tool_result {head}>\n{body}\n</tool_result>"


def args_sha(args: dict | None) -> str:
    return hashlib.sha256(json.dumps(args or {}, sort_keys=True, default=str).encode()).hexdigest()


class AgentLoop:
    def __init__(
        self,
        *,
        actor: str,
        ledger: Ledger,
        runtime: ToolRuntime,
        llm: LLMClient,
        prompt_name: str,
        header: Callable[[], str],
        budget: Budget | None = None,
        config: LoopConfig | None = None,
        overseer: Any | None = None,
        hide_origin: bool = False,
    ):
        self.actor = actor
        self.ledger = ledger
        self.runtime = runtime
        self.llm = llm
        self.system, self.prompt_sha = load_prompt(prompt_name)
        self.prompt_name = prompt_name
        self.header = header
        self.budget = budget or Budget()
        self.config = config or LoopConfig()
        self.overseer = overseer
        self.hide_origin = hide_origin
        self.actions: dict[str, Action] = {}
        self.history: list[StepRecord] = []
        self.summary = ""
        self.nudges: list[str] = []
        self.mode = "INVESTIGATE"
        self.mode_instruction = ""
        self.allowed: set[str] | None = None
        self.round: int | None = None
        self.step_no = 0

    # ---- action table ------------------------------------------------------------
    def add(self, action: Action) -> None:
        self.actions[action.name] = action

    def tool_defs(self) -> list[dict]:
        names = sorted(n for n in self.actions if self.allowed is None or n in self.allowed)
        return [
            {
                "name": n,
                "description": self.actions[n].description,
                "parameters": self.actions[n].parameters,
            }
            for n in names
        ]

    # ---- context -------------------------------------------------------------------
    def compose(self) -> list[dict]:
        parts = [self.header()]
        parts.append(
            f"MODE: {self.mode}" + (f"\n{self.mode_instruction}" if self.mode_instruction else "")
        )
        parts.append(f"BUDGET: {self.budget.line()}")
        if self.summary:
            parts.append(
                f"SUMMARY OF EARLIER STEPS (convenience; the ledger is authoritative):\n{self.summary}"
            )
        state = self.ledger.state()
        parts.append(
            "LEDGER (authoritative):\n"
            + render(
                state,
                self.ledger.short,
                hide_origin=self.hide_origin,
                max_chars=self.config.ledger_max_chars,
            )
        )
        for n in self.nudges:
            parts.append(f"NOTE FROM THE PROCESS MONITOR: {n}")
        self.nudges = []
        msgs: list[dict] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        for rec in self.history[-self.config.k_recent :]:
            if rec.name is None:
                msgs.append({"role": "assistant", "content": rec.text or ""})
                msgs.append({"role": "user", "content": rec.result_text})
                continue
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": rec.call_id,
                            "type": "function",
                            "function": {
                                "name": rec.name,
                                "arguments": json.dumps(rec.args or {}, default=str),
                            },
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": rec.call_id, "content": rec.result_text})
        return msgs

    # ---- one run -------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)

        def reasoner_node(state: AgentState):
            use_budget = state.get("use_budget", True)
            max_steps = state.get("max_steps")
            taken = state["taken"]
            
            if use_budget and (why := self.budget.exhausted()):
                return {"loop_result": LoopResult("budget_exhausted", why, taken)}
            if max_steps is not None and taken >= max_steps:
                return {"loop_result": LoopResult("step_limit", f"{max_steps} steps", taken)}
                
            try:
                resp, _cid = self.llm.complete(
                    self.actor,
                    self.compose(),
                    self.tool_defs(),
                    prompt_id=self.prompt_name,
                    prompt_sha=self.prompt_sha,
                )
                return {"resp": resp}
            except LLMError as exc:
                return {"loop_result": LoopResult("llm_error", str(exc), taken)}

        def tool_node(state: AgentState):
            resp = state.get("resp")
            taken = state["taken"]
            use_budget = state.get("use_budget", True)
            
            self.budget.tokens += resp.input_tokens + resp.output_tokens
            if not resp.tool_calls:
                self.step_no += 1
                taken += 1
                self.budget.steps += 1
                self.history.append(
                    StepRecord(self.step_no, None, None, "", NO_ACTION, text=resp.text)
                )
                self._trajectory(None, None, {}, "no_action", resp)
                return {"taken": taken}
                
            for tc in resp.tool_calls:
                print(f"[{self.mode} | Step {self.step_no + 1}] -> {tc.name}(...)")
                self.step_no += 1
                taken += 1
                self.budget.steps += 1
                seq_before = self.ledger.last_seq()
                
                if tc.raw_arguments is not None:
                    result, control, meta = (
                        {
                            "error": {
                                "type": "invalid_action",
                                "diagnostic": "arguments were not valid JSON",
                            }
                        },
                        None,
                        {},
                    )
                    text = frame(result, action=tc.name)
                else:
                    result, control, meta, text = self.dispatch(tc.name, tc.arguments)
                self.history.append(
                    StepRecord(
                        self.step_no, tc.name, tc.arguments, tc.id or f"c{self.step_no}", text
                    )
                )
                traj = self._trajectory(tc.name, tc.arguments, meta, None, resp, seq_before)
                if control is not None:
                    return {"loop_result": LoopResult(control, "", taken, result if isinstance(result, dict) else {}), "taken": taken}
                if self.overseer is not None:
                    verdict = self.overseer.after_step(traj, self.ledger)
                    if verdict and verdict[0] == "stop":
                        return {"loop_result": LoopResult("overseer_stop", verdict[1], taken), "taken": taken}
                    if verdict and verdict[0] == "nudge":
                        self.nudges.append(verdict[1])
                if use_budget and self.budget.exhausted():
                    break
            return {"taken": taken}

        def consolidator_node(state: AgentState):
            if self.config.consolidate_every and self.step_no % self.config.consolidate_every == 0:
                self.nudges.append(CONSOLIDATE)
            if self.config.summary_every and self.step_no % self.config.summary_every == 0:
                self._summarise()
            return {}

        def route_after_reasoner(state: AgentState):
            if state.get("loop_result") is not None:
                return END
            return "tool_node"

        def route_after_tools(state: AgentState):
            if state.get("loop_result") is not None:
                return END
            return "consolidator_node"
            
        def route_after_consolidator(state: AgentState):
            return "reasoner_node"

        graph.add_node("reasoner_node", reasoner_node)
        graph.add_node("tool_node", tool_node)
        graph.add_node("consolidator_node", consolidator_node)
        
        graph.add_edge(START, "reasoner_node")
        graph.add_conditional_edges("reasoner_node", route_after_reasoner, {END: END, "tool_node": "tool_node"})
        graph.add_conditional_edges("tool_node", route_after_tools, {END: END, "consolidator_node": "consolidator_node"})
        graph.add_edge("consolidator_node", "reasoner_node")
        
        return graph.compile()

    def run(
        self,
        *,
        mode: str = "INVESTIGATE",
        instruction: str = "",
        allowed: set[str] | None = None,
        max_steps: int | None = None,
        use_budget: bool = True,
    ) -> LoopResult:
        self.mode, self.mode_instruction, self.allowed = mode, instruction, allowed
        
        if not hasattr(self, "_graph"):
            self._graph = self._build_graph()
            
        initial_state = {
            "taken": 0,
            "max_steps": max_steps,
            "use_budget": use_budget,
            "loop_result": None,
            "resp": None,
        }
        
        final_state = self._graph.invoke(initial_state)
        return final_state["loop_result"]

    def dispatch(self, name: str, args: dict) -> tuple[dict, str | None, dict, str]:
        if name not in self.actions or (self.allowed is not None and name not in self.allowed):
            avail = [n for n in self.actions if self.allowed is None or n in self.allowed]
            result = {
                "error": {
                    "type": "invalid_action",
                    "diagnostic": f"unknown or unavailable action {name!r} in mode {self.mode}",
                    "suggestions": nearest(name, avail) or sorted(avail)[:10],
                }
            }
            return result, None, {"error_type": "invalid_action"}, frame(result, action=name)
        act = self.actions[name]
        try:
            result, control = act.handler(dict(args or {}))
        except LedgerError as exc:
            result, control = exc.as_dict(), None
        meta = result.pop("_meta", {}) if isinstance(result, dict) else {}
        attrs = meta.get("frame") or {"action": name}
        return result, control, meta, frame(result, **attrs)

    def _trajectory(self, name, args, meta, note, resp, seq_before=None) -> dict:
        row = {
            "run_id": self.ledger.run_id,
            "step": self.step_no,
            "actor": self.actor,
            "mode": self.mode,
            "action": name,
            "args_sha256": args_sha(args) if name else None,
            "query_id": meta.get("query_id"),
            "row_count": meta.get("row_count"),
            "error_type": meta.get("error_type"),
            "ledger_seq_before": seq_before if seq_before is not None else self.ledger.last_seq(),
            "ledger_seq_after": self.ledger.last_seq(),
            "tokens": resp.input_tokens + resp.output_tokens,
            "note": note,
            "created_at": now_iso(),
        }
        cols = list(row)
        self.ledger.con.execute(
            f"INSERT INTO trajectory ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [row[c] for c in cols],
        )
        return row

    def _summarise(self) -> None:
        older = self.history[: -self.config.k_recent]
        if not older:
            return
        lines = [
            f"step {r.step}: {r.name or 'text'} {json.dumps(r.args or {}, default=str)[:200]}"
            for r in older[-20:]
        ]
        text, sha = load_prompt("step_summary")
        try:
            resp, _ = self.llm.complete(
                "summary",
                [
                    {"role": "system", "content": text},
                    {
                        "role": "user",
                        "content": f"Previous summary:\n{self.summary or '(none)'}\n\nOlder steps:\n"
                        + "\n".join(lines),
                    },
                ],
                prompt_id="step_summary",
                prompt_sha=sha,
            )
            self.budget.tokens += resp.input_tokens + resp.output_tokens
            if resp.text:
                self.summary = resp.text.strip()[:2000]
        except LLMError:
            pass  # the summary is a convenience; the ledger is authoritative


# ---- shared action builders --------------------------------------------------------


def data_tool_action(loop: AgentLoop, name: str, description: str, parameters: dict) -> Action:
    def handler(args: dict):
        env = loop.runtime.call(name, args, actor=loop.actor, step=loop.step_no)
        out = envelope_for_model(env)
        out["_meta"] = {
            "query_id": env.query_id,
            "row_count": env.row_count,
            "error_type": env.error.type if env.error else None,
            "frame": {
                "source": env.provenance.source,
                "query_id": env.query_id,
                "window": env.provenance.window,
            },
        }
        return out, None

    return Action(name, description, parameters, handler, "data")


def ledger_action(
    loop: AgentLoop, op: str, description: str, on_write: Callable[[str, str], None] | None = None
) -> Action:
    from agentic_rca.ledger import models as M

    model = M.OPS[op][0]
    params = model.model_json_schema() if model else {"type": "object", "properties": {}}

    def handler(args: dict):
        out = loop.ledger.apply(op, args, actor=loop.actor, step=loop.step_no, round=loop.round)
        if on_write and out.get("id") and op in ("hypothesis_create", "hypothesis_revise"):
            on_write(op, out["id"])
        return out, None

    return Action(op, description, params, handler, "ledger")


def inspect_action(loop: AgentLoop) -> Action:
    params = {
        "type": "object",
        "properties": {
            "handle": {"type": "string", "description": "a query id / result handle"},
            "offset": {"type": "integer", "default": 0},
            "limit": {"type": "integer", "default": 20},
            "sort": {"type": "string", "description": "column, or -column for descending"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "preview_mode": {"type": "string", "enum": ["sample", "head", "tail", "by_entity"]},
        },
        "required": ["handle"],
    }

    def handler(a: dict):
        env = loop.runtime.inspect(
            a.get("handle", ""),
            offset=int(a.get("offset", 0)),
            limit=int(a.get("limit", 20)),
            sort=a.get("sort"),
            columns=a.get("columns"),
            preview_mode=a.get("preview_mode"),
            actor=loop.actor,
        )
        out = envelope_for_model(env)
        out["_meta"] = {
            "query_id": env.query_id,
            "frame": {"source": env.provenance.source, "query_id": env.query_id},
        }
        return out, None

    return Action(
        "inspect_result",
        "Page through, re-sort or re-preview an existing result handle (no new query).",
        params,
        handler,
    )


def coverage_action(loop: AgentLoop) -> Action:
    params = {
        "type": "object",
        "properties": {"source": {"type": "string"}, "entity": {"type": "string"}},
    }

    def handler(a: dict):
        return loop.runtime.coverage_view(a.get("source"), a.get("entity")), None

    return Action(
        "coverage_view",
        "What has been queried, inspected, attempted, and which requested entities a source never held. Raw and unranked.",
        params,
        handler,
    )


def ledger_view_action(loop: AgentLoop) -> Action:
    params = {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "hypotheses",
                        "evidence",
                        "questions",
                        "objections",
                        "failures",
                        "focus",
                    ],
                },
            }
        },
    }

    def handler(a: dict):
        secs = tuple(
            a.get("sections")
            or ("hypotheses", "evidence", "questions", "objections", "failures", "focus")
        )
        return {
            "ledger": render(
                loop.ledger.state(),
                loop.ledger.short,
                hide_origin=loop.hide_origin,
                sections=secs,
                max_chars=30000,
            )
        }, None

    return Action(
        "ledger_view", "Re-read the ledger, optionally restricted to sections.", params, handler
    )
