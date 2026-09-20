"""The LLM boundary (X5): one `complete()` call shape for every role, with
every call recorded in the run DB's `llm_calls` table (model and prompt
version per call, which the bundle must report).

Tool interfaces are plain JSON Schema; messages are OpenAI-style
role/content/tool_calls dicts, which every provider we route to accepts.
`ScriptedLLM` and `ReplayLLM` make every LLM-driven component testable
without a network and without nondeterminism.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_rca.ledger.rundb import now_iso

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> tuple[str, str]:
    """Return (text, sha256[:12]) for prompts/<name>.md. The hash is the
    prompt version recorded with every call (§4.3, §4.5 audit)."""
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None  # set when the model's JSON didn't parse


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    error: str | None = None


class LLMError(Exception):
    """A provider failure that survived bounded retries."""


class LLMClient:
    """Base: subclasses implement `_complete`. `complete` records the call."""

    def __init__(self) -> None:
        self.con = None
        self.run_id: str | None = None
        self._n = 0

    def bind(self, con, run_id: str) -> LLMClient:
        """Attach to a run DB so every call is recorded in `llm_calls`."""
        self.con, self.run_id = con, run_id
        if con is not None:
            self._n = con.execute(
                "SELECT count(*) FROM llm_calls WHERE run_id = ?", [run_id]
            ).fetchone()[0]
        return self

    def _complete(
        self,
        role: str,
        messages: list[dict],
        tools: list[dict] | None,
        response_format: dict | None,
    ) -> LLMResponse:
        raise NotImplementedError

    def complete(
        self,
        role: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        prompt_id: str = "",
        prompt_sha: str = "",
        response_format: dict | None = None,
    ) -> tuple[LLMResponse, str | None]:
        """Returns (response, call_id). Provider errors are recorded and re-raised."""
        t0 = time.perf_counter()
        err: str | None = None
        resp: LLMResponse | None = None
        try:
            resp = self._complete(role, messages, tools, response_format)
        except LLMError as exc:
            err = str(exc)
        call_id = self._record(role, prompt_id, prompt_sha, messages, tools, resp, err, t0)
        if err is not None:
            raise LLMError(err)
        return resp, call_id

    def _record(self, role, prompt_id, prompt_sha, messages, tools, resp, err, t0) -> str | None:
        if self.con is None:
            return None
        self._n += 1
        call_id = f"{self.run_id}/llm{self._n:05d}"
        request = {"messages": messages, "tools": [t["name"] for t in (tools or [])]}
        response = None
        if resp is not None:
            response = {
                "text": resp.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls
                ],
            }
        self.con.execute(
            "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                self.run_id,
                call_id,
                role,
                resp.provider if resp else "",
                resp.model if resp else "",
                prompt_id,
                prompt_sha,
                json.dumps(request, default=str),
                json.dumps(response, default=str) if response else None,
                resp.input_tokens if resp else 0,
                resp.output_tokens if resp else 0,
                (time.perf_counter() - t0) * 1000,
                err,
                now_iso(),
            ],
        )
        return call_id


Script = Callable[[str, list[dict], list[dict] | None], LLMResponse]


class ScriptedLLM(LLMClient):
    """Deterministic test double. Either a function (role, messages, tools)
    -> LLMResponse, or per-role queues of responses. A queue item may be a
    callable taking (messages, tools)."""

    def __init__(
        self,
        fn: Script | None = None,
        *,
        by_role: dict[str, list[Any]] | None = None,
        default: Callable[[str, list[dict], list[dict] | None], LLMResponse] | None = None,
    ):
        super().__init__()
        self.fn = fn
        self.queues = {k: list(v) for k, v in (by_role or {}).items()}
        self.default = default
        self.seen: list[tuple[str, list[dict], list[dict] | None]] = []

    def _complete(self, role, messages, tools, response_format):
        self.seen.append((role, messages, tools))
        if self.fn is not None:
            out = self.fn(role, messages, tools)
        elif self.queues.get(role):
            item = self.queues[role].pop(0)
            out = item(messages, tools) if callable(item) else item
        elif self.default is not None:
            out = self.default(role, messages, tools)
        else:
            raise LLMError(f"ScriptedLLM: script exhausted for role {role!r}")
        out.model = out.model or "scripted"
        out.provider = out.provider or "scripted"
        return out


def call(name: str, _id: str | None = None, **arguments) -> LLMResponse:
    """Shorthand for a scripted single-tool-call response."""
    return LLMResponse(
        tool_calls=[ToolCall(_id or f"call_{name}", name, arguments)], output_tokens=10
    )


class ReplayLLM(LLMClient):
    """Serves the responses recorded in another run's `llm_calls`, in order,
    per role: a finished run becomes a regression fixture."""

    def __init__(self, recorded_con):
        super().__init__()
        rows = recorded_con.execute(
            "SELECT role, response FROM llm_calls WHERE response IS NOT NULL ORDER BY call_id"
        ).fetchall()
        self.queues: dict[str, list[LLMResponse]] = {}
        for role, resp in rows:
            d = json.loads(resp)
            self.queues.setdefault(role, []).append(
                LLMResponse(
                    text=d.get("text"),
                    tool_calls=[
                        ToolCall(c["id"], c["name"], c["arguments"])
                        for c in d.get("tool_calls", [])
                    ],
                    model="replay",
                    provider="replay",
                )
            )

    def _complete(self, role, messages, tools, response_format):
        if not self.queues.get(role):
            raise LLMError(f"ReplayLLM: no recorded response left for role {role!r}")
        return self.queues[role].pop(0)
