"""OpenAI-compatible HTTP adapter for the llm-client component (#20).

Uses only stdlib (urllib, json, time). No provider SDK is imported.
This file is the only place that makes real network calls; every other
module in the package works against LLMClient's abstract interface.

Base URL and per-role model routing come from llm/routing.toml.
Every call is recorded in the run's llm_calls table by LLMClient.complete().
API keys are read from the environment; they are never logged.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from agentic_rca.llm.base import LLMClient, LLMError, LLMResponse, ToolCall

# ── constants ────────────────────────────────────────────────────────────────

_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_S = 2.0  # exponential backoff base


# ── helpers ──────────────────────────────────────────────────────────────────


def _read_key(env_var: str) -> str:
    """Return the value of *env_var* from the environment.  Strips surrounding
    whitespace and embedded newlines that can sneak in from copy-paste."""
    val = os.environ.get(env_var, "")
    return val.replace("\n", "").replace("\r", "").strip()


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert our internal tool dict to the OpenAI function-calling shape."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _parse_response(data: dict[str, Any]) -> LLMResponse:
    """Parse an OpenAI-style chat completion dict into LLMResponse."""
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    usage = data.get("usage", {})

    text: str | None = msg.get("content") or None
    raw_calls = msg.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for c in raw_calls:
        fn = c.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed = {}
        tool_calls.append(
            ToolCall(
                id=c.get("id", ""),
                name=fn.get("name", ""),
                arguments=parsed,
                raw_arguments=raw_args if not parsed else None,
            )
        )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=data.get("model", ""),
        provider=data.get("provider", ""),  # OpenRouter adds this
    )


# ── main adapter ─────────────────────────────────────────────────────────────


class OpenAICompatAdapter(LLMClient):
    """Thin OpenAI-compatible adapter.

    Parameters
    ----------
    base_url:
        e.g. ``"https://openrouter.ai/api/v1"``
    api_key_env:
        Name of the environment variable holding the Bearer token.
    model:
        Model ID to send in every request, e.g.
        ``"meta/muse-spark-1.3-contributor"``.
    provider_name:
        Human label stored in llm_calls.provider when the API doesn't echo one.
    extra_headers:
        Dict of additional HTTP headers (e.g. ``HTTP-Referer`` for OpenRouter).
    timeout_s:
        Per-request timeout in seconds (not counting retries).
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model: str,
        provider_name: str = "",
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.provider_name = provider_name
        self.extra_headers = extra_headers or {}
        self.timeout_s = timeout_s

    def _complete(
        self,
        role: str,  # noqa: ARG002  (used by routing, not by the HTTP call itself)
        messages: list[dict],
        tools: list[dict] | None,
        response_format: dict | None,
    ) -> LLMResponse:
        api_key = _read_key(self.api_key_env)
        if not api_key:
            raise LLMError(
                f"API key env var {self.api_key_env!r} is not set or empty. "
                "Add it to .env before making live calls."
            )

        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = [_tool_schema(t) for t in tools]
            body["tool_choice"] = "auto"
        if response_format:
            body["response_format"] = response_format

        payload = json.dumps(body, default=str).encode()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        last_err: str = "no attempts made"
        for attempt in range(_MAX_RETRIES):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read())
                    out = _parse_response(data)
                    if not out.provider:
                        out.provider = self.provider_name
                    return out
            except urllib.error.HTTPError as exc:
                body_bytes = exc.read()
                try:
                    err_detail = json.loads(body_bytes).get("error", {})
                    err_msg = err_detail.get("message", body_bytes.decode("utf-8", errors="replace"))
                except Exception:
                    err_msg = body_bytes.decode("utf-8", errors="replace")
                last_err = f"HTTP {exc.code}: {err_msg[:300]}"
                if exc.code not in _TRANSIENT_CODES:
                    raise LLMError(last_err) from exc
                # transient: back off and retry
                wait = _RETRY_BASE_S * (2**attempt)
                time.sleep(wait)
            except OSError as exc:
                last_err = f"Network error: {exc}"
                wait = _RETRY_BASE_S * (2**attempt)
                time.sleep(wait)

        raise LLMError(f"Provider failed after {_MAX_RETRIES} attempts. Last: {last_err}")
