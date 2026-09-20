"""llm/routing.py — loads routing.toml and builds the right LLMClient
per role, with per-provider pacing and a no-silent-switch fallback chain.

Usage
-----
    from agentic_rca.llm.routing import build_client

    client = build_client("lead")      # returns configured LLMClient
    client.bind(con, run_id)
    resp, call_id = client.complete("lead", messages, tools,
                                    prompt_id=pid, prompt_sha=sha)

The returned object is a ``RoutedLLMClient`` that tries each provider
in the role's list and stops at the first success.  If all providers fail
it raises ``LLMError`` — it never silently switches model family.
"""

from __future__ import annotations

import time
from pathlib import Path

from agentic_rca.llm.adapter import OpenAICompatAdapter
from agentic_rca.llm.base import LLMClient, LLMError, LLMResponse

import tomllib

_ROUTING_PATH = Path(__file__).with_name("routing.toml")


def _load_config(path: Path = _ROUTING_PATH) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


# ── pacing tracker (module-level, shared across all instances) ────────────────

_last_call: dict[str, float] = {}  # key: provider_key → last call epoch


def _pace(key: str, min_interval_s: float) -> None:
    """Block until the minimum interval since the last call to *key* has passed."""
    if min_interval_s <= 0:
        return
    now = time.monotonic()
    since = now - _last_call.get(key, 0.0)
    if since < min_interval_s:
        time.sleep(min_interval_s - since)
    _last_call[key] = time.monotonic()


# ── routed client ─────────────────────────────────────────────────────────────


class RoutedLLMClient(LLMClient):
    """Tries each adapter in *adapters* in order.  On a transient failure it
    moves to the next; on a permanent failure it stops immediately.  Never
    switches silently: if all fail, ``LLMError`` is raised with the full
    chain of errors."""

    def __init__(
        self,
        adapters: list[tuple[str, OpenAICompatAdapter]],  # (pacing_key, adapter)
        min_interval_s: float = 0.2,
    ) -> None:
        super().__init__()
        self._adapters = adapters
        self._min_interval_s = min_interval_s

    def _complete(
        self,
        role: str,
        messages: list[dict],
        tools: list[dict] | None,
        response_format: dict | None,
    ) -> LLMResponse:
        errors: list[str] = []
        for pacing_key, adapter in self._adapters:
            _pace(pacing_key, self._min_interval_s)
            try:
                resp = adapter._complete(role, messages, tools, response_format)
                _last_call[pacing_key] = time.monotonic()
                return resp
            except LLMError as exc:
                model = getattr(adapter, "model", type(adapter).__name__)
                pname = getattr(adapter, "provider_name", "")
                errors.append(f"[{model}@{pname}] {exc}")
                # Continue to next provider only on provider-level failure.
                # A malformed-response parse error should not fall through.
        raise LLMError(
            f"All {len(self._adapters)} provider(s) failed for role {role!r}:\n"
            + "\n".join(errors)
        )


# ── factory ───────────────────────────────────────────────────────────────────


def build_client(role: str, config_path: Path = _ROUTING_PATH) -> LLMClient:
    """Return a configured ``RoutedLLMClient`` for *role*.

    Raises ``KeyError`` if *role* is absent from routing.toml.
    """
    cfg = _load_config(config_path)
    pacing_cfg = cfg.get("pacing", {})
    min_interval_s: float = float(pacing_cfg.get("min_interval_s", 0.2))

    roles_cfg = cfg.get("roles", {})
    if role not in roles_cfg:
        raise KeyError(
            f"Role {role!r} not found in routing.toml. "
            f"Available roles: {sorted(roles_cfg)}"
        )

    providers_cfg = roles_cfg[role].get("providers", [])
    if not providers_cfg:
        raise ValueError(f"Role {role!r} has no providers in routing.toml")

    adapters: list[tuple[str, OpenAICompatAdapter]] = []
    for p in providers_cfg:
        extra: dict[str, str] = {}
        if "http_referer" in p:
            extra["HTTP-Referer"] = p["http_referer"]
        adapter = OpenAICompatAdapter(
            base_url=p["base_url"],
            api_key_env=p["api_key_env"],
            model=p["model"],
            provider_name=p.get("provider_name", ""),
            extra_headers=extra,
        )
        # Pacing key = base_url + api_key_env so different keys to the same
        # host pace independently, but the same key is shared across roles.
        pacing_key = f"{p['base_url']}|{p['api_key_env']}"
        adapters.append((pacing_key, adapter))

    return RoutedLLMClient(adapters, min_interval_s=min_interval_s)
