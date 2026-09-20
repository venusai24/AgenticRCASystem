"""Tests for llm-client (#20).

Offline tests (default suite, no network):
- ScriptedLLM / ReplayLLM contract: same shape as a real LLMClient
- RoutedLLMClient fallback logic
- routing.toml loads; all roles present; no silent cross-family switch config
- Every call is recorded in llm_calls with model and prompt version

Live test (requires Meta_API_key in env, excluded by default):
- @pytest.mark.live: a real chat completion to Muse Spark 1.3 Contributor
"""

from __future__ import annotations

import os

import duckdb
import pytest

from agentic_rca.llm.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    ReplayLLM,
    ScriptedLLM,
    ToolCall,
    call,
    load_prompt,
)
from agentic_rca.llm.adapter import OpenAICompatAdapter, _parse_response
from agentic_rca.llm.routing import RoutedLLMClient, build_client
from agentic_rca.ledger import Ledger


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_run_con():
    """Create a throw-away run ledger and return (con, run_id) for recording."""
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    led = Ledger.create(tmp / "runs")
    return led.con, led.run_id


# ── ScriptedLLM tests ─────────────────────────────────────────────────────────


def test_scripted_llm_fn():
    """ScriptedLLM with a function returns the right shape."""
    llm = ScriptedLLM(fn=lambda role, msgs, tools: LLMResponse(text=f"ok:{role}", output_tokens=5))
    resp, _ = llm.complete("lead", [{"role": "user", "content": "hi"}])
    assert resp.text == "ok:lead"
    assert resp.error is None


def test_scripted_llm_tool_call():
    """call() shorthand produces a valid tool-call response."""
    llm = ScriptedLLM(fn=lambda r, m, t: call("my_tool", key="val"))
    resp, _ = llm.complete("lead", [], [])
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.name == "my_tool"
    assert tc.arguments == {"key": "val"}


def test_scripted_llm_by_role_queue():
    """Per-role queues are consumed in order."""
    r1 = LLMResponse(text="first", output_tokens=1)
    r2 = LLMResponse(text="second", output_tokens=1)
    llm = ScriptedLLM(by_role={"lead": [r1, r2]})
    a, _ = llm.complete("lead", [])
    b, _ = llm.complete("lead", [])
    assert a.text == "first"
    assert b.text == "second"


def test_scripted_llm_exhausted_raises():
    llm = ScriptedLLM(by_role={"lead": []})
    with pytest.raises(LLMError, match="exhausted"):
        llm.complete("lead", [])


def test_scripted_llm_records_call():
    """complete() writes a row to llm_calls when bound to a run DB."""
    con, run_id = _make_run_con()
    llm = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="x", output_tokens=3))
    llm.bind(con, run_id)
    _, call_id = llm.complete(
        "lead",
        [{"role": "user", "content": "q"}],
        prompt_id="test_prompt",
        prompt_sha="abc123",
    )
    assert call_id is not None
    row = con.execute(
        "SELECT role, prompt_id, prompt_sha, output_tokens FROM llm_calls WHERE call_id=?",
        [call_id],
    ).fetchone()
    assert row == ("lead", "test_prompt", "abc123", 3)


def test_scripted_llm_records_error():
    """LLMError is recorded and re-raised; the call_id is still returned."""

    class _Fail(LLMClient):
        def _complete(self, role, messages, tools, response_format):
            raise LLMError("boom")

    con, run_id = _make_run_con()
    client = _Fail()
    client.bind(con, run_id)
    with pytest.raises(LLMError):
        client.complete("lead", [])
    row = con.execute("SELECT error FROM llm_calls").fetchone()
    assert row is not None
    assert "boom" in row[0]


# ── ReplayLLM ─────────────────────────────────────────────────────────────────


def test_replay_llm_serves_recorded_responses():
    """ReplayLLM replays responses in order per role."""
    src_con, src_run_id = _make_run_con()
    llm = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text=f"r:{r}", output_tokens=2))
    llm.bind(src_con, src_run_id)
    llm.complete("lead", [])
    llm.complete("critic", [])
    llm.complete("lead", [])

    replay = ReplayLLM(src_con)
    a, _ = replay.complete("lead", [])
    b, _ = replay.complete("critic", [])
    c, _ = replay.complete("lead", [])
    assert a.text == "r:lead"
    assert b.text == "r:critic"
    assert c.text == "r:lead"


# ── adapter unit tests (no network) ──────────────────────────────────────────


def test_parse_response_text():
    data = {
        "choices": [{"message": {"content": "hello", "tool_calls": []}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": "test-model",
    }
    resp = _parse_response(data)
    assert resp.text == "hello"
    assert resp.tool_calls == []
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5


def test_parse_response_tool_call():
    import json

    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {
                                "name": "foo",
                                "arguments": json.dumps({"x": 1}),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "model": "m",
    }
    resp = _parse_response(data)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "foo"
    assert resp.tool_calls[0].arguments == {"x": 1}


def test_parse_response_bad_json_args():
    """Malformed JSON in tool arguments sets raw_arguments; arguments is empty."""
    import json

    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "x", "function": {"name": "bad", "arguments": "not json {"}}
                    ]
                }
            }
        ],
        "usage": {},
        "model": "m",
    }
    resp = _parse_response(data)
    tc = resp.tool_calls[0]
    assert tc.arguments == {}
    assert tc.raw_arguments == "not json {"


# ── RoutedLLMClient ───────────────────────────────────────────────────────────


def test_routed_uses_first_provider():
    """RoutedLLMClient returns the first provider's response on success."""
    first = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="first", output_tokens=1))
    second = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="second", output_tokens=1))
    routed = RoutedLLMClient([("k1", first), ("k2", second)], min_interval_s=0)
    resp, _ = routed.complete("lead", [])
    assert resp.text == "first"


def test_routed_falls_back_on_error():
    """RoutedLLMClient tries the next provider when the first raises LLMError."""

    class _AlwaysFail(LLMClient):
        def _complete(self, r, m, t, rf):
            raise LLMError("down")

    fallback = ScriptedLLM(fn=lambda r, m, t: LLMResponse(text="fallback", output_tokens=1))
    routed = RoutedLLMClient([("k1", _AlwaysFail()), ("k2", fallback)], min_interval_s=0)
    resp, _ = routed.complete("lead", [])
    assert resp.text == "fallback"


def test_routed_raises_when_all_fail():
    """RoutedLLMClient raises LLMError if every provider fails."""

    class _AlwaysFail(LLMClient):
        def _complete(self, r, m, t, rf):
            raise LLMError("down")

    routed = RoutedLLMClient([("k1", _AlwaysFail()), ("k2", _AlwaysFail())], min_interval_s=0)
    with pytest.raises(LLMError, match="All 2 provider"):
        routed.complete("lead", [])


# ── routing.toml ──────────────────────────────────────────────────────────────

EXPECTED_ROLES = {
    "lead",
    "critic",
    "reader",
    "dedup",
    "stepback",
    "summary",
    "entailment",
    "bundle_qa",
    "eval_judge",
}


def test_routing_toml_all_roles_present():
    from agentic_rca.llm.routing import _load_config

    cfg = _load_config()
    present = set(cfg.get("roles", {}).keys())
    missing = EXPECTED_ROLES - present
    assert not missing, f"Missing roles in routing.toml: {missing}"


def test_routing_toml_no_empty_providers():
    from agentic_rca.llm.routing import _load_config

    cfg = _load_config()
    for role, rcfg in cfg.get("roles", {}).items():
        providers = rcfg.get("providers", [])
        assert providers, f"Role {role!r} has no providers"
        for p in providers:
            assert p.get("model"), f"Role {role!r} has a provider with no model"
            assert p.get("base_url"), f"Role {role!r} has a provider with no base_url"
            assert p.get("api_key_env"), f"Role {role!r} has a provider with no api_key_env"


def test_routing_no_silent_cross_family():
    """All fallback providers for a role must use the same model family.

    Family is derived as:
    - For ``org/model-name`` style (e.g. ``meta/muse-spark``): the ``org`` part.
    - For ``ModelFamily-version`` style (e.g. ``Llama-4-Maverick``): the first
      hyphen-separated word (``Llama``).

    This ensures a Llama fallback can't silently switch to GPT or Claude.
    """
    import re
    from agentic_rca.llm.routing import _load_config

    def _family(model: str) -> str:
        if "/" in model:
            return model.split("/")[0].lower()
        # Strip version/variant suffixes: keep only the base family name
        return re.split(r"[-_\d]", model)[0].lower()

    cfg = _load_config()
    for role, rcfg in cfg.get("roles", {}).items():
        providers = rcfg.get("providers", [])
        families = {_family(p.get("model", "")) for p in providers}
        assert len(families) <= 1, (
            f"Role {role!r} has providers from different model families: {families}. "
            "A silent cross-family switch is forbidden."
        )


def test_build_client_returns_routed_client():
    client = build_client("lead")
    assert isinstance(client, RoutedLLMClient)


def test_build_client_unknown_role_raises():
    with pytest.raises(KeyError, match="not found"):
        build_client("nonexistent_role_xyz")


def test_load_prompt_returns_text_and_sha():
    """load_prompt loads a real prompt file and returns (text, sha[:12])."""
    text, sha = load_prompt("lead_system")
    assert text.strip()
    assert len(sha) == 12


# ── live smoke test ───────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_smoke_muse_spark():
    """One real completion to Muse Spark 1.3 Contributor via OpenRouter/Llama API.

    Run with: PYTHONPATH=src pytest tests/test_llm_client.py -m live -s
    Requires Meta_API_key in .env.  The test skips if the key is absent,
    expired, or the provider returns 401/429.
    """
    key = os.environ.get("Meta_API_key", "")
    if not key:
        # Try loading from .env relative to the repo root
        env_path = __file__
        for _ in range(10):
            env_path = os.path.dirname(env_path)
            candidate = os.path.join(env_path, ".env")
            if os.path.exists(candidate):
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("Meta_API_key="):
                            key = line.split("=", 1)[1].strip()
                break

    if not key:
        pytest.skip("Meta_API_key not in environment; set it in .env for live tests")

    os.environ["Meta_API_key"] = key

    client = build_client("lead")
    con, run_id = _make_run_con()
    client.bind(con, run_id)

    try:
        resp, call_id = client.complete(
            "lead",
            [{"role": "user", "content": "Reply with exactly: pong"}],
            prompt_id="live_smoke",
            prompt_sha="smoke0000",
        )
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "invalid_api_key" in msg or "Authentication" in msg:
            pytest.skip(f"API key invalid or expired: {msg[:200]}")
        raise

    assert resp is not None, "No response from the model"
    assert resp.error is None, f"Provider error: {resp.error}"
    assert resp.text is not None or resp.tool_calls, "Model returned neither text nor tool calls"
    print(f"\n[live] model={resp.model} provider={resp.provider}")
    print(f"[live] text={resp.text!r} tokens={resp.input_tokens}→{resp.output_tokens}")

    # Verify the call was recorded
    row = con.execute(
        "SELECT model, provider, input_tokens FROM llm_calls WHERE call_id=?",
        [call_id],
    ).fetchone()
    assert row is not None, "Call not recorded in llm_calls"
    model, provider, in_tok = row
    assert model, "model not recorded"
    print(f"[live] recorded: model={model} provider={provider} in_tokens={in_tok}")

