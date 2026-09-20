from typing import Any, Callable

from agentic_rca.llm.base import LLMClient, LLMResponse, ToolCall, LLMError

class ScriptedLLM(LLMClient):
    """
    Adapter that serves hand-written responses, optionally using a callback function
    to determine the response dynamically based on messages and tools.
    """
    def __init__(self, callback: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], LLMResponse]):
        self.callback = callback
        self.call_count = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.call_count += 1
        return self.callback(messages, tools)
