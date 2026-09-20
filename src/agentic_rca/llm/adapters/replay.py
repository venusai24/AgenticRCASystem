import json
from typing import Any

from agentic_rca.llm.base import LLMClient, LLMResponse, ToolCall, LLMError

class ReplayLLM(LLMClient):
    """
    Adapter that serves recorded responses from a predefined list.
    """
    def __init__(self, responses: list[LLMResponse]):
        self.responses = responses
        self.call_count = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        if self.call_count >= len(self.responses):
            raise LLMError("ReplayLLM exhausted its predefined responses.")
        
        response = self.responses[self.call_count]
        self.call_count += 1
        return response
