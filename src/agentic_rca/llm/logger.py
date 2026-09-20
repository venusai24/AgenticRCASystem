import json
import time
import uuid
import hashlib
from typing import Any

from agentic_rca.llm.base import LLMClient, LLMResponse, ToolCall, LLMError

class LoggedLLMClient(LLMClient):
    """
    Wraps an LLMClient to record every call to a DuckDB connection.
    """
    def __init__(self, client: LLMClient, con: Any, run_id: str, role: str):
        self.client = client
        self.con = con
        self.run_id = run_id
        self.role = role
        self._ensure_table()

    def _ensure_table(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                run_id VARCHAR,
                call_id VARCHAR,
                role VARCHAR,
                model VARCHAR,
                prompt_id VARCHAR,
                prompt_sha VARCHAR,
                request JSON,
                response JSON,
                tokens INTEGER,
                latency DOUBLE,
                error VARCHAR
            )
        """)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        call_id = str(uuid.uuid4())
        
        # Calculate prompt_sha over the messages content to track exactly what was sent
        request_data = {"messages": messages}
        if tools:
            request_data["tools"] = tools
            
        request_json = json.dumps(request_data)
        prompt_sha = hashlib.sha256(request_json.encode('utf-8')).hexdigest()
        prompt_id = f"prompt_{prompt_sha[:8]}"
        
        start_time = time.monotonic()
        try:
            response = self.client.chat(messages, tools)
            latency = time.monotonic() - start_time
            
            response_json = json.dumps({
                "text": response.text,
                "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls]
            })
            tokens = response.input_tokens + response.output_tokens
            
            self.con.execute(
                """
                INSERT INTO llm_calls (run_id, call_id, role, model, prompt_id, prompt_sha, request, response, tokens, latency, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (self.run_id, call_id, self.role, response.model, prompt_id, prompt_sha, 
                 request_json, response_json, tokens, latency, response.error)
            )
            return response
            
        except Exception as e:
            latency = time.monotonic() - start_time
            self.con.execute(
                """
                INSERT INTO llm_calls (run_id, call_id, role, model, prompt_id, prompt_sha, request, response, tokens, latency, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (self.run_id, call_id, self.role, "", prompt_id, prompt_sha, 
                 request_json, None, 0, latency, str(e))
            )
            raise
