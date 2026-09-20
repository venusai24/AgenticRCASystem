"""#11 Reader Agent: A stateless interaction loop to answer bounded questions over a result handle.

Uses `inspect_result` to read rows from a handle and deterministically verifies
that returned row_ids exist and match the inspected content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentic_rca.llm.base import LLMClient, load_prompt
from agentic_rca.tools.registry import ToolRuntime

_MAX_ITERATIONS = 5

@dataclass
class ReaderResult:
    claims: list[dict[str, Any]]
    dropped_claims: int
    iterations: int


class ReaderAgent:
    def __init__(self, client: LLMClient, runtime: ToolRuntime):
        self.client = client
        self.runtime = runtime

    def _verify_and_filter(self, handle: str, claims: list[dict], actor: str) -> tuple[list[dict], int]:
        """Verify that all row_ids in claims exist in the handle and were inspected.
        Drop any claim that cites hallucinated row_ids.
        """
        if not claims:
            return [], 0
            
        handle_rows = self.runtime.lookup(handle)
        if handle_rows is None:
            return [], len(claims)
            
        valid_row_ids = {r["_row"] for r in handle_rows}
        
        valid_claims = []
        dropped = 0
        
        for claim in claims:
            row_ids = claim.get("row_ids", [])
            if not isinstance(row_ids, list):
                dropped += 1
                continue
                
            # Check if all row_ids exist in the handle
            if all(isinstance(rid, int) and rid in valid_row_ids for rid in row_ids):
                valid_claims.append(claim)
            else:
                dropped += 1
                
        return valid_claims, dropped

    def ask(self, handle: str, question: str, max_claims: int, step: int, actor: str = "reader") -> ReaderResult:
        """Answer a bounded question over a result handle."""
        sys_text, sys_sha = load_prompt("reader_system")
        
        # Reader only gets access to inspect_result.
        # If sql over handle was supported, we would add it here.
        tools = [{
            "type": "function",
            "function": {
                "name": "inspect_result",
                "description": "Page through a result handle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "handle_id": {"type": "string"},
                        "offset": {"type": "integer", "default": 0},
                        "limit": {"type": "integer", "default": 20},
                        "sort": {"type": "string"},
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["handle_id"],
                    "additionalProperties": False
                }
            }
        }]
        
        schema = {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "row_ids": {
                                "type": "array",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["text", "row_ids"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["claims"],
            "additionalProperties": False
        }
        
        messages = [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": f"QUESTION: {question}\nMAX CLAIMS: {max_claims}\nHANDLE: {handle}"
            }
        ]
        
        for iteration in range(1, _MAX_ITERATIONS + 1):
            resp, _ = self.client.complete(
                "reader",
                messages,
                tools=tools,
                response_format={"type": "json_schema", "json_schema": {"name": "reader_output", "schema": schema, "strict": True}},
                prompt_id="reader_system",
                prompt_sha=sys_sha,
            )
            
            messages.append({"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls})
            
            if not resp.tool_calls:
                break
                
            for tc in resp.tool_calls:
                if tc["name"] == "inspect_result":
                    args = tc["args"]
                    # Force handle_id to match what was assigned
                    args["handle_id"] = handle
                    # Call runtime.inspect which correctly checks handles without creating new query ids
                    env = self.runtime.inspect(
                        handle=handle,
                        offset=args.get("offset", 0),
                        limit=args.get("limit", 20),
                        sort=args.get("sort"),
                        columns=args.get("columns"),
                        actor=actor,
                        via="inspect_result"
                    )
                    
                    if env.error:
                        res_content = json.dumps({"error": env.error.diagnostic})
                    else:
                        res_content = json.dumps({"preview": env.preview, "truncated": env.truncated, "row_count": env.row_count})
                        
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": res_content
                    })
        
        # Parse final response
        try:
            # We look for the last assistant text that contains JSON
            final_text = resp.text
            if not final_text:
                # If they just output tool calls on the last turn, no claims
                return ReaderResult([], 0, iteration)
                
            data = json.loads(final_text)
            raw_claims = data.get("claims", [])
            # Enforce max_claims
            raw_claims = raw_claims[:max_claims]
            
            valid_claims, dropped = self._verify_and_filter(handle, raw_claims, actor)
            return ReaderResult(valid_claims, dropped, iteration)
            
        except json.JSONDecodeError:
            return ReaderResult([], 0, iteration)

