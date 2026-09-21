import json
import time
import urllib.request
import urllib.error
from typing import Any

from agentic_rca.llm.base import LLMClient, LLMResponse, ToolCall, LLMError

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class OpenAIAdapter(LLMClient):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini", max_retries: int = 3, initial_backoff: float = 1.0):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    @traceable(run_type="llm")
    def _complete(self, role: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, response_format: dict | None = None) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            data["tools"] = [{"type": "function", "function": t} for t in tools]
            
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        retries = 0
        backoff = self.initial_backoff
        
        while True:
            try:
                with urllib.request.urlopen(req) as response:
                    status_code = response.getcode()
                    body = response.read().decode("utf-8")
                    
                if status_code != 200:
                    raise LLMError(f"HTTP {status_code}: {body}")
                
                resp_data = json.loads(body)
                return self._parse_response(resp_data)
                
            except urllib.error.HTTPError as e:
                # Retry on 429 (Too Many Requests) or 5xx (Server Errors)
                if e.code == 429 or 500 <= e.code < 600:
                    if retries >= self.max_retries:
                        return LLMResponse(error=f"Max retries reached. Last HTTP error: {e.code} - {e.read().decode('utf-8', errors='ignore')}")
                    time.sleep(backoff)
                    retries += 1
                    backoff *= 2
                else:
                    return LLMResponse(error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
            except urllib.error.URLError as e:
                # Network error, potentially transient
                if retries >= self.max_retries:
                    return LLMResponse(error=f"Max retries reached. Network error: {e.reason}")
                time.sleep(backoff)
                retries += 1
                backoff *= 2
            except json.JSONDecodeError:
                return LLMResponse(error="Failed to decode JSON response from provider")
            except Exception as e:
                return LLMResponse(error=str(e))

    def _parse_response(self, resp_data: dict[str, Any]) -> LLMResponse:
        try:
            choice = resp_data["choices"][0]["message"]
            text = choice.get("content")
            
            tool_calls = []
            if "tool_calls" in choice:
                for tc in choice["tool_calls"]:
                    if tc.get("type") == "function":
                        func = tc["function"]
                        args = json.loads(func.get("arguments", "{}"))
                        tool_calls.append(ToolCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            args=args
                        ))
            
            usage = resp_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            return LLMResponse(
                text=text,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=resp_data.get("model", self.model),
                provider="openai-compatible"
            )
        except Exception as e:
            return LLMResponse(error=f"Error parsing response: {e}")
