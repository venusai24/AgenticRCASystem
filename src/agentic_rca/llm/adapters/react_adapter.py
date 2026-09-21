import socket
import json
import time
import urllib.request
import urllib.error
import re
import uuid
import logging
from typing import Any

logger = logging.getLogger("react_adapter")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("llm_debug.log")
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

from agentic_rca.llm.base import LLMClient, LLMResponse, ToolCall, LLMError

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class ReActAdapter(LLMClient):
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
        
        # Transform messages for ReAct
        transformed_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "tool":
                # Convert tool response to user message
                tool_name = msg.get("name", "unknown")
                transformed_messages.append({
                    "role": "user",
                    "content": f"<tool_result name=\"{tool_name}\">\n{content}\n</tool_result>"
                })
            elif role == "assistant":
                # Serialize past tool calls into the <tool> format
                new_content = content or ""
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        if tc.get("type") == "function":
                            func = tc["function"]
                            name = func.get("name", "")
                            args = func.get("arguments", "{}")
                            if isinstance(args, dict):
                                args = json.dumps(args)
                            new_content += f"\n<tool name=\"{name}\">\n{args}\n</tool>"
                transformed_messages.append({
                    "role": "assistant",
                    "content": new_content.strip()
                })
            else:
                transformed_messages.append({
                    "role": role,
                    "content": content
                })
        
        if tools:
            # Inject tool schema into a system message
            tools_desc = json.dumps(tools, indent=2)
            sys_msg = (
                "You have access to the following tools:\n"
                f"{tools_desc}\n\n"
                "To use a tool, you MUST output an XML block in your response like this:\n"
                "<tool name=\"tool_name\">\n"
                "{\"arg1\": \"value\"}\n"
                "</tool>\n"
                "You may include reasoning before or after the tool call. "
                "If you call a tool, I will reply with the result in a <tool_result> block."
            )
            
            # Prepend or append to system prompt
            if transformed_messages and transformed_messages[0].get("role") == "system":
                transformed_messages[0]["content"] += "\n\n" + sys_msg
            else:
                transformed_messages.insert(0, {"role": "system", "content": sys_msg})

        data: dict[str, Any] = {
            "model": self.model,
            "messages": transformed_messages,
        }
        
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        retries = 0
        backoff = self.initial_backoff
        
        logger.info(f"Starting complete call to {url} with model {self.model}")
        
        while True:
            try:
                logger.debug(f"Sending request (retry {retries})...")
                with urllib.request.urlopen(req, timeout=60) as response:
                    status_code = response.getcode()
                    body = response.read().decode("utf-8")
                    logger.debug(f"Received HTTP {status_code} response")
                    
                if status_code != 200:
                    logger.error(f"HTTP {status_code} error: {body}")
                    raise LLMError(f"HTTP {status_code}: {body}")
                
                resp_data = json.loads(body)
                logger.debug(f"Successfully parsed JSON response")
                return self._parse_response(resp_data)
                
            except urllib.error.HTTPError as e:
                logger.warning(f"HTTPError {e.code} on request")
                # Retry on 429 (Too Many Requests) or 5xx (Server Errors)
                if e.code == 429 or 500 <= e.code < 600:
                    if retries >= self.max_retries:
                        logger.error("Max retries reached on HTTPError")
                        raise LLMError(f"Max retries reached. Last HTTP error: {e.code} - {e.read().decode('utf-8', errors='ignore')}")
                    logger.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
                    retries += 1
                    backoff *= 2
                else:
                    logger.error(f"Unrecoverable HTTPError: {e.code}")
                    raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
            except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
                error_msg = getattr(e, 'reason', str(e))
                logger.warning(f"URLError (Network/Timeout) on request: {error_msg}")
                # Network error, potentially transient
                if retries >= self.max_retries:
                    logger.error("Max retries reached on URLError")
                    raise LLMError(f"Max retries reached. Network error: {e.reason}")
                logger.info(f"Retrying in {backoff}s...")
                time.sleep(backoff)
                retries += 1
                backoff *= 2
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON response")
                raise LLMError("Failed to decode JSON response from provider")
            except Exception as e:
                logger.error(f"Unexpected exception: {e}")
                raise LLMError(str(e))

    def _parse_response(self, resp_data: dict[str, Any]) -> LLMResponse:
        logger.debug(f"Raw response: {json.dumps(resp_data)}")
        try:
            choice = resp_data["choices"][0]["message"]
            text = choice.get("content") or ""
            
            tool_calls = []
            
            # Extract Native tool_calls if provided by the API
            if "tool_calls" in choice:
                for tc in choice["tool_calls"]:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        tc_id = tc.get("id", str(uuid.uuid4()))
                        name = func.get("name", "")
                        raw_args = func.get("arguments", "{}")
                        try:
                            parsed_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            parsed_args = None
                        tool_calls.append(ToolCall(
                            id=tc_id,
                            name=name,
                            arguments=parsed_args if parsed_args is not None else {},
                            raw_arguments=None if parsed_args is not None else raw_args
                        ))
            
            # Fuzzy Regex Extraction for XML-style tool calls
            pattern = r'<tool\s+name="([^"]+)">\s*(.*?)\s*</tool>'
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                name = match.group(1)
                raw_args = match.group(2).strip()
                
                # Markdown stripping (strip ```json and ``` if present)
                if raw_args.startswith("```json"):
                    raw_args = raw_args[7:]
                elif raw_args.startswith("```"):
                    raw_args = raw_args[3:]
                if raw_args.endswith("```"):
                    raw_args = raw_args[:-3]
                raw_args = raw_args.strip()
                
                try:
                    args = json.loads(raw_args) if raw_args else {}
                    tool_calls.append(ToolCall(
                        id=uuid.uuid4().hex,
                        name=name,
                        arguments=args
                    ))
                except json.JSONDecodeError as e:
                    # Structured Failure Recovery
                    tool_calls.append(ToolCall(
                        id=uuid.uuid4().hex,
                        name="parsing_error",
                        arguments={"error": f"JSON Decode Error: {str(e)}", "raw_content": raw_args}
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
                provider="react-adapter"
            )
        except Exception as e:
            raise LLMError(f"Error parsing response: {e}")
