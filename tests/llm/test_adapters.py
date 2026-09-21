import pytest
from agentic_rca.llm.base import LLMResponse, ToolCall, LLMError
from agentic_rca.llm.adapters.scripted import ScriptedLLM
from agentic_rca.llm.adapters.replay import ReplayLLM

def test_scripted_llm():
    def callback(messages, tools):
        if "hello" in messages[-1].get("content", ""):
            return LLMResponse(text="hi there", input_tokens=10, output_tokens=5, model="test-scripted")
        return LLMResponse(error="unknown")
        
    client = ScriptedLLM(callback)
    
    resp1 = client.chat([{"role": "user", "content": "hello"}])
    assert resp1.text == "hi there"
    assert resp1.input_tokens == 10
    assert resp1.output_tokens == 5
    assert resp1.model == "test-scripted"
    
    resp2 = client.chat([{"role": "user", "content": "bye"}])
    assert resp2.error == "unknown"
    
def test_replay_llm():
    responses = [
        LLMResponse(text="response 1", model="model-1"),
        LLMResponse(text="response 2", model="model-1")
    ]
    client = ReplayLLM(responses)
    
    resp1 = client.chat([{"role": "user", "content": "1"}])
    assert resp1.text == "response 1"
    
    resp2 = client.chat([{"role": "user", "content": "2"}])
    assert resp2.text == "response 2"
    
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "3"}])

def test_react_adapter_parse_response():
    from agentic_rca.llm.adapters.react_adapter import ReActAdapter
    adapter = ReActAdapter(api_key="test")
    
    # Test valid tool call
    resp_data = {
        "choices": [{"message": {"content": "Thinking...\n<tool name=\"search\">\n{\"query\": \"test\"}\n</tool>\nDone."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "model": "test-model"
    }
    
    result = adapter._parse_response(resp_data)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "test"}
    
    # Test markdown stripping and fuzzy parsing
    resp_data2 = {
        "choices": [{"message": {"content": "Here is the call:\n<tool name=\"db_query\">\n```json\n{\"sql\": \"SELECT *\"}\n```\n</tool>"}}]
    }
    result2 = adapter._parse_response(resp_data2)
    assert len(result2.tool_calls) == 1
    assert result2.tool_calls[0].name == "db_query"
    assert result2.tool_calls[0].arguments == {"sql": "SELECT *"}

    # Test malformed JSON recovery
    resp_data3 = {
        "choices": [{"message": {"content": "<tool name=\"bad_tool\">\n{bad json}\n</tool>"}}]
    }
    result3 = adapter._parse_response(resp_data3)
    assert len(result3.tool_calls) == 1
    assert result3.tool_calls[0].name == "parsing_error"
    assert "JSON Decode Error" in result3.tool_calls[0].arguments["error"]
