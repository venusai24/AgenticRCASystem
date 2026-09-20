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
