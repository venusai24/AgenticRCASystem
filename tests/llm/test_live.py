import os
import pytest
from agentic_rca.llm.adapters.react_adapter import ReActAdapter

@pytest.mark.live
def test_live_react_adapter():
    api_key = os.environ.get("META_API_KEY")
    if not api_key:
        pytest.skip("META_API_KEY not set")
        
    base_url = os.environ.get("META_API_BASE", "https://api.openai.com/v1")
    model = "meta-llama/Llama-3.1-8B-Instruct"
    
    adapter = ReActAdapter(api_key=api_key, base_url=base_url, model=model)
    
    resp = adapter.chat([{"role": "user", "content": "Say 'hello world' and nothing else."}])
    
    assert resp.error is None
    assert resp.text is not None
    assert "hello world" in resp.text.lower()
