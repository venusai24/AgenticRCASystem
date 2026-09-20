import os
import pytest
from agentic_rca.llm.adapters.openai_adapter import OpenAIAdapter

@pytest.mark.live
def test_live_openai_adapter():
    api_key = os.environ.get("META_API_KEY")
    if not api_key:
        pytest.skip("META_API_KEY not set")
        
    base_url = os.environ.get("META_API_BASE", "https://api.openai.com/v1")
    model = "gpt-4o-mini" # or fallback to whatever model is desired, e.g. meta-llama/Llama-3.1-8B-Instruct
    
    adapter = OpenAIAdapter(api_key=api_key, base_url=base_url, model=model)
    
    resp = adapter.chat([{"role": "user", "content": "Say 'hello world' and nothing else."}])
    
    assert resp.error is None
    assert resp.text is not None
    assert "hello world" in resp.text.lower()
