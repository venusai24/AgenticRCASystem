import pytest
import json
import urllib.error
from unittest.mock import patch, MagicMock

from agentic_rca.llm.adapters.openai_adapter import OpenAIAdapter

def test_openai_adapter_success():
    adapter = OpenAIAdapter(api_key="test")
    
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": "gpt-4o-mini"
    }).encode("utf-8")
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        
        resp = adapter._complete("lead", [{"role": "user", "content": "hi"}])
        
        assert resp.text == "hello"
        assert resp.error is None
        assert mock_urlopen.call_count == 1

@patch("time.sleep") # Don't actually sleep in tests
def test_openai_adapter_retry_transient(mock_sleep):
    adapter = OpenAIAdapter(api_key="test", max_retries=2)
    
    error_429 = urllib.error.HTTPError(url="", code=429, msg="Too Many Requests", hdrs={}, fp=MagicMock())
    error_429.read.return_value = b"Rate limit"
    
    mock_success = MagicMock()
    mock_success.getcode.return_value = 200
    mock_success.read.return_value = json.dumps({
        "choices": [{"message": {"content": "success after retry"}}],
        "usage": {},
        "model": "gpt-4o-mini"
    }).encode("utf-8")
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        # Fail twice with 429, then succeed
        mock_urlopen.side_effect = [
            error_429,
            error_429,
            MagicMock(__enter__=lambda self: mock_success, __exit__=lambda *args: None)
        ]
        
        resp = adapter._complete("lead", [{"role": "user", "content": "hi"}])
        
        assert resp.text == "success after retry"
        assert resp.error is None
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

@patch("time.sleep")
def test_openai_adapter_no_retry_400(mock_sleep):
    adapter = OpenAIAdapter(api_key="test", max_retries=2)
    
    error_400 = urllib.error.HTTPError(url="", code=400, msg="Bad Request", hdrs={}, fp=MagicMock())
    error_400.read.return_value = b"Invalid JSON"
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = error_400
        
        resp = adapter._complete("lead", [{"role": "user", "content": "hi"}])
        
        assert resp.text is None
        assert "HTTP 400" in resp.error
        assert mock_urlopen.call_count == 1
        assert mock_sleep.call_count == 0
