import os
from typing import Any

from agentic_rca.llm.base import LLMClient
from agentic_rca.llm.adapters.react_adapter import ReActAdapter
from agentic_rca.llm.logger import LoggedLLMClient

def get_client(role: str, con: Any, run_id: str) -> LLMClient:
    """
    Returns an LLMClient configured for the given role, logging calls to the provided duckdb connection.
    Defaults to Muse Spark 1.3 Contributor for all roles.
    """
    # X5 routing dictates every role uses Muse Spark 1.3 Contributor on free tier.
    model = "meta-llama/Llama-3.1-8B-Instruct" # Or another specific identifier for Muse Spark 1.3
    api_key = os.environ.get("META_API_KEY", "dummy")
    # Example URL for Meta-compatible API if there is one, or standard OpenAI if mocked
    base_url = os.environ.get("META_API_BASE", "https://api.openai.com/v1") 
    
    adapter = ReActAdapter(api_key=api_key, base_url=base_url, model=model)
    return LoggedLLMClient(adapter, con, run_id, role)
