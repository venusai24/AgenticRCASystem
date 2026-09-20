import json
import pytest
from unittest.mock import Mock, patch

from agentic_rca.agents.reader import ReaderAgent, ReaderResult
from agentic_rca.llm.base import LLMResponse
from agentic_rca.tools.registry import ToolRuntime
from agentic_rca.tools.envelope import ToolEnvelope, ToolError, ResultHandle, Provenance

@pytest.fixture
def mock_runtime():
    runtime = Mock(spec=ToolRuntime)
    
    # Setup lookup mock
    runtime.lookup.return_value = [
        {"_row": 0, "val": "A"},
        {"_row": 1, "val": "B"},
        {"_row": 2, "val": "C"}
    ]
    
    # Setup inspect mock
    runtime.inspect.return_value = ToolEnvelope(
        query_id="q123",
        result_handle=ResultHandle("handle1", 3),
        row_count=3,
        preview=[{"_row": 0, "val": "A"}],
        summary_stats={},
        truncated=True,
        coverage={},
        provenance=Provenance("test")
    )
    return runtime


@pytest.fixture
def mock_client():
    client = Mock()
    return client


def test_reader_ask_no_tools(mock_client, mock_runtime):
    agent = ReaderAgent(mock_client, mock_runtime)
    
    # LLM responds directly with claims
    resp = LLMResponse(text='{"claims": [{"text": "Found A", "row_ids": [0]}]}', input_tokens=10, output_tokens=10)
    mock_client.complete.return_value = (resp, None)
    
    result = agent.ask("handle1", "What did you find?", 5, 1)
    
    assert len(result.claims) == 1
    assert result.claims[0]["text"] == "Found A"
    assert result.claims[0]["row_ids"] == [0]
    assert result.dropped_claims == 0
    assert result.iterations == 1


def test_reader_ask_with_inspect(mock_client, mock_runtime):
    agent = ReaderAgent(mock_client, mock_runtime)
    
    # First turn: LLM calls inspect_result
    tc1 = {"id": "call_1", "name": "inspect_result", "args": {"handle_id": "handle1", "limit": 1}}
    resp1 = LLMResponse(text="", tool_calls=[tc1], input_tokens=10, output_tokens=10)
    
    # Second turn: LLM gives claims
    resp2 = LLMResponse(text='{"claims": [{"text": "Found A", "row_ids": [0]}]}', input_tokens=10, output_tokens=10)
    
    mock_client.complete.side_effect = [(resp1, None), (resp2, None)]
    
    result = agent.ask("handle1", "What did you find?", 5, 1)
    
    assert len(result.claims) == 1
    assert mock_runtime.inspect.called
    assert result.iterations == 2


def test_reader_filters_hallucinated_rows(mock_client, mock_runtime):
    agent = ReaderAgent(mock_client, mock_runtime)
    
    # LLM hallucinated row_id 99
    resp = LLMResponse(
        text='{"claims": [{"text": "Found A", "row_ids": [0]}, {"text": "Fake", "row_ids": [99]}]}', 
        input_tokens=10, 
        output_tokens=10
    )
    mock_client.complete.return_value = (resp, None)
    
    result = agent.ask("handle1", "What did you find?", 5, 1)
    
    assert len(result.claims) == 1
    assert result.claims[0]["text"] == "Found A"
    assert result.dropped_claims == 1


def test_reader_max_claims(mock_client, mock_runtime):
    agent = ReaderAgent(mock_client, mock_runtime)
    
    resp = LLMResponse(
        text='{"claims": [{"text": "Found A", "row_ids": [0]}, {"text": "Found B", "row_ids": [1]}]}', 
        input_tokens=10, 
        output_tokens=10
    )
    mock_client.complete.return_value = (resp, None)
    
    # Request max 1 claim
    result = agent.ask("handle1", "What did you find?", 1, 1)
    
    assert len(result.claims) == 1
    assert result.claims[0]["text"] == "Found A"
    assert result.dropped_claims == 0


def test_reader_handles_bad_json(mock_client, mock_runtime):
    agent = ReaderAgent(mock_client, mock_runtime)
    
    resp = LLMResponse(text='not json', input_tokens=10, output_tokens=10)
    mock_client.complete.return_value = (resp, None)
    
    result = agent.ask("handle1", "What did you find?", 5, 1)
    
    assert len(result.claims) == 0
    assert result.dropped_claims == 0


def test_reader_returns_empty_when_handle_missing(mock_client, mock_runtime):
    # Lookup returns None
    mock_runtime.lookup.return_value = None
    agent = ReaderAgent(mock_client, mock_runtime)
    
    resp = LLMResponse(text='{"claims": [{"text": "Found A", "row_ids": [0]}]}', input_tokens=10, output_tokens=10)
    mock_client.complete.return_value = (resp, None)
    
    result = agent.ask("handle_missing", "What did you find?", 5, 1)
    
    assert len(result.claims) == 0
    assert result.dropped_claims == 1
