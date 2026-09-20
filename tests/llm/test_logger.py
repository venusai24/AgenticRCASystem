import duckdb
import json
from agentic_rca.llm.base import LLMResponse, ToolCall
from agentic_rca.llm.adapters.scripted import ScriptedLLM
from agentic_rca.llm.logger import LoggedLLMClient

def test_logger_records_successful_call():
    con = duckdb.connect(":memory:")
    
    def callback(messages, tools):
        return LLMResponse(
            text="hello from llm",
            tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "test"})],
            input_tokens=10,
            output_tokens=5,
            model="test-model"
        )
        
    client = ScriptedLLM(callback)
    logged_client = LoggedLLMClient(client, con, run_id="run-123", role="lead")
    
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "search"}}]
    
    resp = logged_client.chat(messages, tools)
    assert resp.text == "hello from llm"
    
    # Check the database
    rows = con.execute("SELECT * FROM llm_calls").fetchall()
    assert len(rows) == 1
    
    row = rows[0]
    # schema: run_id, call_id, role, model, prompt_id, prompt_sha, request, response, tokens, latency, error
    assert row[0] == "run-123" # run_id
    assert row[2] == "lead" # role
    assert row[3] == "test-model" # model
    
    # Check request JSON
    request_data = json.loads(row[6])
    assert request_data["messages"] == messages
    assert request_data["tools"] == tools
    
    # Check response JSON
    response_data = json.loads(row[7])
    assert response_data["text"] == "hello from llm"
    assert len(response_data["tool_calls"]) == 1
    assert response_data["tool_calls"][0]["name"] == "search"
    
    assert row[8] == 15 # tokens
    assert row[9] > 0 # latency
    assert row[10] is None # error
