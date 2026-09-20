import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import duckdb

from agentic_rca.answer import answer_question, AnswerError
from agentic_rca.llm.base import LLMClient, LLMResponse

class MockLLMClient(LLMClient):
    def __init__(self, response_text: str):
        super().__init__()
        self.response_text = response_text
        
    def _complete(self, role, messages, tools, response_format):
        return LLMResponse(text=self.response_text, input_tokens=10, output_tokens=10)

@pytest.fixture
def mock_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_bundle"
    bundle_dir.mkdir()
    
    # Create fake report and coverage
    (bundle_dir / "report.md").write_text("Fake Report with e123")
    (bundle_dir / "coverage_human.md").write_text("Fake Coverage with c456")
    
    # Create a fake run.duckdb
    db_path = bundle_dir / "run.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE ledger_events (id VARCHAR)")
    con.execute("CREATE TABLE coverage (id VARCHAR)")
    con.execute("INSERT INTO ledger_events VALUES ('e123')")
    con.execute("INSERT INTO coverage VALUES ('c456')")
    con.close()
    
    return bundle_dir

def test_answer_success_evidence(mock_bundle):
    llm = MockLLMClient(response_text='{"shape": "evidence", "citations": ["e123", "c456"]}')
    
    with patch("agentic_rca.answer.verify_bundle") as mock_verify:
        ans = answer_question(mock_bundle, "Did you find it?", llm)
        
    assert mock_verify.called
    assert ans["shape"] == "evidence"
    assert ans["citations"] == ["e123", "c456"]

def test_answer_refuted(mock_bundle):
    llm = MockLLMClient(response_text='{"shape": "refuted", "citations": ["e123"]}')
    
    with patch("agentic_rca.answer.verify_bundle") as mock_verify:
        ans = answer_question(mock_bundle, "Did you find it?", llm)
        
    assert ans["shape"] == "refuted"
    assert ans["citations"] == ["e123"]

def test_answer_not_examined(mock_bundle):
    llm = MockLLMClient(response_text='{"shape": "not_examined", "citations": []}')
    
    with patch("agentic_rca.answer.verify_bundle") as mock_verify:
        ans = answer_question(mock_bundle, "Is it real?", llm)
        
    assert ans["shape"] == "not_examined"
    assert ans["citations"] == []

def test_answer_rejects_bad_citation(mock_bundle):
    # LLM hallucinated 'e999' which is not in the db
    llm = MockLLMClient(response_text='{"shape": "evidence", "citations": ["e999"]}')
    
    with patch("agentic_rca.answer.verify_bundle") as mock_verify:
        with pytest.raises(AnswerError, match="not found in ledger"):
            answer_question(mock_bundle, "Question?", llm)

def test_answer_rejects_invalid_json(mock_bundle):
    llm = MockLLMClient(response_text='not a json')
    
    with patch("agentic_rca.answer.verify_bundle") as mock_verify:
        with pytest.raises(AnswerError, match="valid JSON"):
            answer_question(mock_bundle, "Question?", llm)

def test_answer_rejects_invalid_shape(mock_bundle):
    llm = MockLLMClient(response_text='{"shape": "maybe", "citations": []}')
    
    with patch("agentic_rca.answer.verify_bundle") as mock_verify:
        with pytest.raises(AnswerError, match="Invalid shape"):
            answer_question(mock_bundle, "Question?", llm)

def test_answer_fails_verify_bundle(mock_bundle):
    llm = MockLLMClient(response_text='{"shape": "not_examined", "citations": []}')
    from agentic_rca.report.bundle import BundleError
    
    with patch("agentic_rca.answer.verify_bundle", side_effect=BundleError("bad manifest")):
        with pytest.raises(AnswerError, match="bundle verification failed"):
            answer_question(mock_bundle, "Question?", llm)
