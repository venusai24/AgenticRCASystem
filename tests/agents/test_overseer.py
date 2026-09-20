import pytest
from agentic_rca.agents.overseer import Overseer

class MockLLM:
    def __init__(self):
        self.calls = 0
    def complete(self, prompt_id, msgs, tool_defs, **kwargs):
        self.calls += 1
        class MockResp:
            text = "You are repeating queries."
        return MockResp(), None

def test_overseer_no_stagnation():
    llm = MockLLM()
    overseer = Overseer(llm, nudge_threshold=3, stop_threshold=8)
    # Step 1: ledger moves to 1
    v = overseer.after_step({"action": "q", "ledger_seq_after": 1}, None)
    assert v is None
    assert overseer.stagnant_steps == 0
    # Step 2: ledger moves to 2
    v = overseer.after_step({"action": "q", "ledger_seq_after": 2}, None)
    assert v is None
    assert overseer.stagnant_steps == 0
    assert llm.calls == 0

def test_overseer_stagnation_nudge_and_stop():
    llm = MockLLM()
    overseer = Overseer(llm, nudge_threshold=3, stop_threshold=8)
    
    # Establish base seq
    overseer.after_step({"action": "query", "ledger_seq_after": 1}, None)
    
    # 2 steps no change -> no nudge
    assert overseer.after_step({"action": "query", "ledger_seq_after": 1}, None) is None
    assert overseer.after_step({"action": "query", "ledger_seq_after": 1}, None) is None
    
    # 3rd step no change -> nudge
    v = overseer.after_step({"action": "query", "ledger_seq_after": 1}, None)
    assert v is not None
    assert v[0] == "nudge"
    assert "repeating queries" in v[1]
    assert llm.calls == 1
    
    # 4-7 steps no change -> no nudge
    for i in range(4):
        v = overseer.after_step({"action": "query", "ledger_seq_after": 1}, None)
        assert v is None
        
    # 8th step no change -> stop
    v = overseer.after_step({"action": "query", "ledger_seq_after": 1}, None)
    assert v is not None
    assert v[0] == "stop"
    assert "Maximum stagnation threshold" in v[1]

def test_overseer_ignores_no_action():
    overseer = Overseer(None, nudge_threshold=3, stop_threshold=8)
    overseer.after_step({"action": "query", "ledger_seq_after": 1}, None)
    # no_actions shouldn't count towards stagnation
    v = overseer.after_step({"action": "no_action", "ledger_seq_after": 1}, None)
    assert v is None
    assert overseer.stagnant_steps == 0
