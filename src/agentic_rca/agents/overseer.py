from typing import Any
import json

class Overseer:
    def __init__(self, llm_client: Any, nudge_threshold: int = 3, stop_threshold: int = 8):
        self.llm = llm_client
        self.nudge_threshold = nudge_threshold
        self.stop_threshold = stop_threshold
        self.stagnant_steps = 0
        self.last_ledger_seq = -1
        self.recent_actions = []

    def after_step(self, traj: dict, ledger: Any) -> tuple[str, str] | None:
        action = traj.get("action")
        # Ignore non-actions or explicit loop controls that don't query
        if not action or action == "no_action":
            return None
            
        seq_after = traj.get("ledger_seq_after", 0)
        
        # Did the ledger sequence advance, or is this the first step?
        if seq_after > self.last_ledger_seq or self.last_ledger_seq == -1:
            self.stagnant_steps = 0
            self.last_ledger_seq = seq_after
            self.recent_actions.clear()
            return None

        self.last_ledger_seq = seq_after
        self.stagnant_steps += 1
        
        # Track the action for summarization
        # Don't record ledger views or inspect_result as strong stagnation indicators 
        # unless they dominate, but we'll include them in the summary.
        action_name = traj.get("action")
        self.recent_actions.append(action_name)
        
        if self.stagnant_steps >= self.stop_threshold:
            return ("stop", "Maximum stagnation threshold reached: no ledger changes for 8 steps.")
            
        if self.stagnant_steps >= self.nudge_threshold:
            # We hit the nudge threshold. Generate a nudge message.
            # Reset counter so we don't spam nudges every single step, 
            # though if it hits stop_threshold, it will stop.
            # Wait, if we reset it, it won't hit stop_threshold!
            # Instead of resetting, we can nudge only at exactly nudge_threshold,
            # or every N steps. Let's nudge exactly at the threshold.
            if self.stagnant_steps == self.nudge_threshold:
                summary = self._summarize_stagnation()
                return ("nudge", f"Your last {self.nudge_threshold} steps haven't changed the ledger. {summary} Review open questions and unexamined coverage.")
            
        return None

    def _summarize_stagnation(self) -> str:
        if not self.llm:
            return "You appear to be stuck in a loop."
            
        # Use a very brief LLM call to summarize what they are repeating.
        prompt = (
            "You are a process monitor. The investigator has made the following sequence of tool calls "
            "without making any updates to the ledger. In one short sentence (under 15 words), "
            "summarize what they are repeating (e.g. 'You are repeatedly querying the same metric.').\n\n"
            f"Recent calls: {', '.join(self.recent_actions)}"
        )
        try:
            resp, _ = self.llm.complete("overseer", [{"role": "user", "content": prompt}], [])
            if resp.text:
                return resp.text.strip()
        except Exception:
            pass
        return "You appear to be repeating the same queries."
