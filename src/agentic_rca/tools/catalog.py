"""Importing this module registers every data tool (populates registry.TOOLS).
The loop, the verifier and re-execution all import it, so the set of tools a
run can use is the set a verifier can re-execute."""

from agentic_rca.tools import discovery, sql, sandbox, metrics, changepoints, traces, logs, crosssource  # noqa: F401
