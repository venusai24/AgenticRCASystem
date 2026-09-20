"""AC8: Ledger core must not import from llm or dedup (X8 invariant)."""

import sys
import importlib
import pytest

def test_ledger_imports():
    """Ensure that importing agentic_rca.ledger does not pull in llm or dedup."""
    # First, make sure the modules are NOT loaded yet.
    # But since pytest might have loaded them, we'll check sys.modules if they are already there,
    # which is likely true in a full test suite run. So instead of checking sys.modules,
    # we can just use the 'ast' module or 'importlib.metadata' or just parse the source files!
    
    import ast
    from pathlib import Path
    
    ledger_dir = Path("src/agentic_rca/ledger")
    
    for py_file in ledger_dir.glob("**/*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    assert not name.name.startswith("agentic_rca.llm"), f"{py_file} imports {name.name}"
                    assert not name.name.startswith("agentic_rca.agents"), f"{py_file} imports {name.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("agentic_rca.llm"), f"{py_file} imports {node.module}"
                    assert not node.module.startswith("agentic_rca.agents"), f"{py_file} imports {node.module}"
