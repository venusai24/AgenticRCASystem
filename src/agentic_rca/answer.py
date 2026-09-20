import json
from pathlib import Path
from typing import Any
import duckdb

from agentic_rca.llm.base import LLMClient
from agentic_rca.report.bundle import verify_bundle, BundleError

class AnswerError(Exception):
    pass

def answer_question(bundle_dir: Path, question: str, llm: LLMClient) -> dict[str, Any]:
    """
    Perform read-only Q&A over a frozen bundle.
    """
    # 1. Verify bundle integrity first (refuse on mismatch)
    try:
        verify_bundle(bundle_dir)
    except BundleError as e:
        raise AnswerError(f"Refused: bundle verification failed: {e}")

    # 2. Extract context from bundle
    report_path = bundle_dir / "report.md"
    coverage_path = bundle_dir / "coverage_human.md"
    
    if not report_path.exists():
        raise AnswerError("report.md not found in bundle.")
        
    context_text = f"--- REPORT ---\n{report_path.read_text('utf-8')}\n"
    if coverage_path.exists():
        context_text += f"--- COVERAGE ---\n{coverage_path.read_text('utf-8')}\n"

    # 3. Ask the LLM
    prompt = f"""You are a strict, read-only question answerer for an incident investigation bundle.
Your job is to answer the user's question using ONLY the provided Report and Coverage context.

You must respond with EXACTLY ONE of the following three shapes in valid JSON (no markdown wrapping, no extra keys):

Shape 1: If there is evidence in the report confirming the question/hypothesis.
{{
  "shape": "evidence",
  "citations": ["e123", "c45"]
}}

Shape 2: If there is evidence in the report refuting the question/hypothesis.
{{
  "shape": "refuted",
  "citations": ["e123", "c45"]
}}

Shape 3: If the entity or concept was never examined or there is no coverage for it.
{{
  "shape": "not_examined",
  "citations": []
}}

RULES:
- "not examined" must NOT be reasoned into a plausible yes.
- Citations must be the exact IDs (e.g., "e123" or "c45") found in the text.
- Do NOT include any other keys in the JSON object. Do not explain your answer outside the JSON.

Context:
{context_text}

Question:
{question}
"""
    
    response, _ = llm.complete("reader", [{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    if response.error:
        raise AnswerError(f"LLM Error: {response.error}")
        
    try:
        data = json.loads(response.text or "{}")
    except json.JSONDecodeError:
        raise AnswerError("LLM failed to return valid JSON.")
        
    shape = data.get("shape")
    citations = data.get("citations", [])
    
    if shape not in ("evidence", "refuted", "not_examined"):
        raise AnswerError(f"Invalid shape returned: {shape}")
        
    if not isinstance(citations, list):
        raise AnswerError("citations must be a list of strings.")

    # 4. Deterministic post-check
    # Check that every cited evidence/coverage ID exists in the bundle's duckdb
    if citations:
        db_path = bundle_dir / "run.duckdb"
        if not db_path.exists():
            raise AnswerError("run.duckdb not found in bundle.")
            
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            for cit in citations:
                # Check ledger_events (evidence) or coverage table
                if cit.startswith("e"):
                    row = con.execute("SELECT 1 FROM ledger_events WHERE id = ?", [cit]).fetchone()
                    if not row:
                        raise AnswerError(f"Citation {cit} rejected: evidence ID not found in ledger.")
                elif cit.startswith("c"):
                    row = con.execute("SELECT 1 FROM coverage WHERE id = ?", [cit]).fetchone()
                    if not row:
                        raise AnswerError(f"Citation {cit} rejected: coverage ID not found in ledger.")
                else:
                    raise AnswerError(f"Citation {cit} rejected: unrecognized ID format (must start with 'e' or 'c').")
        finally:
            con.close()

    return {"shape": shape, "citations": citations}
