"""CLI: python -m agentic_rca ingest | run | verify-bundle"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import signal
    def sigint_handler(sig, frame):
        print("\nInterrupted by user (Ctrl-C). Exiting immediately.", file=sys.stderr)
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    p = argparse.ArgumentParser(prog="agentic_rca")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest", help="build the DuckDB store from a data directory")
    pi.add_argument("--data", default="incident_data")
    pi.add_argument("--store", default=".cache/store.duckdb")
    pr = sub.add_parser("run", help="run one investigation and freeze its bundle")
    pr.add_argument("--store", default=".cache/store.duckdb")
    pr.add_argument("--data", default="incident_data")
    pr.add_argument("--runs", default="runs")
    pr.add_argument("--incident-ts", required=True, help="epoch seconds or ISO-8601 (UTC)")
    pr.add_argument("--hypothesis", action="append", default=[], help="a human hunch (repeatable)")
    pr.add_argument("--max-steps", type=int, default=60)
    pv = sub.add_parser("verify-bundle", help="recompute a bundle's hashes and parent chain")
    pv.add_argument("bundle")
    pv.add_argument("--runs", default="runs")
    pa = sub.add_parser("answer", help="ask a question over a frozen bundle")
    pa.add_argument("bundle")
    pa.add_argument("question")
    pam = sub.add_parser("amend", help="amend a frozen run bundle")
    pam.add_argument("bundle")
    pam.add_argument("--runs", default="runs")
    pam.add_argument("--hypothesis", action="append", default=[])
    pam.add_argument("--question")
    pam.add_argument("--max-steps", type=int, default=20)
    a = p.parse_args(argv)
    if a.cmd == "ingest":
        from agentic_rca.ingest.pipeline import build_index

        Path(a.store).parent.mkdir(parents=True, exist_ok=True)
        r = build_index(Path(a.data), Path(a.store))
        r.con.close()
        print(json.dumps(r.row_counts))
        return 0
    if a.cmd == "verify-bundle":
        from agentic_rca.report.bundle import BundleError, verify_bundle

        try:
            print(json.dumps(verify_bundle(Path(a.bundle))))
            return 0
        except BundleError as exc:
            print(f"BUNDLE INVALID: {exc}", file=sys.stderr)
            return 1
            
    if a.cmd == "answer":
        import os
        from agentic_rca.answer import answer_question, AnswerError
        try:
            from agentic_rca.llm.adapters.react_adapter import ReActAdapter
        except ImportError:
            print("LLM adapter could not be imported.", file=sys.stderr)
            return 2

        model = os.environ.get("META_API_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        api_key = os.environ.get("META_API_KEY", "dummy")
        base_url = os.environ.get("META_API_BASE", "https://api.openai.com/v1")
        
        if api_key == "dummy":
            from agentic_rca.llm.base import ScriptedLLM, LLMResponse
            llm = ScriptedLLM(default=lambda r, m, t: LLMResponse(text='{"shape": "not_examined", "citations": []}', input_tokens=10, output_tokens=10))
        else:
            llm = ReActAdapter(api_key=api_key, base_url=base_url, model=model)
            
        try:
            print(json.dumps(answer_question(Path(a.bundle), a.question, llm), default=str))
            return 0
        except AnswerError as exc:
            print(f"ANSWER ERROR: {exc}", file=sys.stderr)
            return 1
    from agentic_rca.agents.loop import Budget
    from agentic_rca.run import RunConfig, run_investigation
    from agentic_rca.tools._common import _to_epoch_s

    import os
    try:
        from agentic_rca.llm.adapters.openai_adapter import OpenAIAdapter
        from agentic_rca.llm.adapters.react_adapter import ReActAdapter
    except ImportError:
        print("LLM adapter could not be imported.", file=sys.stderr)
        return 2

    model = os.environ.get("META_API_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = os.environ.get("META_API_KEY", "dummy")
    base_url = os.environ.get("META_API_BASE", "https://api.openai.com/v1")
    
    if api_key == "dummy":
        from agentic_rca.llm.base import ScriptedLLM, call
        llm = ScriptedLLM(default=lambda r, m, t: call("request_termination"))
    else:
        if "muse-spark" in model:
            llm = ReActAdapter(api_key=api_key, base_url=base_url, model=model)
        else:
            llm = OpenAIAdapter(api_key=api_key, base_url=base_url, model=model)
    
    if a.cmd == "amend":
        from agentic_rca.report.bundle import verify_bundle, BundleError
        try:
            verify_bundle(Path(a.bundle))
        except BundleError as exc:
            print(f"BUNDLE INVALID: {exc}", file=sys.stderr)
            return 1
            
        manifest_path = Path(a.bundle) / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        if manifest.get("depth", 0) >= 3:
            print("this question needs a fresh full run (depth cap exceeded)", file=sys.stderr)
            return 1
            
        root_run_id = manifest.get("root_run_id")
        count = 0
        if root_run_id:
            for m in Path(a.runs).glob("*/manifest.json"):
                try:
                    with open(m, "r") as mf:
                        if json.load(mf).get("root_run_id") == root_run_id:
                            count += 1
                except Exception:
                    pass
        if count >= 5:
            print("this question needs a fresh full run (count cap exceeded)", file=sys.stderr)
            return 1
            
        incident_ts = None
        # Retrieve incident_ts from parent run's DB because it is required for RunConfig
        import duckdb
        with duckdb.connect(str(Path(a.bundle) / "run.duckdb"), read_only=True) as con:
            res = con.execute("SELECT payload FROM run_inputs WHERE name = 'incident_ts'").fetchone()
            if res:
                incident_ts = json.loads(res[0])
                
        if incident_ts is None:
            print("Failed to recover incident_ts from parent bundle", file=sys.stderr)
            return 1
            
        hypotheses = a.hypothesis.copy()
        if a.question:
            hypotheses.append(a.question)
            
        cfg = RunConfig(
            store_path=Path(manifest.get("files", {}).get("store.duckdb", ".cache/store.duckdb")), # Fake store path, but usually passed or not needed if read-only
            runs_root=Path(a.runs),
            incident_ts=incident_ts,
            data_dir=None,
            human_hypotheses=hypotheses,
            budget=Budget(max_steps=a.max_steps, max_tokens=1500000), # amendment budget
            parent_run_dir=Path(a.bundle)
        )
        print(json.dumps(run_investigation(cfg, llm), default=str))
        return 0
        
    cfg = RunConfig(store_path=Path(a.store), runs_root=Path(a.runs), incident_ts=_to_epoch_s(a.incident_ts),
                    data_dir=Path(a.data), human_hypotheses=a.hypothesis, budget=Budget(max_steps=a.max_steps))
    print(json.dumps(run_investigation(cfg, llm), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
