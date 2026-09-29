import asyncio
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agentic_rca.run import RunConfig, run_investigation
from agentic_rca.agents.loop import Budget
from agentic_rca.tools._common import _to_epoch_s

try:
    from agentic_rca.llm.adapters.react_adapter import ReActAdapter
    from agentic_rca.llm.adapters.openai_adapter import OpenAIAdapter
except ImportError:
    ReActAdapter = None
    OpenAIAdapter = None

app = FastAPI(title="Agentic RCA GUI")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path("incident_data")
RUNS_DIR = Path("runs")
STORE_PATH = Path(".cache/store.duckdb")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def clean_thought_text(text: str) -> str:
    """Extract clean human reasoning by stripping tool tags and trailing JSON blocks."""
    if not text:
        return ""
    text = re.sub(r'<tool[^>]*>.*?</tool>', '', text, flags=re.DOTALL)
    text = re.sub(r'\s*\{.*$', '', text, flags=re.DOTALL)
    return text.strip()

class ActiveRunManager:
    def __init__(self):
        self.status = "idle"  # "idle" or "running"
        self.thoughts = []
        self.listeners = set()
        self.loop = None

    def start_run(self, loop):
        self.status = "running"
        self.thoughts = []
        self.listeners = set()
        self.loop = loop

    def add_thought(self, content):
        if content and (not self.thoughts or self.thoughts[-1] != content):
            self.thoughts.append(content)
            msg = {"type": "thought", "content": content}
            for q in list(self.listeners):
                asyncio.run_coroutine_threadsafe(q.put(msg), self.loop)

    def add_event(self, msg):
        for q in list(self.listeners):
            asyncio.run_coroutine_threadsafe(q.put(msg), self.loop)

    def finish_run(self, msg):
        self.status = "idle"
        self.add_event(msg)

active_run = ActiveRunManager()

class ObservableLLM(ReActAdapter if ReActAdapter else object):
    """Wrapper that intercepts LLM calls to stream clean reasoning steps."""
    def __init__(self, base_llm):
        self._base = base_llm
        self.con = base_llm.con
        self.run_id = base_llm.run_id

    def bind(self, con, run_id: str):
        self._base.bind(con, run_id)
        self.con = self._base.con
        self.run_id = self._base.run_id
        return self

    def _complete(self, role, messages, tools, response_format):
        resp = self._base._complete(role, messages, tools, response_format)
        if resp.text:
            cleaned = clean_thought_text(resp.text)
            if cleaned:
                active_run.add_thought(cleaned)
        return resp

    def complete(self, *args, **kwargs):
        try:
            resp, call_id = self._base.complete(*args, **kwargs)
            if resp.text:
                cleaned = clean_thought_text(resp.text)
                if cleaned:
                    active_run.add_thought(cleaned)
            return resp, call_id
        except Exception as e:
            active_run.add_event({"type": "error", "content": str(e)})
            raise

@app.get("/")
def read_root():
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text())

@app.get("/api/active_run")
def get_active_run():
    return {
        "status": active_run.status,
        "thoughts": active_run.thoughts
    }

@app.get("/api/runs")
def list_runs():
    runs = []
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.glob("r*"), key=lambda p: p.stat().st_mtime, reverse=True):
            manifest_file = run_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
                    runs.append({
                        "id": run_dir.name,
                        "timestamp": run_dir.stat().st_mtime,
                        "manifest": manifest
                    })
    return {"runs": runs}

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run_dir = RUNS_DIR / run_id
    manifest_file = run_dir / "manifest.json"
    report_file = run_dir / "report.md"
    duckdb_file = run_dir / "run.duckdb"
    
    if not manifest_file.exists():
        return JSONResponse(status_code=404, content={"error": "Run not found"})
        
    with open(manifest_file, "r") as f:
        manifest = json.load(f)
        
    report = ""
    if duckdb_file.exists():
        try:
            from agentic_rca.report.render import render_report
            report = render_report(str(duckdb_file))
        except Exception as e:
            if report_file.exists():
                with open(report_file, "r") as f:
                    report = f.read()
            else:
                report = f"Failed to render report: {str(e)}"
    elif report_file.exists():
        with open(report_file, "r") as f:
            report = f.read()

    thoughts = []
    if duckdb_file.exists():
        import duckdb
        try:
            with duckdb.connect(str(duckdb_file), read_only=True) as con:
                rows = con.execute("SELECT response FROM llm_calls WHERE response IS NOT NULL ORDER BY call_id").fetchall()
                for (resp_str,) in rows:
                    try:
                        d = json.loads(resp_str)
                        txt = d.get("text")
                        if txt:
                            cleaned = clean_thought_text(txt)
                            if cleaned and (not thoughts or thoughts[-1] != cleaned):
                                thoughts.append(cleaned)
                    except Exception:
                        pass
        except Exception:
            pass
            
    return {"manifest": manifest, "report": report, "thoughts": thoughts}

@app.post("/api/ingest")
async def ingest_files(
    baseline_app: UploadFile = File(None),
    baseline_container: UploadFile = File(None),
    cluster_app: UploadFile = File(None),
    cluster_container: UploadFile = File(None),
    incident_traces: UploadFile = File(None),
    incident_logs: UploadFile = File(None)
):
    DATA_DIR.mkdir(exist_ok=True)
    files_to_save = [
        (baseline_app, "baseline_app_metrics.csv"),
        (baseline_container, "baseline_container_metrics.csv"),
        (cluster_app, "cluster_app_metrics.csv"),
        (cluster_container, "container_metrics.csv"),
        (incident_traces, "incident_traces.csv"),
        (incident_logs, "cluster_incident_logs.csv"),
    ]
    
    saved_count = 0
    for file, expected_name in files_to_save:
        if file and file.filename:
            path = DATA_DIR / expected_name
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_count += 1
            
    if saved_count > 0:
        from agentic_rca.ingest.pipeline import build_index
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        r = build_index(DATA_DIR, STORE_PATH)
        r.con.close()
        return {"status": "success", "counts": r.row_counts}
    
    return JSONResponse(status_code=400, content={"error": "No files uploaded"})

def run_investigation_thread(cfg, llm):
    def on_hypothesis_write(action, uid):
        active_run.add_event({"type": "hypothesis", "action": action, "id": uid})
        
    try:
        res = run_investigation(cfg, llm, on_hypothesis_write=on_hypothesis_write)
        active_run.finish_run({"type": "done", "result": res})
    except Exception as e:
        active_run.finish_run({"type": "error", "content": str(e)})

@app.get("/api/start_run")
async def start_run(incident_ts: str, max_steps: int = 60, hypotheses: str = ""):
    if active_run.status == "running":
        return {"status": "already_running"}
        
    active_run.start_run(asyncio.get_running_loop())
    
    cfg = RunConfig(
        store_path=STORE_PATH, 
        runs_root=RUNS_DIR, 
        incident_ts=_to_epoch_s(incident_ts),
        data_dir=DATA_DIR if DATA_DIR.exists() else None, 
        human_hypotheses=[h.strip() for h in hypotheses.split(",") if h.strip()], 
        budget=Budget(max_steps=max_steps)
    )
    
    model = os.environ.get("META_API_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = os.environ.get("META_API_KEY", "dummy")
    base_url = os.environ.get("META_API_BASE", "https://api.openai.com/v1")
    
    if api_key == "dummy":
        from agentic_rca.llm.base import ScriptedLLM, call
        base_llm = ScriptedLLM(default=lambda r, m, t: call("request_termination"))
    else:
        if "muse-spark" in model and ReActAdapter:
            base_llm = ReActAdapter(api_key=api_key, base_url=base_url, model=model)
        elif OpenAIAdapter:
            base_llm = OpenAIAdapter(api_key=api_key, base_url=base_url, model=model)
        else:
            base_llm = None
            
    llm = ObservableLLM(base_llm)
    threading.Thread(target=run_investigation_thread, args=(cfg, llm)).start()
    return {"status": "started"}

@app.get("/api/start_amend")
async def start_amend(bundle: str, question: str = "", max_steps: int = 20):
    if active_run.status == "running":
        return {"status": "already_running"}
        
    active_run.start_run(asyncio.get_running_loop())
    
    parent_bundle = RUNS_DIR / bundle
    manifest_path = parent_bundle / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    incident_ts = None
    import duckdb
    with duckdb.connect(str(parent_bundle / "run.duckdb"), read_only=True) as con:
        res = con.execute("SELECT payload FROM run_inputs WHERE name = 'incident_ts'").fetchone()
        if res:
            incident_ts = json.loads(res[0])
            
    cfg = RunConfig(
        store_path=Path(manifest.get("files", {}).get("store.duckdb", ".cache/store.duckdb")),
        runs_root=RUNS_DIR,
        incident_ts=incident_ts,
        data_dir=None,
        human_hypotheses=[question] if question else [],
        budget=Budget(max_steps=max_steps, max_tokens=1500000),
        parent_run_dir=parent_bundle
    )
    
    model = os.environ.get("META_API_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = os.environ.get("META_API_KEY", "dummy")
    base_url = os.environ.get("META_API_BASE", "https://api.openai.com/v1")
    
    if api_key == "dummy":
        from agentic_rca.llm.base import ScriptedLLM, call
        base_llm = ScriptedLLM(default=lambda r, m, t: call("request_termination"))
    else:
        if "muse-spark" in model and ReActAdapter:
            base_llm = ReActAdapter(api_key=api_key, base_url=base_url, model=model)
        elif OpenAIAdapter:
            base_llm = OpenAIAdapter(api_key=api_key, base_url=base_url, model=model)
            
    llm = ObservableLLM(base_llm)
    threading.Thread(target=run_investigation_thread, args=(cfg, llm)).start()
    return {"status": "started"}

@app.get("/api/stream")
async def stream():
    queue = asyncio.Queue()
    active_run.listeners.add(queue)
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                msg = await queue.get()
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ["done", "error"]:
                    break
        finally:
            active_run.listeners.discard(queue)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
