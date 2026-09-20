"""Bundle freezing and verification."""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
import duckdb

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

class BundleError(Exception):
    pass

def freeze_bundle(partial_dir: str | Path, base_dir: str | Path = "runs", extra: dict | None = None) -> tuple[Path, str]:
    """Implements the 6-step freeze procedure (docs/design-d6-d7.md §13.3)."""
    partial_dir = Path(partial_dir)
    base_dir = Path(base_dir)
    run_id = partial_dir.name.replace(".partial", "")
    final_dir = base_dir / run_id
    
    if final_dir.exists():
        raise ValueError(f"Target bundle {final_dir} already exists.")
        
    if not partial_dir.exists():
        raise ValueError(f"Partial bundle {partial_dir} not found.")
        
    db_path = str(partial_dir / "run.duckdb")
    if os.path.exists(db_path + ".wal"):
        raise ValueError("run.duckdb.wal exists, the database was not cleanly checkpointed.")
        
    # 2. Render reports
    from agentic_rca.report.render import render_report, render_coverage_human
    report_path = os.path.join(partial_dir, "report.md")
    coverage_path = os.path.join(partial_dir, "coverage_human.md")
    
    with open(report_path, "w") as f:
        f.write(render_report(db_path))
        
    with open(coverage_path, "w") as f:
        f.write(render_coverage_human(db_path))
        
    # Copy prompts (in a real run we'd read llm_calls.prompt_sha, here we stub the copy logic)
    prompts_dir = partial_dir / "prompts"
    if not prompts_dir.exists():
        os.makedirs(prompts_dir)
        
    # 3. Build manifest
    manifest_path = partial_dir / "manifest.json"
    
    # Read run metadata from DB
    conn = duckdb.connect(db_path, read_only=True)
    
    # Defaults in case the run row doesn't exist (e.g. tests)
    run_info = {
        "kind": "full",
        "parent": None,
        "root_run_id": run_id,
        "depth": 0,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "dataset_digest": "{}",
        "code_version": "unknown",
        "duckdb_version": duckdb.__version__
    }
    
    res = conn.execute("SELECT * FROM run WHERE run_id = ?", (run_id,)).fetchone()
    if res:
        cols = [desc[0] for desc in conn.description]
        row = dict(zip(cols, res))
        run_info["created_at"] = row.get("created_at") or run_info["created_at"]
        run_info["dataset_digest"] = row.get("dataset_digest") or run_info["dataset_digest"]
        run_info["code_version"] = row.get("code_version") or run_info["code_version"]
        run_info["depth"] = row.get("depth") or run_info["depth"]
        run_info["root_run_id"] = row.get("root_run_id") or run_info["root_run_id"]
        
        parent_path = row.get("parent_run_path")
        if parent_path:
            parent_id = os.path.basename(parent_path)
            parent_manifest = base_dir / parent_id / "manifest.json"
            parent_hash = compute_sha256(str(parent_manifest)) if parent_manifest.exists() else ""
            run_info["parent"] = {"run_id": parent_id, "manifest_sha256": parent_hash}
            run_info["kind"] = "addendum"

    # Outcome
    outcome = {"kind": "inconclusive", "reason": "unknown", "accepted_hypothesis_id": None}
    res = conn.execute("SELECT payload FROM ledger_events WHERE record_type = 'scaffold_event' AND op = 'run_outcome' ORDER BY seq DESC LIMIT 1").fetchone()
    if res:
        payload = json.loads(res[0])
        outcome["kind"] = payload.get("kind", outcome["kind"])
        outcome["reason"] = payload.get("reason", outcome["reason"])
        outcome["accepted_hypothesis_id"] = payload.get("accepted_hypothesis_id")
        
    # Models
    models = [r[0] for r in conn.execute("SELECT DISTINCT model FROM llm_calls WHERE model IS NOT NULL").fetchall()]
    
    # Budget
    budget = {"limits": {}, "used": {"steps": 0, "tokens": 0, "wall_clock_s": 0, "extensions": 0}}
    res = conn.execute("SELECT SUM(input_tokens + output_tokens) FROM llm_calls").fetchone()
    if res and res[0]:
        budget["used"]["tokens"] = int(res[0])
    res = conn.execute("SELECT COUNT(*) FROM trajectory WHERE action IS NOT NULL").fetchone()
    if res and res[0]:
        budget["used"]["steps"] = int(res[0])
        
    files = {}
    for root, _, filenames in os.walk(partial_dir):
        for name in filenames:
            if name == "manifest.json":
                continue
            path = os.path.join(root, name)
            relpath = os.path.relpath(path, partial_dir)
            files[relpath] = {
                "sha256": compute_sha256(path),
                "bytes": os.path.getsize(path)
            }
            
    if extra:
        if "budget" in extra:
            budget["used"] = extra["budget"]
        if "kind" in extra:
            run_info["kind"] = extra["kind"]
            
    manifest = {
        "bundle_format": 1,
        "run_id": run_id,
        "kind": run_info["kind"],
        "parent": run_info["parent"],
        "root_run_id": run_info["root_run_id"],
        "depth": run_info["depth"],
        "created_at": run_info["created_at"],
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "outcome": outcome,
        "dataset_digest": run_info["dataset_digest"],
        "code_version": run_info["code_version"],
        "duckdb_version": run_info["duckdb_version"],
        "models": models,
        "budget": budget,
        "files": files
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        
    # Re-read to get manifest hash
    manifest_hash = compute_sha256(manifest_path)
    
    # 4. Set permissions (bottom-up to avoid locking ourselves out)
    for root, dirs, filenames in os.walk(partial_dir, topdown=False):
        for f in filenames:
            os.chmod(os.path.join(root, f), 0o444)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o555)
            
    # 5. Rename
    os.rename(partial_dir, final_dir)
    os.chmod(final_dir, 0o555)
    
    # fsync parent dir
    fd = os.open(str(base_dir), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
        
    # 6. Print
    print(f"run_id={run_id} manifest_sha256={manifest_hash}")
    return final_dir, manifest_hash


def verify_bundle(bundle_dir: str) -> dict:
    bundle_dir = str(bundle_dir)
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    
    if not os.path.exists(manifest_path):
        raise BundleError(f"Manifest not found at {manifest_path}")
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    for relpath, meta in manifest.get("files", {}).items():
        path = os.path.join(bundle_dir, relpath)
        if not os.path.exists(path):
            raise BundleError(f"File missing: {relpath}")
        
        actual_hash = compute_sha256(path)
        if actual_hash != meta["sha256"]:
            raise BundleError(f"Hash mismatch for {relpath}. Expected {meta['sha256']}, got {actual_hash}")
            
    # Check parent chain
    parent = manifest.get("parent")
    if parent:
        parent_id = parent["run_id"]
        base_dir = os.path.dirname(bundle_dir)
        parent_dir = os.path.join(base_dir, parent_id)
        # Recursively verify
        verify_bundle(parent_dir)
        parent_manifest = os.path.join(parent_dir, "manifest.json")
        if compute_sha256(parent_manifest) != parent["manifest_sha256"]:
            raise BundleError(f"Parent manifest hash mismatch for {parent_id}")
            
    return {"ok": True}
