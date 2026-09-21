"""The run orchestrator: design-d6-d7 §11's state machine.

INIT -> ORIENT -> INVESTIGATE <-> PRECHECK -> CRITIC(r) -> RESOLVE(r) ...
-> VERIFY_DET -> ENTAIL -> FINALCHECK (-> REPAIR) -> REPORT -> FREEZE,
with budget exhaustion / overseer stop going through CLOSING (ledger-only)
to verification and an inconclusive outcome. Verification and the final
check sit outside the investigation budget, so they can never be skipped.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentic_rca.agents.critic import run_critic_round
from agentic_rca.agents.lead import LEDGER_ONLY, build_lead
from agentic_rca.agents.loop import Budget, LoopConfig
from agentic_rca.donecheck import DoneCheckInput, done_check, render_unmet, run_coverage
from agentic_rca.ledger import Ledger
from agentic_rca.llm import LLMClient
from agentic_rca.report.bundle import freeze_bundle
from agentic_rca.tools._common import SOURCES, iso
from agentic_rca.tools.registry import ToolRuntime, open_store
from agentic_rca.verify.deterministic import verify

TASK = (
    "TASK: Determine what caused the symptoms observed around the incident time, and state it only as "
    "strongly as the evidence allows.\n"
    "FIRST STEP (step-back): before any query, write down the range of conditions that could produce the "
    "symptoms visible at the incident time, each as a hypothesis (`hypothesis_create`). Then investigate."
)


@dataclass
class RunConfig:
    store_path: Path
    runs_root: Path
    incident_ts: float
    data_dir: Path | None = None
    human_hypotheses: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    loop: LoopConfig = field(default_factory=lambda: LoopConfig(summary_every=6))
    critic_rounds: int = 2
    critic_steps: int = 25
    resolve_steps: int = 15
    repair_rounds: int = 1
    repair_steps: int = 10
    closing_steps: int = 2
    residual_k: float = 2.0
    system_facts: list[dict] = field(default_factory=list)
    parent_run_dir: Path | None = None


def dataset_digest(data_dir: Path | None) -> dict:
    if not data_dir:
        return {}
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(data_dir).glob("*.csv"))
    }


def code_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _snapshot(led: Ledger, cfg: RunConfig) -> dict:
    store = open_store(cfg.store_path)
    cur = store.execute(
        "SELECT host_a, host_b, offset_ms, residual_std_ms, n_pairs, method FROM clock_offsets ORDER BY host_a, host_b"
    )
    offsets = [dict(zip([d[0] for d in cur.description], r, strict=True)) for r in cur.fetchall()]
    from agentic_rca.ingest.declaration import build_declaration

    declaration = build_declaration(store)
    store.close()
    units = {t: v["unit_s"] for t, v in SOURCES.items()}
    snap = {
        "clock_offsets": offsets,
        "source_ts_unit_s": units,
        "declaration": declaration,
        "incident_ts": cfg.incident_ts,
        "human_hypotheses": cfg.human_hypotheses,
    }
    for k, v in snap.items():
        led.con.execute(
            "INSERT INTO run_inputs VALUES (?,?,?)", [led.run_id, k, json.dumps(v, default=str)]
        )
    return snap


def _final_check(led: Ledger, snap: dict, cfg: RunConfig, rounds: int):
    cov, qs = run_coverage(led)
    res = done_check(
        DoneCheckInput(
            led.state(),
            cov,
            qs,
            snap["clock_offsets"],
            snap["source_ts_unit_s"],
            "final",
            rounds,
            cfg.residual_k,
        )
    )
    led.apply("done_check_result", asdict(res), actor="scaffold")
    return res


def run_investigation(
    cfg: RunConfig,
    llm: LLMClient,
    *,
    overseer=None,
    reader=None,
    on_hypothesis_write=None,
    entail=None,
) -> dict:
    partial_root = Path(cfg.runs_root) / ".partial"
    led = Ledger.create(
        partial_root,
        store_path=str(cfg.store_path),
        dataset_digest=dataset_digest(cfg.data_dir),
        code_version=code_version(),
        incident_ts=cfg.incident_ts,
        params={
            "budget": {k: v for k, v in asdict(cfg.budget).items() if k != "started"},
            "critic_rounds": cfg.critic_rounds,
            "repair_rounds": cfg.repair_rounds,
            "residual_k": cfg.residual_k,
        },
        parent_run_dir=cfg.parent_run_dir,
    )
    rt = ToolRuntime(cfg.store_path, led)
    llm.bind(led.con, led.run_id)
    snap = _snapshot(led, cfg)
    rt.declaration, rt.system_facts = snap["declaration"], cfg.system_facts
    rounds = {"done": 0}

    if cfg.parent_run_dir:
        # Amendment profile: skip ORIENT, inject questions directly
        for h in cfg.human_hypotheses:
            out = led.apply("hypothesis_create_human", {"statement": h}, actor="human")
            if on_hypothesis_write:
                on_hypothesis_write("hypothesis_create", out["id"])
    else:
        # ORIENT: standing hypothesis first, then human hunches
        led.apply("standing_seed", {}, actor="scaffold")
        for h in cfg.human_hypotheses:
            out = led.apply("hypothesis_create_human", {"statement": h}, actor="human")
            if on_hypothesis_write:
                on_hypothesis_write("hypothesis_create", out["id"])

    def header() -> str:
        decl = json.dumps(snap["declaration"], default=str)
        return f"{TASK}\nINCIDENT TIME: {iso(cfg.incident_ts)} ({cfg.incident_ts})\nDATA-SUFFICIENCY DECLARATION: {decl[:6000]}"

    from agentic_rca.donecheck import precheck_for

    pre = precheck_for(
        led,
        clock_offsets=snap["clock_offsets"],
        source_ts_unit_s=snap["source_ts_unit_s"],
        residual_k=cfg.residual_k,
        rounds=lambda: rounds["done"],
    )
    lead = build_lead(
        ledger=led,
        runtime=rt,
        llm=llm,
        header=header,
        precheck=pre,
        reader=reader,
        on_hypothesis_write=on_hypothesis_write,
        budget=cfg.budget,
        config=cfg.loop,
        overseer=overseer,
    )
    forced = None  # a reason the outcome must be inconclusive
    kind_extra = "full"
    res = lead.run()
    if res.status != "terminated":
        forced = res.status
        lead.run(
            mode="CLOSING",
            allowed=LEDGER_ONLY,
            max_steps=cfg.closing_steps,
            use_budget=False,
            instruction="The investigation budget is spent. Only ledger operations remain: set statuses, "
            "and propose an inconclusive conclusion naming what would discriminate the candidates.",
        )
    else:
        # Check parent outcome if amendment
        parent_outcome = None
        if cfg.parent_run_dir:
            from agentic_rca.ledger import Ledger as _Ledger
            parent_led = _Ledger.open(cfg.parent_run_dir, read_only=True)
            res_po = parent_led.con.execute("SELECT payload FROM ledger_events WHERE op = 'run_outcome' ORDER BY seq DESC LIMIT 1").fetchone()
            if res_po:
                parent_outcome = json.loads(res_po[0])
            parent_led.close()
            
            concl = led.state().latest_conclusion() or {}
            
            p_kind = parent_outcome.get("kind") if parent_outcome else None
            p_id = parent_outcome.get("accepted_hypothesis_id") if parent_outcome else None
            c_kind = concl.get("kind")
            c_id = concl.get("accepted_hypothesis_id")
            
            if p_kind == c_kind and p_id == c_id:
                # Addendum: unchanged conclusion, skip critic
                kind_extra = "addendum"
                cfg.critic_rounds = 0
            else:
                # Superseding: changed conclusion, force exactly one critic round
                kind_extra = "superseding"
                cfg.critic_rounds = 1
        
        for r in range(1, cfg.critic_rounds + 1):
            run_critic_round(
                ledger=led,
                runtime=rt,
                llm=llm,
                round_no=r,
                declaration=snap["declaration"],
                max_steps=cfg.critic_steps,
                on_hypothesis_write=on_hypothesis_write,
            )
            rounds["done"] = r
            open_obj = [o for o in led.state().objections.values() if o["status"] == "open"]
            if not open_obj:
                break
            listing = "\n".join(f"- {led.short(o['id'])}: {o['text']}" for o in open_obj)
            rr = lead.run(
                mode="RESOLVE",
                max_steps=cfg.resolve_steps,
                instruction="Resolve each open critic objection by evidence, revision or concession "
                f"(objection_resolve), then request termination again:\n{listing}",
            )
            if rr.status != "terminated":
                forced = "objections_unresolved" if rr.status == "step_limit" else rr.status
                break
        if any(o["status"] == "open" for o in led.state().objections.values()):
            forced = forced or "objections_unresolved"

    # VERIFY_DET -> ENTAIL -> FINALCHECK (-> REPAIR), outside the budget
    verify(led, rt, current_dataset_digest=dataset_digest(cfg.data_dir) or None)
    if entail:
        entail(led)
    final = _final_check(led, snap, cfg, rounds["done"])
    repairs = 0
    while not final.ok and not forced and repairs < cfg.repair_rounds:
        repairs += 1
        rep = lead.run(
            mode="REPAIR",
            max_steps=cfg.repair_steps,
            use_budget=False,
            instruction="The final check (after verification) is unmet:\n"
            + "\n".join(render_unmet(final, led.short))
            + "\nFix what you can, then request termination again.",
        )
        if rep.status != "terminated":
            break
        done = set(led.state().verdicts)
        verify(led, rt, evidence_ids=[e for e in led.state().evidence if e not in done])
        if entail:
            entail(led)
        final = _final_check(led, snap, cfg, rounds["done"])
    concl = led.state().latest_conclusion() or {}
    if final.ok and not forced and concl.get("kind") == "conclusive":
        outcome = {
            "kind": "conclusive",
            "reason": "all criteria met",
            "accepted_hypothesis_id": concl.get("accepted_hypothesis_id"),
        }
    else:
        reason = forced or ("final_check_unmet" if not final.ok else "proposed_inconclusive")
        outcome = {"kind": "inconclusive", "reason": reason, "accepted_hypothesis_id": None}
    led.apply("run_outcome", dict(outcome, done_check_seq=led.last_seq()), actor="scaffold")
    budget_used = {
        "steps": lead.budget.steps,
        "tokens": lead.budget.tokens,
        "extensions": lead.budget.extensions,
    }
    partial_dir = led.run_dir
    rt.close()
    led.close()
    extra_freeze = {"budget": budget_used}
    if cfg.parent_run_dir:
        extra_freeze["kind"] = kind_extra
    bundle, manifest_sha = freeze_bundle(partial_dir, cfg.runs_root, extra=extra_freeze)
    return {
        "run_id": bundle.name,
        "bundle": str(bundle),
        "manifest_sha256": manifest_sha,
        "outcome": outcome,
    }
