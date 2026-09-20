#!/usr/bin/env python3
"""Debugging-budget ledger for the dev protocol (.claude/skills/dev-protocol).

One active debugging session at a time, kept in debugging/.active.json at the
MAIN checkout (so a session in a component worktree sees the same budget).
Every finished session -- Claude's or Gemini's -- becomes one row of
debugging/metrics.csv, the data behind protocol §9/§18.

  start  --task T --stage S --issue "..." [--failure-class C] [--agent sonnet|opus]
         [--minutes 30] [--max-loops 6]          (--minutes 0 = no time/loop cap)
  loop                                           one Run->Observe->Diagnose->Modify->Test cycle
  status [--short]
  check                                          exit 2 with a reason if the budget is spent
  stop   --outcome resolved|escalated|handed-off|blocked [finish fields]
  record --agent gemini --task T --stage S --issue "..." --minutes-used M --loops N
         --outcome ... [finish fields]           log a session that ran outside Claude Code

finish fields: --root-cause --resolution --hypothesis --fixes-attempted N
               --successful-fixes N --tests N --reproduced y|n --regression y|n --trace PATH
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = [
    "task_id", "stage", "issue", "failure_class", "agent", "start", "end",
    "duration_min", "loops", "tests_executed", "fixes_attempted", "successful_fixes",
    "failed_attempts", "reproduced", "regression", "current_hypothesis",
    "final_root_cause", "final_resolution", "outcome", "trace_file",
]  # fmt: skip


def state_root() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent,
    ).stdout.strip()  # fmt: skip
    return Path(common).parent if common else Path(__file__).resolve().parent.parent


DEBUG_DIR = state_root() / "debugging"
ACTIVE = DEBUG_DIR / ".active.json"
METRICS = DEBUG_DIR / "metrics.csv"


def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def load() -> dict | None:
    try:
        return json.loads(ACTIVE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(s: dict) -> None:
    DEBUG_DIR.mkdir(exist_ok=True)
    ACTIVE.write_text(json.dumps(s, indent=2))


def elapsed_min(s: dict) -> float:
    return round((now() - datetime.fromisoformat(s["start"])).total_seconds() / 60, 1)


def exhausted(s: dict) -> str | None:
    if not s["minutes"]:
        return None
    if elapsed_min(s) >= s["minutes"]:
        return f"time budget spent ({elapsed_min(s)} of {s['minutes']} min)"
    if s["loops"] >= s["max_loops"]:
        return f"loop budget spent ({s['loops']} of {s['max_loops']} loops)"
    return None


def describe(s: dict) -> str:
    cap = f"{s['minutes']} min / {s['max_loops']} loops" if s["minutes"] else "no cap"
    return (
        f"DEBUG SESSION {s['task']} [{s['stage']}] agent={s['agent']}: "
        f"{elapsed_min(s)} min, {s['loops']} loops used (budget {cap}) -- {s['issue']}"
    )


def append_row(row: dict) -> None:
    DEBUG_DIR.mkdir(exist_ok=True)
    new = not METRICS.exists()
    with METRICS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def finish_fields(a) -> dict:
    return {
        "fixes_attempted": a.fixes_attempted,
        "successful_fixes": a.successful_fixes,
        "failed_attempts": max(a.fixes_attempted - a.successful_fixes, 0),
        "reproduced": a.reproduced, "regression": a.regression,
        "current_hypothesis": a.hypothesis, "final_root_cause": a.root_cause,
        "final_resolution": a.resolution, "outcome": a.outcome, "trace_file": a.trace,
    }  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("start")
    rec = sub.add_parser("record")
    for sp in (st, rec):
        sp.add_argument("--task", required=True)
        sp.add_argument("--stage", required=True)
        sp.add_argument("--issue", required=True)
        sp.add_argument("--failure-class", default="unclassified")
    st.add_argument("--agent", default="sonnet")
    st.add_argument("--minutes", type=int, default=30)
    st.add_argument("--max-loops", type=int, default=6)
    rec.add_argument("--agent", default="gemini")
    rec.add_argument("--minutes-used", type=float, required=True)
    rec.add_argument("--loops", type=int, required=True)

    sub.add_parser("loop")
    sub.add_parser("check")
    sub.add_parser("status").add_argument("--short", action="store_true")

    sp_stop = sub.add_parser("stop")
    for sp in (sp_stop, rec):
        sp.add_argument(
            "--outcome", required=True, choices=["resolved", "escalated", "handed-off", "blocked"]
        )
        for f in ("--root-cause", "--resolution", "--hypothesis", "--trace"):
            sp.add_argument(f, default="")
        for f in ("--fixes-attempted", "--successful-fixes", "--tests"):
            sp.add_argument(f, type=int, default=0)
        for f in ("--reproduced", "--regression"):
            sp.add_argument(f, default="", choices=["", "y", "n"])

    a = p.parse_args()
    s = load()

    if a.cmd == "start":
        if s:
            print(f"refused: a session is already active. {describe(s)}", file=sys.stderr)
            return 1
        s = {"task": a.task, "stage": a.stage, "issue": a.issue, "failure_class": a.failure_class,
             "agent": a.agent, "minutes": a.minutes, "max_loops": a.max_loops,
             "start": now().isoformat(), "loops": 0, "tests": 0}  # fmt: skip
        save(s)
        print(describe(s))
    elif a.cmd == "loop":
        if s:
            s["loops"] += 1
            s["tests"] += 1
            save(s)
            print(describe(s))
    elif a.cmd == "check":
        reason = s and exhausted(s)
        if reason:
            print(reason)
            return 2
    elif a.cmd == "status":
        if not s:
            print("" if a.short else "no active debugging session")
        else:
            print(describe(s) + (f" -- EXHAUSTED: {exhausted(s)}" if exhausted(s) else ""))
    elif a.cmd == "stop":
        if not s:
            print("no active debugging session", file=sys.stderr)
            return 1
        append_row({"task_id": s["task"], "stage": s["stage"], "issue": s["issue"],
                    "failure_class": s["failure_class"], "agent": s["agent"],
                    "start": s["start"], "end": now().isoformat(),
                    "duration_min": elapsed_min(s), "loops": s["loops"],
                    "tests_executed": max(a.tests, s["tests"]), **finish_fields(a)})  # fmt: skip
        ACTIVE.unlink()
        print(f"stopped: {describe(s)} -> {a.outcome}")
    elif a.cmd == "record":
        append_row({"task_id": a.task, "stage": a.stage, "issue": a.issue,
                    "failure_class": a.failure_class, "agent": a.agent,
                    "end": now().isoformat(), "duration_min": a.minutes_used,
                    "loops": a.loops, "tests_executed": a.tests, **finish_fields(a)})  # fmt: skip
        print(f"recorded {a.agent} session for {a.task}: {a.outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
