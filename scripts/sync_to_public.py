#!/usr/bin/env python3
"""Sync eligible files from the private AgenticRCA repo to the public AgenticRCASystem repo.

Only the following files are replicated:
  • Python files (*.py) under  src/
  • README files (README* at any depth, case-insensitive)

The public repo's folder tree is kept as a strict subset of the private repo:
  - New/modified files are copied (with parent dirs created as needed).
  - Files that no longer exist in the private repo are removed from the public repo.
  - Unchanged files are skipped (compared via content hash).

Usage:
    python scripts/sync_to_public.py                 # dry-run (default)
    python scripts/sync_to_public.py --apply         # actually copy / delete
    python scripts/sync_to_public.py --apply --push  # copy, commit, and push
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


# ── Paths ────────────────────────────────────────────────────────────────────
PRIVATE_ROOT = Path(__file__).resolve().parent.parent        # AgenticRCA/
PUBLIC_ROOT  = PRIVATE_ROOT.parent / "AgenticRCASystem"      # AgenticRCASystem/

# Directories / patterns that are always excluded from the scan.
# "archive" holds the previous private dev setup; ".worktrees" holds component checkouts.
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "archive", ".worktrees",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

class SyncAction(NamedTuple):
    kind: str          # "copy" | "delete"
    rel_path: str      # path relative to repo root
    reason: str        # human-readable reason


def _file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_eligible(rel_path: str) -> bool:
    """Return True if *rel_path* should be mirrored to the public repo.

    Rules:
      1. Python files (*.py) whose path starts with ``src/``
      2. Any file whose basename matches README* (case-insensitive)
    """
    parts = Path(rel_path).parts

    # Skip anything inside an excluded directory.
    if any(p in EXCLUDE_DIRS for p in parts):
        return False

    basename = os.path.basename(rel_path)

    # Rule 2 – README files anywhere (case-insensitive basename match).
    if basename.upper().startswith("README"):
        return True

    # Rule 1 – Python files under src/.
    if parts and parts[0] == "src" and rel_path.endswith(".py"):
        return True

    return False


def _gather_eligible_files(root: Path) -> dict[str, Path]:
    """Walk *root* and return ``{relative_path: absolute_path}`` for eligible files."""
    result: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded dirs in-place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            abs_path = Path(dirpath) / fname
            rel_path = str(abs_path.relative_to(root))
            if _is_eligible(rel_path):
                result[rel_path] = abs_path
    return result


def _compute_actions(
    private_files: dict[str, Path],
    public_files: dict[str, Path],
) -> list[SyncAction]:
    """Compare the two file-maps and return the list of required sync actions."""
    actions: list[SyncAction] = []

    # Files to copy (new or modified).
    for rel, priv_abs in sorted(private_files.items()):
        pub_abs = public_files.get(rel)
        if pub_abs is None:
            actions.append(SyncAction("copy", rel, "new file"))
        elif _file_hash(priv_abs) != _file_hash(pub_abs):
            actions.append(SyncAction("copy", rel, "modified"))

    # Files to delete (exist in public but not in private).
    for rel in sorted(public_files):
        if rel not in private_files:
            actions.append(SyncAction("delete", rel, "removed from private repo"))

    return actions


# ── Execution ────────────────────────────────────────────────────────────────

def _apply_actions(actions: list[SyncAction]) -> None:
    """Execute copy / delete actions."""
    for action in actions:
        dst = PUBLIC_ROOT / action.rel_path
        if action.kind == "copy":
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = PRIVATE_ROOT / action.rel_path
            shutil.copy2(src, dst)
        elif action.kind == "delete":
            dst.unlink(missing_ok=True)
            # Remove empty parent dirs up to the repo root.
            parent = dst.parent
            while parent != PUBLIC_ROOT:
                try:
                    parent.rmdir()  # Only removes if empty.
                except OSError:
                    break
                parent = parent.parent


def _git_commit_and_push() -> None:
    """Stage all changes in the public repo, commit, and push."""
    env = {**os.environ, "GIT_WORK_TREE": str(PUBLIC_ROOT), "GIT_DIR": str(PUBLIC_ROOT / ".git")}

    subprocess.run(["git", "add", "-A"], cwd=PUBLIC_ROOT, check=True, env=env)

    # Check if there is anything to commit.
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=PUBLIC_ROOT,
        env=env,
    )
    if result.returncode == 0:
        print("\n✅ Nothing staged — working tree already up-to-date.")
        return

    subprocess.run(
        ["git", "commit", "-m", "sync: replicate eligible files from private repo"],
        cwd=PUBLIC_ROOT,
        check=True,
        env=env,
    )
    subprocess.run(["git", "push"], cwd=PUBLIC_ROOT, check=True, env=env)
    print("\n✅ Changes committed and pushed to the public repo.")


# ── CLI ──────────────────────────────────────────────────────────────────────

_COLORS = {
    "copy":   "\033[32m",   # green
    "delete": "\033[31m",   # red
    "reset":  "\033[0m",
}


def _print_actions(actions: list[SyncAction]) -> None:
    """Pretty-print the list of planned actions."""
    for a in actions:
        color = _COLORS.get(a.kind, "")
        reset = _COLORS["reset"]
        symbol = "+" if a.kind == "copy" else "−"
        print(f"  {color}{symbol} {a.rel_path:<60s} ({a.reason}){reset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Python src/ and README files from the private repo to the public repo.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy / delete files (without this flag, only a dry-run is shown).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="After applying, stage, commit, and push to the public repo remote.",
    )
    args = parser.parse_args()

    if not PUBLIC_ROOT.is_dir():
        print(f"❌ Public repo not found at {PUBLIC_ROOT}", file=sys.stderr)
        sys.exit(1)

    print(f"Private repo : {PRIVATE_ROOT}")
    print(f"Public repo  : {PUBLIC_ROOT}")
    print()

    private_files = _gather_eligible_files(PRIVATE_ROOT)
    public_files  = _gather_eligible_files(PUBLIC_ROOT)

    print(f"Eligible files in private repo: {len(private_files)}")
    print(f"Eligible files in public repo : {len(public_files)}")
    print()

    actions = _compute_actions(private_files, public_files)

    if not actions:
        print("✅ Public repo is already in sync — nothing to do.")
        return

    copies  = sum(1 for a in actions if a.kind == "copy")
    deletes = sum(1 for a in actions if a.kind == "delete")
    print(f"Actions: {copies} file(s) to copy, {deletes} file(s) to delete\n")
    _print_actions(actions)

    if not args.apply:
        print("\n⚠️  Dry-run mode. Re-run with --apply to execute the changes.")
        return

    print("\nApplying…")
    _apply_actions(actions)
    print("✅ Files synced successfully.")

    if args.push:
        _git_commit_and_push()


if __name__ == "__main__":
    main()
