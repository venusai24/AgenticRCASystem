#!/usr/bin/env bash
# component_worktree.sh new|merge <component>
#
# One branch + one worktree per component, so a failing component's dirty
# tree cannot contaminate another's and merges happen in dependency order.
#
#   new <name>    create branch component/<name> off $COMPONENT_BASE and a
#                 worktree at <main checkout>/.worktrees/<name>; idempotent,
#                 so re-running resumes a component left dirty by VERIFY_FAIL.
#   merge <name>  merge component/<name> into the base branch, but only if the
#                 worktree is clean and every HARD dependency listed for it in
#                 plans/component-order.md is already merged.
#
# Run the /plan-component, /build-component, /verify-component loop from inside
# the worktree. (The unattended scripts/run_component.sh that used to live here
# is archived under archive/previous-setup/scripts/; see that README.)
# PROJECT_STATE.md, ROADMAP.md and debugging/ are always the MAIN checkout's.
#
# Env: COMPONENT_BASE (default: private) -- the branch that carries the dev
# setup (.claude/, plans/, DECISIONS.md). `main` is the public src/tests-only
# branch and is deliberately NOT a base here; publishing to it is a separate,
# human step this script does not attempt.

set -uo pipefail

die() { echo "error: $*" >&2; exit 1; }

cmd="${1:-}"; name="${2:-}"
[[ "$cmd" == new || "$cmd" == merge ]] || die "usage: $0 new|merge <component>"
[[ "$name" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "component name must be kebab-case (got: '$name')"

MAIN=$(git worktree list --porcelain | awk 'NR==1{print $2}')
[[ -n "$MAIN" ]] || die "not inside a git repository"
BASE="${COMPONENT_BASE:-private}"
git rev-parse --verify -q "refs/heads/$BASE" >/dev/null || die "base branch '$BASE' does not exist"
ORDER="$MAIN/plans/component-order.md"
[[ -f "$ORDER" ]] || die "$ORDER not found; the component list defines valid names and merge order"

BRANCH="component/$name"
WT="$MAIN/.worktrees/$name"

# Emits "<status>|<dep> <dep> ..." for a component, or nothing if it is not listed.
# Hard deps = backticked component names in the deps column, outside any
# sentence marked **soft**. Status = BUILT | IN_FLIGHT | "".
lookup() {
    python3 - "$ORDER" "$1" <<'PY'
import re, sys
order, want = sys.argv[1:3]
def split_cells(line):
    # A "|" inside backticks is content, not a cell boundary (row 4 lists
    # `a | b | c` enum values); a naive split shifted its deps column.
    out, cur, tick = [], "", False
    for ch in line.strip().strip("|"):
        if ch == "`":
            tick = not tick
        if ch == "|" and not tick:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [c.strip() for c in out]

rows = []
for line in open(order):
    if not line.startswith("|"):
        continue
    cells = split_cells(line)
    # Row numbers may carry a letter suffix (15a, 21a) for inserted rows.
    num = re.fullmatch(r"(\d+)[a-z]?", cells[0]) if cells else None
    if len(cells) < 4 or not num:
        continue
    m = re.match(r"`([^`]+)`", cells[1])
    if m:
        rows.append((m.group(1), cells[1], cells[3], int(num.group(1))))
known = {r[0] for r in rows}
band_a = [r[0] for r in rows if 7 <= r[3] <= 13]
for n, c1, deps, _ in rows:
    if n != want:
        continue
    status = "BUILT" if "BUILT" in c1 else "IN_FLIGHT" if "IN FLIGHT" in c1 else ""
    hard = [s for s in deps.split(". ") if "soft" not in s]
    found = [t for s in hard for t in re.findall(r"`([^`]+)`", s) if t in known and t != n]
    if any("band A" in s for s in hard):
        found += band_a  # the table says "band A tools" without naming them
    print(status + "|" + " ".join(dict.fromkeys(found)))
PY
}

info=$(lookup "$name")
[[ -n "$info" ]] || die "'$name' is not listed in plans/component-order.md; add it there first"

case "$cmd" in
new)
    # Local-only repo config, shared by every worktree, so no tracked file changes.
    common=$(git rev-parse --path-format=absolute --git-common-dir)
    mkdir -p "$common/info"
    grep -qxF '.worktrees/' "$common/info/exclude" 2>/dev/null || echo '.worktrees/' >>"$common/info/exclude"
    # DECISIONS.md is append-only: two branches each adding rows would conflict
    # on every merge. `union` keeps both sides' lines.
    grep -qxF 'DECISIONS.md merge=union' "$common/info/attributes" 2>/dev/null \
        || echo 'DECISIONS.md merge=union' >>"$common/info/attributes"

    if [[ -d "$WT" ]]; then
        echo "worktree already exists: $WT"
    else
        if git rev-parse --verify -q "refs/heads/$BRANCH" >/dev/null; then
            git worktree add "$WT" "$BRANCH" || die "worktree add failed"
        else
            git worktree add -b "$BRANCH" "$WT" "$BASE" || die "worktree add failed"
        fi
    fi

    # incident_data/*.csv is gitignored, so a bare worktree has none. The test
    # fixtures SKIP when it is missing, which would let VERIFY_PASS through on
    # zero real checks. Link the data in, and refuse to continue without it.
    shopt -s nullglob
    csvs=("$MAIN"/incident_data/*.csv)
    ((${#csvs[@]})) || die "no incident_data/*.csv in $MAIN; a worktree without data would skip its tests, not run them"
    mkdir -p "$WT/incident_data"
    for f in "${csvs[@]}"; do ln -sfn "$f" "$WT/incident_data/$(basename "$f")"; done

    echo "component: $name   branch: $BRANCH   base: $BASE@$(git rev-parse --short "$BASE")"
    echo "worktree:  $WT   (${#csvs[@]} data files linked)"
    echo "Uncommitted work in $MAIN is NOT in this worktree; commit it to $BASE first if the component needs it."
    ;;
merge)
    git rev-parse --verify -q "refs/heads/$BRANCH" >/dev/null || die "no branch $BRANCH"
    if [[ -d "$WT" ]] && [[ -n "$(git -C "$WT" status --porcelain)" ]]; then
        die "$WT has uncommitted changes. A failed verify leaves the tree dirty on purpose; only merge a verified, committed component."
    fi
    [[ "$(git rev-list --count "$BASE..$BRANCH")" -gt 0 ]] || die "$BRANCH has no commits beyond $BASE"

    deps="${info#*|}"
    for dep in $deps; do
        if git rev-parse --verify -q "refs/heads/component/$dep" >/dev/null; then
            git merge-base --is-ancestor "component/$dep" "$BASE" \
                || die "hard dependency '$dep' (component/$dep) is not merged into $BASE yet; merge it first"
        else
            # No branch: built directly on $BASE before this scheme existed.
            grep -qE '^\|[^|]*\|[^|]*`'"$dep"'`[^|]*(BUILT|IN FLIGHT)' "$ORDER" \
                || die "hard dependency '$dep' has no branch and is not marked BUILT/IN FLIGHT in $ORDER"
            echo "note: '$dep' has no component branch; assumed already on $BASE (marked built/in-flight)"
        fi
    done

    # Merge inside a worktree of $BASE so the main checkout is never switched.
    int=$(git worktree list --porcelain | awk -v b="refs/heads/$BASE" '/^worktree /{w=$2} $1=="branch" && $2==b{print w}')
    if [[ -z "$int" ]]; then
        int="$MAIN/.worktrees/_integration"
        git worktree add "$int" "$BASE" || die "could not check out $BASE for merging"
    fi
    [[ -z "$(git -C "$int" status --porcelain --untracked-files=no)" ]] || die "$int has uncommitted changes to tracked files"
    git -C "$int" merge --no-ff -m "Merge $BRANCH" "$BRANCH" || die "merge conflict in $int; resolve there, then commit"
    echo "merged $BRANCH into $BASE (in $int). Worktree $WT and the branch are left in place."
    ;;
esac
