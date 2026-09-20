#!/usr/bin/env bash
# sync_main.sh [--apply]
#
# Keep `main` (public, src/tests only) a strict subset of `private` (the dev
# branch), in BOTH directions, for the PUBLIC_PATHS only:
#
#   * changed on private only  -> applied to main
#   * changed on main only     -> applied to private   (main is not read-only:
#                                 work done there must not be overwritten)
#   * changed on both, differently -> CONFLICT, nothing is written
#
# "Changed" is measured against the merge-base of the two branches, per file
# (no line-level merging; a genuine both-sides edit is left to a human).
# Everything outside PUBLIC_PATHS is left alone on both branches: main's
# .gitignore stays its own, and the dev files never reach main.
#
# Result: both branches end with identical public paths, and main is recorded
# as an ancestor of private. That is done in one merge commit on private whose
# tree is private's own plus any main-only public changes. (A plain
# `git merge main` would delete the dev files: main's history contains the
# commits that removed them.)
#
# Dry-run by default; --apply writes. Pure plumbing (temporary indexes,
# commit-tree, update-ref) on COMMITTED state: no checkout, no worktree, and
# nobody's uncommitted work is read or touched. It refuses to change the tree
# of a branch that is checked out somewhere, because moving that ref under a
# live tree would make the tree look like it had reverted the change.
#
# Env: SYNC_SRC (default private), SYNC_DST (default main).

set -uo pipefail

die() { echo "error: $*" >&2; exit 1; }

PRIV="${SYNC_SRC:-private}"; PUB="${SYNC_DST:-main}"
PUBLIC_PATHS=(src tests incident_data/filter_baselines.py)
APPLY=0
case "${1:-}" in "") ;; --apply) APPLY=1 ;; *) die "usage: $0 [--apply]" ;; esac

for b in "$PRIV" "$PUB"; do git rev-parse --verify -q "refs/heads/$b" >/dev/null || die "no branch '$b'"; done
priv_sha=$(git rev-parse "$PRIV"); pub_sha=$(git rev-parse "$PUB")
base=$(git merge-base "$PRIV" "$PUB") || die "'$PRIV' and '$PUB' share no history"
checked_out() { git worktree list --porcelain | grep -qx "branch refs/heads/$1"; }

echo "$PRIV@${priv_sha:0:7}  $PUB@${pub_sha:0:7}  merge-base ${base:0:7}"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# --- 1. Three-way merge of the public paths (file level) ---------------------
GIT_INDEX_FILE="$tmp/merged" git read-tree -m -i --aggressive "$base" "$priv_sha" "$pub_sha" \
    || die "read-tree failed"
conflicts=$(GIT_INDEX_FILE="$tmp/merged" git ls-files -u -- "${PUBLIC_PATHS[@]}" | cut -f2 | sort -u)
if [[ -n "$conflicts" ]]; then
    echo "CONFLICT: changed differently on both branches since ${base:0:7}; nothing written:" >&2
    echo "$conflicts" | sed 's/^/  /' >&2
    exit 1
fi
GIT_INDEX_FILE="$tmp/merged" git ls-files -s -z -- "${PUBLIC_PATHS[@]}" >"$tmp/resolved"

# Tree = <branch>'s tree with the public paths replaced by the resolved ones.
rebuild() {
    local ix="$tmp/build.$1"
    GIT_INDEX_FILE="$ix" git read-tree "$1" || die "read-tree $1 failed"
    GIT_INDEX_FILE="$ix" git rm -r -q -f --cached --ignore-unmatch -- "${PUBLIC_PATHS[@]}" || die "index update failed"
    GIT_INDEX_FILE="$ix" git update-index -z --index-info <"$tmp/resolved" || die "index update failed"
    GIT_INDEX_FILE="$ix" git write-tree
}
priv_tree=$(rebuild "$priv_sha"); pub_tree=$(rebuild "$pub_sha")

# Guard against a script bug: only public paths may differ from the originals.
for pair in "$pub_sha:$pub_tree" "$priv_sha:$priv_tree"; do
    while IFS= read -r f; do
        ok=0; for p in "${PUBLIC_PATHS[@]}"; do [[ "$f" == "$p" || "$f" == "$p"/* ]] && ok=1; done
        ((ok)) || die "refusing: would change '$f', which is not a public path"
    done < <(git diff-tree -r --name-only "${pair%%:*}" "${pair##*:}")
done

report() { # $1=label $2=old commit $3=new tree
    if [[ "$(git rev-parse "$2^{tree}")" == "$3" ]]; then echo "$1: no public-path changes"
    else echo "$1:"; git diff-tree -r --name-status "$2" "$3" | sed 's/^/  /'; fi
}
report "to apply on $PUB (changed on $PRIV only)" "$pub_sha" "$pub_tree"
report "to apply on $PRIV (changed on $PUB only)" "$priv_sha" "$priv_tree"

pub_changes=0; [[ "$(git rev-parse "$pub_sha^{tree}")" != "$pub_tree" ]] && pub_changes=1
priv_changes=0; [[ "$(git rev-parse "$priv_sha^{tree}")" != "$priv_tree" ]] && priv_changes=1
# A new commit on main is by definition not yet on private, so it needs recording too.
need_anc=$pub_changes; git merge-base --is-ancestor "$pub_sha" "$priv_sha" || need_anc=1
((pub_changes || priv_changes || need_anc)) || { echo "in sync: identical public paths, $PUB is an ancestor of $PRIV"; exit 0; }
((need_anc)) && echo "ancestry: $PUB is not yet an ancestor of $PRIV; will be recorded"

((APPLY)) || { echo "(dry run; --apply to write)"; exit 0; }

# --- 2. Apply -----------------------------------------------------------------
((pub_changes)) && checked_out "$PUB" && die "$PUB is checked out in a worktree and would change; check out something else there first."
((priv_changes)) && checked_out "$PRIV" && die "$PRIV is checked out in a worktree and would change (changes from $PUB to apply); merge them from that checkout or check out something else, then re-run."

new_pub=$pub_sha
if ((pub_changes)); then
    new_pub=$(git commit-tree "$pub_tree" -p "$pub_sha" -m "Sync public paths from $PRIV@${priv_sha:0:7}

Source commit: $priv_sha") || die "commit-tree failed"
    git update-ref -m "sync_main: from $PRIV@${priv_sha:0:7}" "refs/heads/$PUB" "$new_pub" "$pub_sha" || die "$PUB moved while syncing; re-run"
    echo "committed ${new_pub:0:7} on $PUB"
fi

if ((priv_changes || need_anc)); then
    parents=(-p "$priv_sha"); git merge-base --is-ancestor "$new_pub" "$priv_sha" || parents+=(-p "$new_pub")
    merge=$(git commit-tree "$priv_tree" "${parents[@]}" -m "Sync public paths with $PUB (dev files unchanged)

Applies public-path changes made on $PUB only (if any) and records $PUB as
merged. Deliberately not a content merge: $PUB's history deletes the dev files.") || die "commit-tree failed"
    git update-ref -m "sync_main: merge public paths with $PUB" "refs/heads/$PRIV" "$merge" "$priv_sha" || die "$PRIV moved while syncing; re-run"
    echo "committed ${merge:0:7} on $PRIV ($( ((priv_changes)) && echo 'includes changes from '"$PUB" || echo 'tree unchanged'))"
fi
