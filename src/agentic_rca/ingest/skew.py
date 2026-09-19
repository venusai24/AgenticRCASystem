"""Cross-host clock-offset estimation from parent/child span timing.

Verified on this dataset (stage0-ingest plan): 54% of cross-host child spans
appear to start before their parent, with per-host-pair deltas that are
near-constant (std 20-230ms) despite magnitudes up to 321,000ms -- the
signature of a pure clock offset, not jitter. MG01 and the docker hosts are
about 5min21s apart; any cross-host timeline correlation in this 30-minute
window without correcting for this is wrong by up to 5.4 minutes, which can
reverse apparent causal order.

Plan underspecified-point #5 (OPEN as of this build, see plans/stage0-ingest.md):
this estimator embeds a causal assumption -- a child span cannot begin
before its parent, plus symmetric network delay -- which is an assumption
about call semantics, not about incident causes. It self-validates on this
data (see tests), but it is still an assumption, reported with its method
and residual so a model can reject it. This module reports the offset; it
NEVER rewrites a timestamp. Correction is left to whoever consumes
clock_offsets -- rewriting here would be interpretation and would destroy
the original observation.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def _ordered_pair_deltas(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per ordered (parent_host, child_host) pair, the child-minus-parent
    timestamp delta for every cross-host linked span pair."""
    return con.execute(
        """
        SELECT
            parent.cmdb_id AS parent_host,
            child.cmdb_id AS child_host,
            child.timestamp_ms - parent.timestamp_ms AS delta_ms
        FROM spans child
        JOIN spans parent ON child.parent_id = parent.span_id
        WHERE child.cmdb_id != parent.cmdb_id
        """
    ).fetchdf()


def same_host_median_delay_ms(con: duckdb.DuckDBPyConnection) -> float | None:
    """Median parent->child delta for SAME-host links, used only as an
    external self-check in tests -- never fed back into the offset estimate
    itself, so it cannot silently bias which pairs get accepted.

    Excludes span_id == parent_id rows (cycle-8 VERIFY_FAIL, same root cause
    as topology.py's edge count): a self-referential span isn't a distinct
    parent->child link and contributes a spurious zero-delta -- 16,973 such
    rows sit in this same-host population (6.3%). The median happens to be
    2.0 ms with or without them on this dataset, but the query claimed to
    measure only "links," which self-parent rows are not.
    """
    row = con.execute(
        """
        SELECT MEDIAN(child.timestamp_ms - parent.timestamp_ms)
        FROM spans child
        JOIN spans parent ON child.parent_id = parent.span_id
        WHERE child.cmdb_id = parent.cmdb_id AND child.span_id != child.parent_id
        """
    ).fetchone()
    return row[0] if row else None


def estimate_clock_offsets(con: duckdb.DuckDBPyConnection) -> int:
    """Populate clock_offsets: one row per unordered host pair where both
    call directions were observed, via the NTP-style decomposition
    offset = (delta_ab - delta_ba) / 2, one_way_delay = (delta_ab + delta_ba) / 2.

    Pairs with data in only one direction cannot be reciprocally estimated
    and are skipped (absent from the table), not filled with a zero or a
    one-directional guess -- an absence here is honestly "couldn't check,"
    not "checked, offset is zero."
    """
    raw = _ordered_pair_deltas(con)
    if raw.empty:
        return 0

    grouped = raw.groupby(["parent_host", "child_host"])["delta_ms"].agg(
        median_delta="median", n="count", std="std"
    )

    rows: list[dict] = []
    seen_pairs: set[frozenset[str]] = set()
    hosts = set(raw["parent_host"]) | set(raw["child_host"])
    for h1 in hosts:
        for h2 in hosts:
            if h1 >= h2:
                continue
            pair = frozenset((h1, h2))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            fwd = (h1, h2)  # h1 parent, h2 child
            rev = (h2, h1)  # h2 parent, h1 child
            if fwd not in grouped.index or rev not in grouped.index:
                continue  # only one direction observed; cannot reciprocally estimate

            delta_ab = grouped.loc[fwd, "median_delta"]
            delta_ba = grouped.loc[rev, "median_delta"]
            n_ab, n_ba = grouped.loc[fwd, "n"], grouped.loc[rev, "n"]
            std_ab = grouped.loc[fwd, "std"] or 0.0
            std_ba = grouped.loc[rev, "std"] or 0.0

            offset = (delta_ab - delta_ba) / 2
            one_way_delay = (delta_ab + delta_ba) / 2

            rows.append(
                {
                    "host_a": h1,
                    "host_b": h2,
                    "offset_ms": float(offset),
                    "n_pairs": int(n_ab + n_ba),
                    "implied_one_way_delay_ms": float(one_way_delay),
                    "residual_std_ms": float(max(std_ab, std_ba)),
                    "method": "ntp_style_parent_child_reciprocal",
                }
            )

    if not rows:
        return 0

    offsets_df = pd.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO clock_offsets SELECT * FROM offsets_df")
    return len(rows)
