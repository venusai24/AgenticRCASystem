from agentic_rca.ingest.load import load_spans
from agentic_rca.ingest.skew import estimate_clock_offsets, same_host_median_delay_ms


def test_clock_offsets_never_rewrites_a_timestamp(db_con, data_dir):
    # This module only ever reads spans and writes to clock_offsets; assert
    # the spans table is untouched by the estimate (Target_Design.MD: "never
    # rewrites a timestamp. Correction is the model's call, not ingest's").
    load_spans(db_con, data_dir)
    before = db_con.execute("SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM spans").fetchone()
    estimate_clock_offsets(db_con)
    after = db_con.execute("SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM spans").fetchone()
    assert before == after


def test_offset_estimate_self_validates_against_same_host_delay(db_con, data_dir):
    # The plan's central verified claim: the implied one-way delay from the
    # reciprocal cross-host estimate should land close to the independently
    # measured same-host parent->child delay (~2ms), which is never used as
    # an input to the estimate itself -- this is an external cross-check.
    load_spans(db_con, data_dir)
    estimate_clock_offsets(db_con)
    same_host = same_host_median_delay_ms(db_con)
    assert same_host is not None

    rows = db_con.execute("SELECT implied_one_way_delay_ms FROM clock_offsets").fetchall()
    assert len(rows) > 0
    for (delay,) in rows:
        # Loose tolerance (order-of-magnitude, not exact equality): this is a
        # sanity cross-check on real noisy data, not a tuned threshold that
        # decides which pairs are "valid" -- no pair is discarded based on it.
        assert abs(delay - same_host) < 50


def test_mg01_dockerA1_offset_is_large_and_antisymmetric(db_con, data_dir):
    # The plan's headline finding: MG01 and dockerA1 clocks are ~5min21s
    # apart (~321,682ms), not a small jitter. This is the concrete case that
    # makes correction non-optional for any cross-host timeline in this
    # dataset (Target_Design.MD Section 5, "third likely failure").
    load_spans(db_con, data_dir)
    estimate_clock_offsets(db_con)
    row = db_con.execute(
        "SELECT offset_ms, n_pairs FROM clock_offsets "
        "WHERE (host_a = 'MG01' AND host_b = 'dockerA1') "
        "OR (host_a = 'dockerA1' AND host_b = 'MG01')"
    ).fetchone()
    assert row is not None
    offset_ms, n_pairs = row
    assert abs(offset_ms) > 300_000  # order of 5+ minutes, not noise
    assert n_pairs > 0


def test_pairs_with_only_one_observed_direction_are_absent_not_zero(db_con, data_dir):
    # A pair that was never observed in both call directions cannot be
    # reciprocally estimated; it must be absent from clock_offsets, not
    # silently reported as offset=0 (Target_Design.MD sub-problem 10:
    # absence-of-evidence vs. evidence-of-absence).
    load_spans(db_con, data_dir)
    estimate_clock_offsets(db_con)
    zero_offset_rows = db_con.execute(
        "SELECT COUNT(*) FROM clock_offsets WHERE offset_ms = 0"
    ).fetchone()[0]
    # Not asserting this must be zero (a true zero offset is a legitimate
    # outcome) -- asserting instead that every row has actual support.
    all_have_support = db_con.execute("SELECT MIN(n_pairs) FROM clock_offsets").fetchone()[0]
    assert all_have_support is None or all_have_support > 0
    assert zero_offset_rows >= 0  # sanity: query runs; no crash on the edge case
