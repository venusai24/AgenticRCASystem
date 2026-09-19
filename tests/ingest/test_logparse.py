from agentic_rca.ingest.load import load_logs
from agentic_rca.ingest.logparse import (
    assign_templates,
    load_access_log_tables,
    make_template_pattern,
    parse_apache_line,
    parse_localhost_line,
)

APACHE_LINE = (
    'IPAddress "POST /UOCP/person/ServiceTest5.json HTTP/1.1" 200 253 "-" '
    '"k6/0.29.0 (api_url 21433 0'
)
LOCALHOST_LINE = (
    "IG01 POST /UOCP/base/ServiceTest7.json HTTP/1.1 200 1403 - k6/0.29.0 (api_url 34 0.034"
)


def test_parse_apache_line_extracts_fields():
    r = parse_apache_line("log1", APACHE_LINE)
    assert r.method == "POST"
    assert r.path == "/UOCP/person/ServiceTest5.json"
    assert r.tc == "ServiceTest5"
    assert r.protocol == "HTTP/1.1"
    assert r.status == "200"
    assert r.bytes == "253"
    # opaque k6 tag never closes its quote -- retained raw, not decomposed
    assert "k6/0.29.0" in r.tail_raw


def test_parse_localhost_line_extracts_gateway_and_duration():
    r = parse_localhost_line("log2", LOCALHOST_LINE)
    assert r.gateway == "IG01"
    assert r.method == "POST"
    assert r.tc == "ServiceTest7"
    assert r.status == "200"
    assert r.self_reported_duration_s == 0.034


def test_parse_functions_never_raise_on_malformed_input():
    # A field splitter that raises on the unexpected turns "couldn't parse"
    # into a crash; raw text must remain the fallback (Target_Design.MD
    # sub-problem 14: don't confuse "couldn't check" with a hard failure).
    junk = "this is not a valid access log line at all"
    apache_r = parse_apache_line("x", junk)
    localhost_r = parse_localhost_line("y", junk)
    assert apache_r.tail_raw == junk
    assert localhost_r.tail_raw == junk
    assert apache_r.method is None
    assert localhost_r.method is None


def test_parse_functions_never_raise_on_none_value_raw():
    # Regression check for cycle-14 VERIFY_FAIL: this test's own name already
    # claimed "never raise on malformed input" but only ever exercised a
    # non-matching *string* -- a NULL value_raw (a blank CSV cell surviving
    # as pandas NaN -> DuckDB NULL) is a different malformed shape and
    # crashed both parse functions with a bare re.match(None) before this
    # fix (reproduced against the real store by NULLing one row).
    apache_r = parse_apache_line("x", None)
    localhost_r = parse_localhost_line("y", None)
    assert apache_r.tail_raw is None
    assert localhost_r.tail_raw is None
    assert apache_r.method is None
    assert localhost_r.method is None


def test_make_template_pattern_never_raises_on_none():
    # Same cycle-14 regression, third call site: assign_templates reads
    # value_raw straight from `logs` and passes it here uncontrolled.
    assert make_template_pattern(None) == "#NULL#"


def test_no_derived_flags_are_computed():
    # Structural naming only -- no is_error / is_slow / severity anywhere in
    # the record. Status is uniformly "200" in this dataset, so any derived
    # error flag would be vacuous as well as encoding (Target_Design.MD).
    r = parse_apache_line("log1", APACHE_LINE)
    assert not hasattr(r, "is_error")
    assert not hasattr(r, "is_slow")


def test_make_template_pattern_masks_digits_and_hex():
    a = make_template_pattern("request took 42 ms for id abc123def456")
    b = make_template_pattern("request took 999 ms for id 000fff111abc")
    assert a == b


def test_load_access_log_tables_and_template_assignment(db_con, data_dir):
    n_logs = load_logs(db_con, data_dir)
    counts = load_access_log_tables(db_con)
    assert counts["apache_access_log"] + counts["localhost_access_log"] < n_logs  # gc excluded
    assert (
        counts["apache_access_log"]
        == db_con.execute(
            "SELECT COUNT(*) FROM logs WHERE log_name = 'apache_access_log'"
        ).fetchone()[0]
    )

    n_templates = assign_templates(db_con)
    # Sanity bound, not a tuned/hardcoded target: this dataset's three formats
    # are highly repetitive (SchemaOfCSVs.MD found ~8 by hand), but the
    # templater's own masking rule -- not that specific number -- is what's
    # under test here, so this stays a loose upper bound rather than equality.
    assert 0 < n_templates < 100

    # Every log row must end up with a template_id; none silently skipped.
    unassigned = db_con.execute("SELECT COUNT(*) FROM logs WHERE template_id IS NULL").fetchone()[0]
    assert unassigned == 0

    # gc lines get NO field extraction (plan underspecified-point #2) but DO
    # get a template, like every other log line.
    gc_templates = db_con.execute(
        "SELECT COUNT(DISTINCT template_id) FROM logs WHERE log_name = 'gc'"
    ).fetchone()[0]
    assert gc_templates > 0


def test_value_raw_always_retained_regardless_of_parse_outcome(db_con, data_dir):
    load_logs(db_con, data_dir)
    load_access_log_tables(db_con)
    total, non_null_raw = db_con.execute("SELECT COUNT(*), COUNT(value_raw) FROM logs").fetchone()
    assert total == non_null_raw


def test_pipeline_survives_a_null_value_raw_row(db_con, data_dir):
    # Regression check for cycle-14 VERIFY_FAIL: a single blank
    # cluster_incident_logs.csv cell (pandas NaN -> DuckDB NULL) crashed
    # load_access_log_tables and assign_templates with an uncaught TypeError,
    # killing all of Stage 0 -- the fourth recurrence of the cycle-3/9/13
    # "broken input kills the pipeline" defect, this time at its actual
    # source (logparse.py's regex functions) rather than a downstream reader.
    load_logs(db_con, data_dir)
    log_id = db_con.execute(
        "SELECT log_id FROM logs WHERE log_name = 'apache_access_log' LIMIT 1"
    ).fetchone()[0]
    db_con.execute("UPDATE logs SET value_raw = NULL WHERE log_id = ?", [log_id])

    load_access_log_tables(db_con)  # must not raise
    n_templates = assign_templates(db_con)  # must not raise
    assert n_templates > 0
    unassigned = db_con.execute("SELECT COUNT(*) FROM logs WHERE template_id IS NULL").fetchone()[0]
    assert unassigned == 0
