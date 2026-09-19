"""Structural field extraction for the two access-log formats, plus a generic
templater for coverage statistics across all three log_name formats.

Target_Design.MD (amended, commit d48b596): structural field extraction is
permitted and distinguished from excluded technology-specific *analysis*.
Accordingly:
  - apache_access_log and localhost_access_log get real field parsers below.
  - gc gets NO field extraction (plan underspecified-point #2): choosing
    which number in a GC line matters is exactly the excluded expertise.
    It participates only in the generic templater, like the other two.
  - No derived is_error / is_slow / severity flags are computed anywhere;
    status is uniformly "200" in this dataset (SchemaOfCSVs.MD), so such a
    flag would be vacuous as well as encoding.
  - value_raw is always retained regardless of parse success.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import pandas as pd

# --- apache_access_log -------------------------------------------------
# 'IPAddress "POST /UOCP/person/ServiceTest5.json HTTP/1.1" 200 253 "-" "k6/0.29.0 (api_url 21433 0'
# The trailing quoted field never closes (SchemaOfCSVs.MD: "an opaque
# (api_url <n> <n> ... k6 iteration metadata -- not a reliable RCA signal,
# don't over-interpret it") -- captured whole into tail_raw, not decomposed
# further.
_APACHE_RE = re.compile(
    r"^(?P<ip_literal>\S+) "
    r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>[^"]+)" '
    r"(?P<status>\d+) (?P<bytes>\S+) "
    r'"(?P<referrer>[^"]*)" '
    r'"(?P<tail_raw>.*)$'
)

# --- localhost_access_log -----------------------------------------------
# 'IG01 POST /UOCP/base/ServiceTest7.json HTTP/1.1 200 1403 - k6/0.29.0 (api_url 34 0.034'
# The gateway hop (IG01/IG02) exists ONLY here -- never in cmdb_id, which on
# these rows is always the Tomcat host. Self-reported duration is the last
# whitespace token, a second independent latency measurement (SchemaOfCSVs.MD).
_LOCALHOST_RE = re.compile(
    r"^(?P<gateway>\S+) "
    r"(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+) "
    r"(?P<status>\d+) (?P<bytes>\S+) (?P<referrer>\S+) "
    r"(?P<agent>\S+) \(api_url (?P<api_url_n>\S+) (?P<duration>[\d.]+)$"
)

_TC_FROM_PATH_RE = re.compile(r"/(ServiceTest\d+)\.json", re.IGNORECASE)


@dataclass(frozen=True)
class ApacheAccessRecord:
    log_id: str
    method: str | None
    path: str | None
    tc: str | None
    protocol: str | None
    status: str | None
    bytes: str | None
    agent: str | None
    tail_raw: str | None


@dataclass(frozen=True)
class LocalhostAccessRecord:
    log_id: str
    gateway: str | None
    method: str | None
    path: str | None
    tc: str | None
    protocol: str | None
    status: str | None
    bytes: str | None
    agent: str | None
    self_reported_duration_s: float | None
    tail_raw: str | None


def _extract_tc(path: str | None) -> str | None:
    if path is None:
        return None
    m = _TC_FROM_PATH_RE.search(path)
    return m.group(1) if m else None


def parse_apache_line(log_id: str, value_raw: str | None) -> ApacheAccessRecord:
    m = _APACHE_RE.match(value_raw) if value_raw is not None else None
    if not m:
        # Malformed line: structural parse failed, raw text is still retained
        # via `logs.value_raw`, so nothing is lost -- this record just carries
        # no extracted fields.
        return ApacheAccessRecord(log_id, None, None, None, None, None, None, None, value_raw)
    path = m.group("path")
    return ApacheAccessRecord(
        log_id=log_id,
        method=m.group("method"),
        path=path,
        tc=_extract_tc(path),
        protocol=m.group("protocol"),
        status=m.group("status"),
        bytes=m.group("bytes"),
        agent=None,  # the quoted agent field never closes; not structurally extractable
        tail_raw=m.group("tail_raw"),
    )


def parse_localhost_line(log_id: str, value_raw: str | None) -> LocalhostAccessRecord:
    m = _LOCALHOST_RE.match(value_raw) if value_raw is not None else None
    if not m:
        return LocalhostAccessRecord(
            log_id, None, None, None, None, None, None, None, None, None, value_raw
        )
    path = m.group("path")
    duration_raw = m.group("duration")
    return LocalhostAccessRecord(
        log_id=log_id,
        gateway=m.group("gateway"),
        method=m.group("method"),
        path=path,
        tc=_extract_tc(path),
        protocol=m.group("protocol"),
        status=m.group("status"),
        bytes=m.group("bytes"),
        agent=m.group("agent"),
        self_reported_duration_s=float(duration_raw) if duration_raw else None,
        tail_raw=None,
    )


def load_access_log_tables(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Parse every apache_access_log / localhost_access_log row already in
    `logs` into the two structured tables. gc rows are intentionally skipped."""
    apache_rows = con.execute(
        "SELECT log_id, value_raw FROM logs WHERE log_name = 'apache_access_log'"
    ).fetchall()
    localhost_rows = con.execute(
        "SELECT log_id, value_raw FROM logs WHERE log_name = 'localhost_access_log'"
    ).fetchall()

    # DataFrame + INSERT...SELECT, not executemany: executemany() issuing one
    # parameterized statement per row was pathologically slow here (hung well
    # past a 90s budget on 17k rows) -- the same reason assign_templates below
    # uses a bulk join-update rather than a per-row UPDATE. DuckDB is
    # columnar/OLAP-shaped; set-based operations are the correct shape for it,
    # matching the pattern load.py already uses successfully at 324k rows.
    apache_records = [parse_apache_line(log_id, raw) for log_id, raw in apache_rows]
    if apache_records:
        apache_df = pd.DataFrame(  # noqa: F841 (used via SQL below, not directly)
            [
                {
                    "log_id": r.log_id,
                    "method": r.method,
                    "path": r.path,
                    "tc": r.tc,
                    "protocol": r.protocol,
                    "status": r.status,
                    "bytes": r.bytes,
                    "agent": r.agent,
                    "tail_raw": r.tail_raw,
                }
                for r in apache_records
            ]
        )
        con.execute("INSERT INTO log_access_apache SELECT * FROM apache_df")

    localhost_records = [parse_localhost_line(log_id, raw) for log_id, raw in localhost_rows]
    if localhost_records:
        localhost_df = pd.DataFrame(  # noqa: F841 (used via SQL below, not directly)
            [
                {
                    "log_id": r.log_id,
                    "gateway": r.gateway,
                    "method": r.method,
                    "path": r.path,
                    "tc": r.tc,
                    "protocol": r.protocol,
                    "status": r.status,
                    "bytes": r.bytes,
                    "agent": r.agent,
                    "self_reported_duration_s": r.self_reported_duration_s,
                    "tail_raw": r.tail_raw,
                }
                for r in localhost_records
            ]
        )
        con.execute("INSERT INTO log_access_localhost SELECT * FROM localhost_df")

    return {
        "apache_access_log": len(apache_records),
        "localhost_access_log": len(localhost_records),
    }


# --- generic templater ---------------------------------------------------
# Mask numeric and hex-ish runs so structurally identical lines collapse to
# one template. This is coverage bookkeeping ("log volume by template is
# queryable"), not interpretation: it assigns no meaning to any template.
_DIGIT_RUN_RE = re.compile(r"\d+")
_HEX_RUN_RE = re.compile(r"\b[0-9a-fA-F]{6,}\b")


_NULL_VALUE_TEMPLATE = "#NULL#"


def make_template_pattern(value_raw: str | None) -> str:
    # A NULL log line (e.g. a blank CSV cell) is not "the same text as
    # another NULL line" in any structural sense the templater assigns
    # meaning to elsewhere -- it's the absence of text -- but it must still
    # get a template rather than crash the regex substitution below.
    if value_raw is None:
        return _NULL_VALUE_TEMPLATE
    masked = _HEX_RUN_RE.sub("#HEX#", value_raw)
    masked = _DIGIT_RUN_RE.sub("#", masked)
    return masked


def assign_templates(con: duckdb.DuckDBPyConnection) -> int:
    """Compute a template pattern per log line, group by (log_name, pattern),
    populate log_templates, and back-fill logs.template_id. Returns the
    number of distinct templates found."""
    rows = con.execute("SELECT log_id, log_name, value_raw FROM logs").fetchall()

    pattern_to_template_id: dict[tuple[str, str], str] = {}
    template_counts: dict[str, int] = {}
    template_example: dict[str, str] = {}
    template_log_name: dict[str, str] = {}
    assignments: list[tuple[str, str]] = []  # (template_id, log_id)

    next_id = 1
    for log_id, log_name, value_raw in rows:
        pattern = make_template_pattern(value_raw)
        key = (log_name, pattern)
        if key not in pattern_to_template_id:
            template_id = f"t{next_id:04d}"
            next_id += 1
            pattern_to_template_id[key] = template_id
            template_example[template_id] = log_id
            template_log_name[template_id] = log_name
        template_id = pattern_to_template_id[key]
        template_counts[template_id] = template_counts.get(template_id, 0) + 1
        assignments.append((template_id, log_id))

    if pattern_to_template_id:
        con.executemany(
            "INSERT INTO log_templates VALUES (?, ?, ?, ?, ?)",
            [
                (
                    tid,
                    template_log_name[tid],
                    pattern,
                    template_counts[tid],
                    template_example[tid],
                )
                for (log_name, pattern), tid in pattern_to_template_id.items()
            ],
        )
        # Bulk join-update rather than one UPDATE per row: with 34k log lines,
        # executemany() issuing a separate point-UPDATE per row took long
        # enough to time out a 120s test run. DuckDB is columnar/OLAP-shaped;
        # a single set-based UPDATE...FROM is the correct shape for it.
        # DataFrame + INSERT...SELECT for the same reason as
        # load_access_log_tables above: executemany() here (34k rows) took
        # 42s where the DataFrame form takes under a second.
        assignments_df = pd.DataFrame(  # noqa: F841 (used via SQL below)
            assignments, columns=["template_id", "log_id"]
        )
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _template_assignments AS SELECT * FROM assignments_df"
        )
        con.execute(
            "UPDATE logs SET template_id = _template_assignments.template_id "
            "FROM _template_assignments WHERE logs.log_id = _template_assignments.log_id"
        )
        con.execute("DROP TABLE _template_assignments")

    return len(pattern_to_template_id)
