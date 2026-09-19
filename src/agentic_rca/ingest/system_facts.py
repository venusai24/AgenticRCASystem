"""Optional loader for human-curated system facts (Target_Design.MD:14,152):
dated statements about the environment that telemetry can't show (shared
infrastructure, known-benign chronic errors, deploy schedules). Absence is
the common, expected case for incident_data/ today -- nothing downstream
breaks; callers get an empty list.

This module only loads and returns records. It does not interpret, filter,
or rank them -- Target_Design.MD:152 is explicit that nothing from
system_facts is injected into prompts; the system_facts_lookup tool (a
later-stage consumer, out of scope here) is the query-only access path.
"""

from __future__ import annotations

import json
from pathlib import Path

_FACTS_FILENAME = "system_facts.json"


def load_system_facts(data_dir: Path) -> list[dict]:
    """Load data_dir/system_facts.json if present; [] if absent.

    Expected format: a JSON array of objects, each a dated, human-authored
    statement -- e.g. {"date": "2021-03-01", "author": "...", "topic": "...",
    "fact": "..."}. No field is required or validated here; this loader's
    only job is "file present and parseable -> return its contents as-is."

    A malformed (unparseable JSON, unreadable, or not valid UTF-8) or
    wrong-shaped (parses but isn't a JSON array) file degrades the same way
    as an absent one -- [] -- rather than raising. This file is
    advisory-only and nothing downstream depends on it being present, so a
    broken file shouldn't be able to break ingest. Cycle-9 VERIFY_FAIL: the
    open() call and its UTF-8 decode were outside the try, so a non-UTF-8
    file still raised UnicodeDecodeError out of build_index -- the same
    "broken file kills all of Stage 0" bug the cycle-3 fix was meant to
    close, via a different exception type.
    """
    path = data_dir / _FACTS_FILENAME
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []
