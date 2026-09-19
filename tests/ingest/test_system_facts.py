import json

from agentic_rca.ingest.system_facts import load_system_facts


def test_returns_empty_list_when_absent(data_dir):
    # The real, common case against this project's actual incident_data/:
    # no system_facts.json exists there today.
    assert load_system_facts(data_dir) == []


def test_parses_facts_file_when_present(tmp_path):
    facts = [
        {
            "date": "2021-03-01",
            "author": "someone",
            "topic": "shared-infra",
            "fact": "Mysql01 and Mysql02 share a physical host.",
        }
    ]
    (tmp_path / "system_facts.json").write_text(json.dumps(facts), encoding="utf-8")

    loaded = load_system_facts(tmp_path)

    assert loaded == facts


def test_malformed_json_syntax_returns_empty_list(tmp_path):
    (tmp_path / "system_facts.json").write_text("{not valid json", encoding="utf-8")

    assert load_system_facts(tmp_path) == []


def test_valid_json_but_not_a_list_returns_empty_list(tmp_path):
    (tmp_path / "system_facts.json").write_text(
        json.dumps({"date": "2021-03-01", "fact": "not wrapped in a list"}),
        encoding="utf-8",
    )

    assert load_system_facts(tmp_path) == []


def test_non_utf8_file_returns_empty_list_instead_of_raising(tmp_path):
    # Regression check for cycle-9 VERIFY_FAIL: open()+decode sat outside the
    # try, so a non-UTF-8 file raised UnicodeDecodeError straight out of
    # build_index -- the same "broken advisory file kills all of Stage 0"
    # bug the cycle-3 fix was meant to close, via a different exception.
    (tmp_path / "system_facts.json").write_bytes(b"\xff\xfe\x00binary")

    assert load_system_facts(tmp_path) == []
