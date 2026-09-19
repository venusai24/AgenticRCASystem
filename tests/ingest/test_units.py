from agentic_rca.ingest.units import to_ms, to_seconds


def test_to_seconds_matches_verified_trace_range():
    # incident_traces.csv verified range (SchemaOfCSVs.MD): 13-digit ms epochs
    # spanning the 30-minute incident window, e.g. 1614852000016 -> ~1614852000.0s
    assert to_seconds(1614852000016) == 1614852000.016


def test_to_ms_round_trips_to_seconds():
    original_ms = 1614852000016
    assert to_ms(to_seconds(original_ms)) == original_ms


def test_to_seconds_is_the_only_ms_boundary_documented_in_this_test():
    # incident_traces.csv is the one 13-digit-ms source; every other source
    # file is already 10-digit seconds and needs no conversion at all.
    assert to_seconds(1000) == 1.0
