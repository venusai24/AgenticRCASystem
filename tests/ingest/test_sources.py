from agentic_rca.ingest.sources import SOURCES, SOURCES_BY_KEY


def test_six_sources_registered():
    assert len(SOURCES) == 6


def test_only_incident_traces_is_milliseconds():
    ms_sources = [s.key for s in SOURCES if s.timestamp_unit == "ms"]
    assert ms_sources == ["incident_traces"]


def test_baseline_and_incident_windows_both_present():
    labels = {s.window_label for s in SOURCES}
    assert labels == {"baseline", "incident"}


def test_app_metrics_identity_key_is_tc():
    assert SOURCES_BY_KEY["cluster_app_metrics"].identity_key == ("tc",)
    assert SOURCES_BY_KEY["baseline_app_metrics"].identity_key == ("tc",)


def test_container_metrics_identity_key_is_cmdb_id_kpi_name():
    assert SOURCES_BY_KEY["container_metrics"].identity_key == ("cmdb_id", "kpi_name")
    assert SOURCES_BY_KEY["baseline_container_metrics"].identity_key == ("cmdb_id", "kpi_name")
