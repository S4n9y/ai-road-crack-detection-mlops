"""Tests for the OpenLineage/Marquez integration (lineage_helpers).

Unit tests stub all network calls; the live integration test at the end
only runs when a real Marquez is reachable via OPENLINEAGE_URL.
"""

import pytest

import lineage_helpers as lh


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OPENLINEAGE_URL", raising=False)
    monkeypatch.delenv("OPENLINEAGE_NAMESPACE", raising=False)
    monkeypatch.delenv("OPENLINEAGE_TIMEOUT", raising=False)


def test_config_defaults_and_absent_url():
    url, ns, timeout = lh.get_lineage_config()
    assert url is None
    assert ns == "road_crack"
    assert timeout == lh.TIMEOUT_DEFAULT


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENLINEAGE_URL", "http://127.0.0.1:5000/")
    monkeypatch.setenv("OPENLINEAGE_NAMESPACE", "custom_ns")
    monkeypatch.setenv("OPENLINEAGE_TIMEOUT", "99")
    url, ns, timeout = lh.get_lineage_config()
    assert url == "http://127.0.0.1:5000/"
    assert ns == "custom_ns"
    # Timeout is clamped to the 1-10 s safety band.
    assert timeout == 10.0


def test_send_event_without_url_is_graceful():
    ok, msg = lh.send_lineage_event("00000000-0000-0000-0000-000000000000", "START")
    assert ok is False
    assert "not configured" in msg


def test_send_event_with_bad_type(monkeypatch):
    monkeypatch.setenv("OPENLINEAGE_URL", "http://127.0.0.1:59999")
    ok, msg = lh.send_lineage_event("00000000-0000-0000-0000-000000000000", "BOGUS")
    assert ok is False
    assert "Invalid event type" in msg


def test_send_event_to_dead_marquez_is_bounded_and_graceful(monkeypatch):
    """A refused connection must return (False, msg) quickly, never raise."""
    import time

    monkeypatch.setenv("OPENLINEAGE_URL", "http://127.0.0.1:59999")
    monkeypatch.setenv("OPENLINEAGE_TIMEOUT", "1")
    started = time.perf_counter()
    ok, msg = lh.send_lineage_event(
        "11111111-1111-1111-1111-111111111111",
        "START",
        model_info={"model_file": "x.pt"},
    )
    elapsed = time.perf_counter() - started
    assert ok is False
    assert "unavailable" in msg
    assert elapsed < 15  # bounded, generous CI margin


def test_model_facet_carries_model_identity():
    facet = lh._model_facet(
        {
            "model_file": "best.pt",
            "model_path": "/x/best.pt",
            "model_task": "detect",
            "model_classes": ["alligator cracking"],
            "model_fingerprint_sha256": "abc123",
            "model_id": "v2",
        }
    )
    assert facet.modelFile == "best.pt"
    assert facet.modelTask == "detect"
    assert facet.sha256 == "abc123"
    assert facet.modelId == "v2"


def test_build_run_event_structure(monkeypatch):
    """START/COMPLETE/FAIL events carry job, inputs/outputs, and facets."""
    import uuid

    run_uuid = str(uuid.uuid4())
    monkeypatch.setenv("OPENLINEAGE_NAMESPACE", "road_crack")
    model_info = {
        "model_file": "best.pt",
        "model_path": "best.pt",
        "model_task": "detect",
        "model_classes": ["alligator cracking"],
        "model_fingerprint_sha256": "abc123",
        "model_id": "v2",
    }

    start = lh.build_run_event(run_uuid, "START", model_info=model_info, image_name="road.jpg")
    assert start.eventType == "START"
    assert start.job.namespace == "road_crack"
    assert start.job.name == lh.JOB_NAME
    assert start.run.facets["model"].modelFile == "best.pt"
    assert start.run.facets["input_context"].imageName == "road.jpg"
    assert start.inputs == []
    assert start.outputs == []

    complete = lh.build_run_event(
        run_uuid,
        "COMPLETE",
        model_info=model_info,
        mlflow_run_id="mlflow-abc",
        detection_count=3,
        severity_counts={"Severe": 2, "Low": 1},
    )
    assert complete.eventType == "COMPLETE"
    assert complete.run.facets["mlflow"].mlflowRunId == "mlflow-abc"
    assert [d.name for d in complete.inputs] == [lh.INPUT_DATASET]
    assert [d.name for d in complete.outputs] == [lh.OUTPUT_DATASET]
    out_facet = complete.outputs[0].facets["detection_output"]
    assert out_facet.detectionCount == 3
    assert out_facet.severityCounts == {"Severe": 2, "Low": 1}
    job_facet = complete.job.facets
    assert job_facet is not None
    assert job_facet.modelSelection.startswith("ROAD_CRACK_MODEL_PATH")

    fail = lh.build_run_event(run_uuid, "FAIL", model_info=model_info, image_name="road.jpg")
    assert fail.eventType == "FAIL"
    assert [d.name for d in fail.inputs] == [lh.INPUT_DATASET]
    assert fail.outputs == []


def test_build_run_event_rejects_bad_type():
    with pytest.raises(ValueError):
        lh.build_run_event("00000000-0000-0000-0000-000000000000", "BOGUS")


def test_fetch_run_without_url_is_graceful():
    ok, msg = lh.fetch_lineage_run("22222222-2222-2222-2222-222222222222")
    assert ok is False
    assert "not configured" in msg


def test_send_and_fetch_real_marquez_if_available(monkeypatch):
    """Integration: only runs when OPENLINEAGE_URL points at a live Marquez."""
    import os

    url = os.environ.get("OPENLINEAGE_URL")
    if not url:
        pytest.skip("OPENLINEAGE_URL not set; live Marquez integration test skipped")
    run_id = "33333333-3333-3333-3333-333333333333"
    ok, _ = lh.send_lineage_event(
        run_id,
        "START",
        model_info={
            "model_file": "best.pt",
            "model_path": "best.pt",
            "model_task": "detect",
            "model_classes": [],
            "model_fingerprint_sha256": "",
        },
    )
    assert ok is True
    import time

    time.sleep(2)
    fetched, payload = lh.fetch_lineage_run(run_id)
    assert fetched is True
    assert str(payload["id"]).lower() == run_id
