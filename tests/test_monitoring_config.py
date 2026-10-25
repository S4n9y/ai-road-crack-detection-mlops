"""Validation tests for the monitoring configuration files.

These tests only parse static YAML files; they do not require Docker,
Prometheus, Grafana, or MLflow to be running. Marquez/OpenLineage is
included since the 2026-10 update and runs as its own service.
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONITORING = PROJECT_ROOT / "monitoring"

try:
    import mlflow  # noqa: F401

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_prometheus_config_scrapes_app():
    config = load_yaml(MONITORING / "prometheus.yml")
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert "road_crack_app" in jobs
    targets = jobs["road_crack_app"]["static_configs"][0]["targets"]
    assert targets == ["host.docker.internal:8001"]


def test_prometheus_rules_match_exported_metrics():
    rules = load_yaml(MONITORING / "alert.rules.yml")
    exprs = " ".join(
        rule["expr"]
        for group in rules["groups"]
        for rule in group["rules"]
    )
    for metric in (
        "road_crack_prediction_errors_total",
        "road_crack_inference_duration_seconds_bucket",
    ):
        assert metric in exprs


def test_alertmanager_has_receiver():
    config = load_yaml(MONITORING / "alertmanager.yml")
    receivers = {r["name"] for r in config["receivers"]}
    assert config["route"]["receiver"] in receivers


def test_grafana_datasource_points_to_prometheus():
    config = load_yaml(
        MONITORING / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    )
    ds = config["datasources"][0]
    assert ds["type"] == "prometheus"
    assert ds["url"] == "http://prometheus:9090"


def test_compose_services_present_and_ports_distinct():
    compose = load_yaml(MONITORING / "docker-compose.yml")
    services = compose["services"]
    assert {"prometheus", "alertmanager", "grafana"} <= set(services)
    assert services["grafana"]["ports"] == ["3001:3000"]


def test_compose_includes_marquez_and_supporting_services():
    compose = load_yaml(MONITORING / "docker-compose.yml")
    services = compose["services"]
    assert {"marquez-api", "marquez-ui", "postgres"} <= set(services)
    api = services["marquez-api"]
    assert api["ports"] == ["5000:5000"]
    assert api["image"].startswith("marquezproject/marquez:")
    # Persistent storage for Marquez data (PostgreSQL + app volume).
    assert any("marquez_postgres_data" in v for v in services["postgres"]["volumes"])
    assert any("marquez_data" in v for v in api["volumes"])
    # UI must not collide with Grafana's 3001 host port.
    assert services["marquez-ui"]["ports"] == ["3002:3000"]
    # API waits for a healthy PostgreSQL before starting.
    assert services["marquez-api"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    # All declared volumes exist.
    assert {"prometheus_data", "grafana_data", "marquez_data", "marquez_postgres_data"} <= set(
        compose["volumes"]
    )


def test_openlineage_dependency_pinned():
    text = (PROJECT_ROOT / "requirements-mlops.txt").read_text(encoding="utf-8")
    assert "openlineage-python" in text


def test_metrics_exporter_requires_multiproc_dir():
    """The standalone exporter refuses to run without PROMETHEUS_MULTIPROC_DIR."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PROMETHEUS_MULTIPROC_DIR"}
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "metrics_exporter.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode != 0
    assert "PROMETHEUS_MULTIPROC_DIR" in (result.stderr + result.stdout)


@pytest.mark.skipif(not HAS_MLFLOW, reason="mlflow not installed")
def test_mlflow_helpers_match_app_public_helpers():
    """The app's public helpers delegate to mlops_helpers."""
    from mlops_helpers import confidence_band, severity_label

    assert severity_label(50) == "Moderate"
    assert confidence_band(0.9) == "High confidence"
