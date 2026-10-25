"""Standalone Prometheus metrics exporter for the road-crack app.

Root cause this fixes: Streamlit executes app.py lazily, only when the
first browser session connects. A freshly started `streamlit run` process
has not run the script, so a metrics server defined at module level in
app.py never binds port 8001 and Prometheus reports
`road_crack_app ... connection refused`.

Fix: the app writes its metrics to a shared multiprocess directory
(PROMETHEUS_MULTIPROC_DIR), and this tiny standalone process exposes them
on port 8001 for as long as the environment is configured. The exporter
merges all app-worker metric files, so metric names stay identical:
road_crack_prediction_runs_total, road_crack_detections_total,
road_crack_inference_duration_seconds_*, road_crack_detection_confidence_*,
road_crack_selected_confidence_threshold, road_crack_prediction_errors_total.

Run:  python metrics_exporter.py
Env:  ROAD_CRACK_METRICS_PORT (default 8001), PROMETHEUS_MULTIPROC_DIR
"""

import os
import sys
import time

DEFAULT_PORT = 8001
POLL_SECONDS = 5


def main():
    from prometheus_client import CollectorRegistry, multiprocess, start_http_server

    port = int(os.environ.get("ROAD_CRACK_METRICS_PORT", DEFAULT_PORT))
    mdir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not mdir:
        print(
            "metrics_exporter: PROMETHEUS_MULTIPROC_DIR is not set; "
            "the Streamlit app defines it at startup. Re-run via start_mlops_stack.ps1 "
            "or set it manually.",
            file=sys.stderr,
        )
        sys.exit(1)

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    start_http_server(port, registry=registry)
    print(f"metrics_exporter: serving on 0.0.0.0:{port} (dir={mdir})")
    # Live-forever loop; MergeCollector reads the dir on every scrape.
    while True:
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
