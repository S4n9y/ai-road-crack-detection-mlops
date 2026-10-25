# Road Crack Detection — MLOps starter kit

This starter kit integrates monitoring/tracking into the existing Streamlit + YOLO road-crack app while keeping its existing user-facing functions (`load_model`, `get_severity_label`, `get_confidence_band`, `run_auto_detection`, and `run_custom_detection`). It does not retrain `best.pt`.

## What is included

- `app.py`: updated app with Prometheus metrics, MLflow inference tracking, and Marquez/OpenLineage run events.
- `mlops_helpers.py`: pure helper logic imported by the app and tested independently.
- `tests/test_mlops_helpers.py`: Pytest tests for threshold boundaries and invalid values.
- `mlops/log_model_artifact.py`: logs the existing `best.pt` file to MLflow without retraining.
- `monitoring/`: Prometheus, alert rules, Alertmanager, Grafana datasource, and Docker Compose configuration.

## 0. Important

Back up your current `app.py` before using this starter. Copy `app.py` and `mlops_helpers.py` into the folder that contains your current `best.pt`. Merge the `monitoring/`, `mlops/`, `tests/`, and `requirements-mlops.txt` files into that same project folder. Do not overwrite `best.pt`.

The app remains usable when MLflow or Marquez is not running, but that tool's logging will report as unavailable in the app's **MLOps integration status** section.

## 1. Install Python dependencies

In PowerShell, from the project root:

```powershell
python -m pip install -r requirements-mlops.txt
```

Keep your existing project dependencies too. If you already maintain a `requirements.txt`, merge these packages into it rather than accidentally deleting existing dependencies.

## 2. Run the tests

```powershell
python -m pytest -q
```

The tests run without importing Streamlit or loading `best.pt`.

## 3. Start monitoring (Prometheus, Alertmanager, Grafana)

Make sure Docker Desktop is running. Open PowerShell in the project root:

```powershell
docker compose -f monitoring/docker-compose.yml up -d
```

Then start/restart the Streamlit app in a separate terminal:

```powershell
python -m streamlit run app.py
```

The app exposes a separate metrics endpoint on port 8001. Open `http://localhost:8001/metrics` to see raw Prometheus metrics.

- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3001 (user `admin`; change the sample password before use)
- Streamlit: http://localhost:8501

Prometheus is configured to scrape `host.docker.internal:8001` for Windows/Docker Desktop. In Grafana, the Prometheus datasource is provisioned automatically. Create a dashboard and use queries such as:

```promql
rate(road_crack_prediction_runs_total[5m])
```

```promql
rate(road_crack_prediction_errors_total[5m])
```

```promql
histogram_quantile(0.95, sum by (le) (rate(road_crack_inference_duration_seconds_bucket[5m])))
```

```promql
sum by (crack_class) (road_crack_detections_total)
```

The starter Alertmanager receiver only displays incoming alerts in the UI; it does not send email or chat messages. Configure an actual email/webhook receiver in `monitoring/alertmanager.yml` to enable notifications. Do not commit SMTP passwords or API tokens.

## 4. Start MLflow tracking

Open a separate PowerShell terminal in the project root:

```powershell
python -m mlflow server --host 127.0.0.1 --port 5005 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Open `http://127.0.0.1:5005`. Every successful prediction attempts to log an MLflow run with the confidence mode, selected threshold, detection count, distinct class count, mean detection confidence, inference duration, and model filename.

To upload the existing weights once as an artifact (not retrain them), in another terminal run:

```powershell
python mlops/log_model_artifact.py best.pt
```

Because these weights already exist, the artifact-import step cannot reconstruct historical training metrics. If you have Ultralytics `results.csv` from the training run, log it as an artifact too, or add MLflow metric logging to the training script next time. This starter uploads the `.pt` file as an artifact; it does not automatically wrap YOLO in an MLflow model flavor or register it in the model registry.

You can point the app to a different MLflow server by setting `MLFLOW_TRACKING_URI` before starting Streamlit, for example:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5005"
```

## 5. Start Marquez (data lineage)
Marquez runs as part of the bundled monitoring stack; no separate clone is needed.
```powershell
docker compose -f monitoring/docker-compose.yml up -d
```
This starts PostgreSQL (persistent Marquez metadata store), the Marquez API, and the Marquez UI alongside the existing Prometheus, Alertmanager, and Grafana containers. Existing volumes and dashboards are untouched.

The app posts OpenLineage events to `http://127.0.0.1:5000/api/v1/lineage` (set `OPENLINEAGE_URL` to change it) and emits START, COMPLETE, or FAIL for the job `road_crack_inference` in namespace `road_crack`. The lineage graph uses logical datasets `road_crack/uploaded-road-images` (input) and `road_crack/crack-detection-results` (output). The actual model identity (file, path, SHA-256, classes, `ROAD_CRACK_MODEL_ID`) travels in a run facet, and the MLflow tracking run id is carried in the events so lineage correlates with MLflow.

Endpoints:
- Marquez API: http://localhost:5000
- Marquez UI: http://localhost:3002
- Grafana: http://localhost:3001 (no port conflict with the Marquez UI)

If Marquez is unreachable, the app reports the event as unavailable and predictions continue unaffected.

## 6. Test the complete flow

1. Ensure Docker monitoring containers, MLflow, and Marquez are running.
2. Start the Streamlit app.
3. Upload a test image and click **Check and Detect Cracks**.
4. Check Prometheus at `http://localhost:9090/targets` — `road_crack_app` should be UP.
5. Check MLflow at `http://127.0.0.1:5005` — a `Road Crack Detection` run should appear after an inference.
6. Check Marquez at `http://localhost:3002` — search for job `road_crack_inference`.
7. Verify lineage via the API: `curl http://localhost:5000/api/v1/namespaces/road_crack/jobs/road_crack_inference/runs`.
8. Check Alertmanager at `http://localhost:9093` for alerts. Alerts depend on their rules firing; not all alerts are expected during a normal run.

## What the current severity labels mean

The current app maps YOLO confidence percentage into Low/Moderate/Severe solely for the requested demo behavior. This is not an engineering severity calculation. Physical severity requires independently validated measurements or labeled severity training data.
