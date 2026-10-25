"""Verify the MLflow inference run recorded during the audit."""

import os

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5005")
exp = mlflow.get_experiment_by_name("Road Crack Detection")
print("experiment:", exp.experiment_id if exp else None, exp.name if exp else None)
runs = mlflow.search_runs(
    [exp.experiment_id], order_by=["start_time DESC"], max_results=3
)
for _, run in runs.iterrows():
    print("---")
    print("run_id:", run["run_id"])
    print("status:", run["status"])
    print("tags.model_file:", run.get("tags.model_file"))
    print("tags.model_path:", run.get("tags.model_path"))
    print("tags.model_task:", run.get("tags.model_task"))
    print("tags.model_fingerprint_sha256:", run.get("tags.model_fingerprint_sha256"))
    print("tags.model_classes:", run.get("tags.model_classes"))
    print("params.confidence_mode:", run.get("params.confidence_mode"))
    print("params.model_task:", run.get("params.model_task"))
    print("metrics.detection_count:", run.get("metrics.detection_count"))
    print(
        "metrics.inference_duration_seconds:",
        run.get("metrics.inference_duration_seconds"),
    )
