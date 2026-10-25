"""Controlled end-to-end inference run through the real pipeline.

Loads the configured model (best.pt unless ROAD_CRACK_MODEL_PATH is set),
runs YOLO11 inference on a synthetic road-like image, logs the run to
MLflow, and sends START/COMPLETE OpenLineage events to Marquez - the same
sequence the Streamlit app performs.
"""

import os
import time
import uuid

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

import mlflow
from model_registry import validate_and_load_model, collect_model_metadata, get_model_id
from lineage_helpers import send_lineage_event, fetch_lineage_run
from PIL import Image, ImageDraw

run_id = str(uuid.uuid4())
print(f"lineage run id: {run_id}")

model_path = os.environ.get("ROAD_CRACK_MODEL_PATH", "best.pt")
model = validate_and_load_model(model_path)
model_info = {**collect_model_metadata(model, model_path), "model_id": get_model_id()}
print(f"model: {model_info['model_file']} task={model_info['model_task']} "
      f"classes={model_info['model_classes']} sha256={model_info['model_fingerprint_sha256'][:12]}...")

# Synthetic dark road surface with light "crack" lines (demo input only).
img = Image.new("RGB", (640, 640), (55, 55, 55))
draw = ImageDraw.Draw(img)
for x in range(40, 600, 25):
    draw.line([(x, 30), (x + 8, 610)], fill=(150, 145, 140), width=3)
image_name = "e2e_synthetic_road.png"
img.save(image_name)

# START event (as the app does before rendering results).
ok, msg = send_lineage_event(run_id, "START", model_info=model_info, image_name=image_name)
print("lineage START:", ok, msg)

started = time.perf_counter()
result = model.predict(source=img, conf=0.01, verbose=False)[0]
duration = time.perf_counter() - started

boxes = result.boxes
detections = []
severity_counts = {}
if boxes is not None and len(boxes):
    for coords, cls, score in zip(
        boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()
    ):
        name = result.names.get(int(cls), str(int(cls)))
        sev = "Low" if score * 100 <= 35 else ("Moderate" if score * 100 <= 70 else "Severe")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        detections.append({"class": name, "score": round(float(score), 4)})
print(f"inference: {len(detections)} detections in {duration:.2f}s")
print(f"severity counts: {severity_counts}")

# MLflow logging (same as log_prediction_to_mlflow in app.py).
mlflow_run_id = None
try:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5005"))
    mlflow.set_experiment("Road Crack Detection")
    with mlflow.start_run(run_name="e2e-inference-verify") as r:
        mlflow_run_id = r.info.run_id
        mlflow.set_tag("lineage_run_id", run_id)
        mlflow.set_tag("model_file", model_info["model_file"])
        mlflow.log_metric("detection_count", len(detections))
    print(f"mlflow run: {mlflow_run_id}")
except Exception as exc:
    print("mlflow unavailable:", exc)

# COMPLETE event carrying the MLflow run reference.
ok, msg = send_lineage_event(
    run_id, "COMPLETE",
    model_info=model_info,
    mlflow_run_id=mlflow_run_id,
    image_name=image_name,
    detection_count=len(detections),
    severity_counts=severity_counts,
)
print("lineage COMPLETE:", ok, msg)

import json
fetched, payload = fetch_lineage_run(run_id)
print("marquez run fetched:", fetched)
if fetched:
    print(json.dumps({k: payload[k] for k in ("id", "state", "durationMs")}, indent=2))
else:
    print(payload)
