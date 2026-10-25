"""Start a Streamlit app script in bare mode and probe its metrics endpoint.

Run from the project root. Loads app.py directly (the streamlit runner on
this machine executes the script lazily, so module-level side effects never
fire), then queries the metrics server on port 8001 and performs one real
prediction through the app's own functions.
"""

import io
import time
import urllib.request

import app  # module-level: starts metrics server on 8001

print("metrics server_started:", app.METRICS.get("server_started"))
try:
    body = (
        urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=8)
        .read()
        .decode()
    )
    lines = [l for l in body.splitlines() if l.startswith("road_crack_")]
    print("metrics endpoint OK; road_crack_* lines:", len(lines))
    print("\n".join(lines[:10]))
except Exception as e:
    print("metrics endpoint FAILED:", e)

# --- Real test prediction using the app's own pipeline ---
import uuid
from PIL import Image

from model_registry import (
    validate_and_load_model,
    resolve_model_path,
    check_class_compatibility,
)

model = validate_and_load_model(resolve_model_path())
print("model loaded:", model.task, list(model.names.values()))
print("class warnings:", check_class_compatibility(model))

# Synthetic road-like test image (a small gradient with a dark crack line)
img = Image.new("RGB", (320, 240))
px = img.load()
for y in range(240):
    for x in range(320):
        px[x, y] = (120 + x // 8, 110 + y // 8, 100)
for x in range(320):  # a dark horizontal "crack"
    px[x, 118] = (15, 12, 10)
    px[x, 119] = (20, 15, 12)

started = time.perf_counter()
result = model.predict(source=img, conf=0.01, verbose=False)[0]
duration = time.perf_counter() - started
boxes = result.boxes
n = len(boxes) if boxes is not None else 0
print(f"PREDICTION_OK boxes={n} duration={duration:.2f}s")

app.METRICS["prediction_runs"].inc()
app.METRICS["inference_duration"].observe(duration)

run_id = str(uuid.uuid4())
ok, msg = app.log_prediction_to_mlflow(
    run_id=run_id,
    confidence_mode="automatic",
    used_threshold=0.05,
    detections=[
        {
            "Crack class": "test-class",
            "Confidence (%)": 50.0,
        }
    ],
    duration_seconds=duration,
    model_info={
        "model_file": resolve_model_path().name,
        "model_path": str(resolve_model_path()),
        "model_task": model.task,
        "model_classes": sorted(str(v) for v in model.names.values()),
        "model_fingerprint_sha256": "AUDIT",
        "model_id": "",
    },
)
print("mlflow log:", ok, msg, "run_id:", run_id)

try:
    body = (
        urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=8)
        .read()
        .decode()
    )
    hits = [
        l
        for l in body.splitlines()
        if "road_crack_prediction_runs_total" in l
        or "road_crack_inference_duration" in l
    ]
    print("post-prediction metric lines:", len(hits))
    print("\n".join(hits[:6]))
except Exception as e:
    print("metrics re-read FAILED:", e)
