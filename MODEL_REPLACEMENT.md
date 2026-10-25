# Replacing the YOLO model without rebuilding the MLOps stack

The model is selected by **configuration**, not source code. You never edit
`app.py`, `mlops_helpers.py`, the Prometheus config, the Grafana dashboards,
or the Docker Compose files to swap models.

## 1. How model selection works

- Environment variable `ROAD_CRACK_MODEL_PATH` points to the `.pt` file to use.
- If it is unset or empty, the app uses the default `best.pt` in the project root.
- The model is loaded once per app session by `model_registry.validate_and_load_model()`
  (the only place a YOLO model is instantiated), then cached by Streamlit's
  `@st.cache_resource`. It is **not** reloaded per prediction.
- Optional `ROAD_CRACK_MODEL_ID` lets you tag runs with a version label
  (e.g. `v2-finetune-2026-10`) that appears in MLflow metadata.

## 2. Selecting a replacement model (PowerShell)

```powershell
# Point at the new model (any location; the file is never moved or overwritten)
$env:ROAD_CRACK_MODEL_PATH = "E:\models\crack_v2\best_v2.pt"

# Optional version identifier for MLflow runs
$env:ROAD_CRACK_MODEL_ID = "v2-finetune-2026-10"

# Restart the Streamlit app (required: the model is cached for the session)
python -m streamlit run app.py
```

To go back to the original model:

```powershell
Remove-Item Env:ROAD_CRACK_MODEL_PATH
Remove-Item Env:ROAD_CRACK_MODEL_ID
python -m streamlit run app.py
```

## 3. Validation performed before activation

`model_registry.validate_and_load_model()` checks, in order:

1. **File exists** and has a `.pt` extension.
2. **Loads successfully** with Ultralytics YOLO (corrupt or foreign files are rejected).
3. **Inference interface**: callable `predict()` and a non-empty `model.names` mapping.
4. **Task type**: must be `detect` (bounding boxes with per-box confidence).
   A segmentation/classification/pose model is refused with an explanation.

If any check fails:

- A clear error is reported (in the app UI and/or Streamlit error).
- When a custom path fails, the app **falls back to the default `best.pt`**
  and shows a warning — the working configuration is preserved.
- No file is ever deleted, renamed, or overwritten.

## 4. Class-compatibility check

`model_registry.check_class_compatibility()` compares the model's class names
with the expected four crack classes (`alligator`, `edge`, `longitudinal`,
`transverse cracking`).

- Matching classes → predictions behave exactly as before.
- Different classes → the model still activates, but the app shows a prominent
  warning that its predictions are **not** the four crack classes, so you must
  adapt the severity mapping, report tables, and Grafana panels before relying
  on them. Nothing is silently relabeled.

The Low/Moderate/Severe severity logic maps confidence (0–1) to bands, which
works for any `detect`-task model with per-box confidence; only class semantics
need adaptation.

## 5. MLOps metadata per inference

Every prediction run logs the **actual** loaded model to MLflow (tags + params):

- `model_file`, `model_path` — filename and configured path
- `model_id` — when `ROAD_CRACK_MODEL_ID` is set
- `model_fingerprint_sha256` — SHA-256 of the model file
- `model_task`, `model_classes` — task type and class-name list
- Existing params/metrics: `confidence_mode`, `selected_threshold`,
  `detection_count`, `distinct_classes`, `mean_detection_confidence`,
  `inference_duration_seconds`

The sidebar and the "MLOps integration status" expander show the same
metadata. Prometheus metrics and Grafana dashboards are model-agnostic
(they count detections by class label and confidence bands), so they keep
working across model changes without edits.

## 6. Safe switching — important constraints

- **Restart required.** The model is cached for the app session; changing the
  environment variable takes effect on the next `streamlit run`.
- **Only one model in memory.** The fallback to `best.pt` loads at most one
  model at a time on your 16 GB laptop; the configured model and the fallback
  are never held simultaneously.
- **Explicit change only.** Switching is done via the environment variable —
  the app never auto-replaces a model.
- **Artifacts preserved.** `best.pt`, MLflow data (`mlflow.db`, `mlartifacts/`),
  Docker volumes, and monitoring configs are untouched.

## 7. Testing

```powershell
python -m pytest -q tests/test_model_registry.py   # targeted tests
python -m pytest -q                                # full suite
```

Covered: default vs env-var path selection, missing/invalid paths, corrupt
files, wrong task type, missing class names, class-mismatch warnings,
metadata fingerprinting, and model-ID configuration. All tests use stub
models; `best.pt` is never loaded by the test suite.
