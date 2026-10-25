from pathlib import Path
from io import BytesIO
import os
import tempfile
import time
import uuid

# Must be set before importing mlflow so its HTTP client picks up the
# bounded timeout and retry settings on initialization.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")

import mlflow
from prometheus_client import Counter, Gauge, Histogram
from mlops_helpers import (
    CONFIDENCE_FLOOR,
    severity_label,
    confidence_band,
    run_auto_detection,
    select_auto_threshold,
    filter_detection_records,
    build_threshold_search_log,
)
import pandas as pd
import streamlit as st
from model_registry import (
    MODEL_PATH_ENV_VAR,
    ModelValidationError,
    check_class_compatibility,
    collect_model_metadata,
    get_model_id,
    resolve_model_path,
    validate_and_load_model,
)
from lineage_helpers import fetch_lineage_run, get_lineage_config, send_lineage_event
from ultralytics import YOLO  # noqa: F401  (kept for API compatibility)
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="AI Road Crack Detection",
    page_icon="🛣️",
    layout="wide",
)

# Model selection is configuration-driven: ROAD_CRACK_MODEL_PATH overrides
# the default best.pt. best.pt itself is never moved or overwritten.
MODEL_PATH = resolve_model_path()
MODEL_PATH_ENV = MODEL_PATH_ENV_VAR


# --- Prometheus metrics (multiprocess pattern) ---
#
# Root cause of the audit's "road_crack_app DOWN" finding: Streamlit runs
# app.py lazily, only when the first browser session connects, so an
# in-process HTTP server bound at import time never exists in a freshly
# started `streamlit run` process. Fix: metrics are written to a shared
# multiprocess directory; the standalone metrics_exporter.py process
# exposes them on port 8001 for as long as the app environment is up.
# Metric names are unchanged.

MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
if not MULTIPROC_DIR:
    MULTIPROC_DIR = os.path.join(tempfile.gettempdir(), "road_crack_prom")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = MULTIPROC_DIR
os.makedirs(MULTIPROC_DIR, exist_ok=True)


@st.cache_resource(show_spinner=False)
def initialize_prometheus_metrics():
    """Create metric objects on the default registry in multiprocess mode.

    In multiprocess mode metrics MUST use the default REGISTRY; values are
    written to PROMETHEUS_MULTIPROC_DIR on each update and merged/decision
    served by the standalone metrics_exporter.py process on port 8001.
    No HTTP server is started here.
    """
    metrics = {
        "prediction_runs": Counter(
            "road_crack_prediction_runs",
            "Total prediction attempts made by the Streamlit app.",
        ),
        "prediction_errors": Counter(
            "road_crack_prediction_errors",
            "Prediction attempts that raised an error.",
        ),
        "inference_duration": Histogram(
            "road_crack_inference_duration_seconds",
            "Time spent loading the model and producing a prediction result.",
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
        ),
        "detections": Counter(
            "road_crack_detections",
            "Total detections returned, partitioned by predicted crack class.",
            labelnames=("crack_class",),
        ),
        "detection_confidence": Histogram(
            "road_crack_detection_confidence",
            "YOLO confidence values for returned detections (0 to 1).",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        ),
        "selected_threshold": Gauge(
            "road_crack_selected_confidence_threshold",
            "Most recently selected confidence threshold (0 to 1).",
        ),
    }
    metrics["server_started"] = True  # exposition handled by metrics_exporter.py
    return metrics


METRICS = initialize_prometheus_metrics()


def log_prediction_to_mlflow(run_id, confidence_mode, used_threshold, detections, duration_seconds, model_info=None):
    """Log one inference run if an MLflow tracking server is available.

    A short HTTP timeout keeps a dead tracking server from stalling the
    prediction flow; the MLflow client's internal retry loop would
    otherwise hang for minutes against an unreachable endpoint.
    """
    try:
        # Must be set before mlflow makes its first HTTP request. The
        # 3 s timeout plus a single retry bounds a dead tracking server
        # to a few seconds; MLflow's default of 7 retries with
        # exponential backoff would otherwise hang for minutes.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5005")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Road Crack Detection")
        confidence_values = [item["Confidence (%)"] / 100 for item in detections]
        mean_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values else 0.0
        )
        with mlflow.start_run(run_name="road-crack-inference") as _mlflow_run:
            st.session_state["crack_mlflow_run_id"] = _mlflow_run.info.run_id
            mlflow.set_tag("lineage_run_id", run_id)
            # Metadata describing the ACTUAL loaded model, not a default
            # filename, so every model replacement stays traceable.
            info = model_info or {}
            mlflow.set_tag("model_file", info.get("model_file", MODEL_PATH.name))
            mlflow.set_tag("model_path", info.get("model_path", str(MODEL_PATH)))
            mlflow.set_tag("model_task", info.get("model_task", "unknown"))
            if info.get("model_id"):
                mlflow.set_tag("model_id", info["model_id"])
            if info.get("model_fingerprint_sha256"):
                mlflow.set_tag("model_fingerprint_sha256", info["model_fingerprint_sha256"])
            if info.get("model_classes"):
                mlflow.set_tag("model_classes", ", ".join(info["model_classes"]))
            mlflow.log_param("confidence_mode", confidence_mode)
            mlflow.log_param("selected_threshold", float(used_threshold))
            if info.get("model_task"):
                mlflow.log_param("model_task", info["model_task"])
            if info.get("model_fingerprint_sha256"):
                mlflow.log_param("model_fingerprint_sha256", info["model_fingerprint_sha256"])
            mlflow.log_metric("detection_count", len(detections))
            mlflow.log_metric("distinct_classes", len({x["Crack class"] for x in detections}))
            mlflow.log_metric("mean_detection_confidence", mean_confidence)
            mlflow.log_metric("inference_duration_seconds", float(duration_seconds))
        return True, "MLflow run recorded."
    except Exception as exc:
        # Optional observability services must not break road-crack inference.
        return False, f"MLflow logging unavailable: {exc}"


# OpenLineage events: START in run_once_and_cache(), COMPLETE or FAIL from
# annotate_and_render() / the error handler. See lineage_helpers.py. Graceful
# when Marquez is unreachable.


@st.cache_resource(show_spinner="Loading road-crack model...")
def load_model(model_path: str):
    """Load and cache the configured YOLO model (centralized function).

    The model is loaded and cached once, not per prediction. When a custom
    ROAD_CRACK_MODEL_PATH is configured but fails validation, the app
    falls back to the default best.pt and reports the problem, so the
    existing working model is preserved.
    """
    try:
        model = validate_and_load_model(model_path)
        return model, None, str(Path(model_path).resolve())
    except ModelValidationError as exc:
        default_path = Path(__file__).resolve().parent / "best.pt"
        if Path(model_path).resolve() != default_path and default_path.is_file():
            fallback_model = validate_and_load_model(default_path)
            return (
                fallback_model,
                f"Configured model failed validation ({exc}). Fell back to default best.pt.",
                str(default_path),
            )
        raise


def get_severity_label(severity_score):
    """Preserved public helper: map the requested demo score to a label."""
    return severity_label(severity_score)


def get_confidence_band(score):
    """Preserved public helper: categorize YOLO confidence (0 to 1)."""
    return confidence_band(score)


def run_custom_detection(model, image, threshold):
    """Run detection once at the user-selected confidence threshold."""
    result = model.predict(
        source=image,
        conf=threshold,
        verbose=False,
    )[0]

    boxes = result.boxes
    count = len(boxes) if boxes is not None else 0

    search_log = [
        {
            "Threshold tested (%)": round(threshold * 100, 2),
            "Detections found": count,
        }
    ]

    return result, threshold, search_log


st.title("🛣️ AI-Based Road Crack Detection")
st.write(
    "Upload a road image to identify alligator, edge, "
    "longitudinal, and transverse cracks."
)
st.caption(
    "Confidence indicates the model's certainty in a prediction. "
    "The Low/Moderate/Severe label below is currently mapped from "
    "confidence for demonstration; it is not a physical crack-severity measurement."
)

# Check for the configured YOLO model.
if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        f"Set the {MODEL_PATH_ENV} environment variable to a valid YOLO "
        ".pt file, or unset it to use the default best.pt."
    )
    st.stop()

try:
    MODEL_INFO = collect_model_metadata(None, MODEL_PATH)
except Exception:
    MODEL_INFO = {"model_file": MODEL_PATH.name, "model_path": str(MODEL_PATH)}
MODEL_ID = get_model_id()

# Sidebar settings.
with st.sidebar:
    st.header("Prediction settings")

    custom_mode = st.radio(
        "Would you like to set a custom confidence threshold?",
        options=["No — automatic", "Yes — custom"],
        index=0,
        help=(
            "Choose No to scan thresholds automatically. "
            "Choose Yes to set your own threshold."
        ),
    )

    custom_confidence = None
    if custom_mode == "Yes — custom":
        custom_confidence = st.slider(
            "Custom confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            format="%.2f",
            help=(
                "Only detections meeting your selected threshold "
                "will be retained."
            ),
        )
        st.write(f"**Selected threshold:** {custom_confidence:.0%}")
    else:
        st.info(
            "Automatic mode tests thresholds from 100% down to 0%, "
            "in 5-percentage-point steps, stopping at the first "
            "threshold that returns at least one detection."
        )

    st.markdown("---")
    st.write(f"**Model file:** {MODEL_PATH.name}")
    st.write(f"**Model path:** `{MODEL_PATH}`")
    if MODEL_ID:
        st.write(f"**Model ID:** {MODEL_ID}")
    if MODEL_PATH.name != "best.pt":
        st.warning(
            "A custom model is configured via "
            f"{MODEL_PATH_ENV}. Changing it requires an app restart; "
            "predictions use the new model's own classes."
        )

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear road-surface image.",
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open image: {exc}")
        st.stop()

    left, right = st.columns(2)

    with left:
        st.subheader("Original image")
        st.image(original_image, use_container_width=True)

    # Cache key for the raw inference of the currently uploaded image.
    image_key = f"{uploaded_file.name}_{uploaded_file.size}"

    def run_once_and_cache():
        """Run YOLO once per image; cache candidate detections + names.

        Called only from the Detect button. Slider reruns reuse the
        cached candidates and never call model.predict() again.
        """
        model, load_warning, loaded_path = load_model(str(MODEL_PATH))
        class_warnings = check_class_compatibility(model)
        started = time.perf_counter()

        with st.spinner("Analyzing road image..."):
            result = model.predict(
                source=original_image,
                conf=CONFIDENCE_FLOOR,
                verbose=False,
            )[0]

        duration = time.perf_counter() - started
        METRICS["inference_duration"].observe(duration)
        METRICS["selected_threshold"].set(CONFIDENCE_FLOOR)

        candidates = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for coords, class_id, score in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.cls.cpu().tolist(),
                boxes.conf.cpu().tolist(),
            ):
                class_id = int(class_id)
                candidates.append(
                    {
                        "score": float(score),
                        "class_name": result.names.get(class_id, str(class_id)),
                        "x1": coords[0],
                        "y1": coords[1],
                        "x2": coords[2],
                        "y2": coords[3],
                    }
                )

        if custom_mode == "Yes — custom" and custom_confidence is not None:
            initial_threshold = float(custom_confidence)
            confidence_mode = "custom"
        else:
            initial_threshold = select_auto_threshold(
                [item["score"] for item in candidates]
            )
            confidence_mode = "automatic"

        st.session_state["crack_candidates"] = candidates
        st.session_state["crack_image_key"] = image_key
        st.session_state["crack_image_name"] = uploaded_file.name
        st.session_state["crack_inference_seconds"] = duration
        st.session_state["crack_run_id"] = str(uuid.uuid4())
        st.session_state["crack_slider_default"] = initial_threshold
        st.session_state["crack_confidence_mode"] = confidence_mode
        st.session_state["crack_threshold_slider"] = int(
            round(min(max(initial_threshold, 0.0), 1.0) * 100)
        )
        st.session_state["crack_model_info"] = {
            **collect_model_metadata(model, loaded_path),
            "model_id": get_model_id(),
            "class_warnings": class_warnings,
        }
        if load_warning:
            st.session_state["crack_load_warning"] = load_warning
        st.session_state["crack_logged_threshold"] = None
        st.session_state["crack_mlflow_message"] = None
        st.session_state["crack_mlflow_run_id"] = None
        st.session_state["crack_lineage_start_msg"] = None
        st.session_state["crack_lineage_msg"] = None
        # OpenLineage START: emitted as soon as the run is identified so the
        # run exists in Marquez even if a later step fails (then a FAIL event
        # closes it). Sent here and not in annotate_and_render because the
        # prediction may raise before any rendering happens.
        _, st.session_state["crack_lineage_start_msg"] = send_lineage_event(
            st.session_state["crack_run_id"],
            "START",
            model_info=st.session_state["crack_model_info"],
            image_name=uploaded_file.name,
        )

    def annotate_and_render(threshold):
        """Filter cached candidates at `threshold` and render everything.

        Pure Python: no model.predict(), no MLflow logging, and no
        Prometheus increment happens on slider reruns.
        """
        candidates = st.session_state.get("crack_candidates", [])
        inference_seconds = st.session_state.get("crack_inference_seconds", 0.0)
        run_id = st.session_state.get("crack_run_id", str(uuid.uuid4()))
        model_info = st.session_state.get("crack_model_info") or {}

        if model_info.get("class_warnings"):
            st.warning(
                "Model-class mismatch: "
                + " | ".join(model_info["class_warnings"])
            )
        load_warning = st.session_state.get("crack_load_warning")
        if load_warning:
            st.error(load_warning)

        retained = filter_detection_records(
            candidates,
            threshold,
            image_size=original_image.size,
        )

        detections = []
        annotated = original_image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        for record in retained:
            score = record["score"]
            class_name = record["class_name"]

            # Requested demonstration mapping: confidence percentage -> severity label.
            severity_score = score * 100
            severity = get_severity_label(severity_score)
            METRICS["detections"].labels(crack_class=class_name).inc()
            METRICS["detection_confidence"].observe(score)

            detections.append(
                {
                    "Crack class": class_name,
                    "Confidence (%)": round(severity_score, 2),
                    "Confidence category": get_confidence_band(score),
                    "Severity label (from confidence)": severity,
                    "x1": round(record["x1"], 1),
                    "y1": round(record["y1"], 1),
                    "x2": round(record["x2"], 1),
                    "y2": round(record["y2"], 1),
                    "Threshold used (%)": round(threshold * 100, 2),
                }
            )

            x1, y1, x2, y2 = map(int, (record["x1"], record["y1"], record["x2"], record["y2"]))
            label = (
                f"{class_name} | Conf: {score:.0%} "
                f"| Severity: {severity}"
            )
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            label_y = max(0, y1 - 22)
            text_box = draw.textbbox((x1, label_y), label, font=font)
            draw.rectangle(text_box, fill="red")
            draw.text((x1, label_y), label, fill="white", font=font)

        # Log to MLflow and OpenLineage once per actual inference, not per slider move.
        logged_threshold = st.session_state.get("crack_logged_threshold")
        if logged_threshold is None:
            st.session_state["crack_logged_threshold"] = threshold
            mlflow_ok, st.session_state["crack_mlflow_message"] = log_prediction_to_mlflow(
                run_id=run_id,
                confidence_mode="interactive-slider",
                used_threshold=threshold,
                detections=detections,
                duration_seconds=inference_seconds,
                model_info=model_info,
            )
            # OpenLineage COMPLETE for this inference run; the START event was
            # already emitted in run_once_and_cache(). The MLflow tracking run
            # id (if logging succeeded) is carried in the events so the lineage
            # graph correlates with the MLflow experiment.
            image_name = st.session_state.get("crack_image_name", "")
            severity_counts = {}
            for item in detections:
                sev = item["Severity label (from confidence)"]
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            _, st.session_state["crack_lineage_msg"] = send_lineage_event(
                run_id,
                "COMPLETE",
                model_info=model_info,
                mlflow_run_id=st.session_state.get("crack_mlflow_run_id") if mlflow_ok else None,
                image_name=image_name,
                detection_count=len(detections),
                severity_counts=severity_counts,
            )

        # --- Render (identical layout to the previous version) ---
        with right:
            st.subheader("Detection result")
            st.image(annotated, use_container_width=True)

        st.markdown("---")
        st.subheader("Detection summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detections retained", len(detections))
        with col2:
            st.metric("Threshold used", f"{threshold:.0%}")
        with col3:
            st.metric(
                "Distinct crack classes",
                len({item["Crack class"] for item in detections}),
            )

        # Document the confidence distribution of all candidates.
        st.subheader("Confidence threshold search log")
        log_df = pd.DataFrame(build_threshold_search_log([c["score"] for c in candidates]))
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        log_csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download threshold search log (CSV)",
            data=log_csv,
            file_name="confidence_threshold_log.csv",
            mime="text/csv",
            key="download_threshold_log",
        )

        if detections:
            st.subheader("Detected cracks")
            detections_df = pd.DataFrame(detections).sort_values(
                "Confidence (%)",
                ascending=False,
            )
            st.dataframe(
                detections_df,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Class-wise detection count")
            counts = (
                detections_df["Crack class"]
                .value_counts()
                .rename_axis("Crack class")
                .reset_index(name="Detection count")
            )
            st.dataframe(counts, use_container_width=True, hide_index=True)

            report_csv = detections_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download crack detection report (CSV)",
                data=report_csv,
                file_name="road_crack_report.csv",
                mime="text/csv",
                key="download_detection_report",
            )

            image_buffer = BytesIO()
            annotated.save(image_buffer, format="PNG")
            st.download_button(
                "Download annotated image",
                data=image_buffer.getvalue(),
                file_name="road_crack_detection.png",
                mime="image/png",
                use_container_width=True,
                key="download_annotated_image",
            )

            st.info(
                "The Low/Moderate/Severe label is mapped from YOLO "
                "confidence using your requested bands: 0–35 = Low, "
                ">35–70 = Moderate, and >70 = Severe. This is a "
                "confidence-based demo label, not a validated measurement "
                "of physical crack severity."
            )
        else:
            st.warning(
                "No detections at this threshold. Lower the slider to "
                "reveal lower-confidence boxes; if nothing appears even "
                f"near 0%, the model found nothing at the "
                f"{CONFIDENCE_FLOOR:.0%} confidence floor."
            )

        with st.expander("MLOps integration status"):
            st.write(
                "**Prometheus:** metrics endpoint active at "
                "http://localhost:8001/metrics."
                if METRICS["server_started"]
                else "**Prometheus:** metrics server could not start on port 8001; check whether the port is already in use."
            )
            st.write(
                f"**MLflow:** {st.session_state.get('crack_mlflow_message') or 'not logged yet'}"
            )
            st.write(
                f"**Model in use:** {model_info.get('model_file', MODEL_PATH.name)} "
                f"(task: {model_info.get('model_task', 'unknown')})"
            )
            if model_info.get("model_fingerprint_sha256"):
                st.caption(
                    "SHA-256: "
                    + model_info["model_fingerprint_sha256"][:16]
                    + "..."
                )
            lineage_url, _, _ = get_lineage_config()
            if lineage_url:
                start_msg = st.session_state.get("crack_lineage_start_msg") or "no START event sent yet"
                lineage_msg = st.session_state.get("crack_lineage_msg") or "no lineage event sent yet"
                st.write(f"**Marquez/OpenLineage START:** {start_msg} (target: {lineage_url})")
                st.write(f"**Marquez/OpenLineage COMPLETE:** {lineage_msg} (target: {lineage_url})")
                st.caption(
                    "Lineage graph: http://localhost:3002 (job `road_crack/road_crack_inference`)"
                )
            else:
                st.write(
                    "**Marquez/OpenLineage:** not configured; set OPENLINEAGE_URL to "
                    "http://127.0.0.1:5000 (the bundled Marquez API) to enable lineage."
                )

    if st.button(
        "Check and Detect Cracks",
        type="primary",
        use_container_width=True,
    ):
        METRICS["prediction_runs"].inc()
        try:
            run_once_and_cache()
            st.session_state["crack_show_results"] = True
        except Exception as exc:
            METRICS["prediction_errors"].inc()
            st.error(f"Prediction failed: {exc}")
            # OpenLineage FAIL: close the run that already sent START so the
            # lineage graph does not show a permanently running job.
            fail_run_id = st.session_state.get("crack_run_id")
            if fail_run_id:
                _, st.session_state["crack_lineage_msg"] = send_lineage_event(
                    fail_run_id,
                    "FAIL",
                    model_info=st.session_state.get("crack_model_info"),
                    image_name=st.session_state.get("crack_image_name"),
                )

    # Render results (and re-render on every slider move) from the cache.
    if st.session_state.get("crack_show_results") and st.session_state.get("crack_image_key") == image_key:
        with right:
            st.subheader("Interactive confidence threshold")
        slider_default_pct = int(
            round(
                min(
                    max(st.session_state.get("crack_slider_default", CONFIDENCE_FLOOR), 0.0),
                    1.0,
                )
                * 100
            )
        )
        slider_threshold = st.slider(
            "Confidence threshold (%)",
            min_value=0,
            max_value=100,
            value=slider_default_pct,
            step=1,
            format="%d%%",
            key="crack_threshold_slider",
            help=(
                "Filters the cached detections from the single inference pass. "
                "Moving the slider does not rerun the model. Boxes with "
                "confidence >= the selected threshold are shown."
            ),
        )
        annotate_and_render(slider_threshold / 100)
else:
    st.info("Upload an image to begin.")

st.markdown("---")
st.caption(
    "For research and demonstration use. Automatic threshold selection "
    "does not guarantee correct detections. Validate on unseen road images."
)
