from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ----------------------------
# App configuration
# ----------------------------
st.set_page_config(
    page_title="AI Road Crack Detection",
    page_icon="🛣️",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"


# ----------------------------
# Model and helper functions
# ----------------------------
@st.cache_resource(show_spinner="Loading road-crack model...")
def load_model(model_path: str):
    """Load and cache the trained YOLO model."""
    return YOLO(model_path)


def get_severity_label(confidence_percent):
    """
    Demo-only mapping from confidence to a label.
    This is NOT a physical crack-severity measurement.
    """
    if confidence_percent is None or not 0 <= confidence_percent <= 100:
        return "Invalid score"
    if confidence_percent <= 35:
        return "Low"
    if confidence_percent <= 70:
        return "Moderate"
    return "Severe"


def get_confidence_band(score):
    """Convert YOLO confidence (0-1) to a descriptive band."""
    percent = float(score) * 100
    if percent <= 35:
        return "Low confidence"
    if percent <= 70:
        return "Moderate confidence"
    return "High confidence"


def predict_at_low_threshold(model, image):
    """
    Run YOLO once at a low threshold. The app then applies the selected
    threshold to the returned boxes, avoiding up to 21 repeated inferences.
    """
    return model.predict(
        source=image,
        conf=0.001,
        verbose=False,
    )[0]


def choose_automatic_threshold(result):
    """
    Find the highest 5-percentage-point threshold (100%, 95%, ... 0%)
    at which at least one returned detection remains.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0.0, []

    scores = boxes.conf.detach().cpu().tolist()
    thresholds = [value / 100 for value in range(100, -1, -5)]

    search_log = []
    chosen_threshold = 0.0

    for threshold in thresholds:
        count = sum(float(score) >= threshold for score in scores)
        search_log.append(
            {
                "Threshold tested (%)": int(threshold * 100),
                "Detections found": count,
            }
        )
        if count > 0:
            chosen_threshold = threshold
            break

    return chosen_threshold, search_log


def make_search_log(result, threshold):
    """Return a short log for a custom confidence threshold."""
    boxes = result.boxes
    scores = [] if boxes is None else boxes.conf.detach().cpu().tolist()
    count = sum(float(score) >= threshold for score in scores)
    return [
        {
            "Threshold tested (%)": round(threshold * 100, 2),
            "Detections found": count,
        }
    ]


def get_selected_indices(result, threshold):
    """Indices of boxes whose confidence meets the selected threshold."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    scores = boxes.conf.detach().cpu().tolist()
    return [
        index for index, score in enumerate(scores)
        if float(score) >= threshold
    ]


def annotate_image(original_image, result, selected_indices):
    """Draw only the detections retained by the selected threshold."""
    annotated = original_image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    boxes = result.boxes
    if boxes is None:
        return annotated

    coordinates = boxes.xyxy.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    names = result.names

    for index in selected_indices:
        coords = coordinates[index]
        class_id = int(class_ids[index])
        score = float(scores[index])
        class_name = names.get(class_id, str(class_id))
        severity = get_severity_label(score * 100)

        x1, y1, x2, y2 = [int(value) for value in coords]
        label = f"{class_name} | Conf: {score:.0%} | Severity: {severity}"

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        text_box = draw.textbbox((x1, max(0, y1 - 22)), label, font=font)
        # Keep the label background within the image as much as possible.
        label_y = max(0, y1 - (text_box[3] - text_box[1]) - 6)
        text_box = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text((x1, label_y), label, fill="white", font=font)

    return annotated


def build_detection_rows(result, selected_indices, used_threshold):
    """Build the detailed detection table."""
    rows = []
    boxes = result.boxes
    if boxes is None:
        return rows

    coordinates = boxes.xyxy.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    names = result.names

    for index in selected_indices:
        coords = coordinates[index]
        class_id = int(class_ids[index])
        score = float(scores[index])
        confidence_percent = score * 100
        class_name = names.get(class_id, str(class_id))

        rows.append(
            {
                "Crack class": class_name,
                "Confidence (%)": round(confidence_percent, 2),
                "Confidence category": get_confidence_band(score),
                "Severity label (from confidence)": get_severity_label(
                    confidence_percent
                ),
                "x1": round(coords[0], 1),
                "y1": round(coords[1], 1),
                "x2": round(coords[2], 1),
                "y2": round(coords[3], 1),
                "Threshold used (%)": round(used_threshold * 100, 2),
            }
        )

    return rows


# ----------------------------
# Page content
# ----------------------------
st.title("🛣️ AI-Based Road Crack Detection")
st.write(
    "Upload a road image to identify alligator, edge, longitudinal, "
    "and transverse cracks."
)
st.caption(
    "Confidence indicates the model's certainty in a prediction. "
    "The Low/Moderate/Severe label is mapped from confidence for "
    "demonstration; it is not a physical crack-severity measurement."
)

if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Place best.pt in the same folder as app.py."
    )
    st.stop()

with st.sidebar:
    st.header("Prediction settings")
    custom_mode = st.radio(
        "Would you like to set a custom confidence threshold?",
        options=["No — automatic", "Yes — custom"],
        index=0,
        help=(
            "Automatic mode selects the highest tested 5%-step threshold "
            "that retains at least one detection. Custom mode uses your "
            "chosen threshold."
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
            help="Only detections at or above this confidence are retained.",
        )
        st.write(f"**Selected threshold:** {custom_confidence:.0%}")
    else:
        st.info(
            "The model runs once at a low diagnostic threshold. The app "
            "then tests the returned confidence scores in 5-point steps."
        )

    st.markdown("---")
    st.write("**Model:** YOLO11n")
    st.write("**Model file:** best.pt")

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

    # A new upload invalidates any previous result.
    image_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("last_image_key") != image_key:
        st.session_state["last_image_key"] = image_key
        st.session_state.pop("prediction_output", None)

    if st.button(
        "Check and Detect Cracks",
        type="primary",
        use_container_width=True,
    ):
        try:
            model = load_model(str(MODEL_PATH))

            with st.spinner("Analyzing road image..."):
                result = predict_at_low_threshold(model, original_image)

                if custom_mode == "Yes — custom":
                    used_threshold = float(custom_confidence)
                    search_log = make_search_log(result, used_threshold)
                else:
                    used_threshold, search_log = choose_automatic_threshold(result)

                selected_indices = get_selected_indices(result, used_threshold)
                annotated = annotate_image(
                    original_image, result, selected_indices
                )
                detections = build_detection_rows(
                    result, selected_indices, used_threshold
                )

                st.session_state["prediction_output"] = {
                    "image_key": image_key,
                    "annotated": annotated,
                    "detections": detections,
                    "search_log": search_log,
                    "used_threshold": used_threshold,
                    "custom_mode": custom_mode,
                    "model_names": model.names,
                    "raw_count": (
                        len(result.boxes)
                        if result.boxes is not None
                        else 0
                    ),
                }
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.exception(exc)

    # Render saved output on every rerun so it doesn't disappear when
    # Streamlit reruns the script after a widget interaction.
    output = st.session_state.get("prediction_output")
    if output and output.get("image_key") == image_key:
        with right:
            st.subheader("Detection result")
            st.image(output["annotated"], use_container_width=True)

        detections = output["detections"]
        used_threshold = output["used_threshold"]
        search_log = output["search_log"]

        st.markdown("---")
        st.subheader("Detection summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detections retained", len(detections))
        with col2:
            st.metric("Threshold used", f"{used_threshold:.0%}")
        with col3:
            st.metric(
                "Distinct crack classes",
                len({item["Crack class"] for item in detections}),
            )

        st.write("**Model classes:**", output["model_names"])
        st.write("**Raw boxes returned at diagnostic threshold:**", output["raw_count"])

        st.subheader("Confidence threshold search log")
        log_df = pd.DataFrame(search_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download threshold search log (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="confidence_threshold_log.csv",
            mime="text/csv",
            key="download_threshold_log",
        )

        if detections:
            st.subheader("Detected cracks")
            detections_df = pd.DataFrame(detections).sort_values(
                "Confidence (%)", ascending=False
            )
            st.dataframe(
                detections_df,
                use_container_width=True,
                hide_index=True,
            )

            counts = (
                detections_df["Crack class"]
                .value_counts()
                .rename_axis("Crack class")
                .reset_index(name="Detection count")
            )
            st.subheader("Class-wise detection count")
            st.dataframe(counts, use_container_width=True, hide_index=True)

            st.download_button(
                "Download crack detection report (CSV)",
                data=detections_df.to_csv(index=False).encode("utf-8"),
                file_name="road_crack_report.csv",
                mime="text/csv",
                key="download_detection_report",
            )

            image_buffer = BytesIO()
            output["annotated"].save(image_buffer, format="PNG")
            st.download_button(
                "Download annotated image",
                data=image_buffer.getvalue(),
                file_name="road_crack_detection.png",
                mime="image/png",
                use_container_width=True,
                key="download_annotated_image",
            )

            if output["custom_mode"] == "No — automatic":
                st.success(
                    f"Automatic search selected {used_threshold:.0%}, "
                    "the highest tested 5-point threshold that retained "
                    "at least one detection from the returned boxes."
                )

            st.info(
                "Low/Moderate/Severe is mapped from YOLO confidence: "
                "0–35 = Low, >35–70 = Moderate, >70 = Severe. "
                "This is a confidence-based demonstration label, not a "
                "validated measurement of physical crack severity."
            )
        else:
            st.warning(
                "No detections met the selected threshold. The app ran YOLO "
                "at a low diagnostic threshold, but no returned boxes met "
                "the chosen threshold. Check the model's class mapping, "
                "training data, and a few known test images."
            )
else:
    st.info("Upload an image to begin.")

st.markdown("---")
st.caption(
    "For research and demonstration use. Confidence thresholds do not "
    "guarantee correct detections. Validate on unseen road images."
)

