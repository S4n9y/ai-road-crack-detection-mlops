from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Road Crack Detection",
    page_icon="🛣️",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "best.pt"


@st.cache_resource(show_spinner="Loading road-crack model...")
def load_model(model_path: str):
    """Load and cache the trained YOLO model."""
    return YOLO(model_path)


def get_severity_label(severity_score):
    """Map a 0-100 score to the requested demo severity bands.

    In this version, the value supplied is YOLO confidence converted to
    a percentage. This is only a confidence-based demo label, not a
    validated assessment of physical crack severity.
    """
    if severity_score is None or not 0 <= severity_score <= 100:
        return "Invalid score"

    if severity_score <= 35:
        return "Low"
    elif severity_score <= 70:
        return "Moderate"
    else:
        return "Severe"


def get_confidence_band(score):
    """Return a descriptive category for YOLO's confidence score (0-1)."""
    percent = score * 100

    if percent <= 35:
        return "Low confidence"
    elif percent <= 70:
        return "Moderate confidence"
    else:
        return "High confidence"


def run_auto_detection(model, image, progress_bar):
    """Test thresholds from 100% down to 0% in 5-point steps.

    Return the result at the highest tested threshold that produces at
    least one detection, together with the threshold search log.
    """
    thresholds = [value / 100 for value in range(100, -1, -5)]
    search_log = []
    result = None

    for index, threshold in enumerate(thresholds):
        result = model.predict(
            source=image,
            conf=threshold,
            verbose=False,
        )[0]

        boxes = result.boxes
        detection_count = len(boxes) if boxes is not None else 0

        search_log.append(
            {
                "Threshold tested (%)": int(threshold * 100),
                "Detections found": detection_count,
            }
        )

        progress_bar.progress(
            (index + 1) / len(thresholds),
            text=f"Testing confidence threshold: {threshold:.0%}",
        )

        if detection_count > 0:
            return result, threshold, search_log

    return result, 0.0, search_log


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

# Check for the trained YOLO model.
if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Place best.pt in the same folder as app.py."
    )
    st.stop()

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

    if st.button(
        "Check and Detect Cracks",
        type="primary",
        use_container_width=True,
    ):
        try:
            model = load_model(str(MODEL_PATH))
            progress_bar = None

            with st.spinner("Analyzing road image..."):
                if custom_mode == "Yes — custom":
                    result, used_threshold, search_log = run_custom_detection(
                        model,
                        original_image,
                        custom_confidence,
                    )
                else:
                    progress_bar = st.progress(
                        0,
                        text="Starting automatic confidence search...",
                    )
                    result, used_threshold, search_log = run_auto_detection(
                        model,
                        original_image,
                        progress_bar,
                    )

            if progress_bar is not None:
                progress_bar.empty()

            boxes = result.boxes
            detections = []

            # Start with the model's boxes, without its default text labels.
            annotated_array = result.plot(
                labels=False,
                conf=False,
            )[:, :, ::-1].copy()
            annotated = Image.fromarray(annotated_array)
            draw = ImageDraw.Draw(annotated)

            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except OSError:
                font = ImageFont.load_default()

            if boxes is not None and len(boxes) > 0:
                coordinates = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.cpu().tolist()
                scores = boxes.conf.cpu().tolist()

                for coords, class_id, score in zip(
                    coordinates,
                    class_ids,
                    scores,
                ):
                    class_id = int(class_id)
                    score = float(score)
                    class_name = result.names.get(class_id, str(class_id))

                    # Requested demonstration mapping: confidence percentage -> severity label.
                    severity_score = score * 100
                    severity = get_severity_label(severity_score)

                    detections.append(
                        {
                            "Crack class": class_name,
                            "Confidence (%)": round(severity_score, 2),
                            "Confidence category": get_confidence_band(score),
                            "Severity label (from confidence)": severity,
                            "x1": round(coords[0], 1),
                            "y1": round(coords[1], 1),
                            "x2": round(coords[2], 1),
                            "y2": round(coords[3], 1),
                            "Threshold used (%)": round(used_threshold * 100, 2),
                        }
                    )

                    x1, y1, x2, y2 = map(int, coords)
                    label = (
                        f"{class_name} | Conf: {score:.0%} "
                        f"| Severity: {severity}"
                    )

                    draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        outline="red",
                        width=3,
                    )

                    label_y = max(0, y1 - 22)
                    text_box = draw.textbbox(
                        (x1, label_y),
                        label,
                        font=font,
                    )
                    draw.rectangle(text_box, fill="red")
                    draw.text(
                        (x1, label_y),
                        label,
                        fill="white",
                        font=font,
                    )

            # Display result.
            with right:
                st.subheader("Detection result")
                st.image(annotated, use_container_width=True)

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

            # Document the threshold search.
            st.subheader("Confidence threshold search log")
            log_df = pd.DataFrame(search_log)
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

                if custom_mode == "No — automatic":
                    st.success(
                        f"Automatic search selected {used_threshold:.0%}, "
                        "the highest tested threshold that returned "
                        "at least one detection."
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
                    "No detections were found, even at 0% confidence. "
                    "Check the model, image quality, and training data."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.info("Upload an image to begin.")

st.markdown("---")
st.caption(
    "For research and demonstration use. Automatic threshold selection "
    "does not guarantee correct detections. Validate on unseen road images."
)
