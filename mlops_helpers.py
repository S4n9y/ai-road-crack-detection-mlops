"""Pure helper functions used by the Streamlit app and pytest.

Also contains the single-inference automatic threshold search: the model
runs once at a low confidence floor, then the highest 5-point threshold
that retains at least one detection is selected from the returned scores.
"""

from copy import copy

CONFIDENCE_FLOOR = 0.01
AUTO_THRESHOLD_STEP = 0.05


def severity_label(severity_score):
    """Map a 0-100 demo score to Low / Moderate / Severe.

    The current app uses YOLO confidence percentage as the demo score; it is
    not a validated physical road-damage severity measurement.
    """
    if severity_score is None or not isinstance(severity_score, (int, float)):
        return "Invalid score"
    if not 0 <= severity_score <= 100:
        return "Invalid score"
    if severity_score <= 35:
        return "Low"
    if severity_score <= 70:
        return "Moderate"
    return "Severe"


def confidence_band(score):
    """Map YOLO confidence (0-1) to a readable confidence category."""
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return "Invalid confidence"
    percent = score * 100
    if percent <= 35:
        return "Low confidence"
    if percent <= 70:
        return "Moderate confidence"
    return "High confidence"


def auto_threshold_candidates(step=AUTO_THRESHOLD_STEP):
    """Threshold grid from 1.00 down to 0.00 in `step` increments."""
    count = int(round(1.0 / step))
    return [round(index * step, 10) for index in range(count, -1, -1)]


def select_auto_threshold(scores, step=AUTO_THRESHOLD_STEP):
    """Highest threshold on the grid retaining at least one detection.

    Returns CONFIDENCE_FLOOR when `scores` is empty so callers can
    distinguish "no detections at all" from a real selection.
    """
    for threshold in auto_threshold_candidates(step):
        if any(score >= threshold for score in scores):
            return threshold
    return CONFIDENCE_FLOOR


def build_threshold_search_log(scores, step=AUTO_THRESHOLD_STEP):
    """Detection count at every grid threshold, from one inference pass."""
    return [
        {
            "Threshold tested (%)": int(round(threshold * 100)),
            "Detections found": sum(1 for score in scores if score >= threshold),
        }
        for threshold in auto_threshold_candidates(step)
    ]


def filter_boxes_at_threshold(boxes, scores, threshold):
    """Keep only boxes whose confidence is >= threshold.

    `boxes` must support indexing with a list of integer positions
    (Ultralytics Boxes does). Returns `boxes` unchanged when nothing
    needs filtering. Scores whose value equals the threshold are kept.
    """
    if boxes is None or not scores:
        return boxes
    keep = [index for index, score in enumerate(scores) if score >= threshold]
    if len(keep) == len(scores):
        return boxes
    return boxes[keep]


def validate_box_coords(box, image_width, image_height):
    """Sanitize one raw model box without inventing geometry.

    Returns a cleaned (x1, y1, x2, y2) tuple, or None when the box is
    unusable. Rules, applied in order:

    - Any NaN/inf coordinate, or any negative coordinate, rejects the box
      (the model output is corrupt; we do not guess a location).
    - Coordinates are clipped to the image bounds. Ultralytics should
      already do this, but a clipped edge box is still real evidence, so
      clipping is preferred over dropping it. Clipping cannot shrink a
      valid interior box.
    - A degenerate box (zero or negative width/height after clipping)
      is rejected.
    """
    x1, y1, x2, y2 = box
    values = (x1, y1, x2, y2)
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        return None
    # Clip to image bounds FIRST; a box partially outside remains real
    # evidence for its in-image part. Only then reject negatives, which
    # after clipping can only mean the box was entirely outside.
    cx1 = min(max(x1, 0.0), float(image_width))
    cy1 = min(max(y1, 0.0), float(image_height))
    cx2 = min(max(x2, 0.0), float(image_width))
    cy2 = min(max(y2, 0.0), float(image_height))
    if any(v < 0 for v in (x1, y1, x2, y2)) and (cx1 == cx2 or cy1 == cy2):
        # Negative source coords that clip to a degenerate box = fully outside.
        return None
    if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
        return None
    return (cx1, cy1, cx2, cy2)


def filter_detection_records(records, threshold, image_size=None):
    """Retain cached detection records whose confidence >= threshold.

    `records` are plain dicts produced at inference time (one per
    candidate box, including a "score" key). Filtering is pure Python
    so the UI slider never reruns the model.

    When `image_size` = (width, height) is given, each retained record's
    coordinates are validated/first with `validate_box_coords`; invalid
    records are dropped rather than drawn. Coordinate values are never
    otherwise altered, so valid boxes keep the exact geometry the model
    produced.
    """
    retained = [record for record in records if record.get("score", 0.0) >= threshold]
    if image_size is not None:
        width, height = image_size
        validated = []
        for record in retained:
            cleaned = validate_box_coords(
                (record["x1"], record["y1"], record["x2"], record["y2"]),
                width,
                height,
            )
            if cleaned is None:
                continue
            record = {**record, "x1": cleaned[0], "y1": cleaned[1], "x2": cleaned[2], "y2": cleaned[3]}
            validated.append(record)
        return validated
    return retained


def _extract_scores(boxes):
    """Confidence values from an Ultralytics Boxes object or a mock."""
    if boxes is None or not len(boxes):
        return []
    conf = boxes.conf
    if hasattr(conf, "tolist"):
        return list(conf.tolist())
    return list(conf)


def run_auto_detection(model, image, progress_bar=None, conf_floor=CONFIDENCE_FLOOR, step=AUTO_THRESHOLD_STEP):
    """Run YOLO inference once at `conf_floor`, then select the highest
    5-point threshold that retains at least one detection.

    Returns (result, threshold, search_log). The returned result's boxes
    are filtered to the selected threshold so annotations, tables, and
    the reported threshold stay consistent. `progress_bar` is optional
    so this is testable without Streamlit.
    """
    result = model.predict(
        source=image,
        conf=conf_floor,
        verbose=False,
    )[0]

    if progress_bar is not None:
        progress_bar.progress(
            0.5,
            text=f"Analyzing detections at {conf_floor:.0%} confidence floor...",
        )

    scores = _extract_scores(result.boxes)
    threshold = select_auto_threshold(scores, step)
    search_log = build_threshold_search_log(scores, step)

    filtered_boxes = filter_boxes_at_threshold(result.boxes, scores, threshold)
    if filtered_boxes is not result.boxes:
        result = copy(result)
        result.boxes = filtered_boxes

    if progress_bar is not None:
        progress_bar.progress(
            1.0,
            text=f"Selected confidence threshold: {threshold:.0%}",
        )

    return result, threshold, search_log
