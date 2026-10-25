"""Unit tests for the single-inference automatic threshold search.

YOLO inference is mocked; no model file or torch is required.
"""

from types import SimpleNamespace

import pytest

from mlops_helpers import (
    CONFIDENCE_FLOOR,
    auto_threshold_candidates,
    build_threshold_search_log,
    filter_boxes_at_threshold,
    run_auto_detection,
    select_auto_threshold,
)


class FakeBoxes:
    """Minimal stand-in for Ultralytics Boxes supporting conf and indexing."""

    def __init__(self, scores):
        self.conf = scores
        self._scores = list(scores)

    def __len__(self):
        return len(self._scores)

    def __getitem__(self, indices):
        return FakeBoxes([self._scores[i] for i in indices])


def make_mock_model(scores):
    """Model whose single predict() call returns FakeBoxes with `scores`."""
    boxes = FakeBoxes(scores)
    result = SimpleNamespace(boxes=boxes, names={})
    calls = []

    def predict(source=None, conf=None, verbose=False):
        calls.append({"conf": conf})
        return [result]

    return SimpleNamespace(predict=predict), calls, result


def test_auto_threshold_candidates_grid():
    candidates = auto_threshold_candidates()
    assert candidates[0] == 1.0
    assert candidates[-1] == 0.0
    assert len(candidates) == 21
    # 5-percentage-point spacing.
    assert all(
        abs(candidates[i] - candidates[i + 1] - 0.05) < 1e-9
        for i in range(len(candidates) - 1)
    )


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.97], 0.95),
        ([0.73, 0.71], 0.70),
        ([0.69], 0.65),
        ([0.36, 0.51], 0.50),
        ([0.35], 0.35),
        ([0.34], 0.30),
        ([0.02], 0.0),
    ],
)
def test_select_auto_threshold_picks_highest_retaining(scores, expected):
    assert select_auto_threshold(scores) == expected


def test_select_auto_threshold_empty_scores_returns_floor():
    assert select_auto_threshold([]) == CONFIDENCE_FLOOR


def test_build_threshold_search_log_from_single_inference():
    scores = [0.92, 0.44, 0.31]
    log = build_threshold_search_log(scores)
    by_threshold = {row["Threshold tested (%)"]: row["Detections found"] for row in log}
    assert len(log) == 21
    assert by_threshold[100] == 0
    assert by_threshold[95] == 0  # 0.92 < 0.95
    assert by_threshold[90] == 1
    assert by_threshold[45] == 1  # 0.44 < 0.45 drops out
    assert by_threshold[40] == 2
    assert by_threshold[30] == 3
    assert by_threshold[0] == 3


def test_filter_boxes_at_threshold_keeps_passing_boxes():
    boxes = FakeBoxes([0.91, 0.63, 0.72])
    filtered = filter_boxes_at_threshold(boxes, [0.91, 0.63, 0.72], 0.70)
    assert isinstance(filtered, FakeBoxes)
    assert list(filtered.conf) == [0.91, 0.72]


def test_filter_boxes_at_threshold_noop_when_all_pass():
    boxes = FakeBoxes([0.8, 0.9])
    assert filter_boxes_at_threshold(boxes, [0.8, 0.9], 0.70) is boxes


def test_filter_boxes_at_threshold_handles_none():
    assert filter_boxes_at_threshold(None, [], 0.5) is None


def test_run_auto_detection_calls_predict_exactly_once():
    model, calls, _ = make_mock_model([0.87, 0.52])
    _, threshold, _ = run_auto_detection(model, "image")
    assert len(calls) == 1
    assert calls[0]["conf"] == CONFIDENCE_FLOOR
    assert threshold == 0.85


def test_run_auto_detection_filters_result_boxes_to_threshold():
    model, _, original = make_mock_model([0.91, 0.63, 0.72])
    returned, threshold, _ = run_auto_detection(model, "image")
    assert threshold == 0.90
    # Returned result carries only detections >= threshold...
    assert list(returned.boxes.conf) == [0.91]
    # ...while the original model output is left untouched.
    assert list(original.boxes.conf) == [0.91, 0.63, 0.72]


def test_run_auto_detection_no_detections():
    model, calls, _ = make_mock_model([])
    _, threshold, search_log = run_auto_detection(model, "image")
    assert len(calls) == 1
    assert threshold == CONFIDENCE_FLOOR
    assert all(row["Detections found"] == 0 for row in search_log)


def test_run_auto_detection_progress_bar_updated():
    model, _, _ = make_mock_model([0.4])
    updates = []

    class FakeProgress:
        def progress(self, value, text=None):
            updates.append(value)

    run_auto_detection(model, "image", progress_bar=FakeProgress())
    assert updates == [0.5, 1.0]
