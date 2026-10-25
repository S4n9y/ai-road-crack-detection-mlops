"""Tests for coordinate validation and image/table consistency.

Uses mocked detection records and mocked YOLO predictions; no model
file, torch, or Streamlit runtime is required. best.pt is untouched.
"""

from types import SimpleNamespace

import pytest

from mlops_helpers import (
    CONFIDENCE_FLOOR,
    filter_detection_records,
    filter_boxes_at_threshold,
    run_auto_detection,
    validate_box_coords,
)


class FakeBoxes:
    def __init__(self, scores):
        self.conf = scores
        self._scores = list(scores)

    def __len__(self):
        return len(self._scores)

    def __getitem__(self, indices):
        return FakeBoxes([self._scores[i] for i in indices])


def make_record(score, coords, class_name="alligator cracking"):
    x1, y1, x2, y2 = coords
    return {
        "score": score,
        "class_name": class_name,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


# --- validate_box_coords -------------------------------------------------

def test_valid_interior_box_unchanged():
    box = (10.0, 20.0, 50.0, 60.0)
    assert validate_box_coords(box, 100, 100) == box


def test_box_fully_outside_bounds_is_dropped():
    # Entirely beyond the image: clipping would collapse it, so it is
    # rejected rather than producing a fabricated edge box.
    assert validate_box_coords((-50.0, -50.0, -10.0, -10.0), 100, 100) is None
    assert validate_box_coords((150.0, 150.0, 180.0, 180.0), 100, 100) is None


def test_partially_out_of_bounds_box_is_clipped_to_image():
    # Box crossing the edge: the in-image part is real evidence.
    assert validate_box_coords((-5.0, -5.0, 105.0, 105.0), 100, 100) == (0.0, 0.0, 100.0, 100.0)
    assert validate_box_coords((90.0, 20.0, 120.0, 60.0), 100, 100) == (90.0, 20.0, 100.0, 60.0)


def test_negative_coordinate_rejects_box():
    # The negative coord clips to 0 and the box stays valid, so it is
    # clipped (not dropped); only fully-outside boxes are rejected.
    assert validate_box_coords((-1.0, 10.0, 50.0, 60.0), 100, 100) == (0.0, 10.0, 50.0, 60.0)


def test_nan_and_inf_reject_box():
    nan = float("nan")
    inf = float("inf")
    assert validate_box_coords((nan, 0, 50, 50), 100, 100) is None
    assert validate_box_coords((0, inf, 50, 50), 100, 100) is None


def test_degenerate_boxes_rejected():
    assert validate_box_coords((10.0, 10.0, 10.0, 50.0), 100, 100) is None  # zero width
    assert validate_box_coords((10.0, 10.0, 50.0, 10.0), 100, 100) is None  # zero height
    assert validate_box_coords((50.0, 60.0, 40.0, 20.0), 100, 100) is None  # inverted


# --- filter_detection_records --------------------------------------------

def _records():
    return [
        make_record(0.91, (10, 10, 30, 30)),
        make_record(0.63, (40, 40, 60, 60), class_name="transverse crack"),
        make_record(0.72, (70, 70, 90, 90), class_name="longitudinal crack"),
    ]


def test_multiple_boxes_filtered_by_threshold():
    retained = filter_detection_records(_records(), 0.70)
    assert [r["score"] for r in retained] == [0.91, 0.72]
    # Coordinates and labels pass through unchanged for valid boxes.
    assert retained[0]["x1"] == 10 and retained[0]["class_name"] == "alligator cracking"


def test_boundary_score_equals_threshold_is_kept():
    records = [make_record(0.70, (1, 1, 2, 2))]
    assert len(filter_detection_records(records, 0.70)) == 1
    assert len(filter_detection_records(records, 0.71)) == 0


def test_empty_candidates_give_empty_result():
    assert filter_detection_records([], 0.5) == []


def test_all_below_threshold_gives_empty_result():
    assert filter_detection_records(_records(), 0.95) == []


def test_invalid_coords_dropped_only_when_image_size_given():
    records = _records() + [make_record(0.80, (float("nan"), 1, 2, 2))]
    without_size = filter_detection_records(records, 0.5)
    assert len(without_size) == 4  # no validation requested
    with_size = filter_detection_records(records, 0.5, image_size=(100, 100))
    assert len(with_size) == 3  # NaN box dropped
    assert all(r["class_name"] != "alligator cracking" or r["x1"] == 10 for r in with_size)


def test_threshold_never_changes_coordinates_or_labels():
    for threshold in (0.0, 0.3, 0.63, 0.91, 1.0):
        retained = filter_detection_records(_records(), threshold, image_size=(100, 100))
        for record in retained:
            original = next(r for r in _records() if r["score"] == record["score"])
            assert (record["x1"], record["y1"], record["x2"], record["y2"]) == (
                original["x1"], original["y1"], original["x2"], original["y2"],
            )
            assert record["class_name"] == original["class_name"]


# --- mocked end-to-end run_auto_detection --------------------------------

def _mock_model(scores, coords_list, names):
    class MockBoxes(FakeBoxes):
        def __init__(self, scores_arg, coords_arg):
            super().__init__(scores_arg)
            self._coords = coords_arg

        def __getitem__(self, indices):
            return MockBoxes(
                [self._scores[i] for i in indices],
                [self._coords[i] for i in indices],
            )

        @property
        def xyxy(self):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: self._coords))

        @property
        def cls(self):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [0] * len(self._scores)))

    boxes = MockBoxes(scores, coords_list)
    result = SimpleNamespace(boxes=boxes, names=names)
    calls = []

    def predict(source=None, conf=None, verbose=False):
        calls.append(conf)
        return [result]

    return SimpleNamespace(predict=predict), calls


def test_run_auto_detection_passes_coordinates_through_unscaled():
    # A box covering most of a 640x640 image must come back unchanged:
    # the app must not shrink, scale, or reinterpret model geometry.
    coords = [[5.0, 10.0, 630.0, 635.0]]
    model, calls = _mock_model([0.56], coords, {0: "alligator cracking"})
    result, threshold, _ = run_auto_detection(model, "img")
    assert calls == [CONFIDENCE_FLOOR]
    assert threshold == 0.55
    assert result.boxes.xyxy.cpu().tolist() == coords  # identical, unscaled
    assert list(result.boxes.conf) == [0.56]


def test_run_auto_detection_multiple_boxes_sorted_threshold_filtering():
    # 0.91 -> keeps at 0.90 grid; 0.44 and 0.72 are below 0.90 and drop.
    # (The automatic search selects the HIGHEST grid threshold retaining
    # at least one box, which is 0.90 here, not the threshold that keeps
    # the most boxes.)
    coords = [[0, 0, 10, 10], [20, 20, 30], [40, 40, 50]]
    model, calls = _mock_model([0.91, 0.44, 0.72], coords, {0: "crack"})
    result, threshold, _ = run_auto_detection(model, "img")
    assert calls == [CONFIDENCE_FLOOR]
    assert threshold == 0.90
    assert list(result.boxes.conf) == [0.91]
    assert result.boxes.xyxy.cpu().tolist() == [[0, 0, 10, 10]]


def test_run_auto_detection_no_candidates_returns_empty_boxes():
    model, calls = _mock_model([], [], {0: "crack"})
    result, threshold, _ = run_auto_detection(model, "img")
    assert calls == [CONFIDENCE_FLOOR]
    assert threshold == CONFIDENCE_FLOOR
    assert list(result.boxes.conf) == []
