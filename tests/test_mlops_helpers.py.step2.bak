import pytest
from mlops_helpers import severity_label, confidence_band


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "Low"),
        (35, "Low"),
        (36, "Moderate"),
        (70, "Moderate"),
        (70.01, "Severe"),
        (71, "Severe"),
        (100, "Severe"),
    ],
)
def test_severity_label_boundaries(score, expected):
    assert severity_label(score) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "Low confidence"),
        (0.35, "Low confidence"),
        (0.36, "Moderate confidence"),
        (0.70, "Moderate confidence"),
        (0.71, "High confidence"),
        (1.0, "High confidence"),
    ],
)
def test_confidence_band(score, expected):
    assert confidence_band(score) == expected


def test_invalid_severity_scores_are_rejected():
    assert severity_label(None) == "Invalid score"
    assert severity_label(-1) == "Invalid score"
    assert severity_label(101) == "Invalid score"


def test_invalid_confidence_values_are_rejected():
    assert confidence_band(-0.1) == "Invalid confidence"
    assert confidence_band(1.1) == "Invalid confidence"
