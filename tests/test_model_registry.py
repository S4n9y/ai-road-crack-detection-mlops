"""Targeted tests for configuration-driven model selection (model_registry).

These tests never load a real .pt file; the YOLO constructor is monkeypatched
so the suite stays fast and independent of best.pt.
"""

import hashlib
from pathlib import Path

import pytest

import model_registry as mr


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(mr.MODEL_PATH_ENV_VAR, raising=False)
    monkeypatch.delenv("ROAD_CRACK_MODEL_ID", raising=False)


class FakeModel:
    def __init__(self, names="default", task="detect", fail=False):
        self.names = (
            {
                0: "alligator cracking",
                1: "edge cracking",
                2: "longitudinal cracking",
                3: "transverse cracking",
            }
            if names == "default"
            else names
        )
        self.task = task
        self._fail = fail

    def predict(self, source, conf, verbose):
        return []


def test_default_path_is_best_pt(monkeypatch):
    assert mr.resolve_model_path() == mr.PROJECT_ROOT / "best.pt"


def test_env_var_overrides_path(monkeypatch, tmp_path):
    custom = tmp_path / "new_best.pt"
    monkeypatch.setenv(mr.MODEL_PATH_ENV_VAR, str(custom))
    assert mr.resolve_model_path() == custom.resolve()


def test_invalid_path_raises_and_names_default(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    monkeypatch.setenv(mr.MODEL_PATH_ENV_VAR, str(missing))
    with pytest.raises(mr.ModelValidationError, match="not found"):
        mr.validate_and_load_model(missing)
    # Default configuration remains best.pt (resolve ignores the env change here)
    assert mr.resolve_model_path.__name__ == "resolve_model_path"


def test_non_pt_file_rejected(tmp_path):
    bad = tmp_path / "model.onnx"
    bad.write_bytes(b"x")
    with pytest.raises(mr.ModelValidationError, match="Unsupported model file"):
        mr.validate_and_load_model(bad)


def test_corrupt_pt_file_rejected(monkeypatch, tmp_path):
    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"not a real yolo model")
    monkeypatch.setattr(
        "ultralytics.YOLO",
        lambda path: (_ for _ in ()).throw(RuntimeError("bad pickle")),
    )
    with pytest.raises(mr.ModelValidationError, match="Could not load model"):
        mr.validate_and_load_model(bad)


def test_wrong_task_rejected(monkeypatch, tmp_path):
    path = tmp_path / "seg.pt"
    path.write_bytes(b"stub")
    monkeypatch.setattr(
        "ultralytics.YOLO", lambda path: FakeModel(task="segment")
    )
    with pytest.raises(mr.ModelValidationError, match="task type"):
        mr.validate_and_load_model(path)


def test_missing_names_rejected(monkeypatch, tmp_path):
    path = tmp_path / "nolabels.pt"
    path.write_bytes(b"stub")
    monkeypatch.setattr(
        "ultralytics.YOLO", lambda path: FakeModel(names={})
    )
    with pytest.raises(mr.ModelValidationError, match="class-name mapping"):
        mr.validate_and_load_model(path)


def test_valid_model_loads(monkeypatch, tmp_path):
    path = tmp_path / "good.pt"
    path.write_bytes(b"stub")
    monkeypatch.setattr(
        "ultralytics.YOLO", lambda path: FakeModel()
    )
    model = mr.validate_and_load_model(path)
    assert model.task == "detect"


def test_class_mismatch_warns_and_match_is_silent(monkeypatch):
    mismatched = FakeModel(names={0: "cat", 1: "dog"})
    warnings = mr.check_class_compatibility(mismatched)
    assert warnings and "NOT the four crack classes" in warnings[0]
    assert mr.check_class_compatibility(FakeModel()) == []


def test_metadata_identifies_actual_model(tmp_path):
    path = tmp_path / "actual_model.pt"
    payload = b"deterministic payload for hash"
    path.write_bytes(payload)
    fake = FakeModel()
    meta = mr.collect_model_metadata(fake, path)
    assert meta["model_file"] == "actual_model.pt"
    assert meta["model_path"] == str(path)
    assert meta["model_task"] == "detect"
    assert meta["model_fingerprint_sha256"] == hashlib.sha256(payload).hexdigest()
    assert "alligator cracking" in meta["model_classes"]


def test_model_id_from_env(monkeypatch):
    assert mr.get_model_id() == ""
    monkeypatch.setenv("ROAD_CRACK_MODEL_ID", "v2-finetune-2026")
    assert mr.get_model_id() == "v2-finetune-2026"
