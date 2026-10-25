"""Centralized YOLO model configuration, loading, validation, and metadata.

This module is the ONLY place in the project that resolves a model path or
instantiates a YOLO model. To replace the trained model in the future:

    $env:ROAD_CRACK_MODEL_PATH = "C:\\path\\to\\new_best.pt"
    python -m streamlit run app.py

No source code needs to change. `best.pt` is never moved, renamed,
overwritten, or retrained.
"""

import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Environment variable that selects the model file. Unset/empty -> best.pt.
MODEL_PATH_ENV_VAR = "ROAD_CRACK_MODEL_PATH"

# Expected detection classes of the current road-crack pipeline. A
# replacement model that reports different class names still loads and
# predicts, but the app must show an explicit adaptation warning instead of
# silently treating its output as these four crack classes.
EXPECTED_CLASS_NAMES = frozenset(
    {
        "alligator cracking",
        "edge cracking",
        "longitudinal cracking",
        "transverse cracking",
    }
)

# The severity-assessment logic maps confidence (0-1) to Low/Moderate/Severe
# bands. That works for ANY detect-task model that returns per-box
# confidence in [0, 1]; only the class names above are semantic constraints.
EXPECTED_TASK = "detect"


class ModelValidationError(Exception):
    """Raised when a configured model cannot be safely activated."""


def resolve_model_path():
    """Resolve the configured model path.

    Precedence: ROAD_CRACK_MODEL_PATH env var -> <project root>/best.pt.
    Returns an absolute Path; existence is validated separately so callers
    can produce a precise error message.
    """
    configured = os.environ.get(MODEL_PATH_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "best.pt"


def sha256_fingerprint(path, chunk_size=1024 * 1024):
    """SHA-256 hex digest of a model file (streamed, 1 MiB chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_load_model(model_path):
    """Validate and load the configured model through a single function.

    Checks, in order:
      1. The file exists and is a .pt file.
      2. The file loads with Ultralytics YOLO (a corrupt or non-YOLO
         .pt file raises here).
      3. The loaded model exposes the standard inference interface
         (a `predict` method and a `names` dict of class ids -> names).
      4. The model task is `detect` (the pipeline expects bounding-box
         detection with per-box confidence).

    Raises ModelValidationError with a clear, actionable message on any
    failure. The caller's existing model configuration is untouched: this
    function never deletes, renames, or overwrites any file.
    """
    from ultralytics import YOLO  # imported lazily; heavy import

    path = Path(model_path)

    if not path.is_file():
        raise ModelValidationError(
            f"Model file not found: {path}\n"
            f"Set the {MODEL_PATH_ENV_VAR} environment variable to a valid "
            "YOLO .pt file, or unset it to use the default best.pt."
        )
    if path.suffix.lower() != ".pt":
        raise ModelValidationError(
            f"Unsupported model file type: {path.suffix!r}. "
            "This pipeline expects a trained Ultralytics YOLO .pt file."
        )

    try:
        model = YOLO(str(path))
    except Exception as exc:
        raise ModelValidationError(
            f"Could not load model {path.name}: {exc}\n"
            "The file exists but is not a loadable Ultralytics YOLO .pt "
            "model (it may be corrupt or from an incompatible framework). "
            "The previously working model configuration is unchanged."
        ) from exc

    if not hasattr(model, "predict") or not callable(model.predict):
        raise ModelValidationError(
            f"Model {path.name} has no callable predict() method and is "
            "incompatible with the inference pipeline."
        )
    names = getattr(model, "names", None)
    if not isinstance(names, dict) or not names:
        raise ModelValidationError(
            f"Model {path.name} exposes no class-name mapping "
            "(model.names is empty or missing), so predictions cannot be "
            "labeled. Incompatible with the pipeline."
        )

    task = getattr(model, "task", None)
    if task != EXPECTED_TASK:
        raise ModelValidationError(
            f"Model {path.name} has task type {task!r}, but this pipeline "
            f"requires {EXPECTED_TASK!r} (bounding-box detection with "
            "per-box confidence). A segmentation/classification/pose model "
            "cannot be activated here without adapting the prediction and "
            "annotation code first."
        )

    return model


def check_class_compatibility(model):
    """Compare a loaded model's class names with the pipeline's expectations.

    Returns a list of human-readable adaptation warnings (empty when the
    model matches the expected four crack classes). Warnings never block
    activation, but the UI must surface them so predictions from a model
    with different output semantics are not mistaken for crack classes.
    """
    warnings = []
    names = getattr(model, "names", {}) or {}
    actual = {str(v).strip().lower() for v in names.values()}
    missing = EXPECTED_CLASS_NAMES - actual
    if missing:
        warnings.append(
            "Model classes do not match the expected crack classes "
            f"({', '.join(sorted(missing))} missing). Its predictions are "
            "NOT the four crack classes; adapt severity mapping, reports, "
            "and dashboards before relying on them."
        )
    return warnings


def collect_model_metadata(model, model_path):
    """Metadata describing the ACTUAL loaded model for MLflow/OpenLineage.

    Includes filename, configured path, SHA-256 fingerprint, task type,
    class names, and an optional user-supplied identifier from
    ROAD_CRACK_MODEL_ID.
    """
    path = Path(model_path)
    names = getattr(model, "names", {}) or {}
    metadata = {
        "model_file": path.name,
        "model_path": str(path),
        "model_task": getattr(model, "task", "unknown"),
        "model_classes": sorted(str(v) for v in names.values()),
        "model_fingerprint_sha256": sha256_fingerprint(path),
        "model_id": os.environ.get("ROAD_CRACK_MODEL_ID", "").strip() or "",
        "ultralytics_task_supported": getattr(model, "task", None) == EXPECTED_TASK,
    }
    return metadata


def get_model_id():
    """User-supplied model identifier (e.g. a version tag), or empty string."""
    return os.environ.get("ROAD_CRACK_MODEL_ID", "").strip()
