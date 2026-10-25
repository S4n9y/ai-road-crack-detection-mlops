"""OpenLineage lineage events for road-crack inference runs.

Sends START/COMPLETE (or FAIL) events for each inference to a Marquez
backend using the OpenLineage v2 HTTP transport. Design constraints:

- Environment-based configuration (no hard-coded endpoints):
    OPENLINEAGE_URL       e.g. http://127.0.0.1:5000  (Marquez API base)
    OPENLINEAGE_NAMESPACE default "road_crack"
    OPENLINEAGE_TIMEOUT   seconds, default 3 (bounded, clamped 1-10)
- Bounded requests: timeout is clamped and the client's retry policy is
  reduced to a single attempt so a dead Marquez cannot stall inference.
- Graceful: if Marquez is unreachable, send_lineage_event returns
  (False, reason) and the prediction flow is never interrupted.

The workflow represented by each run:

    inputs:  uploaded-road-images       (the road image / input dataset)
    job:     road_crack_inference       (YOLO11 inference + severity)
    outputs: crack-detection-results    (detections + severity assessment)

The actual model artifact (file, path, SHA-256, classes, configured
ROAD_CRACK_MODEL_ID) travels as a run facet, and the MLflow tracking run
id, when available, is carried both in a run facet and on the parent job.
"""

import os

NAMESPACE_DEFAULT = "road_crack"
JOB_NAME = "road_crack_inference"
INPUT_DATASET = "uploaded-road-images"
OUTPUT_DATASET = "crack-detection-results"
TIMEOUT_DEFAULT = 3
PRODUCER = "https://github.com/road-crack/road_crack_mlops_starter"


def get_lineage_config():
    """Resolve OpenLineage/Marquez settings from the environment.

    Returns (url, namespace, timeout) or (None, namespace, timeout) when
    lineage is not configured.
    """
    url = (os.environ.get("OPENLINEAGE_URL") or "").strip()
    namespace = (os.environ.get("OPENLINEAGE_NAMESPACE") or NAMESPACE_DEFAULT).strip()
    try:
        timeout = float(os.environ.get("OPENLINEAGE_TIMEOUT", TIMEOUT_DEFAULT))
    except ValueError:
        timeout = TIMEOUT_DEFAULT
    timeout = max(1.0, min(timeout, 10.0))
    return url or None, namespace, timeout


def _model_facet(model_info):
    """RunFacet carrying the actual model identity (file, path, hash, classes)."""
    from attrs import define, field
    from openlineage.client.facet_v2 import RunFacet

    @define
    class ModelRunFacet(RunFacet):
        modelFile: str = field()
        modelPath: str = field()
        modelTask: str = field()
        modelClasses: list = field()
        sha256: str = field()
        modelId: str = field(default="")

    return ModelRunFacet(
        modelFile=model_info.get("model_file", ""),
        modelPath=model_info.get("model_path", ""),
        modelTask=model_info.get("model_task", ""),
        modelClasses=list(model_info.get("model_classes", [])),
        sha256=model_info.get("model_fingerprint_sha256", ""),
        modelId=model_info.get("model_id", ""),
    )


def _run_facets(model_info, mlflow_run_id, image_name):
    """Run facets: model identity, MLflow correlation, and input context."""
    facets = {}
    if model_info:
        facets["model"] = _model_facet(model_info)
    if mlflow_run_id:
        from attrs import define, field
        from openlineage.client.facet_v2 import RunFacet

        @define
        class MlflowRunFacet(RunFacet):
            mlflowRunId: str = field()
            mlflowTrackingUri: str = field(default="")

        facets["mlflow"] = MlflowRunFacet(
            mlflowRunId=str(mlflow_run_id),
            mlflowTrackingUri=os.environ.get("MLFLOW_TRACKING_URI", ""),
        )
    if image_name:
        from attrs import define, field
        from openlineage.client.facet_v2 import RunFacet

        @define
        class InputContextFacet(RunFacet):
            imageName: str = field()

        facets["input_context"] = InputContextFacet(imageName=str(image_name))
    return facets


def _job_facets(mlflow_run_id):
    """Job-level documentation facet describing the inference pipeline."""
    from attrs import define, field
    from openlineage.client.facet_v2 import JobFacet

    @define
    class InferenceJobFacet(JobFacet):
        pipeline: str = field()
        modelSelection: str = field()
        mlflowRunId: str = field(default="")

    return InferenceJobFacet(
        pipeline=(
            "YOLO11 road-crack inference with automatic confidence threshold "
            "selection and confidence-based severity labels"
        ),
        modelSelection="ROAD_CRACK_MODEL_PATH env var, default best.pt",
        mlflowRunId=str(mlflow_run_id) if mlflow_run_id else "",
    )


def _input_dataset():
    """Dataset facet for the road image input."""
    from attrs import define, field
    from openlineage.client.facet_v2 import DatasetFacet
    from openlineage.client.event_v2 import Dataset

    @define
    class ImageDatasetFacet(DatasetFacet):
        mediaType: str = field(default="image/*")
        description: str = field(default="")

    return Dataset(
        namespace=namespace_from_config(),
        name=INPUT_DATASET,
        facets={"image": ImageDatasetFacet(mediaType="image/*")},
    )


def _output_dataset(detection_count, severity_counts):
    """Dataset facet describing the detection + severity output."""
    from attrs import define, field
    from openlineage.client.facet_v2 import DatasetFacet
    from openlineage.client.event_v2 import Dataset

    @define
    class DetectionOutputFacet(DatasetFacet):
        detectionCount: int = field()
        severityCounts: dict = field()

    return Dataset(
        namespace=namespace_from_config(),
        name=OUTPUT_DATASET,
        facets={
            "detection_output": DetectionOutputFacet(
                detectionCount=int(detection_count),
                severityCounts={
                    str(k): int(v) for k, v in (severity_counts or {}).items()
                },
            )
        },
    )


def namespace_from_config():
    """Current namespace (kept separate so build_run_event stays pure)."""
    _, namespace, _ = get_lineage_config()
    return namespace


def build_run_event(
    run_id,
    event_type,
    model_info=None,
    mlflow_run_id=None,
    image_name=None,
    detection_count=0,
    severity_counts=None,
):
    """Build (but do not emit) a RunEvent for the inference workflow.

    Pure construction so tests can inspect the event without any network.
    """
    from datetime import datetime, timezone

    from openlineage.client.event_v2 import Job, Run, RunEvent

    if event_type not in ("START", "COMPLETE", "FAIL"):
        raise ValueError(f"Invalid event type: {event_type}")

    inputs = []
    outputs = []
    if event_type in ("COMPLETE", "FAIL"):
        inputs = [_input_dataset()]
    if event_type == "COMPLETE":
        outputs = [_output_dataset(detection_count, severity_counts)]

    return RunEvent(
        eventType=event_type,
        eventTime=datetime.now(timezone.utc).isoformat(),
        run=Run(
            runId=str(run_id),
            facets=_run_facets(model_info, mlflow_run_id, image_name),
        ),
        job=Job(
            namespace=namespace_from_config(),
            name=JOB_NAME,
            facets=_job_facets(mlflow_run_id),
        ),
        producer=PRODUCER,
        inputs=inputs,
        outputs=outputs,
    )


def fetch_lineage_run(run_id):
    """Retrieve a lineage run from Marquez's API (read-only verification).

    Matches on exact run id; returns (ok, payload_or_message).
    """
    import json
    import urllib.request

    url, namespace, timeout = get_lineage_config()
    if not url:
        return False, "OpenLineage not configured."
    try:
        api = url.rstrip("/")
        endpoint = f"{api}/api/v1/namespaces/{namespace}/jobs/{JOB_NAME}/runs"
        req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=max(3.0, timeout)) as resp:
            body = json.loads(resp.read().decode())
        runs = body.get("runs", [])
        match = next(
            (r for r in runs if str(r.get("id", "")).lower() == str(run_id).lower()),
            None,
        )
        if match:
            return True, match
        return False, f"Run {run_id} not found among {len(runs)} recorded runs."
    except Exception as exc:
        return False, f"Could not query Marquez: {exc}"


def fetch_lineage_job():
    """Fetch the job definition (with facets) from Marquez's API."""
    import json
    import urllib.request

    url, namespace, timeout = get_lineage_config()
    if not url:
        return False, "OpenLineage not configured."
    try:
        api = url.rstrip("/")
        endpoint = f"{api}/api/v1/namespaces/{namespace}/jobs/{JOB_NAME}"
        req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=max(3.0, timeout)) as resp:
            body = json.loads(resp.read().decode())
        return True, body
    except Exception as exc:
        return False, f"Could not query Marquez: {exc}"


def send_lineage_event(
    run_id,
    event_type,
    model_info=None,
    mlflow_run_id=None,
    image_name=None,
    detection_count=0,
    severity_counts=None,
):
    """Send one OpenLineage RUN event to Marquez.

    event_type: "START", "COMPLETE", or "FAIL".
    Returns (ok, message). Never raises.
    """
    url, namespace, timeout = get_lineage_config()
    if not url:
        return False, "OpenLineage not configured (set OPENLINEAGE_URL)."

    try:
        from openlineage.client import OpenLineageClient
        from openlineage.client.transport import HttpConfig, HttpTransport

        event = build_run_event(
            run_id,
            event_type,
            model_info=model_info,
            mlflow_run_id=mlflow_run_id,
            image_name=image_name,
            detection_count=detection_count,
            severity_counts=severity_counts,
        )

        http_config = HttpConfig(
            url=url,
            timeout=timeout,
            retry={"total": 1, "read": 1, "connect": 1, "backoff_factor": 0.1},
        )
        client = OpenLineageClient(transport=HttpTransport(http_config))
        client.emit(event)
        return True, f"OpenLineage {event_type} sent to {url}."
    except Exception as exc:
        # Bounded and non-fatal: Marquez being down must not break inference.
        return False, f"Marquez/OpenLineage unavailable: {exc}"


def fetch_lineage_run_fixed(run_id):
    """Deprecated alias kept for compatibility; use fetch_lineage_run."""
    return fetch_lineage_run(run_id)
