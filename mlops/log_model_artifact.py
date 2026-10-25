"""Log the existing YOLO weights to MLflow without retraining."""
from pathlib import Path
import os
import sys
import mlflow


def main():
    model_path = Path(sys.argv[1] if len(sys.argv) > 1 else "best.pt").resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. Pass the path to best.pt."
        )

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5005")
    )
    mlflow.set_experiment("Road Crack Detection")
    with mlflow.start_run(run_name="existing-yolo-weights-import"):
        mlflow.log_param("framework", "Ultralytics YOLO")
        mlflow.log_param("model_file", model_path.name)
        mlflow.log_param("artifact_only_import", True)
        mlflow.log_artifact(str(model_path), artifact_path="model_weights")
    print("Uploaded best.pt as an MLflow artifact.")
    print("Note: training metrics are not recreated by importing the weight file.")


if __name__ == "__main__":
    main()
