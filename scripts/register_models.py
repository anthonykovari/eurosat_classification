"""
Register trained model weights into the MLflow Model Registry and promote to Production.

Run once after training (or whenever weights are updated):
    python3 scripts/register_models.py

Connects to MLflow at http://localhost:5001 (the host-side port from docker-compose).
"""

import os
import sys

import mlflow
import mlflow.pytorch
import torch
import torchvision.models as tvm

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_URI)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = mlflow.tracking.MlflowClient()


def _promote(name: str, version: str) -> None:
    client.transition_model_version_stage(name=name, version=version, stage="Production",
                                          archive_existing_versions=True)
    print(f"  promoted {name} v{version} → Production")


def register_resnet18() -> None:
    weights_path = os.path.join(PROJECT_ROOT, "outputs", "resnet18_eurosat.pth")
    if not os.path.isfile(weights_path):
        print(f"resnet18_eurosat.pth not found at {weights_path} — skipping")
        return

    model = tvm.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    mlflow.set_experiment("model-registration")
    with mlflow.start_run(run_name="resnet18-eurosat"):
        mlflow.log_params({
            "architecture":  "resnet18",
            "dataset":       "EuroSAT",
            "n_classes":     10,
            "val_accuracy":  0.986,
            "tile_size_px":  64,
            "tile_size_m":   640,
            "normalization": "sentinel2-chicagoland",
            "mean":          "[0.344, 0.380, 0.408]",
            "std":           "[0.177, 0.150, 0.142]",
            "aoi":           "chicagoland",
        })
        result = mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="resnet18-eurosat",
        )
        version = result.registered_model_version
        print(f"  logged resnet18-eurosat v{version}")

    _promote("resnet18-eurosat", version)


def register_segformer() -> None:
    config_path = os.path.join(PROJECT_ROOT, "outputs", "segformer_config")
    weights_path = os.path.join(PROJECT_ROOT, "outputs", "segformer_b2_loveda.pth")
    if not os.path.isdir(config_path):
        print(f"segformer_config dir not found at {config_path} — skipping")
        return

    from transformers import SegformerForSemanticSegmentation
    seg_model = SegformerForSemanticSegmentation.from_pretrained(config_path)
    if os.path.isfile(weights_path):
        seg_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    seg_model.eval()

    mlflow.set_experiment("model-registration")
    with mlflow.start_run(run_name="segformer-b2-loveda"):
        mlflow.log_params({
            "architecture":  "segformer-b2",
            "dataset":       "LoveDA",
            "n_classes":     7,
            "resolution_m":  10,
            "aoi":           "chicagoland",
        })
        result = mlflow.pytorch.log_model(
            seg_model,
            artifact_path="model",
            registered_model_name="segformer-b2-loveda",
        )
        version = result.registered_model_version
        print(f"  logged segformer-b2-loveda v{version}")

    _promote("segformer-b2-loveda", version)


if __name__ == "__main__":
    print(f"Connecting to MLflow at {MLFLOW_URI}")
    try:
        mlflow.search_experiments()
    except Exception as e:
        print(f"Cannot reach MLflow: {e}")
        sys.exit(1)

    print("Registering ResNet-18...")
    register_resnet18()

    print("Registering SegFormer-B2...")
    register_segformer()

    print("Done. View registry at http://localhost:5001/#/models")
