"""Export ResNet-18 EuroSAT weights to ONNX for edge/runtime deployment."""

import argparse
from pathlib import Path

import torch
from torchvision import models

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Sentinel-2 channel stats used at train and serve time
MEAN = [0.344, 0.380, 0.408]
STD  = [0.177, 0.150, 0.142]


def load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def export(weights_path: Path, output_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(weights_path, device)

    dummy = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported → {output_path}")
    print(f"Input:  image  [batch, 3, 224, 224]  float32")
    print(f"        Preprocess: CLAHE → resize 224×224 → normalize(mean={MEAN}, std={STD})")
    print(f"Output: logits [batch, 10]  — argmax → class index")
    print(f"Classes: {', '.join(CLASS_NAMES)}")

    # Verify round-trip with onnxruntime if available
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"image": dummy.cpu().numpy()})[0]

        with torch.no_grad():
            pt_out = model(dummy).cpu().numpy()

        max_diff = float(abs(ort_out - pt_out).max())
        print(f"Verified: max logit diff PyTorch vs ONNX Runtime = {max_diff:.2e}")
    except ImportError:
        print("onnxruntime not installed — skipping verification (pip install onnxruntime)")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(project_root / "outputs" / "resnet18_eurosat.pth"))
    parser.add_argument("--output", default=str(project_root / "outputs" / "resnet18_eurosat.onnx"))
    args = parser.parse_args()
    export(Path(args.weights), Path(args.output))


if __name__ == "__main__":
    main()
