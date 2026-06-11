"""
EuroSAT ResNet-18 fine-tuning script.

Runs locally or as a SageMaker Training Job without code changes.
SageMaker passes hyperparameters as CLI args and injects SM_* env vars
for data/model paths. Metrics and the final model are logged to MLflow;
the model is registered in the MLflow Model Registry on completion.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import EuroSAT, ImageFolder


CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
DEFAULT_MEAN = [0.344, 0.380, 0.408]
DEFAULT_STD = [0.177, 0.150, 0.142]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Hyperparameters — SageMaker forwards these as CLI args
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler-step", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # SageMaker injects these; fall back to local paths for dev
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "outputs"))
    parser.add_argument("--training", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", ""))
    parser.add_argument("--manifests", type=str, default=os.environ.get("SM_CHANNEL_MANIFESTS", ""))

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_norm_stats(manifests_dir: str) -> tuple[list, list]:
    """Use ETL-computed per-channel stats when available, else fall back to defaults."""
    if manifests_dir:
        stats_path = Path(manifests_dir) / "normalization_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            return stats["mean_rgb"], stats["std_rgb"]
    return DEFAULT_MEAN, DEFAULT_STD


def build_dataloaders(
    training_dir: str, seed: int, batch_size: int, mean: list, std: list
) -> tuple[DataLoader, DataLoader, dict]:
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if training_dir and Path(training_dir).exists():
        # SageMaker: data pre-downloaded from S3 channel into training_dir
        train_ds = ImageFolder(training_dir, transform=train_tf)
        val_ds = ImageFolder(training_dir, transform=val_tf)
    else:
        # Local dev: torchvision auto-download
        local_root = Path(__file__).resolve().parent.parent / "data"
        train_ds = EuroSAT(root=str(local_root), download=True, transform=train_tf)
        val_ds = EuroSAT(root=str(local_root), download=False, transform=val_tf)

    n = len(train_ds)
    train_n = int(0.8 * n)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_idx, val_idx = indices[:train_n], indices[train_n:]

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    # Weighted sampler for class imbalance
    train_targets = [train_ds.targets[i] for i in train_idx]
    class_counts = Counter(train_targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, class_weights


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "eurosat-resnet18"))

    mean, std = load_norm_stats(args.manifests)
    train_loader, val_loader, class_weights = build_dataloaders(
        args.training, args.seed, args.batch_size, mean, std
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    weights_tensor = torch.tensor([class_weights[i] for i in range(10)], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "scheduler_step": args.scheduler_step,
            "scheduler_gamma": args.scheduler_gamma,
            "seed": args.seed,
            "architecture": "resnet18",
            "pretrained": "imagenet1k_v1",
            "norm_mean": mean,
            "norm_std": std,
            "device": str(device),
        })

        best_val_acc = 0.0

        for epoch in range(args.epochs):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            run_loss, run_correct, total = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                run_loss += loss.item() * inputs.size(0)
                run_correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_loss = run_loss / total
            train_acc = run_correct / total

            # ── Validate ───────────────────────────────────────────────────
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total
            best_val_acc = max(best_val_acc, val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch + 1)

            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

            scheduler.step()

        model_path = os.path.join(args.model_dir, "resnet18_eurosat.pth")
        torch.save(model.state_dict(), model_path)

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="eurosat-resnet18",
        )
        print(f"Saved to {model_path} | Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
