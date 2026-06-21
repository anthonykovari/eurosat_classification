"""
Fine-tune SegFormer-B2 on the LoveDA dataset for per-pixel land cover segmentation.

LoveDA classes: 0=Background, 1=Building, 2=Road, 3=Water,
                4=Barren, 5=Forest, 6=Agriculture  (255=ignore)

Usage:
  python3 scripts/train_segformer.py                        # full 40-epoch run
  python3 scripts/train_segformer.py --epochs 5             # smoke test
  MLFLOW_TRACKING_URI=http://localhost:5001 python3 ...     # log to local MLflow
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import urllib.request
import zipfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
LOVEDA_URL_TRAIN = "https://zenodo.org/record/5706578/files/Train.zip"
LOVEDA_URL_VAL   = "https://zenodo.org/record/5706578/files/Val.zip"

LOVEDA_ID2LABEL = {
    0: "Background", 1: "Building", 2: "Road",   3: "Water",
    4: "Barren",     5: "Forest",   6: "Agriculture",
}
LOVEDA_LABEL2ID = {v: k for k, v in LOVEDA_ID2LABEL.items()}
NUM_LABELS   = 7
IGNORE_INDEX = 255

# ImageNet stats — matches the mit-b2 backbone pre-training
SEGFORMER_MEAN = [0.485, 0.456, 0.406]
SEGFORMER_STD  = [0.229, 0.224, 0.225]


# ── download ──────────────────────────────────────────────────────────────────
def _report(block, block_size, total):
    done = block * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        print(f"\r  {pct:3d}%  {done // 1_000_000} / {total // 1_000_000} MB", end="", flush=True)


def download_loveda(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for url, split in [(LOVEDA_URL_TRAIN, "Train"), (LOVEDA_URL_VAL, "Val")]:
        if any((root / split).rglob("*.png")):
            log.info("LoveDA %s already present — skipping", split)
            continue
        zip_path = root / f"{split}.zip"
        log.info("Downloading LoveDA %s …", split)
        urllib.request.urlretrieve(url, zip_path, _report)
        print()
        log.info("Extracting …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        zip_path.unlink()
        log.info("LoveDA %s ready", split)


# ── dataset ───────────────────────────────────────────────────────────────────
class LoveDADataset(Dataset):
    def __init__(self, root_dir: Path, crop_size: int = 512, augment: bool = True):
        self.crop_size = crop_size
        self.augment   = augment
        self.normalize = transforms.Normalize(SEGFORMER_MEAN, SEGFORMER_STD)
        self.to_tensor = transforms.ToTensor()

        self.image_paths: list[Path] = []
        self.mask_paths:  list[Path] = []
        for img in sorted(root_dir.rglob("images_png/*.png")):
            mask = Path(str(img).replace("images_png", "masks_png"))
            if mask.exists():
                self.image_paths.append(img)
                self.mask_paths.append(mask)

        if not self.image_paths:
            raise RuntimeError(f"No images found under {root_dir}")
        log.info("LoveDADataset: %d images from %s", len(self.image_paths), root_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx])

        # Synchronized random crop — both image and mask get identical coordinates
        i, j, h, w = transforms.RandomCrop.get_params(image, (self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        mask  = TF.crop(mask,  i, j, h, w)

        if self.augment:
            if random.random() > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() > 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)

        image = self.normalize(self.to_tensor(image))
        mask  = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask


# ── loss ──────────────────────────────────────────────────────────────────────
def seg_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: (B, C, H/4, W/4) — SegFormer outputs quarter-resolution
    up = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return F.cross_entropy(up, labels, ignore_index=IGNORE_INDEX)


# ── training loop ─────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_dir)
    download_loveda(data_root)

    train_ds = LoveDADataset(data_root / "Train", crop_size=args.crop_size, augment=True)
    val_ds   = LoveDADataset(data_root / "Val",   crop_size=args.crop_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Start from ADE20K-finetuned weights; only reinitialize the final 7-class head
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_LABELS,
        id2label=LOVEDA_ID2LABEL,
        label2id=LOVEDA_LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.model_dir, exist_ok=True)
    best_miou = 0.0

    import evaluate
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("segformer-b2-loveda")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "grad_accum": args.grad_accum,
            "crop_size": args.crop_size, "seed": args.seed,
            "backbone": "nvidia/segformer-b2-finetuned-ade-512-512",
            "dataset": "LoveDA", "num_labels": NUM_LABELS,
        })

        for epoch in range(args.epochs):
            # ── train ─────────────────────────────────────────────────────────
            model.train()
            optimizer.zero_grad()
            train_loss, n_steps = 0.0, 0

            for step, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    loss = seg_loss(model(pixel_values=images).logits, masks) / args.grad_accum
                scaler.scale(loss).backward()

                if (step + 1) % args.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * args.grad_accum
                n_steps += 1
                if step % 50 == 0:
                    log.info("Epoch %d/%d  step %d/%d  loss=%.4f",
                             epoch + 1, args.epochs, step, len(train_loader),
                             loss.item() * args.grad_accum)

            train_loss /= n_steps

            # ── validate ──────────────────────────────────────────────────────
            model.eval()
            val_loss, n_val = 0.0, 0
            iou_metric = evaluate.load("mean_iou")

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        logits = model(pixel_values=images).logits
                        val_loss += seg_loss(logits, masks).item()
                    n_val += 1

                    up   = F.interpolate(logits, size=masks.shape[-2:],
                                         mode="bilinear", align_corners=False)
                    pred = up.argmax(1).cpu().numpy()
                    true = masks.cpu().numpy()
                    valid = true != IGNORE_INDEX
                    iou_metric.add_batch(
                        predictions=pred[valid].tolist(),
                        references=true[valid].tolist(),
                    )

            val_loss /= n_val
            mean_iou = iou_metric.compute(
                num_labels=NUM_LABELS, ignore_index=IGNORE_INDEX, reduce_labels=False
            )["mean_iou"]
            scheduler.step()

            log.info("Epoch %d/%d  train=%.4f  val=%.4f  mIoU=%.4f",
                     epoch + 1, args.epochs, train_loss, val_loss, mean_iou)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "mean_iou": mean_iou},
                step=epoch + 1,
            )

            if mean_iou > best_miou:
                best_miou = mean_iou
                torch.save(model.state_dict(),
                           os.path.join(args.model_dir, "segformer_b2_loveda.pth"))
                # Save config to disk so inference never needs Hub access
                model.save_pretrained(os.path.join(args.model_dir, "segformer_config"))
                log.info("  → best mIoU=%.4f saved", best_miou)

        mlflow.log_metric("best_mean_iou", best_miou)
        mlflow.pytorch.log_model(model, "model", registered_model_name="segformer-b2-loveda")

    log.info("Done. Best mIoU=%.4f  weights: %s/segformer_b2_loveda.pth",
             best_miou, args.model_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SegFormer-B2 on LoveDA")
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch-size", type=int,   default=4,    dest="batch_size")
    p.add_argument("--lr",         type=float, default=6e-5)
    p.add_argument("--grad-accum", type=int,   default=4,    dest="grad_accum")
    p.add_argument("--crop-size",  type=int,   default=512,  dest="crop_size")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--data-dir",   type=str,   default="data/loveda", dest="data_dir")
    p.add_argument("--model-dir",  type=str,   default="outputs",     dest="model_dir")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
