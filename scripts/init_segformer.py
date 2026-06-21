"""
Download SegFormer-B2 from HuggingFace, reinitialize the head for the 7
LoveDA classes, and save weights + config — no dataset download, no training.

Use this to bootstrap the backend/frontend dev loop before the full fine-tune
on LoveDA completes.  Predictions will be random (ImageNet-initialized head)
but the entire pipeline runs end-to-end.

Usage:
  python3 scripts/init_segformer.py
  # then: python3 scripts/generate_seg_masks.py
  # then: make serve-local
"""

import logging
import os

import torch
from transformers import SegformerForSemanticSegmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LOVEDA_ID2LABEL = {
    0: "Background", 1: "Building", 2: "Road",   3: "Water",
    4: "Barren",     5: "Forest",   6: "Agriculture",
}
LOVEDA_LABEL2ID = {v: k for k, v in LOVEDA_ID2LABEL.items()}

OUT_DIR    = "outputs"
WEIGHTS    = os.path.join(OUT_DIR, "segformer_b2_loveda.pth")
CONFIG_DIR = os.path.join(OUT_DIR, "segformer_config")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    log.info("Downloading SegFormer-B2 backbone from HuggingFace …")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=7,
        id2label=LOVEDA_ID2LABEL,
        label2id=LOVEDA_LABEL2ID,
        ignore_mismatched_sizes=True,  # reinitializes the classification head
    )
    torch.save(model.state_dict(), WEIGHTS)
    model.save_pretrained(CONFIG_DIR)
    log.info("Saved weights  → %s", WEIGHTS)
    log.info("Saved config   → %s", CONFIG_DIR)
    log.info(
        "Head is randomly initialized — predictions will be noise.\n"
        "Run 'make train-segformer' to fine-tune on LoveDA when ready."
    )


if __name__ == "__main__":
    main()
