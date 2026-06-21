#!/usr/bin/env bash
# Full pipeline: download LoveDA → train SegFormer → regenerate masks → restart backend
# Runs entirely in the background. Check progress: tail -f /tmp/train_pipeline.log

set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
LOG=/tmp/train_pipeline.log
exec >> "$LOG" 2>&1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [pipeline] $*"; }

cd "$PROJ"
source venv/bin/activate

# ── 1. Download LoveDA (resume if partial) ────────────────────────────────────
log "Downloading LoveDA Train.zip …"
wget -c -q --show-progress \
  "https://zenodo.org/record/5706578/files/Train.zip" \
  -O data/loveda/Train.zip

log "Downloading LoveDA Val.zip …"
wget -c -q --show-progress \
  "https://zenodo.org/record/5706578/files/Val.zip" \
  -O data/loveda/Val.zip

log "Downloads complete."

# ── 2. Train SegFormer-B2 on LoveDA ──────────────────────────────────────────
log "Starting SegFormer training (40 epochs) …"
MLFLOW_TRACKING_URI=./mlruns python3 scripts/train_segformer.py --epochs 40
log "Training complete."

# ── 3. Regenerate masks with trained weights ──────────────────────────────────
log "Regenerating Chicagoland masks …"
python3 scripts/generate_seg_masks.py
log "Masks regenerated."

# ── 4. Restart the backend ────────────────────────────────────────────────────
log "Restarting backend …"
pkill -f "uvicorn backend.main" || true
sleep 2
LOCAL_DATA_DIR=local_s3 \
  MODEL_PATH=outputs/resnet18_eurosat.pth \
  SEG_MODEL_PATH=outputs/segformer_b2_loveda.pth \
  nohup python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 \
  >> /tmp/backend.log 2>&1 &
log "Backend restarted. Pipeline done — real predictions are live."
