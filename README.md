# eurosat-classification

Chicagoland land-use change tracker. Ingests Sentinel-2 L2A imagery via the Copernicus CDSE API, classifies it with a fine-tuned ResNet-18 (EuroSAT, 10 classes), and serves an interactive Leaflet map with a year slider and split-screen compare mode covering 2019–2026.

**AOI:** `(-88.4, 41.45, -87.5, 42.75)` — Chicagoland + Gary IN + Kenosha WI  
**Grid:** 4 cols × 6 rows = 24 tiles at 10 m/px, each ~1900×2380 px

---

## Architecture

![Architecture diagram](docs/architecture.png)

| Layer | Tech |
|-------|------|
| Ingestion | Apache Airflow · Copernicus CDSE Sentinel Hub Process API |
| Storage | MinIO (local) / AWS S3 (prod) — same boto3 client, `AWS_ENDPOINT_URL` only variable |
| Model serving | FastAPI · PyTorch · MLflow Model Registry |
| Frontend | Vanilla JS · Leaflet |
| Observability | Prometheus · Grafana |
| Orchestration | Kubernetes (Kustomize) · Docker Compose |
| CI/CD | GitHub Actions · AWS ECR · OIDC |

---

## Running the stack

Everything runs from the root `docker-compose.yml`:

```bash
docker compose up -d          # start full stack
docker compose up -d --build  # rebuild after Dockerfile changes
docker compose down           # stop (data persists in named volumes)
docker compose down -v        # stop + wipe all data
```

Do not use `etl/docker-compose.yml` directly — it is superseded by the root file.

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | — |
| Backend API | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | admin / admin |
| MinIO console | http://localhost:9011 | minioadmin / minioadmin |
| MinIO API | http://localhost:9010 | — |
| MLflow | http://localhost:5001 | — |
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | — |

### GPU setup

The scheduler and backend containers both request the GPU via `deploy.resources.reservations.devices`. They share it — no exclusive allocation. Requires `nvidia-container-toolkit` on the host:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -qq && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Verify GPU is visible inside the scheduler container:

```bash
docker exec eurosat_classification-airflow-scheduler-1 \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-container-toolkit` is not installed the containers still start — GPU is just unavailable.

---

## DAGs

Three DAGs, kept separate so a classification failure can never trigger a CDSE re-fetch.

| DAG | File | Cost |
|-----|------|------|
| `chicago_fetch_year` | `etl/dags/fetch_year_dag.py` | ~18 PU/tile |
| `chicago_classify_year` | `etl/dags/classify_year_dag.py` | zero — local GPU |
| `chicago_compute_sprawl` | `etl/dags/compute_sprawl_dag.py` | zero |

Trigger via CLI inside the scheduler container:

```bash
docker exec eurosat_classification-airflow-scheduler-1 \
  airflow dags trigger chicago_fetch_year --conf '{"year": 2023}'

docker exec eurosat_classification-airflow-scheduler-1 \
  airflow dags trigger chicago_classify_year --conf '{"year": 2023}'

docker exec eurosat_classification-airflow-scheduler-1 \
  airflow dags trigger chicago_compute_sprawl
```

### `chicago_fetch_year` conf options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `year` | int | required | Year to fetch |
| `max_tiles` | int | all | Limit tile count (useful for testing) |
| `patch_zeros` | bool | `true` | Re-fetch only bad tiles if mosaic exists |
| `zero_threshold` | float | `0.02` | Tiles with more zero pixels than this are re-fetched |
| `max_cloud_pct` | float | `0.10` | Per-tile SCL cloud fraction ceiling; triggers retry up to 5× |

**Patch mode:** if `imagery/{year}/rgb.npy` already exists in S3, only tiles exceeding `zero_threshold` are re-fetched and patched in-place. Costs only the bad tiles' PU budget.

### CDSE quirks

- Always use `mosaicking: "ORBIT"` with `maxcc=1.0` — `SIMPLE` mosaicking silently returns null samples on CDSE for every pixel.
- Single-scene approach: catalog-search per tile, pick least-cloudy scene, request a 1-day window. ~18 PU/tile vs 300–1700 PU for a full-season composite.
- Resolution is fixed at 10 m — do not change it.

---

## S3 layout

Bucket: `chicago-land-use`

```
imagery/{year}/rgb.npy              # stitched full-AOI mosaic (H×W×3 uint8)
imagery/{year}/tiles/r{r}_c{c}.npy # per-tile staging (deleted after stitch)
classifications/{year}/grid.npy     # (n_rows × n_cols) uint8 class indices
viz/{year}/satellite.png            # cached downsampled satellite PNG
viz/{year}/map.png                  # cached classification map PNG
catalog/latest.json                 # timeseries catalog
catalog/sprawl_stats.json           # per-class area + year-over-year deltas
```

---

## Backend API

Base URL: `http://localhost:8000`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness — returns device and status |
| GET | `/available` | Years in S3 with satellite/classification flags + class metadata |
| GET | `/satellite/{year}.png` | Downsampled Sentinel-2 RGB (~1000 px wide), cached in `viz/` |
| GET | `/overlay/{year}.png` | Classification grid as RGBA PNG, sized (n_cols × n_rows) |
| GET | `/grid/{year}` | Raw grid as JSON `{rows, cols, data: [flat uint8 list]}` |
| GET | `/map/{year}.png` | Colourised land-use map |
| GET | `/change/{year_a}/{year_b}.png` | Change-detection diff between two years, built on demand |
| GET | `/timeseries` | Full catalog from `catalog/latest.json` |
| GET | `/sprawl` | Sprawl stats from `catalog/sprawl_stats.json` |
| POST | `/predict/` | Classify a single image patch (multipart upload) |
| GET | `/metrics` | Prometheus metrics endpoint |

The backend loads the `resnet18-eurosat/Production` model from MLflow at startup, falling back to `outputs/resnet18_eurosat.pth` if the registry is unreachable.

---

## Model

**ResNet-18 fine-tuned on EuroSAT** — 10 land-use classes, 64×64 px tiles (640 m blocks).

Registered in MLflow as `resnet18-eurosat` v3 (Production stage). Weights at `outputs/resnet18_eurosat.pth`.

Normalization: `mean=[0.344, 0.380, 0.408]`, `std=[0.177, 0.150, 0.142]` (Sentinel-2 channel stats, not ImageNet).

| Class | Val accuracy |
|-------|-------------|
| SeaLake | 99.6% |
| Industrial | 99.4% |
| Forest | 99.3% |
| Highway | 99.2% |
| Residential | 99.0% |
| Pasture | 98.7% |
| River | 98.2% |
| HerbaceousVegetation | 98.0% |
| PermanentCrop | 97.4% |
| AnnualCrop | 97.3% |

![Validation confusion matrix](docs/validate_output_0_2.png)

To export to ONNX:

```bash
make export-onnx  # → outputs/resnet18_eurosat.onnx
```

---

## Observability

Custom Prometheus metrics exposed at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `landuse_predictions_total` | Counter | Prediction count by class label |
| `landuse_inference_seconds` | Histogram | ResNet-18 forward-pass duration (8 buckets) |
| `landuse_info` | Info | Model version and device loaded at startup |

All FastAPI endpoints are auto-instrumented (request count, latency) via `prometheus_fastapi_instrumentator`. Grafana dashboard auto-provisioned from `monitoring/grafana/dashboards/eurosat.json`.

---

## Kubernetes (minikube dev)

```bash
make k8s-minikube-start   # start cluster (Docker driver, 4 CPU / 4 GB)
make k8s-minikube-deploy  # build images, apply dev overlay
make k8s-minikube-seed    # sync local MinIO bucket → in-cluster MinIO
make k8s-minikube-url     # print service URLs
make k8s-minikube-stop    # pause (data survives in PVC)
```

Dev overlay uses MinIO in-cluster — same bucket name and key layout as prod S3, only `AWS_ENDPOINT_URL` differs. No code changes between environments.

---

## CI

GitHub Actions runs on every push:

- `ruff` lint
- Airflow DAG parse validation
- Docker builds (backend + frontend)
- `kustomize build` dry-run (dev + prod overlays)
- `terraform validate`

CD deploys to AWS EKS via OIDC (no long-lived keys), tags images with commit SHA, and runs `kubectl rollout` with automatic rollback on failure.
