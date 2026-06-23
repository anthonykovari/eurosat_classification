# CLAUDE.md — Project rules for Claude Code

## Engineering at enterprise scale

Operate as a real ML engineer on a production system, not a demo hacker.

- **Use Airflow for all pipeline orchestration** — trigger DAGs via CLI (`airflow dags trigger <dag_id>`), not the UI and not one-off Python scripts
- **Use the actual infrastructure** — MinIO S3, Airflow scheduler, Kubernetes. Never bypass these with local file hacks when the real system is available
- **CLI over UI** — trigger, monitor, and debug via `airflow` CLI inside the scheduler container
- **Never run inference on CPU if GPU is available** — always use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`


## No external data shortcuts

Every pixel, polygon, and label must come from our own pipeline. Do not substitute external services to make the map look richer than the model actually produces.

**Forbidden:**
- ESRI World Imagery, OpenStreetMap tiles, or any third-party tile layer as a base map
- OpenStreetMap / Natural Earth GeoJSON overlays (rivers, roads, buildings, borders)
- Any external WMS/WFS/tile feed used as a stand-in for model output
- Interpolating / copying pixels to fill NoData gaps — that is fabricating satellite data

**What is allowed:**
- Sentinel-2 RGB imagery served from the backend (`/satellite/{year}.png` ← `imagery/{year}/rgb.npy` in S3)
- ResNet-18 tile classifications (`/overlay/{year}.png` ← `classifications/{year}/grid.npy`)
- SegFormer-B2 pixel masks (`/seg/overlay/{year}.png` ← `seg/{year}/mask.npy`)

If the pipeline does not yet produce a layer, the layer does not appear in the UI. Build the pipeline first.


## Stack — how to start and stop

Everything runs from one docker-compose.yml at the project root:

```bash
docker compose up -d        # start full stack
docker compose up -d --build  # after Dockerfile changes
docker compose down         # stop everything
```

**Do not** use `etl/docker-compose.yml` directly — it is superseded by the root compose file.

Services and ports:
- Airflow UI: `localhost:8080` (admin/admin)
- Backend API: `localhost:8000`
- Frontend: `localhost:3000`
- MinIO S3: `localhost:9010` (API) / `localhost:9011` (console, minioadmin/minioadmin)
- MLflow: `localhost:5001`
- Grafana: `localhost:3001` (admin/admin)
- Prometheus: `localhost:9090`


## GPU setup

The RTX 3060 Ti is the only GPU. Both `airflow-scheduler` and `backend` containers request it via `deploy.resources.reservations.devices` in docker-compose.yml. They share it — no exclusive allocation.

**Prerequisite — nvidia-container-toolkit must be installed on the host:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -qq && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

If `nvidia-container-toolkit` is not yet installed, the containers still start but GPU is unavailable inside them. Check with:
```bash
docker exec eurosat_classification-airflow-scheduler-1 python3 -c "import torch; print(torch.cuda.is_available())"
```


## CDSE / Sentinel Hub quirks

- **SIMPLE mosaicking returns null samples on CDSE** — `samples[0]` is null for every pixel even when the catalog has valid scenes. Always use `mosaicking: "ORBIT"` with `maxcc=1.0`.
- **Single-scene approach**: catalog-search per tile first, pick least-cloudy scene, request just that 1-day window. Costs ~18 PU/tile vs 300–1700 PU for a full-season composite.
- **Swath edge zeros**: after fetching, check zero coverage. If a tile has >2% zeros, retry with the next-best scene (up to 3 attempts). This is already implemented in `fetch_year_dag.py`.
- **Resolution must never be changed** — always 10m. Do not suggest or implement resolution reduction.


## DAG architecture

Two separate DAGs — kept separate intentionally so a classify failure can never trigger a CDSE re-fetch:

| DAG | File | Trigger | Cost |
|-----|------|---------|------|
| `chicago_fetch_year` | `etl/dags/fetch_year_dag.py` | `{"year": 2019}` | ~18 PU/tile |
| `chicago_classify_year` | `etl/dags/classify_year_dag.py` | `{"year": 2019}` | zero — local GPU |

**`chicago_fetch_year` conf options:**
- `year` — required
- `max_tiles` — int, limit for testing (e.g. 2)
- `patch_zeros` — bool (default `true`): if mosaic exists, scan for bad tiles and re-fetch only those
- `zero_threshold` — float (default `0.02`): tiles with more zeros than this are re-fetched in patch mode
- `max_cloud_pct` — float (default `0.10`): per-tile SCL cloud fraction ceiling; scenes above this trigger retry (up to 5 attempts)

**Patch mode** (mosaic already in S3): re-fetches only tiles with too many black pixels, patches them in-place, re-uploads. Costs only the bad tiles' PU budget, not the full grid.

**`chicago_classify_year`**: reads `imagery/{year}/rgb.npy`, runs ResNet-18 in 64×64 windows on GPU, writes `classifications/{year}/grid.npy`. Safe to retry freely.


## Models

| Model | Weights | Classes | Status |
|-------|---------|---------|--------|
| ResNet-18 (EuroSAT) | `outputs/resnet18_eurosat.pth` | 10 land-use classes | Trained, 98.6% val acc — registered in MLflow as `resnet18-eurosat` v3 Production |
| SegFormer-B2 (LoveDA) | `outputs/segformer_b2_loveda.pth` | 7 land-cover classes | Trained, not yet run on Chicagoland |

ResNet-18 classifies 64×64 px tiles (640m blocks). SegFormer produces per-pixel masks at full 10m resolution. Both are loaded by the backend at startup.


## Current pipeline state (as of 2026-06-22)

- `imagery/2019/rgb.npy` — real Sentinel-2 data, full 24-tile grid (14268×7486), ~95%+ nonzero
- `classifications/2019/grid.npy` — real ResNet-18 output (222×116, 27.8s on RTX 3060 Ti)
- All other years — not yet fetched
- SegFormer inference — not yet run on any year

**Next steps:**
1. Install nvidia-container-toolkit (requires sudo — user must do this)
2. `docker compose up -d --build` from project root to rebuild with CUDA torch
3. Trigger `chicago_classify_year --conf '{"year": 2019}'` to replace dummy grid with real predictions
4. Fetch additional years: `chicago_fetch_year --conf '{"year": 2020}'` etc.


## S3 layout (LocalStack bucket: `chicago-land-use`)

```
imagery/{year}/rgb.npy              # stitched full-AOI mosaic (H×W×3 uint8)
imagery/{year}/tiles/r{r}_c{c}.npy # staging tiles (deleted after stitch)
classifications/{year}/grid.npy     # (n_rows × n_cols) uint8 class indices
seg/{year}/mask.npy                 # (H × W) uint8 SegFormer class indices
viz/{year}/satellite.png            # cached downsampled satellite PNG
viz/{year}/map.png                  # cached classification map PNG
catalog/latest.json                 # timeseries catalog for /timeseries endpoint
```

AOI: `(-88.4, 41.45, -87.5, 42.75)` — Chicagoland + Gary IN + Kenosha WI  
Grid: 4 cols × 6 rows = 24 tiles at 10m, each ~1900×2380 px
