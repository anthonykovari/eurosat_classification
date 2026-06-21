# CLAUDE.md — Project rules for Claude Code

## Engineering at enterprise scale

Operate as a real ML engineer on a production system, not a demo hacker.

- **Use Airflow for all pipeline orchestration** — trigger DAGs via CLI (`airflow dags trigger <dag_id>`), not the UI and not one-off Python scripts
- **Use the actual infrastructure** — LocalStack S3, Airflow scheduler, Kubernetes. Never bypass these with local file hacks when the real system is available
- **CLI over UI** — trigger, monitor, and debug via `airflow` CLI inside the scheduler container


## No external data shortcuts

Every pixel, polygon, and label must come from our own pipeline. Do not substitute external services to make the map look richer than the model actually produces.

**Forbidden:**
- ESRI World Imagery, OpenStreetMap tiles, or any third-party tile layer as a base map
- OpenStreetMap / Natural Earth GeoJSON overlays (rivers, roads, buildings, borders)
- Any external WMS/WFS/tile feed used as a stand-in for model output

**What is allowed:**
- Sentinel-2 RGB imagery served from the backend (`/satellite/{year}.png` ← `imagery/{year}/rgb.npy` in S3)
- ResNet-18 tile classifications (`/overlay/{year}.png` ← `classifications/{year}/grid.npy`)
- SegFormer-B2 pixel masks (`/seg/overlay/{year}.png` ← `seg/{year}/mask.npy`)

If the pipeline does not yet produce a layer, the layer does not appear in the UI. Build the pipeline first.
