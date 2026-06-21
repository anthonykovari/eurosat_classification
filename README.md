# Chicagoland Urban Land Use Intelligence Platform

Production ML system I designed and built end-to-end: Sentinel-2 satellite ingestion → dual-architecture computer vision → Kubernetes-native serving on AWS EKS → full Prometheus/Grafana observability. Built to demonstrate the same lifecycle ownership and engineering judgment I'd bring to a senior ML role — architecture decisions, MLOps discipline, fault-tolerant production design, and infrastructure a team could actually operate.

**Live demo:** Chicagoland land-use change, 2019–2024 · two model architectures · interactive Leaflet map with timeline + compare mode

---

## Why two models — an architecture decision, not a default

Most portfolio projects ship one model. I deliberately ran two, because the throughput-vs-fidelity trade-off between them is exactly what a lead engineer navigates on a real team:

- **ResNet-18** — fast, coarse tile classification (10 EuroSAT classes, ~3 min/year on CPU). Drives the time-series chart and change-detection catalog.
- **SegFormer-B2** — slow, precise pixel segmentation (7 LoveDA classes, full Sentinel-2 10 m/px resolution, sliding-window inference with overlap-averaging to eliminate seam artifacts at tile boundaries).

Same FastAPI backend, same Kubernetes serving infrastructure, single UI toggle. The system is architected to support both without forking the pipeline — because in production you rarely get to pick just one.

---

## Owning the full lifecycle

**Sentinel-2 ingestion via Copernicus CDSE API** (Apache Airflow-orchestrated, yearly schedule, ORBIT cloud-free median compositing, multi-tile AOI stitching around the 2,500 px API limit) → **training on AWS SageMaker or local GPU** (PyTorch, AMP, MLflow experiment tracking) → **MLflow Model Registry** (manual promotion gate, S3-versioned artifacts) → **FastAPI on AWS EKS** (Dockerized, autoscaled via HPA) → **Prometheus + Grafana monitoring** (custom metrics, auto-provisioned dashboards) → **all infrastructure as Terraform** (VPC, EKS, ECR, S3, IAM least-privilege).

Each stage fails independently and is independently testable — see fault tolerance below.

---

## MLOps practices implemented

- **CI (GitHub Actions):** ruff lint, Airflow DAG parse validation, Docker builds for backend + frontend, `kustomize build` dry-run for both dev and prod overlays, `terraform validate` — all before merge.
- **CD:** OIDC-authenticated deploys to AWS (no long-lived keys stored in GitHub), commit-SHA-tagged ECR images, `kubectl rollout` with timeout and automatic rollback on failure.
- **Model versioning:** S3-versioned artifacts + MLflow Model Registry, decoupled from application deploys via a Kustomize ConfigMap the CD pipeline patches post-promotion.
- **Experiment tracking:** every run — local RTX 3060 Ti or SageMaker `ml.p3.2xlarge` — logs params, per-epoch metrics, and artifacts to the same MLflow server. Comparable by design, not by convention.
- **Train/serve parity:** identical CLAHE preprocessing and domain-specific normalization (`[0.344, 0.380, 0.408]` Sentinel-2 channel stats, not ImageNet) applied in the Airflow ETL and the live `/predict/` endpoint. Eliminates the quiet accuracy degradation that hits real deployments when preprocessing diverges between training and serving.

---

## Designing for failure, not just success

- Pod readiness probes hit `/health` — a pod that fails to load either model is never added to the EKS load balancer pool. No silent degradation.
- SegFormer is optional at boot: ResNet-18 endpoints stay live even if segmentation weights are absent; `/seg/*` returns a clear 503 instead of crashing the service.
- S3 errors are classified — `NoSuchKey` (404) surfaces to the caller cleanly; transient 5xx errors propagate for upstream retry. No blanket exception swallowing.
- Local/offline fallbacks (LocalStack S3, synthetic Chicagoland imagery, `LOCAL_DATA_DIR` filesystem mode) mean the full pipeline runs in CI and on a fresh laptop without real AWS credentials — saves a new engineer a day of setup friction.

---

## Production engineering

**Kubernetes:** Kustomize base + dev (minikube, NodePort) / prod (EKS, NLB) overlays. HPA scales 2–10 replicas on CPU utilization. Image tags pinned per-deploy by the CD pipeline via `kustomize edit set image`.

**Observability:** custom Prometheus metrics — `landuse_predictions_total` (per-class counter, surfaces distribution drift), `landuse_inference_seconds` (histogram, 8 buckets, alerts if GPU degrades), `landuse_info` (gauge, tracks which model versions are loaded). Auto-provisioned Grafana dashboards at startup.

**Infrastructure:** full Terraform stack — VPC, EKS managed node group, ECR with lifecycle policies, S3 data lake + model registry, IAM roles with least-privilege, GitHub Actions OIDC federation.

---

## Stack

Python · PyTorch · HuggingFace Transformers · FastAPI · Apache Airflow · MLflow · Docker · Kubernetes (AWS EKS) · Terraform · AWS (SageMaker, S3, ECR) · Prometheus · Grafana · GitHub Actions

---

## Quickstart (no AWS account needed)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-training.txt -r backend/requirements-backend.txt

make init-segformer    # pull SegFormer-B2 backbone from HuggingFace (~90 MB)
make generate-masks    # run both models on synthetic Chicagoland data
make serve-local       # FastAPI backend at :8000, open frontend/index.html
make monitoring-up     # Prometheus :9090 · Grafana :3001 · MLflow :5001
```

To train SegFormer on real data (LoveDA dataset, ~5 hrs on RTX 3060 Ti):
```bash
make train-pipeline    # downloads LoveDA → trains → regenerates masks → restarts backend
make train-pipeline-watch  # tail training logs + MLflow metrics
```

---

<details>
<summary>Architecture diagram + detailed runbook</summary>

See [docs/architecture.md](docs/architecture.md) and [docs/runbook.md](docs/runbook.md).

</details>
