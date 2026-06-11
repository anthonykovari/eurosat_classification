# EuroSAT Land Use Classifier

End-to-end ML platform for satellite image classification. A fine-tuned ResNet-18 achieves **97–99% per-class accuracy** across 10 land-use categories, served via a FastAPI backend and web UI. The full stack covers data ingestion, model training, experiment tracking, containerised deployment, and cloud infrastructure — built to demonstrate a production AI/ML Engineer workflow.

## Demo

| Forest | Residential |
|--------|-------------|
| ![Forest](assets/example_forest.png) | ![Residential](assets/example_residential.png) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                              │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌────────────┐   │
│  │ Extract  │───▶│ Validate │───▶│ Transform │───▶│    Load    │   │
│  │          │    │          │    │  (CLAHE)  │    │  S3 catalog│   │
│  └──────────┘    └──────────┘    └───────────┘    └─────┬──────┘   │
│     Apache Airflow DAG (weekly)                         │          │
└─────────────────────────────────────────────────────────┼──────────┘
                                                          │ trigger
┌─────────────────────────────────────────────────────────▼──────────┐
│                         TRAINING PIPELINE                           │
│                                                                     │
│  ┌─────────────────────┐     ┌──────────────────────────────────┐  │
│  │  SageMaker Training │────▶│  MLflow Tracking Server          │  │
│  │  Job (ml.p3.2xlarge)│     │  · params, loss/acc per epoch    │  │
│  │  scripts/train.py   │     │  · model registered in registry  │  │
│  └─────────────────────┘     └──────────────────────────────────┘  │
│              │                                                      │
│              ▼ model artifact                                       │
│       S3 model registry                                             │
└─────────────────────────────────────────────────────────────────────┘
                    │ deploy.py
┌───────────────────▼─────────────────────────────────────────────────┐
│                        SERVING (Kubernetes)                          │
│                                                                      │
│   ┌─────────────────────┐        ┌──────────────────────────────┐   │
│   │  frontend (nginx)   │        │  backend (FastAPI)            │   │
│   │  NodePort :30300    │──────▶│  NodePort :30800              │   │
│   │  (dev) / NLB (prod) │        │  CLAHE → ResNet-18 → JSON     │   │
│   └─────────────────────┘        │  HPA: 2–10 replicas          │   │
│                                  └──────────────────────────────┘   │
│            Kustomize overlays: dev (minikube) / prod (EKS)          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    CLOUD INFRASTRUCTURE (Terraform)                   │
│                                                                       │
│  S3 data lake  ·  S3 model registry  ·  ECR (backend + frontend)     │
│  EKS cluster (VPC + managed node group)  ·  IAM roles                │
│  GitHub Actions OIDC role (no long-lived keys)                        │
└───────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Model | ResNet-18 — ImageNet pre-trained, fine-tuned on EuroSAT RGB |
| Training | PyTorch · weighted sampler · StepLR · 25 epochs |
| Preprocessing | OpenCV CLAHE (applied by ETL pipeline and at inference time) |
| Experiment tracking | MLflow — params, per-epoch metrics, model registry |
| ETL orchestration | Apache Airflow 2.9 — weekly DAG, S3-backed data catalog |
| Training infrastructure | AWS SageMaker Training Jobs |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/JS |
| Container registry | Amazon ECR |
| Kubernetes | EKS (prod) · minikube/kind (dev) · Kustomize overlays · HPA |
| Infrastructure as code | Terraform — S3, ECR, EKS, IAM, GitHub OIDC |
| CI/CD | GitHub Actions — lint, DAG validation, k8s dry-run, ECR push, EKS rollout |
| Local dev | Docker Compose — backend + frontend + MLflow |

## Model Performance

| Class | Val Accuracy |
|---|---|
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

## Project Structure

```
├── backend/                        # FastAPI inference service
│   ├── main.py                     # /predict/ and /health endpoints
│   ├── Dockerfile
│   └── requirements-backend.txt
├── frontend/                       # Static web UI
│   ├── index.html
│   └── Dockerfile
├── scripts/
│   ├── train.py                    # SageMaker-compatible training script (MLflow)
│   ├── sagemaker_train.py          # Submit SageMaker training job from CLI
│   ├── download_data.py            # Local EuroSAT downloader
│   └── requirements.txt            # Extra deps installed in SageMaker container
├── code/
│   └── inference.py                # SageMaker inference handler (model_fn … output_fn)
├── etl/
│   ├── dags/
│   │   └── eurosat_etl_dag.py      # Airflow DAG: extract→validate→transform→load→train
│   ├── Dockerfile                  # Custom Airflow image with OpenCV + sklearn
│   ├── docker-compose.yml          # Local Airflow stack (webserver + scheduler + postgres)
│   └── requirements.txt
├── k8s/
│   ├── base/                       # Shared manifests (Deployment, Service, HPA, ConfigMap)
│   └── overlays/
│       ├── dev/                    # minikube — NodePort services, local images
│       └── prod/                   # EKS — NLB LoadBalancer, ECR images, pinned tags
├── terraform/
│   ├── main.tf                     # AWS provider, S3 remote state
│   ├── s3.tf                       # Data lake + model registry buckets
│   ├── ecr.tf                      # ECR repos + lifecycle policies
│   ├── eks.tf                      # VPC + EKS cluster (terraform-aws-modules)
│   ├── iam.tf                      # SageMaker, Airflow, and GitHub OIDC roles
│   ├── variables.tf
│   └── outputs.tf
├── .github/workflows/
│   ├── ci.yml                      # Lint · DAG parse · Docker build · k8s dry-run · tf validate
│   └── cd.yml                      # ECR push (OIDC) → kustomize deploy → EKS rollout
├── notebooks/
│   ├── validate.ipynb              # Per-class accuracy & confusion matrix
│   └── visual_check.ipynb
├── docker-compose.yml              # Local: backend + frontend + MLflow
├── Makefile
└── requirements-training.txt
```

---

## Runbook

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Docker + Compose | ≥ 24 | [docs.docker.com](https://docs.docker.com/get-docker/) |
| kubectl | ≥ 1.29 | `brew install kubectl` |
| kustomize | ≥ 5 | `brew install kustomize` |
| Terraform | ≥ 1.6 | `brew install terraform` |
| AWS CLI | ≥ 2 | `brew install awscli` |
| minikube | ≥ 1.32 | `brew install minikube` (dev only) |

---

### 1 · Local development

```bash
# Start backend, frontend, and MLflow tracking server
make start

# Services:
#   Frontend  → http://localhost:3000
#   Backend   → http://localhost:8000
#   MLflow UI → http://localhost:5001
```

Train locally and track in MLflow:

```bash
pip install -r requirements-training.txt
MLFLOW_TRACKING_URI=http://localhost:5001 python scripts/train.py --epochs 5
```

---

### 2 · Provision cloud infrastructure

```bash
# Export required vars (use Terraform outputs after first apply)
export AWS_DEFAULT_REGION=us-east-1

# One-time: create the S3 backend bucket manually, then:
make tf-init
make tf-plan   # review
make tf-apply

# Note the outputs — you'll need these below:
terraform -chdir=terraform output
```

Key outputs: `data_lake_bucket`, `model_registry_bucket`, `ecr_backend_url`, `eks_cluster_name`, `sagemaker_role_arn`.

---

### 3 · Run the ETL pipeline

```bash
# Set Airflow Variables before starting (or set them in the UI after)
export EUROSAT_S3_BUCKET=<data_lake_bucket>

make etl-up          # starts Airflow at http://localhost:8080 (admin/admin)
make etl-logs        # tail logs

# Trigger a manual run from the UI, or:
docker compose -f etl/docker-compose.yml exec airflow-scheduler \
  airflow dags trigger eurosat_etl_pipeline
```

Set these Airflow Variables in the UI (`Admin → Variables`):

| Key | Value |
|---|---|
| `eurosat_s3_bucket` | `<data_lake_bucket>` |
| `eurosat_data_lake_bucket` | `<data_lake_bucket>` |
| `eurosat_model_registry_bucket` | `<model_registry_bucket>` |
| `eurosat_sagemaker_role_arn` | `<sagemaker_role_arn>` |
| `eurosat_mlflow_tracking_uri` | MLflow server URL |

The DAG runs weekly. On completion it automatically submits a SageMaker training job.

---

### 4 · Submit a training job manually

```bash
export EUROSAT_DATA_LAKE_BUCKET=<data_lake_bucket>
export EUROSAT_MODEL_REGISTRY_BUCKET=<model_registry_bucket>
export SAGEMAKER_ROLE_ARN=<sagemaker_role_arn>
export MLFLOW_TRACKING_URI=<mlflow_url>

make train-sagemaker
# or: python scripts/sagemaker_train.py --epochs 25 --instance ml.p3.2xlarge --wait

# Train locally instead:
make train-local
```

Track experiments at your MLflow server. The best model is automatically registered in the MLflow Model Registry as `eurosat-resnet18`.

---

### 5 · Deploy to Kubernetes

**Dev (minikube):**

```bash
minikube start
make k8s-dev         # applies overlays/dev — NodePort services

# Create the AWS credentials secret for the model init container:
kubectl create secret generic aws-credentials -n eurosat \
  --from-literal=access-key-id=$AWS_ACCESS_KEY_ID \
  --from-literal=secret-access-key=$AWS_SECRET_ACCESS_KEY

make k8s-status      # watch pods come up

# Access:
#   Frontend → http://$(minikube ip):30300
#   Backend  → http://$(minikube ip):30800
```

**Prod (EKS):**

```bash
aws eks update-kubeconfig --name <eks_cluster_name> --region us-east-1
make k8s-prod        # applies overlays/prod — NLB LoadBalancer services
make k8s-status
```

Image tags for prod are pinned at deploy time by the CD pipeline (`kustomize edit set image`).

---

### 6 · Deploy SageMaker inference endpoint

```bash
export EUROSAT_MODEL_REGISTRY_BUCKET=<model_registry_bucket>
export SAGEMAKER_ROLE_ARN=<sagemaker_role_arn>

make deploy
# or: python deploy.py --instance ml.g4dn.xlarge
```

---

### 7 · CI/CD (GitHub Actions)

Push to any branch → **CI** runs automatically:
- `ruff` lint
- Airflow DAG parse check
- Docker build for both images
- `kustomize build` dry-run for dev + prod overlays
- `terraform validate`

Merge to `main` → **CD** runs:
1. Authenticates to AWS via OIDC (no long-lived keys stored in GitHub)
2. Builds and pushes backend + frontend images to ECR (tagged with short SHA)
3. Pins image tags in `k8s/overlays/prod/kustomization.yaml` via `kustomize edit set image`
4. Applies manifests to EKS and waits for rollout

One-time setup: add `AWS_DEPLOY_ROLE_ARN` to GitHub repository secrets (value from `terraform output`).

---

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 labeled 64×64 Sentinel-2 satellite images across 10 land-use classes. The ETL pipeline downloads, preprocesses, and versions the dataset in S3 weekly.
