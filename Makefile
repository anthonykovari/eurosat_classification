
# ── MLflow Model Registry ────────────────────────────────────────────────────
mlflow-register:
	python3 scripts/register_models.py

# ── Local Docker ──────────────────────────────────────────────────────────────
start:
	docker compose up --build -d

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

restart: down build up

# ── Monitoring ────────────────────────────────────────────────────────────────
monitoring-up:
	docker compose up -d prometheus grafana

monitoring-down:
	docker compose stop prometheus grafana

monitoring-logs:
	docker compose logs -f prometheus grafana

# ── ETL (Airflow + LocalStack) ────────────────────────────────────────────────
etl-up:
	docker compose -f etl/docker-compose.yml up --build -d

etl-down:
	docker compose -f etl/docker-compose.yml down -v

etl-logs:
	docker compose -f etl/docker-compose.yml logs -f

localstack-seed:
	python3 scripts/localstack_seed.py

etl-trigger:
	docker compose -f etl/docker-compose.yml exec airflow-scheduler \
	  airflow dags trigger chicago_land_use_pipeline

etl-status:
	docker compose -f etl/docker-compose.yml exec airflow-scheduler \
	  airflow dags list-runs -d chicago_land_use_pipeline --state all

# ── Training (EuroSAT ResNet-18 — weights reused by the ETL classifier) ──────
export-onnx:
	python3 scripts/export_onnx.py

train-local:
	python3 scripts/train.py

train-mlflow:
	MLFLOW_TRACKING_URI=http://localhost:5001 python3 scripts/train.py --epochs 2

# ── Training (SegFormer-B2 on LoveDA — per-pixel segmentation) ────────────────
init-segformer:
	python3 scripts/init_segformer.py

train-pipeline:
	bash scripts/train_pipeline.sh

train-pipeline-watch:
	tail -f /tmp/train_pipeline.log

train-segformer:
	MLFLOW_TRACKING_URI=./mlruns python3 scripts/train_segformer.py

train-segformer-mlflow:
	MLFLOW_TRACKING_URI=http://localhost:5001 python3 scripts/train_segformer.py --epochs 5

# ── Local demo data (no CDSE / Airflow needed) ────────────────────────────────
generate-masks:
	python3 scripts/generate_seg_masks.py

serve-local:
	LOCAL_DATA_DIR=local_s3 MODEL_PATH=outputs/resnet18_eurosat.pth \
	  SEG_MODEL_PATH=outputs/segformer_b2_loveda.pth \
	  python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# ── Kubernetes ────────────────────────────────────────────────────────────────
k8s-dev:
	kustomize build k8s/overlays/dev | kubectl apply -f -

k8s-dev-down:
	kustomize build k8s/overlays/dev | kubectl delete -f -

k8s-prod:
	kustomize build k8s/overlays/prod | kubectl apply -f -

k8s-status:
	kubectl get pods,svc,hpa -n chicago-land-use

# ── Kubernetes (minikube) ─────────────────────────────────────────────────────
install-k8s-tools:
	curl -LO "https://dl.k8s.io/release/$$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
	  && chmod +x kubectl && sudo mv kubectl /usr/local/bin/
	curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 \
	  && chmod +x minikube-linux-amd64 && sudo mv minikube-linux-amd64 /usr/local/bin/minikube
	curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash \
	  && sudo mv kustomize /usr/local/bin/

k8s-minikube-start:
	minikube start --cpus=4 --memory=4g --driver=docker

k8s-minikube-images:
	eval $$(minikube docker-env) && \
	  docker build -t chicago-landuse-backend:latest -f backend/Dockerfile . && \
	  docker build -t chicago-landuse-frontend:latest ./frontend

k8s-minikube-deploy: k8s-minikube-images
	minikube ssh -- sudo mkdir -p /mnt/chicago-outputs
	minikube cp outputs/resnet18_eurosat.pth minikube:/mnt/chicago-outputs/resnet18_eurosat.pth
	kustomize build k8s/overlays/dev | kubectl apply -f -
	kubectl rollout status deployment/minio   -n eurosat --timeout=60s
	kubectl rollout status deployment/backend -n eurosat --timeout=120s
	kubectl rollout status deployment/frontend -n eurosat --timeout=60s

k8s-minikube-seed:
	kubectl delete job minio-seed -n eurosat --ignore-not-found
	kubectl apply -f k8s/overlays/dev/minio-seed-job.yaml
	kubectl wait --for=condition=complete job/minio-seed -n eurosat --timeout=600s

k8s-minikube-url:
	@echo "Frontend : $$(minikube service frontend -n eurosat --url)"
	@echo "Backend  : $$(minikube service backend  -n eurosat --url)"
	@echo "API docs : $$(minikube service backend  -n eurosat --url)/docs"
	@echo "MinIO    : $$(minikube service minio    -n eurosat --url | head -1)"

k8s-minikube-stop:
	minikube stop

k8s-minikube-delete:
	minikube delete

# ── Terraform ─────────────────────────────────────────────────────────────────
tf-init:
	terraform -chdir=terraform init

tf-plan:
	terraform -chdir=terraform plan

tf-apply:
	terraform -chdir=terraform apply

tf-destroy:
	terraform -chdir=terraform destroy
