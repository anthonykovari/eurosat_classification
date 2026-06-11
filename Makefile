
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
	pip install boto3 -q
	python scripts/localstack_seed.py

etl-trigger:
	docker compose -f etl/docker-compose.yml exec airflow-scheduler \
	  airflow dags trigger eurosat_etl_pipeline

etl-status:
	docker compose -f etl/docker-compose.yml exec airflow-scheduler \
	  airflow dags list-runs -d eurosat_etl_pipeline --state all

# ── Kubernetes ────────────────────────────────────────────────────────────────
k8s-dev:
	kustomize build k8s/overlays/dev | kubectl apply -f -

k8s-dev-down:
	kustomize build k8s/overlays/dev | kubectl delete -f -

k8s-prod:
	kustomize build k8s/overlays/prod | kubectl apply -f -

k8s-status:
	kubectl get pods,svc,hpa -n eurosat

# ── Training ──────────────────────────────────────────────────────────────────
train-local:
	python scripts/train.py

train-mlflow:
	MLFLOW_TRACKING_URI=http://localhost:5001 python scripts/train.py --epochs 2

train-sagemaker:
	python scripts/sagemaker_train.py

deploy:
	python deploy.py

# ── Kubernetes (minikube) ─────────────────────────────────────────────────────
k8s-minikube-images:
	eval $$(minikube docker-env) && \
	  docker build -t eurosat-backend:latest -f backend/Dockerfile . && \
	  docker build -t eurosat-frontend:latest ./frontend

k8s-minikube-deploy: k8s-minikube-images
	kubectl create secret generic aws-credentials -n eurosat \
	  --from-literal=access-key-id=test \
	  --from-literal=secret-access-key=test \
	  --dry-run=client -o yaml | kubectl apply -f -
	kustomize build k8s/overlays/dev | kubectl apply -f -
	kubectl rollout status deployment/backend -n eurosat --timeout=120s
	kubectl rollout status deployment/frontend -n eurosat --timeout=60s

# ── Terraform ─────────────────────────────────────────────────────────────────
tf-init:
	terraform -chdir=terraform init

tf-plan:
	terraform -chdir=terraform plan

tf-apply:
	terraform -chdir=terraform apply

tf-destroy:
	terraform -chdir=terraform destroy
