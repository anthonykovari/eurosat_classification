
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

# ── ETL (Airflow) ─────────────────────────────────────────────────────────────
etl-up:
	docker compose -f etl/docker-compose.yml up --build -d

etl-down:
	docker compose -f etl/docker-compose.yml down

etl-logs:
	docker compose -f etl/docker-compose.yml logs -f

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

train-sagemaker:
	python scripts/sagemaker_train.py

deploy:
	python deploy.py

# ── Terraform ─────────────────────────────────────────────────────────────────
tf-init:
	terraform -chdir=terraform init

tf-plan:
	terraform -chdir=terraform plan

tf-apply:
	terraform -chdir=terraform apply

tf-destroy:
	terraform -chdir=terraform destroy
