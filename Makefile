.DEFAULT_GOAL := help

API := api
CORE := core_ml
PY := python

# No default: the account ID is not versioned. CI passes it as a protected
# variable; locally, `make docker-push AWS_ACCOUNT_ID=123456789012`.
AWS_ACCOUNT_ID ?=
AWS_REGION ?= us-east-1
ECR_REGISTRY ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
API_IMAGE := $(ECR_REGISTRY)/predictive-maintenance-mlops-api
# Immutable, traceable tag: the git commit by default (CI_COMMIT_SHA in GitLab),
# never "latest" for a real release.
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
K8S_OVERLAY := kubernetes/overlays/production
EKS_CLUSTER_NAME ?= predictive-maintenance-mlops

.PHONY: help install install-api install-core \
	format format-check lint lint-fix typecheck \
	test test-api test-core coverage \
	precommit-install precommit-run \
	build-toy-dataset prepare-toy prepare-data train train-toy smoke-test \
	dvc-pull dvc-push dvc-use-localstack \
	compose-up compose-down compose-destroy compose-logs compose-ps localstack-env \
	docker-build docker-push \
	k8s-build k8s-diff deploy \
	terraform-fmt terraform-validate terraform-plan terraform-apply \
	ci-local \
	ci clean

help: ## Show this help
	@echo "Single command interface (local and CI). Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## ---------------------------------------------------------------------
## Deterministic dependencies (Poetry + lockfile)
## ---------------------------------------------------------------------

install: install-api install-core ## Install runtime + dev dependencies of api/ and core_ml/

install-api: ## Install api/ dependencies (dev group included)
	poetry -C $(API) install --with dev --no-interaction

install-core: ## Install core_ml/ dependencies (dev group included)
	poetry -C $(CORE) install --with dev --no-interaction

## ---------------------------------------------------------------------
## Shift-left validation (Python)
## ---------------------------------------------------------------------

format: ## Apply ruff format (rewrites files) in api/ and core_ml/
	poetry -C $(API) run ruff format .
	poetry -C $(CORE) run ruff format .

format-check: ## Check formatting without changing files (used in CI)
	poetry -C $(API) run ruff format --check .
	poetry -C $(CORE) run ruff format --check .

lint: ## Run ruff (lint) in api/ and core_ml/
	poetry -C $(API) run ruff check .
	poetry -C $(CORE) run ruff check .

lint-fix: ## Run ruff with autofix in api/ and core_ml/
	poetry -C $(API) run ruff check --fix .
	poetry -C $(CORE) run ruff check --fix .

typecheck: ## Run mypy in api/ and core_ml/ (advisory, does not block CI)
	poetry -C $(API) run mypy .
	poetry -C $(CORE) run mypy src

test: test-api test-core ## Run pytest in api/ and core_ml/

test-api: ## Run pytest in api/ only
	poetry -C $(API) run pytest

test-core: ## Run pytest in core_ml/ only
	poetry -C $(CORE) run pytest

coverage: ## Run pytest with coverage in api/ and core_ml/
	poetry -C $(API) run pytest --cov=. --cov-report=term-missing
	poetry -C $(CORE) run pytest --cov=src --cov-report=term-missing

## ---------------------------------------------------------------------
## Data and reproducibility
## ---------------------------------------------------------------------

build-toy-dataset: ## Rebuild the fixed toy dataset (1,116 rows) from the full dataset
	poetry -C $(CORE) run python scripts/build_toy_dataset.py

prepare-toy: ## Build engine_features.parquet + feature_names.json for the toy dataset
	poetry -C $(CORE) run python src/prepare_training_data.py --data-dir data_toy

prepare-data: ## Build engine_features.parquet + feature_names.json for the full dataset
	poetry -C $(CORE) run python src/prepare_training_data.py --data-dir data

train: prepare-data ## Train on the full dataset (quality gate enforced)
	poetry -C $(CORE) run python src/train.py --data-dir data --epochs 25 --patience 5

# Defaults MLFLOW_TRACKING_URI to a throwaway local SQLite store, so the target
# runs on its own. An exported value wins (e.g. the docker-compose MLflow:
# export MLFLOW_TRACKING_URI=http://localhost:5001).
train-toy: prepare-toy ## Train end to end on the toy dataset: seconds, no GPU
	MLFLOW_TRACKING_URI="$${MLFLOW_TRACKING_URI:-sqlite:///mlruns/smoke_test.db}" \
	poetry -C $(CORE) run python src/train.py \
		--data-dir data_toy --no-enforce-quality-gate

smoke-test: train-toy ## Validate the pipeline end to end (data, contracts, MLflow) before real compute
	@echo "OK: pipeline verified end to end on the toy dataset."

dvc-pull: ## Download the versioned datasets from the DVC S3 remote
	poetry -C $(CORE) run dvc pull

dvc-push: ## Upload the versioned datasets (data/, data_toy/) to the DVC S3 remote
	poetry -C $(CORE) run dvc push

dvc-use-localstack: ## Point the DVC S3 remote at LocalStack (this machine only, via .dvc/config.local)
	poetry -C $(CORE) run dvc remote modify --local s3remote url s3://predictive-maintenance-mlops-artifacts-local
	poetry -C $(CORE) run dvc remote modify --local s3remote endpointurl http://localhost:4567
	poetry -C $(CORE) run dvc remote modify --local s3remote access_key_id test
	poetry -C $(CORE) run dvc remote modify --local s3remote secret_access_key test
	@echo "OK: core_ml/.dvc/config.local points 's3remote' at LocalStack (http://localhost:4567, the port docker-compose publishes)."
	@echo "    core_ml/.dvc/config (shared in git) is unchanged."

## ---------------------------------------------------------------------
## Local containers (Docker Compose + LocalStack)
## ---------------------------------------------------------------------

compose-up: ## Start the whole local stack: Postgres, MLflow, API, LocalStack
	docker compose up -d --build
	@echo "API:        http://localhost:8001 (503 on /health until a 'champion' model exists)"
	@echo "MLflow UI:  http://localhost:5001"
	@echo "LocalStack: http://localhost:4567"

compose-down: ## Stop the local stack (volumes, i.e. Postgres/LocalStack data, are kept)
	docker compose down

compose-destroy: ## Stop the local stack and DELETE its volumes (clean restart)
	docker compose down --volumes

compose-logs: ## Follow the logs of every local service
	docker compose logs -f

compose-ps: ## Show the state and healthcheck of every local service
	docker compose ps

localstack-env: ## Print the variables that point a local shell at LocalStack
	@echo "export AWS_ENDPOINT_URL=http://localhost:4567"
	@echo "export AWS_ACCESS_KEY_ID=test"
	@echo "export AWS_SECRET_ACCESS_KEY=test"
	@echo "export AWS_DEFAULT_REGION=us-east-1"

## ---------------------------------------------------------------------
## API image build/push and Kubernetes deployment
## ---------------------------------------------------------------------
# MLflow has no image of its own to build: it runs the public
# ghcr.io/mlflow/mlflow image (docker-compose.yml, kubernetes/base/mlflow.yaml).

docker-build: ## Build the API image (root Dockerfile)
	@if [ -z "$(AWS_ACCOUNT_ID)" ]; then echo "Usage: make $@ AWS_ACCOUNT_ID=123456789012"; exit 1; fi
	docker build -t $(API_IMAGE):$(IMAGE_TAG) -f Dockerfile .

docker-push: ## Push the API image to ECR (needs a prior `docker login` to ECR)
	@if [ -z "$(AWS_ACCOUNT_ID)" ]; then echo "Usage: make $@ AWS_ACCOUNT_ID=123456789012"; exit 1; fi
	docker push $(API_IMAGE):$(IMAGE_TAG)

k8s-build: ## Render the production overlay (kustomize build) for inspection
	kubectl kustomize $(K8S_OVERLAY)

k8s-diff: ## Diff the overlay against the live cluster (kubectl diff -k)
	kubectl diff -k $(K8S_OVERLAY) || true

deploy: ## Pin IMAGE_TAG in the overlay and apply it to the cluster (kubectl apply -k)
	@if [ -z "$(AWS_ACCOUNT_ID)" ]; then echo "Usage: make $@ AWS_ACCOUNT_ID=123456789012"; exit 1; fi
	cd $(K8S_OVERLAY) && kustomize edit set image predictive-maintenance-mlops-api=$(API_IMAGE):$(IMAGE_TAG)
	kubectl apply -k $(K8S_OVERLAY)

## ---------------------------------------------------------------------
## Infrastructure as code (Terraform)
## ---------------------------------------------------------------------

terraform-fmt: ## Check the canonical format of every .tf file (terraform/bootstrap included)
	terraform -chdir=terraform fmt -check -diff -recursive

terraform-validate: ## Validate terraform/ syntax and types (needs a prior `terraform init`)
	terraform -chdir=terraform validate

terraform-plan: ## Plan against real AWS (needs valid credentials)
	terraform -chdir=terraform plan

terraform-apply: ## Apply against real AWS (needs valid credentials)
	terraform -chdir=terraform apply

## ---------------------------------------------------------------------
## Run the GitLab CI pipeline locally
## ---------------------------------------------------------------------

ci-local: ## Run .gitlab-ci.yml locally with gitlab-ci-local (needs Docker + Node.js)
	@command -v npx >/dev/null 2>&1 || { \
		echo "Node.js/npx is not installed. The community-maintained way to"; \
		echo "run GitLab CI pipelines locally is"; \
		echo "gitlab-ci-local: https://github.com/firecow/gitlab-ci-local"; \
		echo "Install Node.js (nodejs.org) and run 'make ci-local' again."; \
		exit 1; \
	}
	MSYS_NO_PATHCONV=1 npx --yes gitlab-ci-local $(JOB)

## ---------------------------------------------------------------------
## Git hooks
## ---------------------------------------------------------------------

precommit-install: ## Install the pre-commit git hooks in this repo
	pre-commit install --install-hooks
	pre-commit install --hook-type commit-msg || true

precommit-run: ## Run every pre-commit hook over the whole repo
	pre-commit run --all-files

## ---------------------------------------------------------------------
## Aggregates
## ---------------------------------------------------------------------

ci: format-check lint test ## Full quality pipeline (what GitLab CI runs in lint_test)
	@echo "OK: every quality check passed."

clean: ## Remove local tool caches
	find . -type d -name "__pycache__" -not -path "*/.venv/*" -prune -exec rm -rf {} +
	rm -rf $(API)/.mypy_cache $(API)/.ruff_cache $(API)/.pytest_cache $(API)/htmlcov
	rm -rf $(CORE)/.mypy_cache $(CORE)/.ruff_cache $(CORE)/.pytest_cache $(CORE)/htmlcov
