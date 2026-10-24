.PHONY: setup venv install-deps clean-env clean-data smoke test lint run-pipeline run-ingest run-features run-align run-train run-drift closed-loop serve dashboard mlflow-ui

# Variables
VENV_NAME := forge-env
PYTHON := python3
PIP := pip3
PROJECT_DIR := $(shell pwd)
VENV_DIR := $(PROJECT_DIR)/$(VENV_NAME)
VENV_BIN := $(VENV_DIR)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

# Setup virtual environment
setup: venv install-deps

venv:
	@echo "Creating virtual environment: $(VENV_NAME)"
	@$(PYTHON) -m venv $(VENV_NAME)
	@$(VENV_PIP) install --upgrade pip

install-deps:
	@echo "Installing dependencies"
	@$(VENV_PIP) install -r requirements.txt
	@$(VENV_PIP) install pytest flake8 black

# Smoke tests with uv (the whole pipeline on the sample data; a few minutes on CPU)
smoke:
	uv run pytest -q

# Clean environments and data
clean-env:
	@echo "Cleaning virtual environment"
	@rm -rf $(VENV_NAME)

clean-data:
	@echo "Cleaning experiment data (keeps the experiments/*/.gitkeep files)"
	@find experiments -mindepth 1 ! -name .gitkeep ! -type d -delete
	@find experiments -mindepth 1 -type d -empty ! -path 'experiments/features' ! -path 'experiments/models' ! -path 'experiments/monitoring' ! -path 'experiments/feature_store' -delete
	@rm -rf logs

# Run tests
test: test-unit test-integration

test-unit:
	@echo "Running unit tests"
	@$(VENV_PYTHON) -m pytest tests/test_alignment.py tests/test_drift.py tests/test_schema_validation.py enhancements_modules/tests -v

test-integration:
	@echo "Running integration tests"
	@$(VENV_PYTHON) -m pytest tests/test_integration.py tests/test_smoke.py -v

# Lint code
lint:
	@echo "Running linting"
	@$(VENV_BIN)/flake8 src/
	@$(VENV_BIN)/black --check src/

# Run pipeline stages
run-pipeline:
	@echo "Running complete pipeline"
	@$(VENV_PYTHON) -m src.pipeline.pipeline

run-ingest:
	@echo "Running ingestion stage"
	@$(VENV_PYTHON) -m src.ingest.video_ingest
	@$(VENV_PYTHON) -m src.ingest.imu_ingest

run-features:
	@echo "Running feature extraction stage"
	@$(VENV_PYTHON) -m src.features.video_features
	@$(VENV_PYTHON) -m src.features.extract_ins_features

run-align:
	@echo "Running alignment stage"
	@$(VENV_PYTHON) -m src.align.temporal_alignment

run-train:
	@echo "Running model training stage"
	@$(VENV_PYTHON) -m src.models.train_model
	@$(VENV_PYTHON) -m src.models.evaluate_model

run-drift:
	@echo "Running drift detection and retraining"
	@$(VENV_PYTHON) -m src.drift.drift_detector

# Replay the end of the drive with a sensor fault: drift alerts, retraining, promotion
closed-loop:
	@$(VENV_PYTHON) -m src.monitoring.closed_loop

# Serve the champion model on http://127.0.0.1:8000
serve:
	@$(VENV_PYTHON) -m uvicorn src.serve.api:app --host 127.0.0.1 --port 8000

dashboard:
	@$(VENV_BIN)/streamlit run src/monitoring/dashboard.py --server.address 127.0.0.1

mlflow-ui:
	@$(VENV_BIN)/mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db --host 127.0.0.1

# pipeline.yaml is an Argo Workflow for Kubernetes; it needs a container image of this repository

# Default target
all: setup test lint run-pipeline 