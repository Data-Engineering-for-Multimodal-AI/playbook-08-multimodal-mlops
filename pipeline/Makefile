.PHONY: setup venv clean-env clean-data test lint run-pipeline run-ingest run-features run-align run-train run-drift

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

# Clean environments and data
clean-env:
	@echo "Cleaning virtual environment"
	@rm -rf $(VENV_NAME)

clean-data:
	@echo "Cleaning experiment data"
	@rm -rf experiments/*

# Run tests
test: test-unit test-integration

test-unit:
	@echo "Running unit tests"
	@$(VENV_PYTHON) -m pytest tests/test_alignment.py tests/test_drift.py tests/test_schema_validation.py -v

test-integration:
	@echo "Running integration tests"
	@$(VENV_PYTHON) -m pytest tests/test_integration.py -v

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
	@$(VENV_PYTHON) -m src.features.imu_features

run-align:
	@echo "Running alignment stage"
	@$(VENV_PYTHON) -m src.align.temporal_alignment

run-train:
	@echo "Running model training stage"
	@$(VENV_PYTHON) -m src.models.train

run-drift:
	@echo "Running drift detection and retraining"
	@$(VENV_PYTHON) -m src.drift.drift_detector

# Use 'pipeline.yaml' for orchestration with Vertex AI or Kubeflow
create-pipeline-dag:
	@echo "Creating pipeline DAG for Vertex AI or Kubeflow"
	@$(VENV_PYTHON) -m src.utils.create_pipeline_dag

# Default target
all: setup test lint run-pipeline 