# MLOps Pipeline with Automated Drift Detection and Retraining

This document provides an overview of the automated MLOps pipeline that includes drift detection and model retraining capabilities.

## Pipeline Overview

The pipeline consists of the following components:

1. **Data Ingestion**: Ingest INS (Inertial Navigation System) data
2. **Feature Extraction**: Process raw INS data into feature windows
3. **Model Training**: Train a model using the extracted features
4. **Model Evaluation**: Evaluate model performance on test data
5. **Model Serving**: Deploy the model via FastAPI
6. **Model Monitoring**: Monitor for data drift and performance degradation
7. **Automated Retraining**: Retrain the model when significant drift is detected

## Architecture

The pipeline can be executed in two ways:

1. **Standalone Mode**: Using the `run_pipeline.py` script, which runs all components once and exits (`--serve` keeps the API and monitoring active)
2. **Orchestrated Mode**: Using Apache Airflow DAG (`airflow/dags/ins_mlops_pipeline.py`), which provides scheduling, dependency management, and retries

## Drift Detection and Automated Retraining

The system continuously monitors incoming data for drift compared to the training data distribution:

1. Feature statistics are calculated for each batch of predictions
2. Z-scores are computed against the served model's training data (`monitoring` in `config/params.yaml`)
3. When drift is detected (z-score > 2.0) an alert is stored; z-score > 3.0 is severe
4. The `DriftHandler` picks up severe alerts and retrains on the newest labeled data (`python -m src.models.train_model --retrain`)
5. `python -m src.models.evaluate_model` compares the new version with the champion on data neither trained on and promotes it only if its F1 is higher; the API then reloads

`python -m src.monitoring.closed_loop` shows the whole loop on the end of the drive with a simulated sensor fault.

## How to Run the Pipeline

### Quick Start (Standalone Mode)

To run the complete pipeline in standalone mode:

```bash
python run_pipeline.py
```

This will:
- Extract features from the INS data and the camera frames
- Train, register and evaluate the model (MLflow)
- Start the model serving API
- Replay the end of the drive with a sensor fault: drift detection, retraining, promotion
- Exit (with `--serve`: keep serving and monitoring until Ctrl+C)

### Using Airflow (Orchestrated Mode)

To run the pipeline using Airflow 3, see "Orchestration with Airflow" in [README.md](README.md).

### Running Individual Components

To run specific components of the pipeline:

- **Feature Extraction**: `python -m src.features.extract_ins_features`
- **Model Training**: `python -m src.models.train_model`
- **Model Evaluation**: `python -m src.models.evaluate_model`
- **Model Serving**: `uvicorn src.serve.api:app` (the rule-based first-edition service: `uvicorn src.serve.simple_api:app`)
- **Closed-Loop Replay**: `python -m src.monitoring.closed_loop`
- **Generate Synthetic Monitoring Data**: `python -m src.monitoring.force_monitoring_data`
- **Drift Monitoring**: `python -m src.monitoring.drift_handler` (`--once` to check once)

## Directory Structure

```
├── airflow/
│   └── dags/                      # Airflow DAG definition
├── data/
│   └── ins/                       # Raw INS data
├── experiments/
│   ├── features/                  # Extracted features
│   ├── models/                    # Trained model artifacts
│   ├── mlflow.db                  # MLflow runs and Model Registry
│   └── monitoring/                # Monitoring data
│       ├── data_drift/            # Drift detection results
│       ├── performance/           # Performance metrics
│       └── predictions/           # Prediction logs
├── src/
│   ├── features/                  # Feature extraction code
│   ├── models/                    # Model training code
│   ├── monitoring/                # Monitoring code
│   │   ├── drift_handler.py       # Automated retraining
│   │   ├── force_monitoring_data.py # Generate synthetic data
│   │   └── monitoring_service.py  # Model monitoring
│   └── serve/                     # Model serving code
└── run_pipeline.py                # Pipeline runner script
```

## Extending the Pipeline

To extend this pipeline for your own use cases:

1. Replace the data ingestion with your own data source
2. Modify the feature extraction for your specific data
3. Update the model training for your ML task
4. Adjust the drift thresholds to suit your sensitivity requirements

## Troubleshooting

If you encounter issues:

1. Check the log files (`logs/pipeline_*.log` and `logs/api.log`)
2. Verify that directories exist and have appropriate permissions
3. Ensure dependencies are correctly installed
4. Check that ports 8000 (API) and 8501 (Streamlit dashboard) are available 