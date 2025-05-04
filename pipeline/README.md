# Core MLOps Pipeline

This directory contains the core MLOps pipeline, providing a complete workflow from data ingestion to model serving with drift detection and automated retraining.

## Pipeline Components

- **src/**: Source code for all pipeline components
  - **ingest/**: Data ingestion modules
  - **features/**: Feature engineering
  - **align/**: Temporal alignment for multimodal data
  - **models/**: Model training and evaluation
  - **drift/**: Drift detection components
  - **monitoring/**: Model performance monitoring
  - **serve/**: Model serving via API
  - **utils/**: Shared utility functions
  - **pipeline/**: Pipeline orchestration code

- **airflow/**: Airflow DAG configuration for workflow orchestration
- **config/**: Configuration files for the pipeline
- **tests/**: Testing framework for pipeline components
- **run_pipeline.py**: Script to run the complete pipeline in standalone mode
- **pipeline.yaml**: Workflow definition for orchestration engines
- **requirements.txt**: Core dependencies
- **Makefile**: Build automation and convenience commands

## Getting Started

Run the complete pipeline with:

```bash
python run_pipeline.py
```

This will execute all stages from feature extraction to model serving with drift detection.

## Pipeline Execution Flow

1. Feature extraction from INS data
2. Model training with extracted features
3. Model evaluation
4. Model serving via API
5. Continuous drift monitoring
6. Automated retraining when drift is detected

## Extending the Pipeline

To extend this pipeline for your own use cases:

1. Replace the data ingestion with your own data source
2. Modify the feature extraction for your specific data
3. Update the model training for your ML task
4. Adjust the drift thresholds to suit your sensitivity requirements 