# Implementing an end-to-end MLOps pipeline

This repository contains a comprehensive MLOps pipeline for processing, analyzing, and deploying models based on Inertial Navigation System (INS) data. The project demonstrates best practices for building robust, maintainable, and scalable machine learning operations pipelines.

## Project Overview

This pipeline processes data from the Oxford RobotCar dataset, focusing on INS (Inertial Navigation System) data which provides filtered, navigation-ready information about position, velocity, and orientation, making it ideal for autonomous vehicle applications.

The pipeline consists of these key stages:

1. **Data Ingestion**: Processing raw INS data from sensors and video frames
2. **Feature Engineering**: Extracting meaningful features with sliding windows from INS data
3. **Model Training**: Training and validating ML models with aligned multimodal data
4. **Drift Detection**: Monitoring for data distribution changes over time
5. **Automated Retraining**: Using drift signals to trigger model retraining
6. **Model Serving**: Providing predictions through a REST API
7. **MLOps Enhancements**: Schema validation, feature store, and monitoring

The project uses Apache Airflow for workflow orchestration, implementing a full CI/CD pipeline with automated testing and deployment.

## Repository Structure

```
├── MLOps_Enhancements/           # Enhanced MLOps components
│   ├── schemas/                  # Data validation schemas
│   ├── feature_store/            # Feature storage and retrieval
│   ├── tests/                    # Unit tests
│   ├── ci/                       # CI/CD configuration
│   └── demo.py                   # Demo script
├── src/                          # Core source code
│   ├── ingest/                   # Data processing modules
│   │   ├── ins_ingest.py         # INS data ingestion 
│   │   └── video_ingest.py       # Video frame processing
│   ├── features/                 # Feature engineering modules
│   │   ├── ins_features.py       # INS feature extraction
│   │   └── extract_ins_features.py # Combined extraction script
│   ├── align/                    # Data alignment
│   │   └── temporal_alignment.py # Aligns INS and video data
│   ├── models/                   # Model training and evaluation
│   │   ├── train.py              # Model training script
│   │   └── predict.py            # Model inference
│   ├── drift/                    # Drift detection
│   │   └── drift_detector.py     # Detects data distribution shifts
│   ├── monitoring/               # Model monitoring
│   │   ├── monitoring_service.py # Tracks predictions and drift
│   │   ├── drift_handler.py      # Handles drift detection
│   │   └── dashboard.py          # Monitoring visualization
│   └── serve/                    # Model serving
│       ├── model_service.py      # Model loading and prediction
│       ├── simple_model_service.py # Simplified model service
│       └── api.py                # FastAPI application
├── airflow/                      # Airflow DAG definitions
│   └── dags/                     # DAG script for pipeline
│       └── ins_mlops_pipeline.py # Main pipeline DAG
├── config/                       # Configuration files
├── run_pipeline.py               # Script to run the full pipeline
└── setup.py                      # Package installation script
```

## Installation

### Prerequisites

- Python 3.8+
- Apache Airflow 2.0+
- Conda or virtualenv for environment management

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Data-Engineering-for-Multimodal-AI/pb8-mm-mlops-pipeline.git
   cd pb8-mm-mlops-pipeline
   ```

2. Create and activate a virtual environment:
   ```bash
   conda create -n mlops-pipeline python=3.9
   conda activate mlops-pipeline
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   pip install -r MLOps_Enhancements/requirements-enhancements.txt
   pip install -e .
   ```

4. Set up necessary directories:
   ```bash
   mkdir -p experiments/{features,models,monitoring,feature_store}
   ```

## Pipeline Execution

### Running the Complete Pipeline

The simplest way to run the entire pipeline is using the `run_pipeline.py` script:

```bash
python run_pipeline.py
```

This will execute all steps: data ingestion, feature extraction, model training, drift detection, and deployment.

### Step-by-Step Execution

Alternatively, you can execute each stage of the pipeline individually:

#### 1. Data Ingestion

Process INS data from the Oxford RobotCar dataset:

```bash
python -m src.ingest.ins_ingest
```

Process video frames:

```bash
python -m src.ingest.video_ingest
```

#### 2. Feature Extraction

Extract features from INS data:

```bash
python -m src.features.ins_features
```

Extract features from video data:

```bash
python -m src.features.video_features
```

#### 3. Data Alignment

Align INS and video data based on timestamps:

```bash
python -m src.align.temporal_alignment
```

#### 4. Model Training

Train a model using the aligned features:

```bash
python -m src.models.train
```

#### 5. Model Evaluation

Evaluate the model performance:

```bash
python -m src.models.predict
```

#### 6. Model Serving

Start the prediction service:

```bash
python -m src.serve.simple_model_service
```

Start the FastAPI server to serve predictions:

```bash
python -m src.serve.api
```

Test the API with a client:

```bash
python -m src.serve.http_client
```

#### 7. Monitoring

Start the monitoring dashboard:

```bash
streamlit run src/monitoring/dashboard.py
```

Generate simulated traffic for monitoring:

```bash
python -m src.monitoring.simple_simulate_traffic --requests 150 --drift
```

### Orchestration with Airflow

For production deployment, the pipeline is orchestrated using Apache Airflow:

1. Set up Airflow:
   ```bash
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow db reset -y  # Only if starting fresh
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.org
   ```

2. Start Airflow components (each in a separate terminal):
   ```bash
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow scheduler
   
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow webserver -p 8080
   
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow triggerer
   ```

3. Access the Airflow UI at http://localhost:8080 and enable the `ins_mlops_pipeline` DAG.

## MLOps Enhancements

The `MLOps_Enhancements` directory contains lightweight components to improve your MLOps pipeline:

### Schema Validation

```python
from MLOps_Enhancements.schemas.ins_schemas import INSDataPoint

# Validate raw INS data
valid_data = INSDataPoint(
    timestamp=1586346000000000,
    velocity_north=0.5,
    velocity_east=1.2,
    velocity_down=0.1,
    roll=0.01,
    pitch=0.02,
    yaw=0.5
)
```

### Feature Store

Efficiently store and retrieve features with metadata:

```python
from MLOps_Enhancements.feature_store.feature_store import FeatureStore

# Initialize the feature store
feature_store = FeatureStore(base_path="experiments/feature_store")

# Store features
feature_id = feature_store.store_features(
    features_df,
    metadata={"source": "ins_data", "processing_version": "1.0"}
)

# Retrieve features with filtering
recent_features = feature_store.retrieve_features(
    filters=[("timestamp", ">", yesterday_timestamp)]
)
```

## Drift Detection and Automated Retraining

The pipeline includes continuous monitoring for data drift:

1. The `drift_detector.py` component constantly analyzes new data distributions
2. When significant drift is detected, it automatically triggers retraining
3. The `drift_handler.py` manages the retraining workflow
4. New models are evaluated and deployed only if they perform better than the current model

This closed-loop architecture ensures your model stays accurate as real-world data evolves.

## Troubleshooting

### Common Issues

#### Airflow Database Issues

If you encounter SQLite errors or "UNIQUE constraint failed: job.id" when running Airflow:

```bash
# Reset the Airflow database
export AIRFLOW_HOME=$(pwd)/airflow
airflow db reset -y
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.org
```

#### Model Dimension Mismatch

If you encounter "ValueError: X has 6 features, but StandardScaler is expecting 1343 features as input", use the simplified model service:

```bash
# Use the simplified model service that works with the 6 raw INS features
python -m src.serve.simple_model_service
python -m src.serve.simple_api
```

This happens because the full model was trained on engineered features, but for demonstration, we're allowing direct inference with the 6 raw INS values (velocity_north, velocity_east, velocity_down, roll, pitch, yaw).

#### Package Import Errors

If you encounter import errors when running scripts:

```bash
# Make sure to set the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 