# Implementing an end-to-end MLOps pipeline

This repository contains a comprehensive MLOps pipeline for processing, analyzing, and deploying models based on Inertial Navigation System (INS) data. The project demonstrates best practices for building robust, maintainable, and scalable machine learning operations pipelines.

## Project Overview

The MLOps pipeline handles INS data through several stages:

1. **Data Ingestion**: Processing raw INS data from sensors
2. **Feature Engineering**: Extracting meaningful features from INS data
3. **Model Training & Evaluation**: Training and validating ML models
4. **Drift Detection**: Monitoring for data and concept drift
5. **Deployment**: Automated deployment of models
6. **Monitoring**: Continuous monitoring of model performance

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
│   ├── data/                     # Data processing modules
│   ├── features/                 # Feature engineering modules
│   ├── models/                   # Model training and evaluation
│   ├── drift/                    # Drift detection
│   ├── monitoring/               # Model monitoring
│   └── serve/                    # Model serving
├── dags/                         # Airflow DAG definitions
├── experiments/                  # Output directory for experiments
│   ├── features/                 # Extracted features
│   ├── models/                   # Trained models
│   └── monitoring/               # Monitoring data
├── test_schemas.py               # Schema validation test script
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
   pip install -r MLOps_Enhancements/requirements-enhancements.txt
   pip install -e .
   ```

4. Set up Apache Airflow:
   ```bash
   pip install apache-airflow
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
   ```

5. Create necessary directories:
   ```bash
   mkdir -p experiments/{features,models,monitoring,feature_store}
   ```

## Usage Guide

### Running the Pipeline with Airflow

1. Start Airflow services:
   ```bash
   airflow webserver -p 8080 &
   airflow scheduler &
   ```

2. Create a symlink to the DAG files:
   ```bash
   ln -s $(pwd)/dags/* ~/airflow/dags/
   ```

3. Navigate to http://localhost:8080 in your browser and login with the credentials you created.

4. Enable the `ins_mlops_pipeline` DAG from the Airflow UI.

### Using MLOps Enhancements

The `MLOps_Enhancements` directory contains lightweight components that can be integrated with existing ML pipelines:

#### Schema Validation

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

#### Feature Store

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

### Running the Demo

To see all components in action:

```bash
python MLOps_Enhancements/demo.py
```

This will:
1. Generate sample INS data
2. Validate it with schemas
3. Extract features
4. Store them in the feature store
5. Retrieve and display the features

### Running Tests

```bash
pytest MLOps_Enhancements/tests/
python test_schemas.py
```

## CI/CD Pipeline

The repository includes a GitHub Actions workflow that:

1. Runs tests on each push
2. Checks for code quality with flake8 and black
3. Performs drift detection on new data
4. Automatically deploys to staging and production when appropriate

To use it, simply copy the `.github/workflows/mlops-pipeline-ci.yml` file to your own repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 