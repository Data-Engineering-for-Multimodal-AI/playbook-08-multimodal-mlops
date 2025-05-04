# Implementing an end-to-end MLOps pipeline

This repository contains a comprehensive MLOps pipeline for processing, analyzing, and deploying models based on Inertial Navigation System (INS) data. The project demonstrates best practices for building robust, maintainable, and scalable machine learning operations pipelines.

View our [Pipeline Architecture Diagram](docs/images/pipeline_architecture.md)

## Project Overview

This pipeline processes data from the Oxford RobotCar dataset, focusing on INS (Inertial Navigation System) data which provides filtered, navigation-ready information about position, velocity, and orientation, making it ideal for autonomous vehicle applications.

The pipeline consists of these key stages:

1. **Data Ingestion**: Processing raw INS data from sensors and video frames
2. **Feature Engineering**: Extracting meaningful features with sliding windows from INS data
3. **Model Training**: Training and validating ML models with aligned multimodal data
4. **Drift Detection**: Monitoring for data distribution changes over time
5. **Automated Retraining**: Using drift signals to trigger model retraining
6. **Model Serving**: Providing predictions through a REST API

The project uses Apache Airflow for workflow orchestration, implementing a full CI/CD pipeline with automated testing and deployment.

## Repository Structure

```
├── pipeline/                       # Core MLOps Pipeline
│   ├── src/                        # Core source code
│   │   ├── ingest/                 # Data processing modules
│   │   ├── features/               # Feature engineering modules
│   │   ├── align/                  # Data alignment
│   │   ├── models/                 # Model training and evaluation 
│   │   ├── drift/                  # Drift detection
│   │   ├── monitoring/             # Model monitoring
│   │   ├── serve/                  # Model serving
│   │   ├── utils/                  # Utility functions
│   │   └── pipeline/               # Pipeline orchestration
│   ├── airflow/                    # Airflow DAG definitions
│   ├── config/                     # Configuration files
│   ├── tests/                      # Core pipeline tests
│   ├── run_pipeline.py             # Script to run the full pipeline
│   ├── pipeline.yaml               # Workflow definition
│   ├── requirements.txt            # Core dependencies
│   └── Makefile                    # Build automation
│
├── enhancements_modules/           # Enhanced MLOps components
│   ├── schemas/                    # Data validation schemas
│   ├── feature_store/              # Feature storage and retrieval
│   ├── tests/                      # Enhancement-specific tests
│   ├── ci/                         # CI/CD configuration
│   └── demo.py                     # Demo script
│
├── data/                           # Dataset directory
├── experiments/                    # Experiment tracking
├── notebooks/                      # Jupyter notebooks
└── docs/                           # Documentation
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
   pip install -r pipeline/requirements.txt
   pip install -r enhancements_modules/requirements-enhancements.txt
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
python pipeline/run_pipeline.py
```

This will execute all steps: data ingestion, feature extraction, model training, drift detection, and deployment.

### Orchestration with Airflow

For production deployment, the pipeline is orchestrated using Apache Airflow:

1. Set up Airflow:
   ```bash
   export AIRFLOW_HOME=$(pwd)/pipeline/airflow
   airflow db reset -y  # Only if starting fresh
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.org
   ```

2. Start Airflow components (each in a separate terminal):
   ```bash
   export AIRFLOW_HOME=$(pwd)/pipeline/airflow
   airflow scheduler
   
   export AIRFLOW_HOME=$(pwd)/pipeline/airflow
   airflow webserver -p 8080
   
   export AIRFLOW_HOME=$(pwd)/pipeline/airflow
   airflow triggerer
   ```

3. Access the Airflow UI at http://localhost:8080 and enable the `ins_mlops_pipeline` DAG.

## MLOps Enhancements

The `enhancements_modules` directory contains lightweight components to improve your MLOps pipeline, including schema validation, feature store implementation, CI/CD automation, and more. Refer to the directory's README for detailed documentation.

## Drift Detection and Automated Retraining

One of the key features of this pipeline is its ability to detect data drift and automatically retrain models when needed:

1. The monitoring service tracks predictions and calculates statistical distributions
2. When drift is detected beyond configured thresholds, an alert is generated
3. The drift handler component picks up alerts and triggers model retraining
4. New models are evaluated and deployed only if they perform better than current ones

This closed-loop architecture ensures models stay accurate as data distributions evolve over time.

## Troubleshooting

### Common Issues

#### Airflow Database Issues

If you encounter SQLite errors or "UNIQUE constraint failed" when running Airflow:

```bash
# Reset the Airflow database
export AIRFLOW_HOME=$(pwd)/pipeline/airflow
airflow db reset -y
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.org
```

#### Model Dimension Mismatch

If you encounter "ValueError: X has 6 features, but StandardScaler is expecting X features", use the simplified model service:

```bash
# Use the simplified model service that works with raw INS features
python -m pipeline.src.serve.simple_model_service
```

#### Package Import Errors

If you encounter import errors:

```bash
# Make sure to set the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Data Sources

This pipeline works with the Oxford RobotCar dataset:

- **Oxford RobotCar Dataset**: https://robotcar-dataset.robots.ox.ac.uk/
- We specifically use the INS data which provides filtered navigation sensor readings

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Data Engineering for Multimodal AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Acknowledgments

- Oxford Robotics Institute for providing the RobotCar dataset
- Apache Airflow team for their excellent workflow orchestration framework
- The MLOps community for best practices and inspiration