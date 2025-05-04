# MLOps Enhancements

This directory contains enhancements to the INS data processing MLOps pipeline that can be integrated without disrupting the existing functionality.

## Overview of Enhancements

1. **Schema Validation with Pydantic**
   - Robust data validation for INS data points and features
   - Type checking and range validation for numerical values
   - Standardized request/response schemas for the prediction API

2. **Feature Store**
   - Efficient storage and retrieval of feature vectors
   - Partitioned Parquet files for performance
   - Queryable feature registry with metadata
   - Efficient filtering capabilities

3. **Unit Tests**
   - Testing for feature store functionality
   - Test fixtures for temporary storage

4. **CI/CD Pipeline**
   - GitHub Actions workflow for automated testing and deployment
   - Data drift detection in the CI pipeline
   - Conditional model retraining based on drift detection
   - Automated deployment to staging and production

## Usage Guide

### Schema Validation

The schema validation can be used to validate INS data at various stages of the pipeline:

```python
from MLOps_Enhancements.schemas.ins_schemas import INSDataPoint, INSFeatures

# Validate raw INS data
try:
    validated_data = INSDataPoint(
        timestamp=1586346000000000,
        velocity_north=0.5,
        velocity_east=1.2,
        velocity_down=0.1,
        roll=0.01,
        pitch=0.02,
        yaw=0.5
    )
    print("Valid INS data point")
except ValueError as e:
    print(f"Invalid INS data: {e}")

# Validate feature vector
try:
    features = INSFeatures(
        statistical_features=[...],  # 24 feature values
        window_id=123,
        timestamp=1586346000000000
    )
    print("Valid feature vector")
except ValueError as e:
    print(f"Invalid feature vector: {e}")
```

### Feature Store

The feature store provides a consistent interface for storing and retrieving INS features:

```python
from MLOps_Enhancements.feature_store.feature_store import FeatureStore

# Initialize the feature store
feature_store = FeatureStore(base_path="experiments/feature_store")

# Store features
feature_id = feature_store.store_features(
    features_df,
    metadata={"source": "ins_data", "processing_version": "1.0"}
)

# Retrieve features by ID
features = feature_store.retrieve_features(feature_id=feature_id)

# Retrieve features with filtering
recent_features = feature_store.retrieve_features(
    filters=[("timestamp", ">", 1586346000000000)]
)

# Get the latest features
latest_features = feature_store.get_latest_features(limit=10)
```

### CI/CD Pipeline Integration

To use the CI/CD pipeline:

1. Copy the `.github/workflows/mlops-pipeline-ci.yml` file to your repository's `.github/workflows/` directory
2. Customize the workflow as needed for your specific environment
3. Push to GitHub to trigger the workflow

The CI pipeline will:
- Run tests
- Check code quality
- Build the package
- Run drift detection
- Deploy to staging and production

## Integration with Existing Code

These enhancements are designed to be non-disruptive. You can:

1. Start by integrating schema validation into the API
2. Then add the feature store for new feature extraction runs
3. Finally set up the CI/CD pipeline once the other components are working

## Requirements

The enhancements require the following additional dependencies:

```
pydantic>=1.8.0
pyarrow>=7.0.0
pytest>=6.0.0
pytest-cov>=2.12.0
```

Add these to your requirements.txt file before integrating the enhancements. 