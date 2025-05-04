# MLOps Pipeline Enhancements

This directory contains modular components that enhance the core MLOps pipeline with additional capabilities, best practices, and tools.

## Enhancement Components

- **schemas/**: Data validation schema definitions
  - Ensures data integrity at every pipeline stage
  - Provides automatic data validation and documentation
  - Supports schema evolution tracking

- **feature_store/**: Feature storage and retrieval system
  - Centralized feature management
  - Feature versioning and lineage tracking
  - Efficient feature retrieval with metadata filtering

- **ci/**: Continuous Integration/Continuous Deployment
  - GitHub Actions workflows
  - Automated testing
  - Deployment configurations

- **tests/**: Unit and integration tests for enhancements
  - Test coverage for all enhancement components
  - Mocks and fixtures for isolated testing

## Usage Examples

### Schema Validation

```python
from enhancements_modules.schemas.ins_schemas import INSDataPoint

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

```python
from enhancements_modules.feature_store.feature_store import FeatureStore

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

## Installation

Install the enhancement dependencies:

```bash
pip install -r requirements-enhancements.txt
```

## Demo

Run the demo script to see the enhancements in action:

```bash
python demo.py
```

## Integration with Core Pipeline

These enhancements are designed to be plugged into the core pipeline in a modular fashion. See the main repository README for guidance on integrating these components. 