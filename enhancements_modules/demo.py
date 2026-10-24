#!/usr/bin/env python
"""
Demo script showing how to use the MLOps enhancements.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlops_demo")

# Import our enhancements
from enhancements_modules.schemas.ins_schemas import INSDataPoint, INSFeatures
from enhancements_modules.feature_store.feature_store import FeatureStore

def create_sample_ins_data(num_points: int = 100) -> pd.DataFrame:
    """Create sample INS data for demonstration."""
    # Create timestamps - one per second for the last N seconds
    now = datetime.now()
    timestamps = [(now - timedelta(seconds=i)).timestamp() * 1_000_000 for i in range(num_points)]
    
    # Create random velocity and orientation values
    data = {
        "timestamp": timestamps,
        "velocity_north": np.random.normal(0, 1, num_points),
        "velocity_east": np.random.normal(0, 1, num_points),
        "velocity_down": np.random.normal(0, 0.1, num_points),
        "roll": np.random.normal(0, 0.1, num_points),
        "pitch": np.random.normal(0, 0.1, num_points),
        "yaw": np.random.normal(0, 0.5, num_points)
    }
    
    return pd.DataFrame(data)

def validate_ins_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate INS data using Pydantic schema."""
    valid_rows = []
    invalid_rows = []
    
    for _, row in df.iterrows():
        try:
            # Validate row using Pydantic model
            validated_data = INSDataPoint(**row.to_dict())
            valid_rows.append(validated_data.model_dump())  # Use model_dump() instead of dict()
        except Exception as e:
            logger.warning(f"Invalid data row: {e}")
            invalid_rows.append(row.to_dict())
    
    logger.info(f"Validated {len(valid_rows)} rows, found {len(invalid_rows)} invalid rows")
    return pd.DataFrame(valid_rows)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract statistical features from INS data."""
    # Group by windows of 10 samples
    window_size = 10
    features_list = []
    
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        if len(window) < window_size:
            continue  # Skip incomplete windows
            
        # Extract statistical features for each channel
        feature_vector = []
        for channel in ["velocity_north", "velocity_east", "velocity_down", "roll", "pitch", "yaw"]:
            values = window[channel].values
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            feature_vector.extend([mean, std, min_val, max_val])
        
        # Create a feature row
        feature_row = {
            "timestamp": window["timestamp"].min(),
            "window_id": i // window_size,
            "statistical_features": feature_vector
        }
        
        # Validate feature vector
        try:
            INSFeatures(**feature_row)
            features_list.append(feature_row)
        except Exception as e:
            logger.warning(f"Invalid feature vector: {e}")
    
    return pd.DataFrame(features_list)

def main():
    """Run the demo."""
    logger.info("Starting MLOps enhancements demo")
    
    # Create demo data directory
    os.makedirs("demo_output", exist_ok=True)
    
    # 1. Create sample INS data
    logger.info("Creating sample INS data")
    ins_data = create_sample_ins_data(num_points=200)
    ins_data.to_csv("demo_output/sample_ins_data.csv", index=False)
    
    # 2. Validate INS data using schema
    logger.info("Validating INS data using schema")
    valid_data = validate_ins_data(ins_data)
    
    # 3. Extract features
    logger.info("Extracting features from INS data")
    features_df = extract_features(valid_data)
    features_df.to_csv("demo_output/features.csv", index=False)
    
    # 4. Store features in feature store
    logger.info("Storing features in feature store")
    feature_store = FeatureStore(base_path="demo_output/feature_store")
    feature_id = feature_store.store_features(
        features_df,
        metadata={
            "source": "demo",
            "timestamp": datetime.now().isoformat(),
            "num_features": len(features_df)
        }
    )
    
    # 5. Retrieve features from feature store
    logger.info("Retrieving features from feature store")
    retrieved_features = feature_store.retrieve_features(feature_id=feature_id)
    logger.info(f"Retrieved {len(retrieved_features)} feature rows")
    
    # 6. Get latest features
    logger.info("Getting latest features")
    latest_features = feature_store.get_latest_features(limit=5)
    
    # Check if latest_features is empty or has the expected columns
    if not latest_features.empty and 'window_id' in latest_features.columns and 'timestamp' in latest_features.columns:
        logger.info(f"Latest features:\n{latest_features[['window_id', 'timestamp']].head()}")
    else:
        logger.warning("Latest features retrieval returned empty or unexpected format")
    
    # 7. List feature registry
    logger.info("Listing feature registry")
    registry = feature_store.list_features()
    print(json.dumps(registry, indent=2))
    
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    main() 