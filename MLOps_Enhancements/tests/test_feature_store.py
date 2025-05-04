import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import pytest
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from MLOps_Enhancements.feature_store.feature_store import FeatureStore

@pytest.fixture
def temp_feature_store():
    """Fixture to create a temporary feature store for testing."""
    temp_dir = tempfile.mkdtemp()
    feature_store = FeatureStore(base_path=temp_dir)
    
    yield feature_store
    
    # Cleanup
    shutil.rmtree(temp_dir)

def create_sample_features(num_samples: int = 10) -> pd.DataFrame:
    """Create sample INS features for testing."""
    # Create timestamps - one per hour for the last N hours
    now = datetime.now()
    timestamps = [(now - timedelta(hours=i)).timestamp() * 1_000_000 for i in range(num_samples)]
    
    # Create feature vectors - simulate statistical features for 6 channels (mean, std, min, max)
    # Each row has 24 features (6 channels * 4 stats)
    features_list = []
    for _ in range(num_samples):
        # Generate random statistical features (mean, std, min, max) for 6 channels
        features = []
        for _ in range(6):  # 6 channels
            channel_mean = np.random.normal(0, 1)
            channel_std = np.random.uniform(0.1, 1.0)
            channel_min = channel_mean - 2 * channel_std
            channel_max = channel_mean + 2 * channel_std
            features.extend([channel_mean, channel_std, channel_min, channel_max])
        features_list.append(features)
    
    # Create DataFrame with timestamps and features
    df = pd.DataFrame({
        "timestamp": timestamps,
        "window_id": list(range(num_samples))
    })
    
    # Add the features as separate columns for testing
    for i, features in enumerate(features_list):
        for j, feature_value in enumerate(features):
            df.loc[i, f"feature_{j}"] = feature_value
    
    # Add a column with the full feature vector for convenience
    df["statistical_features"] = features_list
    
    return df

def test_feature_store_init(temp_feature_store):
    """Test feature store initialization."""
    # Check that the directories were created
    assert os.path.exists(temp_feature_store.base_path)
    assert os.path.exists(temp_feature_store.ins_features_path)
    assert os.path.exists(temp_feature_store.metadata_path)
    assert os.path.exists(temp_feature_store.registry_file)

def test_store_and_retrieve_features(temp_feature_store):
    """Test storing and retrieving features."""
    # Create sample features
    sample_features = create_sample_features(num_samples=20)
    
    # Store features
    feature_id = temp_feature_store.store_features(
        sample_features, 
        feature_type="ins_features",
        metadata={"source": "test", "description": "Sample features"}
    )
    
    # Check that the feature ID was returned
    assert feature_id
    
    # Retrieve the features
    retrieved_features = temp_feature_store.retrieve_features(feature_id=feature_id)
    
    # Check that we got the expected number of features back
    assert len(retrieved_features) == len(sample_features)
    
    # Check that the timestamps match
    pd.testing.assert_series_equal(
        sample_features["timestamp"].sort_values().reset_index(drop=True),
        retrieved_features["timestamp"].sort_values().reset_index(drop=True)
    )

def test_get_latest_features(temp_feature_store):
    """Test retrieving the latest features."""
    # Create sample features over a few days
    day1_features = create_sample_features(num_samples=5)
    day2_features = create_sample_features(num_samples=5)
    
    # Make day2 timestamps more recent
    day2_features["timestamp"] = day2_features["timestamp"] + 86400 * 1_000_000  # Add 1 day in microseconds
    
    # Store both days
    feature_id1 = temp_feature_store.store_features(day1_features, metadata={"day": "1"})
    feature_id2 = temp_feature_store.store_features(day2_features, metadata={"day": "2"})
    
    # Get latest features (should be day2)
    latest_features = temp_feature_store.get_latest_features(limit=3)
    
    # Should have the latest timestamps
    max_timestamp = max(
        day1_features["timestamp"].max(),
        day2_features["timestamp"].max()
    )
    
    assert max_timestamp in latest_features["timestamp"].values
    assert len(latest_features) == 3  # Requested only 3 latest features

def test_feature_registry(temp_feature_store):
    """Test that the feature registry is updated properly."""
    # Create and store sample features
    sample_features = create_sample_features(num_samples=5)
    feature_id = temp_feature_store.store_features(
        sample_features,
        metadata={"test_metadata": True}
    )
    
    # Check that the registry contains the feature
    registry = temp_feature_store.list_features()
    assert "ins_features" in registry
    assert feature_id in registry["ins_features"]
    assert registry["ins_features"][feature_id]["metadata"]["test_metadata"] is True

def test_filtering_features(temp_feature_store):
    """Test filtering features by conditions."""
    # Create sample features with varying window IDs
    sample_features = create_sample_features(num_samples=10)
    
    # Ensure window_id is a column we can filter on
    sample_features["window_id"] = list(range(10))
    
    # Store features
    feature_id = temp_feature_store.store_features(sample_features)
    
    # Retrieve features with window_id > 5
    filters = [("window_id", ">", 5)]
    filtered_features = temp_feature_store.retrieve_features(filters=filters)
    
    # Check that we only got features with window_id > 5
    assert len(filtered_features) == 4  # window_id 6, 7, 8, 9
    assert all(filtered_features["window_id"] > 5) 