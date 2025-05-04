import os
import pandas as pd
import numpy as np
import pytest
import json
from typing import Dict, List, Tuple

from src.utils.config import load_config, load_schema


def test_schema_structure():
    """Test that the schema has the expected structure."""
    schema = load_schema()
    
    # Check top-level sections
    assert 'video' in schema
    assert 'imu' in schema
    assert 'alignment' in schema
    assert 'feature_store' in schema
    
    # Check video section
    assert 'source_path' in schema['video']
    assert 'format' in schema['video']
    assert 'sampling_rate' in schema['video']
    assert 'features' in schema['video']
    
    # Check video features
    video_features = schema['video']['features']
    assert isinstance(video_features, list)
    assert len(video_features) > 0
    
    for feature in video_features:
        assert 'name' in feature
        assert 'type' in feature
        assert 'shape' in feature
        assert 'description' in feature
    
    # Check IMU section
    assert 'source_path' in schema['imu']
    assert 'sampling_rate' in schema['imu']
    assert 'columns' in schema['imu']
    assert 'features' in schema['imu']
    
    # Check IMU columns
    imu_columns = schema['imu']['columns']
    assert isinstance(imu_columns, list)
    assert len(imu_columns) > 0
    
    required_column_names = ['timestamp', 'acceleration_x', 'acceleration_y', 'acceleration_z',
                           'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    column_names = [col['name'] for col in imu_columns]
    for required_name in required_column_names:
        assert any(name.startswith(required_name) for name in column_names), f"Missing column: {required_name}"
    
    # Check alignment section
    assert 'window_size' in schema['alignment']
    assert 'strategy' in schema['alignment']
    assert 'output_rate' in schema['alignment']
    
    # Check feature store section
    assert 'path' in schema['feature_store']
    assert 'format' in schema['feature_store']


def create_mock_video_metadata() -> pd.DataFrame:
    """Create a mock video metadata dataframe that conforms to the schema."""
    schema = load_schema()
    video_schema = schema['video']
    
    # Create a dataframe with timestamps
    timestamps = np.arange(1000000, 1500000, 100000)
    
    metadata = []
    for i, ts in enumerate(timestamps):
        metadata.append({
            'frame_id': i,
            'timestamp': ts,
            'timestamp_readable': pd.Timestamp(ts * 1000).isoformat(),
            'file_path': f"{ts}.{video_schema['format']}",
            'ingested_path': f"output/{ts}.{video_schema['format']}"
        })
    
    return pd.DataFrame(metadata)


def create_mock_imu_data() -> pd.DataFrame:
    """Create a mock IMU dataframe that conforms to the schema."""
    schema = load_schema()
    imu_schema = schema['imu']
    
    # Create a dataframe with timestamps
    timestamps = np.arange(1000000, 1500000, 10000)
    
    data = {'timestamp': timestamps}
    
    # Add columns based on schema
    for column in imu_schema['columns']:
        if column['name'] != 'timestamp':
            col_type = column.get('type', 'float64')
            if col_type == 'float64':
                data[column['name']] = np.random.normal(0, 1, len(timestamps))
            elif col_type == 'int64':
                data[column['name']] = np.random.randint(0, 100, len(timestamps))
    
    return pd.DataFrame(data)


def test_schema_compatibility():
    """Test that mock data conforms to the schema."""
    # Load schema
    schema = load_schema()
    
    # Create mock data
    video_df = create_mock_video_metadata()
    imu_df = create_mock_imu_data()
    
    # Validate video metadata against schema
    assert 'frame_id' in video_df.columns
    assert 'timestamp' in video_df.columns
    assert video_df['timestamp'].dtype == np.int64
    
    # Validate IMU data against schema
    required_columns = [col['name'] for col in schema['imu']['columns']]
    for col in required_columns:
        if col in imu_df.columns:
            if col == 'timestamp':
                assert imu_df[col].dtype == np.int64
    
    # Check at least one accelerometer and gyroscope column exists
    accel_cols = [col for col in imu_df.columns if col.startswith('acceleration_')]
    gyro_cols = [col for col in imu_df.columns if col.startswith('gyroscope_')]
    
    assert len(accel_cols) > 0, "No acceleration columns found"
    assert len(gyro_cols) > 0, "No gyroscope columns found"


def test_schema_version():
    """Test that the schema has a version."""
    schema = load_schema()
    assert 'version' in schema
    assert isinstance(schema['version'], str)


if __name__ == "__main__":
    test_schema_structure()
    test_schema_compatibility()
    test_schema_version()
    print("All schema validation tests passed!") 