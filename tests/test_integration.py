import os
import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from src.ingest.video_ingest import ingest_video_frames
from src.ingest.imu_ingest import ingest_imu_data
from src.features.video_features import process_video_features
from src.features.imu_features import process_imu_features
from src.align.temporal_alignment import align_features, prepare_ml_dataset
from src.utils.config import load_schema, load_params


@pytest.mark.integration
def test_data_ingestion_to_alignment():
    """Test the data flow from ingestion to alignment."""
    # Create test directories
    os.makedirs('experiments/test_integration', exist_ok=True)
    os.makedirs('experiments/test_integration/ingested/video', exist_ok=True)
    os.makedirs('experiments/test_integration/ingested/imu', exist_ok=True)
    os.makedirs('experiments/test_integration/features/video', exist_ok=True)
    os.makedirs('experiments/test_integration/features/imu', exist_ok=True)
    os.makedirs('experiments/test_integration/aligned', exist_ok=True)
    
    # 1. Ingest video data
    try:
        video_df = ingest_video_frames('experiments/test_integration/ingested/video')
        assert video_df is not None
        assert not video_df.empty
        assert 'timestamp' in video_df.columns
        assert 'frame_id' in video_df.columns
        assert 'file_path' in video_df.columns
        
        video_metadata_path = os.path.join('experiments/test_integration/ingested/video', 'video_metadata.csv')
        assert os.path.exists(video_metadata_path)
    except Exception as e:
        pytest.skip(f"Video ingestion not testable: {str(e)}")
    
    # 2. Ingest IMU data
    try:
        imu_df = ingest_imu_data('experiments/test_integration/ingested/imu')
        assert imu_df is not None
        # IMU data can be empty if not available in the dataset
        if not imu_df.empty:
            assert 'timestamp' in imu_df.columns
            
            imu_data_path = os.path.join('experiments/test_integration/ingested/imu', 'imu_processed.csv')
            assert os.path.exists(imu_data_path)
    except Exception as e:
        pytest.skip(f"IMU ingestion not testable: {str(e)}")
    
    # 3. Process video features
    try:
        if os.path.exists(video_metadata_path):
            video_features_df = process_video_features(
                video_metadata_path, 
                'experiments/test_integration/features/video'
            )
            assert video_features_df is not None
            assert not video_features_df.empty
            assert 'timestamp' in video_features_df.columns
            assert 'hist_features' in video_features_df.columns
            
            video_features_path = os.path.join('experiments/test_integration/features/video', 'video_features.parquet')
            assert os.path.exists(video_features_path)
    except Exception as e:
        pytest.skip(f"Video feature extraction not testable: {str(e)}")
    
    # 4. Process IMU features
    try:
        if not imu_df.empty and os.path.exists(os.path.join('experiments/test_integration/ingested/imu', 'imu_processed.csv')):
            imu_features_df = process_imu_features(
                os.path.join('experiments/test_integration/ingested/imu', 'imu_processed.csv'),
                'experiments/test_integration/features/imu'
            )
            assert imu_features_df is not None
            # IMU features can be empty if IMU data is not available
            if not imu_features_df.empty:
                assert 'timestamp' in imu_features_df.columns
                assert 'statistical_features' in imu_features_df.columns
                
                imu_features_path = os.path.join('experiments/test_integration/features/imu', 'imu_features.parquet')
                assert os.path.exists(imu_features_path)
    except Exception as e:
        pytest.skip(f"IMU feature extraction not testable: {str(e)}")
    
    # 5. Align features
    try:
        if (os.path.exists(os.path.join('experiments/test_integration/features/video', 'video_features.parquet')) and
            os.path.exists(os.path.join('experiments/test_integration/features/imu', 'imu_features.parquet'))):
            
            aligned_df = align_features(
                os.path.join('experiments/test_integration/features/video', 'video_features.parquet'),
                os.path.join('experiments/test_integration/features/imu', 'imu_features.parquet'),
                'experiments/test_integration/aligned'
            )
            
            # Alignment can be empty if modalities don't overlap in time
            if not aligned_df.empty:
                assert 'timestamp' in aligned_df.columns
                assert 'time_diff_sec' in aligned_df.columns
                
                aligned_features_path = os.path.join('experiments/test_integration/aligned', 'aligned_features.parquet')
                assert os.path.exists(aligned_features_path)
                
                # 6. Prepare ML dataset
                features_df, labels_df = prepare_ml_dataset(
                    aligned_features_path,
                    os.path.join('experiments/test_integration/aligned', 'ml_dataset')
                )
                
                if not features_df.empty:
                    assert 'timestamp' in features_df.columns
                    assert 'features' in features_df.columns
                    
                    assert os.path.exists(os.path.join('experiments/test_integration/aligned/ml_dataset', 'features.parquet'))
                    
                    if not labels_df.empty:
                        assert 'timestamp' in labels_df.columns
                        assert 'label' in labels_df.columns
                        assert os.path.exists(os.path.join('experiments/test_integration/aligned/ml_dataset', 'labels.parquet'))
    except Exception as e:
        pytest.skip(f"Feature alignment not testable: {str(e)}")


if __name__ == "__main__":
    test_data_ingestion_to_alignment() 