import os
import pandas as pd
import numpy as np
import pytest
from typing import Dict, List, Tuple

from src.align.temporal_alignment import find_nearest_frame, align_modalities


def test_find_nearest_frame():
    # Create a sample dataframe with timestamps
    video_df = pd.DataFrame({
        'timestamp': [1000000, 2000000, 3000000, 4000000, 5000000],
        'frame_id': [0, 1, 2, 3, 4],
        'hist_features': [[1, 2, 3]] * 5
    })
    
    # Test finding exact matches
    frame = find_nearest_frame(3000000, video_df)
    assert frame is not None
    assert frame['timestamp'] == 3000000
    assert frame['frame_id'] == 2
    
    # Test finding nearest match (3.5M was exactly between two frames in the first edition's
    # test, which then expected the later one; 3.6M is unambiguous)
    frame = find_nearest_frame(3600000, video_df)
    assert frame is not None
    assert frame['timestamp'] == 4000000  # Closest to 3.6M
    assert frame['frame_id'] == 3
    
    # Test with max_time_diff
    # Timestamp diff is 0.4s, so this should find a match
    frame = find_nearest_frame(3600000, video_df, max_time_diff=0.6)
    assert frame is not None
    assert frame['timestamp'] == 4000000
    
    # Test with stricter max_time_diff
    # Timestamp diff is 0.4s, so this shouldn't find a match
    frame = find_nearest_frame(3600000, video_df, max_time_diff=0.3)
    assert frame is None
    
    # The lookup must also work when the dataframe's index isn't 0..n-1 (e.g. after sorting)
    shuffled = video_df.iloc[::-1]
    assert find_nearest_frame(3600000, shuffled)['frame_id'] == 3
    
    # Test with empty dataframe
    empty_df = pd.DataFrame(columns=['timestamp', 'frame_id'])
    frame = find_nearest_frame(3000000, empty_df)
    assert frame is None


def test_align_modalities():
    # Create sample dataframes
    video_df = pd.DataFrame({
        'timestamp': [1000000, 2000000, 3000000, 4000000, 5000000],
        'frame_id': [0, 1, 2, 3, 4],
        'hist_features': [[1, 2, 3]] * 5,
        'embedding_features': [[4, 5, 6]] * 5
    })
    
    imu_df = pd.DataFrame({
        'timestamp': [1200000, 2200000, 3200000, 4200000],
        'statistical_features': [[7, 8, 9]] * 4,
        'acceleration_x_mean': [0.1, 0.2, 0.3, 0.4],
        'acceleration_y_mean': [0.5, 0.6, 0.7, 0.8]
    })
    
    # Test alignment with a tolerance that matches every window (each is 0.2 s from a frame;
    # the default tolerance is 0.1 s, which the first edition's test overlooked)
    aligned_df = align_modalities(video_df, imu_df, max_time_diff=0.5)
    
    # Check if alignment was successful
    assert len(aligned_df) == 4  # Should match all 4 IMU windows
    
    # Check if the correct columns are present
    expected_columns = [
        'timestamp', 'video_timestamp', 'time_diff_us', 'time_diff_sec',
        'imu_statistical_features', 'imu_acceleration_x_mean', 'imu_acceleration_y_mean',
        'video_hist_features', 'video_embedding_features'
    ]
    
    for col in expected_columns:
        assert col in aligned_df.columns
    
    # Check if time differences are calculated correctly
    for i, row in aligned_df.iterrows():
        assert row['time_diff_us'] == abs(row['timestamp'] - row['video_timestamp'])
        assert row['time_diff_sec'] == row['time_diff_us'] / 1e6
    
    # Test with max_time_diff
    aligned_df = align_modalities(video_df, imu_df, max_time_diff=0.1)
    # With a strict time difference limit, we should have fewer matches
    assert len(aligned_df) < 4
    
    # Test with empty dataframes
    empty_df = pd.DataFrame(columns=['timestamp'])
    aligned_empty = align_modalities(empty_df, imu_df)
    assert aligned_empty.empty
    
    aligned_empty = align_modalities(video_df, empty_df)
    assert aligned_empty.empty


if __name__ == "__main__":
    test_find_nearest_frame()
    test_align_modalities()
    print("All alignment tests passed!") 