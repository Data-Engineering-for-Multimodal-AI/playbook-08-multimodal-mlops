import os
import pandas as pd
import numpy as np
import pytest
from typing import Dict, List, Tuple

from src.drift.drift_detector import compute_ks_test, compute_wasserstein_distance, detect_drift, inject_drift


def test_compute_ks_test():
    # Test with identical distributions
    dist1 = np.random.normal(0, 1, 1000)
    dist2 = dist1.copy()
    ks_stat, p_value = compute_ks_test(dist1, dist2)
    
    # For identical distributions, p-value should be high
    assert p_value > 0.05
    assert ks_stat < 0.1
    
    # Test with different distributions
    dist3 = np.random.normal(2, 1, 1000)  # Different mean
    ks_stat, p_value = compute_ks_test(dist1, dist3)
    
    # For different distributions, p-value should be low
    assert p_value < 0.05
    assert ks_stat > 0.1
    
    # Test with empty arrays
    empty = np.array([])
    ks_stat, p_value = compute_ks_test(empty, dist1)
    assert ks_stat == 1.0
    assert p_value == 0.0
    
    ks_stat, p_value = compute_ks_test(dist1, empty)
    assert ks_stat == 1.0
    assert p_value == 0.0


def test_compute_wasserstein_distance():
    # Test with identical distributions
    dist1 = np.random.normal(0, 1, 1000)
    dist2 = dist1.copy()
    distance = compute_wasserstein_distance(dist1, dist2)
    
    # For identical distributions, distance should be close to 0
    assert distance < 0.1
    
    # Test with different distributions
    dist3 = np.random.normal(2, 1, 1000)  # Different mean
    distance = compute_wasserstein_distance(dist1, dist3)
    
    # For different distributions, distance should be larger
    assert distance > 1.0
    
    # Test with empty arrays
    empty = np.array([])
    distance = compute_wasserstein_distance(empty, dist1)
    assert distance == 1.0
    
    distance = compute_wasserstein_distance(dist1, empty)
    assert distance == 1.0


def test_detect_drift():
    # Create sample dataframes
    reference_data = pd.DataFrame({
        'acceleration_x_mean': np.random.normal(0, 1, 100),
        'acceleration_y_mean': np.random.normal(0, 1, 100),
        'acceleration_z_mean': np.random.normal(0, 1, 100)
    })
    
    # Test with identical data (no drift)
    current_data = reference_data.copy()
    monitor_features = ['acceleration_x_mean', 'acceleration_y_mean', 'acceleration_z_mean']
    
    # KS test with threshold 0.05
    drift_results = detect_drift(reference_data, current_data, monitor_features, 'ks_test', 0.05)
    assert not drift_results['drift_detected']
    
    # Test with drifted data (with drift)
    drifted_data = reference_data.copy()
    drifted_data['acceleration_x_mean'] = np.random.normal(2, 1, 100)  # Shifted mean
    
    drift_results = detect_drift(reference_data, drifted_data, monitor_features, 'ks_test', 0.05)
    assert drift_results['drift_detected']
    assert drift_results['features']['acceleration_x_mean']['drift_detected']
    assert not drift_results['features']['acceleration_y_mean']['drift_detected']
    
    # Test with wasserstein distance
    drift_results = detect_drift(reference_data, drifted_data, monitor_features, 'wasserstein', 0.5)
    assert drift_results['drift_detected']
    
    # Test with missing feature
    monitor_features.append('missing_feature')
    drift_results = detect_drift(reference_data, drifted_data, monitor_features, 'ks_test', 0.05)
    assert 'missing_feature' in drift_results['features']
    assert 'error' in drift_results['features']['missing_feature']


def test_inject_drift(tmp_path):
    # Create a test INS dataframe (the first edition's test used accelerometer and gyroscope
    # columns, which the INS data doesn't have)
    timestamps = np.arange(1000000, 2000000, 10000)
    rng = np.random.default_rng(0)
    imu_data = pd.DataFrame({
        'timestamp': timestamps,
        'velocity_north': rng.normal(0, 1, len(timestamps)),
        'velocity_east': rng.normal(0, 1, len(timestamps)),
        'velocity_down': rng.normal(0, 0.1, len(timestamps)),
        'roll': rng.normal(0, 0.01, len(timestamps)),
        'pitch': rng.normal(0, 0.01, len(timestamps)),
        'yaw': rng.normal(1.5, 0.1, len(timestamps))
    })
    
    # Create a temporary file
    temp_dir = tmp_path
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, 'test_imu.csv')
    imu_data.to_csv(temp_file, index=False)
    
    # Test injecting drift
    drifted_df = inject_drift(temp_file, output_path=temp_dir, noise_scale=0.5, frequency_change=0.2)
    
    # Verify that the drifted data is different from the original
    assert not drifted_df.equals(imu_data)
    
    # Verify that the shape is changed due to frequency resampling
    assert drifted_df.shape[0] != imu_data.shape[0]
    
    # Verify that the values are not the same due to noise injection
    original = imu_data.set_index('timestamp')
    kept = drifted_df[drifted_df['timestamp'].isin(original.index)]
    for col in ['velocity_north', 'velocity_east', 'yaw']:
        assert not np.allclose(kept[col].values, original.loc[kept['timestamp'], col].values)
    



if __name__ == "__main__":
    test_compute_ks_test()
    test_compute_wasserstein_distance()
    test_detect_drift()
    print("All drift detection tests passed!") 