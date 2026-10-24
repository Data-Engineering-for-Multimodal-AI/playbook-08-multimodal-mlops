import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import json
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import time

from src.utils import get_pipeline_logger, get_param

logger = get_pipeline_logger("drift_detector")


def compute_ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov test between reference and current distributions.
    
    Args:
        reference: Reference distribution
        current: Current distribution
        
    Returns:
        Tuple of (ks_statistic, p_value)
    """
    # Handle empty arrays
    if len(reference) == 0 or len(current) == 0:
        return 1.0, 0.0
    
    # Compute KS test
    ks_stat, p_value = stats.ks_2samp(reference, current)
    
    return ks_stat, p_value


def compute_wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Compute Wasserstein distance between reference and current distributions.
    
    Args:
        reference: Reference distribution
        current: Current distribution
        
    Returns:
        Wasserstein distance
    """
    # Handle empty arrays
    if len(reference) == 0 or len(current) == 0:
        return 1.0
    
    # Compute Wasserstein distance
    distance = stats.wasserstein_distance(reference, current)
    
    return distance


def detect_drift(reference_data: pd.DataFrame, 
                current_data: pd.DataFrame, 
                monitor_features: List[str],
                metric: str = 'ks_test',
                threshold: float = 0.05) -> Dict[str, Any]:
    """
    Detect drift between reference and current distributions.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        monitor_features: List of feature names to monitor for drift
        metric: Metric to use for drift detection ('ks_test', 'wasserstein')
        threshold: Threshold for drift detection
        
    Returns:
        Dictionary with drift detection results
    """
    drift_results = {}
    drift_detected = False
    
    for feature in monitor_features:
        if feature in reference_data.columns and feature in current_data.columns:
            reference_values = reference_data[feature].dropna().values
            current_values = current_data[feature].dropna().values
            
            if metric == 'ks_test':
                stat, p_value = compute_ks_test(reference_values, current_values)
                feature_drift = p_value < threshold
                drift_results[feature] = {
                    'metric': 'ks_test',
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'threshold': threshold,
                    'drift_detected': feature_drift
                }
            elif metric == 'wasserstein':
                distance = compute_wasserstein_distance(reference_values, current_values)
                feature_drift = distance > threshold
                drift_results[feature] = {
                    'metric': 'wasserstein',
                    'distance': float(distance),
                    'threshold': threshold,
                    'drift_detected': feature_drift
                }
            else:
                raise ValueError(f"Unsupported drift metric: {metric}")
            
            if feature_drift:
                drift_detected = True
        else:
            logger.warning(f"Feature {feature} not found in both datasets")
            drift_results[feature] = {
                'error': 'Feature not found in both datasets'
            }
    
    summary = {
        'drift_detected': drift_detected,
        'metric': metric,
        'threshold': threshold,
        'features': drift_results
    }
    
    return summary


def save_drift_visualization(reference_data: pd.DataFrame, 
                           current_data: pd.DataFrame,
                           monitor_features: List[str],
                           output_path: str,
                           timestamp: Optional[int] = None) -> None:
    """
    Save visualizations of distributions for drift detection.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        monitor_features: List of feature names to visualize
        output_path: Directory to save visualizations
        timestamp: Optional timestamp for the filename
    """
    os.makedirs(output_path, exist_ok=True)
    
    if timestamp is None:
        timestamp = int(time.time())
    
    for feature in monitor_features:
        if feature in reference_data.columns and feature in current_data.columns:
            # Create a histogram plot for the feature
            plt.figure(figsize=(10, 6))
            
            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            plt.hist(ref_values, bins=30, alpha=0.5, label='Reference', density=True)
            plt.hist(curr_values, bins=30, alpha=0.5, label='Current', density=True)
            
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.title(f'Distribution Comparison for {feature}')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save the plot
            plt_path = os.path.join(output_path, f"{feature}_drift_{timestamp}.png")
            plt.savefig(plt_path)
            plt.close()
            
            logger.info(f"Saved drift visualization for {feature} to {plt_path}")


def create_reference_dataset(aligned_features_path: str, 
                           output_path: Optional[str] = None,
                           monitor_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create a reference dataset for drift detection.
    
    Args:
        aligned_features_path: Path to aligned features parquet file
        output_path: Path to save reference dataset
        monitor_features: List of feature names to include in reference dataset
        
    Returns:
        Reference dataset
    """
    # Load aligned features
    logger.info(f"Loading aligned features from {aligned_features_path}")
    aligned_df = pd.read_parquet(aligned_features_path)
    
    # Get default parameters
    if output_path is None:
        output_path = get_param('drift.reference_path')
    
    if monitor_features is None:
        monitor_features = get_param('drift.monitor_features')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Extract relevant features
    ref_features = []
    for col in aligned_df.columns:
        # Include only the monitored features and timestamp
        if col == 'timestamp' or any(f in col for f in monitor_features):
            ref_features.append(col)
    
    # Create reference dataset
    reference_df = aligned_df[ref_features]
    
    # Save reference dataset
    reference_path = os.path.join(output_path, "reference_dataset.parquet")
    reference_df.to_parquet(reference_path, index=False)
    logger.info(f"Saved reference dataset to {reference_path}")
    
    # Save reference summary
    summary = {
        'timestamp': int(time.time()),
        'num_samples': len(reference_df),
        'features': ref_features,
        'monitor_features': monitor_features
    }
    
    summary_path = os.path.join(output_path, "reference_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return reference_df


def inject_drift(imu_data_path: str, 
               output_path: Optional[str] = None,
               noise_scale: float = 0.5,
               frequency_change: float = 0.2) -> pd.DataFrame:
    """
    Inject artificial drift into IMU data for testing.
    
    Args:
        imu_data_path: Path to IMU data CSV
        output_path: Path to save drifted data
        noise_scale: Scale of noise to inject (0-1)
        frequency_change: Scale of frequency change to inject (0-1)
        
    Returns:
        DataFrame with drifted IMU data
    """
    # Load IMU data
    logger.info(f"Loading IMU data from {imu_data_path}")
    imu_df = pd.read_csv(imu_data_path)
    
    # Get default output path if not provided
    if output_path is None:
        output_path = os.path.dirname(imu_data_path)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create a copy of the data
    drifted_df = imu_df.copy()
    
    # Define INS channels to add drift to (the first edition listed accelerometer and gyroscope
    # channels, which the INS data doesn't have, so no noise was added)
    imu_channels = ['velocity_north', 'velocity_east', 'velocity_down',
                   'roll', 'pitch', 'yaw']
    
    # Inject noise
    for channel in imu_channels:
        if channel in drifted_df.columns:
            # Calculate noise based on the standard deviation of the channel
            channel_std = drifted_df[channel].std()
            noise = np.random.normal(0, channel_std * noise_scale, len(drifted_df))
            
            # Add noise to the channel
            drifted_df[channel] = drifted_df[channel] + noise
    
    # Inject frequency drift by resampling a portion of the data
    if 'timestamp' in drifted_df.columns and frequency_change > 0:
        # Convert timestamp to datetime
        drifted_df['datetime'] = pd.to_datetime(drifted_df['timestamp'], unit='us')
        
        # Determine the number of rows to affect with frequency change
        n_rows = int(len(drifted_df) * frequency_change)
        
        # Sort by timestamp
        drifted_df = drifted_df.sort_values('datetime')
        
        # Change the sampling frequency for a portion of the data
        # We'll select the middle portion of the data to simulate drift over time
        start_idx = (len(drifted_df) - n_rows) // 2
        end_idx = start_idx + n_rows
        
        # Extract the portion to resample
        portion = drifted_df.iloc[start_idx:end_idx].copy()
        
        # Resample to a different frequency
        # First, set datetime as index
        portion.set_index('datetime', inplace=True)
        
        # Resample to a faster frequency
        portion = portion.resample('800ms').mean().reset_index()
        
        # Convert back to microsecond timestamps
        portion['timestamp'] = portion['datetime'].dt.as_unit('us').astype(np.int64)
        
        # Remove the portion from the original dataframe
        drifted_df = pd.concat([drifted_df.iloc[:start_idx], drifted_df.iloc[end_idx:]])
        
        # Add back the resampled portion
        drifted_df = pd.concat([drifted_df, portion], ignore_index=True)
        
        # Sort by timestamp
        drifted_df = drifted_df.sort_values('timestamp')
    
    # Save drifted data
    drifted_path = os.path.join(output_path, "imu_drifted.csv")
    drifted_df.to_csv(drifted_path, index=False)
    logger.info(f"Saved drifted IMU data to {drifted_path}")
    
    # Save drift summary
    summary = {
        'timestamp': int(time.time()),
        'original_data': imu_data_path,
        'noise_scale': noise_scale,
        'frequency_change': frequency_change,
        'num_samples': len(drifted_df),
        'affected_channels': imu_channels
    }
    
    summary_path = os.path.join(output_path, "drift_injection_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return drifted_df


def check_for_drift(reference_path: Optional[str] = None,
                   current_data_path: Optional[str] = None,
                   output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Check for drift between reference and current data.
    
    Args:
        reference_path: Path to reference dataset parquet file
        current_data_path: Path to current dataset parquet file
        output_path: Path to save drift detection results
        
    Returns:
        Dictionary with drift detection results
    """
    # Get default paths if not provided
    if reference_path is None:
        reference_path = os.path.join(get_param('drift.reference_path'), "reference_dataset.parquet")
    
    if current_data_path is None:
        current_data_path = os.path.join(get_param('alignment.output_path'), "aligned_features.parquet")
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(reference_path), "drift_results")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load datasets
    logger.info(f"Loading reference dataset from {reference_path}")
    reference_df = pd.read_parquet(reference_path)
    
    logger.info(f"Loading current dataset from {current_data_path}")
    current_df = pd.read_parquet(current_data_path)
    
    # Get drift parameters
    monitor_features = get_param('drift.monitor_features')
    threshold = get_param('drift.threshold')
    metric = get_param('drift.metric')
    
    # Detect drift
    drift_results = detect_drift(reference_df, current_df, monitor_features, metric, threshold)
    
    # Save drift results
    timestamp = int(time.time())
    results_path = os.path.join(output_path, f"drift_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(drift_results, f, indent=2)
    
    logger.info(f"Saved drift detection results to {results_path}")
    
    # Save visualizations
    save_drift_visualization(reference_df, current_df, monitor_features, output_path, timestamp)
    
    # Log drift detection results
    if drift_results['drift_detected']:
        logger.warning("Drift detected in the following features:")
        for feature, result in drift_results['features'].items():
            if result.get('drift_detected', False):
                logger.warning(f"  - {feature}: {result}")
    else:
        logger.info("No drift detected")
    
    return drift_results


if __name__ == "__main__":
    # Create reference dataset
    aligned_features_path = os.path.join(get_param('alignment.output_path'), "aligned_features.parquet")
    create_reference_dataset(aligned_features_path)
    
    # Inject drift for testing
    imu_data_path = os.path.join(get_param('ingest.imu.output_path'), "imu_processed.csv")
    inject_drift(imu_data_path)
    
    # Check for drift
    check_for_drift() 