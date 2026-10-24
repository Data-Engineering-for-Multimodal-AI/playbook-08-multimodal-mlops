import functools
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy import signal
import json

from src.utils import get_pipeline_logger, get_param, load_schema

logger = get_pipeline_logger("ins_features")


def extract_statistical_features(signal_window: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a signal window.
    
    Args:
        signal_window: Window of signal data
        
    Returns:
        Array of statistical features
    """
    # Handle empty or NaN signal windows
    if signal_window.size == 0 or np.isnan(signal_window).all():
        return np.zeros(4)  # Return zeros for mean, std, min, max
    
    # Extract basic statistical features
    mean = np.mean(signal_window)
    std = np.std(signal_window)
    min_val = np.min(signal_window)
    max_val = np.max(signal_window)
    
    return np.array([mean, std, min_val, max_val])


@functools.lru_cache(maxsize=None)
def _hann(length: int) -> np.ndarray:
    return signal.windows.hann(length)


def extract_fft_features(signal_window: np.ndarray, num_components: int = 5) -> np.ndarray:
    """
    Extract frequency domain features from a signal window.
    
    Args:
        signal_window: Window of signal data
        num_components: Number of frequency components to extract
        
    Returns:
        Array of FFT features
    """
    # Handle empty or NaN signal windows
    if signal_window.size == 0 or np.isnan(signal_window).all():
        return np.zeros(num_components)
    
    # Apply Hanning window to reduce spectral leakage (same as scipy.signal.windows.hann)
    windowed_signal = signal_window * _hann(len(signal_window))
    
    # Compute FFT
    fft = np.abs(np.fft.rfft(windowed_signal))
    
    # Get the most significant frequency components
    if len(fft) <= num_components:
        # Pad with zeros if not enough components
        return np.pad(fft, (0, max(0, num_components - len(fft))))
    else:
        # Return the first N components (includes DC and lower frequencies)
        return fft[:num_components]


def create_sliding_windows(df: pd.DataFrame, window_size_sec: float, overlap_sec: float,
                          time_col: str = 'timestamp') -> List[pd.DataFrame]:
    """
    Create sliding windows from a time series dataframe.
    
    Args:
        df: DataFrame with time series data
        window_size_sec: Window size in seconds
        overlap_sec: Overlap between windows in seconds
        time_col: Column containing timestamps in microseconds
        
    Returns:
        List of dataframes, each containing data for one window
    """
    # Ensure the dataframe is sorted by timestamp
    df = df.sort_values(by=time_col)
    
    # Convert time units from seconds to microseconds
    window_size_us = int(window_size_sec * 1e6)
    step_size_us = int((window_size_sec - overlap_sec) * 1e6)
    
    windows = []
    
    if df.empty:
        logger.warning("Empty dataframe, no windows created")
        return windows
    
    # Define the start and end times
    start_time = df[time_col].min()
    end_time = df[time_col].max()
    
    # Create windows. The data is sorted, so each window is a slice found by binary search
    # (the first edition filtered the whole dataframe for every window, which took minutes)
    df = df.reset_index(drop=True)
    times = df[time_col].to_numpy()
    current_start = start_time
    while current_start + window_size_us <= end_time:
        current_end = current_start + window_size_us
        
        # Get data within the current window
        first, last = np.searchsorted(times, [current_start, current_end], side='left')
        
        # Only add window if it has data
        if last > first:
            windows.append(df.iloc[first:last])
            
        # Move to the next window
        current_start += step_size_us
    
    logger.info(f"Created {len(windows)} windows from {len(df)} INS samples")
    return windows


def extract_window_features(window_df: pd.DataFrame, fft_components: int = 5) -> Dict[str, Any]:
    """
    Extract features from a window of INS data.
    
    Args:
        window_df: DataFrame containing INS data for a single window
        fft_components: Number of FFT components to extract
        
    Returns:
        Dictionary of extracted features
    """
    # Extract features for each INS channel
    feature_dict = {}
    
    # Get window timestamp (middle of the window)
    timestamps = window_df['timestamp'].to_numpy()
    window_time = (timestamps.min() + timestamps.max()) // 2
    
    feature_dict['timestamp'] = window_time
    
    # Extract features for each INS channel (velocity and orientation).
    # Work on NumPy arrays: per-column pandas access was most of the run time
    channels = {
        channel: window_df[channel].to_numpy(dtype=float)
        for channel in ['velocity_north', 'velocity_east', 'velocity_down', 'roll', 'pitch', 'yaw']
        if channel in window_df.columns
    }
    
    all_features = []
    
    # Calculate combined velocity magnitude
    if all(col in channels for col in ['velocity_north', 'velocity_east', 'velocity_down']):
        channels['velocity_magnitude'] = np.sqrt(
            channels['velocity_north']**2 + 
            channels['velocity_east']**2 + 
            channels['velocity_down']**2
        )
    
    for channel, channel_data in channels.items():
        # Extract statistical features
        stats = extract_statistical_features(channel_data)
        
        # Extract FFT features
        fft_feats = extract_fft_features(channel_data, fft_components)
        
        # Store features with channel-specific names
        feature_dict[f"{channel}_mean"] = stats[0]
        feature_dict[f"{channel}_std"] = stats[1]
        feature_dict[f"{channel}_min"] = stats[2]
        feature_dict[f"{channel}_max"] = stats[3]
        
        # Combine features for the statistical feature vector
        all_features.extend(stats)
        all_features.extend(fft_feats)
    
    # Store the combined feature vector
    feature_dict['statistical_features'] = all_features
    
    return feature_dict


def process_ins_features(ins_data_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Process INS data to extract features.
    
    Args:
        ins_data_path: Path to processed INS data CSV
        output_path: Path to save extracted features
        
    Returns:
        DataFrame with extracted features
    """
    # Get parameters
    if output_path is None:
        output_path = os.path.join(get_param('features.output_path'), 'imu')
    
    window_size = get_param('features.imu.window_size')
    overlap = get_param('features.imu.overlap')
    fft_components = get_param('features.imu.fft_components')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load INS data
    logger.info(f"Loading INS data from {ins_data_path}")
    ins_df = pd.read_csv(ins_data_path)
    
    if ins_df.empty:
        logger.warning("Empty dataframe, skipping feature extraction")
        return pd.DataFrame()
    
    # Create windows
    windows = create_sliding_windows(ins_df, window_size, overlap)
    
    # Extract features for each window
    features_list = []
    for i, window_df in enumerate(windows):
        if i % 2000 == 0:
            logger.info(f"Processing window {i}/{len(windows)}")
            
        try:
            features = extract_window_features(window_df, fft_components)
            features_list.append(features)
        except Exception as e:
            logger.error(f"Error processing window {i}: {str(e)}")
    
    # Create features DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Save features
    features_path = os.path.join(output_path, "ins_features.parquet")
    features_df.to_parquet(features_path, index=False)
    logger.info(f"Saved INS features to {features_path}")
    
    # For compatibility, also save as imu_features.parquet
    imu_features_path = os.path.join(output_path, "imu_features.parquet")
    features_df.to_parquet(imu_features_path, index=False)
    logger.info(f"Saved INS features to {imu_features_path} (for compatibility)")
    
    # Save summary
    if not features_df.empty:
        # Calculate feature vector size
        ins_channels = [
            'velocity_north', 'velocity_east', 'velocity_down',
            'roll', 'pitch', 'yaw', 'velocity_magnitude'
        ]
        feature_size = len(ins_channels) * (4 + fft_components)  # 4 stats + fft components per channel
        
        summary = {
            'total_windows': len(features_df),
            'window_size_sec': window_size,
            'overlap_sec': overlap,
            'fft_components': fft_components,
            'feature_vector_size': feature_size,
            'start_timestamp': int(features_df['timestamp'].min()),
            'end_timestamp': int(features_df['timestamp'].max()),
            'data_source': 'INS (velocity and orientation)'
        }
    else:
        summary = {'total_windows': 0, 'error': 'No features extracted'}
    
    summary_path = os.path.join(output_path, "ins_features_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save with imu name for compatibility
    imu_summary_path = os.path.join(output_path, "imu_features_summary.json")
    with open(imu_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Feature extraction complete. Processed {len(features_df)} INS windows")
    
    return features_df


if __name__ == "__main__":
    ins_data_path = os.path.join(get_param('ingest.imu.output_path'), "ins_processed.csv")
    process_ins_features(ins_data_path) 