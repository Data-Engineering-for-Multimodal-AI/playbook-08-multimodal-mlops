import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import json

from src.utils import get_pipeline_logger, get_param, load_schema

logger = get_pipeline_logger("temporal_alignment")


def load_feature_data(video_features_path: str, imu_features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load video and IMU feature data.
    
    Args:
        video_features_path: Path to video features parquet file
        imu_features_path: Path to IMU features parquet file
        
    Returns:
        Tuple of (video_features_df, imu_features_df)
    """
    logger.info(f"Loading video features from {video_features_path}")
    video_df = pd.read_parquet(video_features_path)
    
    logger.info(f"Loading IMU features from {imu_features_path}")
    imu_df = pd.read_parquet(imu_features_path)
    
    return video_df, imu_df


def find_nearest_frame(timestamp: int, video_df: pd.DataFrame, 
                       max_time_diff: Optional[float] = None) -> Optional[pd.Series]:
    """
    Find the nearest video frame to the given timestamp.
    
    Args:
        timestamp: Timestamp to find the nearest frame for (microseconds)
        video_df: DataFrame containing video features and timestamps
        max_time_diff: Maximum allowed time difference in seconds
        
    Returns:
        Pandas Series containing the nearest frame data, or None if no match found
    """
    if video_df.empty:
        return None
    
    # Find the index of the frame with the closest timestamp. idxmin returns an index label, so
    # look it up with .loc (the first edition used .iloc, which picks the wrong frame once the
    # dataframe has been re-sorted, as align_modalities does)
    closest_idx = (video_df['timestamp'] - timestamp).abs().idxmin()
    closest_frame = video_df.loc[closest_idx]
    
    # Check if the time difference is within the allowed maximum
    if max_time_diff is not None:
        time_diff_sec = abs(closest_frame['timestamp'] - timestamp) / 1e6
        if time_diff_sec > max_time_diff:
            return None
    
    return closest_frame


def align_modalities(video_df: pd.DataFrame, imu_df: pd.DataFrame, 
                    max_time_diff: float = 0.1,
                    strategy: str = 'nearest') -> pd.DataFrame:
    """
    Align video and IMU data based on timestamps.
    
    Args:
        video_df: DataFrame containing video features
        imu_df: DataFrame containing IMU features
        max_time_diff: Maximum allowed time difference in seconds
        strategy: Alignment strategy ('nearest', 'interpolate', 'average')
        
    Returns:
        DataFrame with aligned multimodal features
    """
    if video_df.empty or imu_df.empty:
        logger.warning("Empty dataframes, cannot perform alignment")
        return pd.DataFrame()
    
    aligned_data = []
    
    # Ensure dataframes are sorted by timestamp
    video_df = video_df.sort_values('timestamp')
    imu_df = imu_df.sort_values('timestamp')
    
    # Only windows within max_time_diff of the camera's time range can match a frame (the sample
    # has 3.6 s of frames in a 42-minute drive; checking every window took minutes)
    if max_time_diff is not None:
        margin = max_time_diff * 1e6
        imu_df = imu_df[(imu_df['timestamp'] >= video_df['timestamp'].min() - margin) &
                        (imu_df['timestamp'] <= video_df['timestamp'].max() + margin)]
    
    logger.info(f"Aligning {len(imu_df)} IMU windows with {len(video_df)} video frames")
    
    # For each IMU window, find matching video frame
    for idx, imu_row in imu_df.iterrows():
        if idx % 2000 == 0:
            logger.info(f"Processing alignment for IMU window {idx}/{len(imu_df)}")
            
        try:
            imu_timestamp = imu_row['timestamp']
            
            # Find the nearest video frame
            nearest_frame = find_nearest_frame(imu_timestamp, video_df, max_time_diff)
            
            if nearest_frame is None:
                logger.debug(f"No matching video frame found for IMU timestamp {imu_timestamp}")
                continue
                
            # Create aligned feature record
            aligned_record = {
                'timestamp': imu_timestamp,
                'video_timestamp': nearest_frame['timestamp'],
                'time_diff_us': abs(nearest_frame['timestamp'] - imu_timestamp),
                'time_diff_sec': abs(nearest_frame['timestamp'] - imu_timestamp) / 1e6
            }
            
            # Add IMU features
            for col in imu_row.index:
                if col.endswith('_mean') or col.endswith('_std') or col == 'statistical_features':
                    aligned_record[f'imu_{col}'] = imu_row[col]
            
            # Add video features
            for col in nearest_frame.index:
                if col == 'hist_features' or col == 'embedding_features':
                    aligned_record[f'video_{col}'] = nearest_frame[col]
            
            aligned_data.append(aligned_record)
            
        except Exception as e:
            logger.error(f"Error aligning data at index {idx}: {str(e)}")
    
    aligned_df = pd.DataFrame(aligned_data)
    logger.info(f"Created {len(aligned_df)} aligned records")
    
    return aligned_df


def align_features(video_features_path: Optional[str] = None, 
                  imu_features_path: Optional[str] = None,
                  output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Perform temporal alignment of video and IMU features.
    
    Args:
        video_features_path: Path to video features parquet file
        imu_features_path: Path to IMU features parquet file
        output_path: Path to save aligned features
        
    Returns:
        DataFrame with aligned features
    """
    # Get default paths if not provided
    if video_features_path is None:
        video_features_path = os.path.join(get_param('features.output_path'), 'video', 'video_features.parquet')
    
    if imu_features_path is None:
        imu_features_path = os.path.join(get_param('features.output_path'), 'imu', 'imu_features.parquet')
        if not os.path.exists(imu_features_path):
            # Written by `python -m src.features.extract_ins_features`
            imu_features_path = os.path.join(get_param('features.output_path'), 'ins', 'ins_features.parquet')
    
    if output_path is None:
        output_path = get_param('alignment.output_path')
    
    # Get alignment parameters
    window_size = get_param('alignment.window_size')
    max_time_diff = get_param('alignment.max_time_diff')
    strategy = get_param('alignment.strategy')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load feature data
    video_df, imu_df = load_feature_data(video_features_path, imu_features_path)
    
    # Align modalities
    aligned_df = align_modalities(video_df, imu_df, max_time_diff, strategy)
    
    # Save aligned features
    if not aligned_df.empty:
        aligned_path = os.path.join(output_path, "aligned_features.parquet")
        aligned_df.to_parquet(aligned_path, index=False)
        logger.info(f"Saved aligned features to {aligned_path}")
        
        # Save alignment summary
        summary = {
            'total_aligned_samples': len(aligned_df),
            'alignment_strategy': strategy,
            'max_time_diff_sec': max_time_diff,
            'window_size_sec': window_size,
            'start_timestamp': int(aligned_df['timestamp'].min()),
            'end_timestamp': int(aligned_df['timestamp'].max()),
            'avg_time_diff_sec': float(aligned_df['time_diff_sec'].mean())
        }
        
        summary_path = os.path.join(output_path, "alignment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        logger.warning("No aligned samples produced")
        
        # Save empty summary
        summary = {
            'total_aligned_samples': 0,
            'error': 'No aligned samples created',
            'alignment_strategy': strategy,
            'max_time_diff_sec': max_time_diff,
            'window_size_sec': window_size
        }
        
        summary_path = os.path.join(output_path, "alignment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    logger.info(f"Alignment complete. Created {len(aligned_df)} aligned samples")
    return aligned_df


def prepare_ml_dataset(aligned_features_path: Optional[str] = None, 
                      output_path: Optional[str] = None,
                      add_synthetic_label: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare ML dataset from aligned features.
    
    Args:
        aligned_features_path: Path to aligned features parquet file
        output_path: Path to save ML dataset
        add_synthetic_label: Whether to add a synthetic label for demonstration
        
    Returns:
        Tuple of (features_df, labels_df)
    """
    # Get default paths if not provided
    if aligned_features_path is None:
        aligned_features_path = os.path.join(get_param('alignment.output_path'), 'aligned_features.parquet')
    
    if output_path is None:
        # The training dataset (ml_dataset) is now built from every INS window by
        # src.models.dataset; this multimodal one has only the windows with a camera frame
        output_path = os.path.join(get_param('alignment.output_path'), 'multimodal_dataset')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load aligned features
    logger.info(f"Loading aligned features from {aligned_features_path}")
    aligned_df = pd.read_parquet(aligned_features_path)
    
    if aligned_df.empty:
        logger.warning("Empty aligned features dataframe")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create feature matrix by exploding the feature vectors
    features = []
    labels = []
    
    for idx, row in aligned_df.iterrows():
        # Extract IMU statistical features
        imu_feats = np.array(row['imu_statistical_features'])
        
        # Extract video features (either histogram, embeddings, or both)
        if 'video_hist_features' in row:
            video_hist = np.array(row['video_hist_features'])
        else:
            video_hist = np.array([])
            
        if 'video_embedding_features' in row:
            video_embed = np.array(row['video_embedding_features'])
        else:
            video_embed = np.array([])
        
        # Combine features
        combined_features = np.concatenate([
            imu_feats, 
            video_hist if video_hist.size > 0 else np.array([]), 
            video_embed if video_embed.size > 0 else np.array([])
        ])
        
        # Create feature record
        feature_record = {
            'timestamp': row['timestamp'],
            'features': combined_features.tolist()
        }
        
        # Add individual feature columns for easier access
        # This is useful for interpretability and drift detection
        imu_columns = ['acceleration_x_mean', 'acceleration_y_mean', 'acceleration_z_mean',
                      'gyroscope_x_mean', 'gyroscope_y_mean', 'gyroscope_z_mean']
        
        for col in imu_columns:
            imu_col = f'imu_{col}'
            if imu_col in row:
                feature_record[col] = row[imu_col]
        
        features.append(feature_record)
        
        # Create synthetic label if requested
        # In a real-world scenario, you would use actual labels
        if add_synthetic_label:
            # Create synthetic labels based on vehicle movement patterns
            # Extract velocity components
            v_north = row.get('imu_velocity_north_mean', 0)
            v_east = row.get('imu_velocity_east_mean', 0)
            v_down = row.get('imu_velocity_down_mean', 0)
            
            # Get orientation values
            roll = row.get('imu_roll_mean', 0)
            pitch = row.get('imu_pitch_mean', 0)
            yaw = row.get('imu_yaw_mean', 0)
            
            # Calculate horizontal velocity
            velocity_horizontal = np.sqrt(v_north**2 + v_east**2)
            
            # Based on the actual data distribution we observed, we need to lower the thresholds
            # Detect if the vehicle is moving in a straight line with higher-than-median speed
            # This is a simplified binary classification task relevant to autonomous driving
            if v_east > 7.0:  # Using the median from the data we observed
                label = 1  # Higher east velocity
            else:
                label = 0  # Lower east velocity
                
            label_record = {
                'timestamp': row['timestamp'],
                'label': label
            }
            labels.append(label_record)
    
    # Create DataFrames
    features_df = pd.DataFrame(features)
    
    if add_synthetic_label:
        labels_df = pd.DataFrame(labels)
    else:
        labels_df = pd.DataFrame()
    
    # Save datasets
    features_path = os.path.join(output_path, "features.parquet")
    features_df.to_parquet(features_path, index=False)
    logger.info(f"Saved feature matrix to {features_path}")
    
    if not labels_df.empty:
        labels_path = os.path.join(output_path, "labels.parquet")
        labels_df.to_parquet(labels_path, index=False)
        logger.info(f"Saved labels to {labels_path}")
    
    # Save dataset schema
    feature_columns = [col for col in features_df.columns if col != 'timestamp' and col != 'features']
    
    schema = {
        'features': {
            'timestamp': 'int64',
            'features': f'list[float] (length={len(combined_features)})',
        },
        'feature_columns': feature_columns,
        'labels': {
            'timestamp': 'int64',
            'label': 'int64 (binary classification)'
        } if add_synthetic_label else {},
        'total_samples': len(features_df)
    }
    
    schema_path = os.path.join(output_path, "dataset_schema.json")
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    logger.info(f"Prepared ML dataset with {len(features_df)} samples")
    return features_df, labels_df


if __name__ == "__main__":
    # First, align the features
    aligned_df = align_features()
    
    # Then, prepare ML dataset
    features_df, labels_df = prepare_ml_dataset() 