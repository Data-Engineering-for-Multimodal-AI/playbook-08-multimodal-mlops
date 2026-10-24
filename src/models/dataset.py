"""Build the training dataset from the INS feature windows (and, optionally, the camera features).

Writes the same files the first edition's alignment step wrote, so train.py reads them unchanged:
  experiments/aligned/ml_dataset/features.parquet   timestamp, features (list of floats)
  experiments/aligned/ml_dataset/labels.parquet     timestamp, label

The first edition built the dataset from the windows that have a camera frame within 0.1 s. The
sample has 3 seconds of camera frames in a 42-minute drive, which left 15 samples. This module uses
every INS window and adds the camera features only when model.use_video_features is true.
"""
import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.models.labels import make_labels
from src.utils import get_param, get_pipeline_logger

logger = get_pipeline_logger("dataset")

INS_FEATURES_PATH = os.path.join('experiments', 'features', 'ins', 'ins_features.parquet')
INS_DATA_PATH = os.path.join('experiments', 'ingested', 'imu', 'ins_processed.csv')


def window_matrix(windows: pd.DataFrame) -> np.ndarray:
    """The model's input matrix: one row of INS statistics and FFT features per window."""
    return np.array(windows['statistical_features'].tolist(), dtype=float)


def add_video_features(windows: pd.DataFrame, X: np.ndarray, video_features_path: str) -> np.ndarray:
    """Append camera features to each window: the nearest frame within max_time_diff, else zeros."""
    video = pd.read_parquet(video_features_path).sort_values('timestamp').reset_index(drop=True)
    frames = np.hstack([np.array(video['hist_features'].tolist(), dtype=float),
                        np.array(video['embedding_features'].tolist(), dtype=float)
                        if 'embedding_features' in video else np.zeros((len(video), 0))])
    times = video['timestamp'].to_numpy()
    nearest = np.clip(np.searchsorted(times, windows['timestamp'].to_numpy()), 1, len(times) - 1)
    nearest = np.where(np.abs(times[nearest - 1] - windows['timestamp'].to_numpy())
                       < np.abs(times[nearest] - windows['timestamp'].to_numpy()), nearest - 1, nearest)
    has_camera = np.abs(times[nearest] - windows['timestamp'].to_numpy()) / 1e6 <= get_param('alignment.max_time_diff')
    camera = np.where(has_camera[:, None], frames[nearest], 0.0)
    logger.info(f"Camera features for {int(has_camera.sum())} of {len(windows)} windows")
    return np.hstack([X, camera, has_camera[:, None].astype(float)])


def build_dataset(ins_features_path: str = INS_FEATURES_PATH,
                  ins_data_path: str = INS_DATA_PATH,
                  output_path: Optional[str] = None,
                  video_features_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Features and labels for every INS window, in time order."""
    if output_path is None:
        output_path = os.path.join(get_param('alignment.output_path'), 'ml_dataset')
    os.makedirs(output_path, exist_ok=True)

    windows = pd.read_parquet(ins_features_path)
    # Windows over gaps in the INS log have no readings after resampling
    windows = windows.dropna(subset=['velocity_north_mean']).sort_values('timestamp').reset_index(drop=True)
    labels = make_labels(windows, pd.read_csv(ins_data_path))
    X = window_matrix(windows)

    if get_param('model.use_video_features', False):
        if video_features_path is None:
            video_features_path = os.path.join(get_param('features.output_path'), 'video', 'video_features.parquet')
        X = add_video_features(windows, X, video_features_path)

    features_df = pd.DataFrame({'timestamp': windows['timestamp'], 'features': list(X)})
    labels_df = pd.DataFrame({'timestamp': windows['timestamp'], 'label': labels})
    features_df.to_parquet(os.path.join(output_path, 'features.parquet'), index=False)
    labels_df.to_parquet(os.path.join(output_path, 'labels.parquet'), index=False)

    # The monitored window features, for drift detection against the training data
    monitor = get_param('monitoring.monitor_features', [])
    windows[['timestamp'] + monitor].to_parquet(os.path.join(output_path, 'monitor_features.parquet'), index=False)

    schema = {
        'label': get_param('label.name'),
        'total_samples': len(features_df),
        'positive_rate': float(labels.mean()),
        'feature_length': int(X.shape[1]),
        'video_features': bool(get_param('model.use_video_features', False)),
    }
    with open(os.path.join(output_path, 'dataset_schema.json'), 'w') as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Prepared ML dataset with {len(features_df)} windows, {X.shape[1]} features, "
                f"{labels.mean():.1%} positive ('{schema['label']}')")
    return features_df, labels_df


if __name__ == "__main__":
    build_dataset()
