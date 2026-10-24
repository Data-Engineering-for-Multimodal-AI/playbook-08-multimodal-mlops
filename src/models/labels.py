"""Labels for the INS windows.

The default label, 'turn_next_1s', asks whether the car turns in the second after a window: the
recorded heading (yaw) changes faster than a threshold. It is computed from readings after the
window, so the model has to anticipate it from the motion so far.

'east_velocity' is the first edition's synthetic label (velocity_east_mean > 7 m/s). It is one of
the model's own inputs, so any model learns it almost perfectly.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_param


def heading_change_rate(ins_df: pd.DataFrame, timestamps: np.ndarray, horizon_sec: float) -> np.ndarray:
    """Absolute heading change per second over [t, t + horizon_sec] for each timestamp (microseconds)."""
    ins_df = ins_df.dropna(subset=['yaw']).sort_values('timestamp')
    times = ins_df['timestamp'].to_numpy()
    yaw = np.unwrap(ins_df['yaw'].to_numpy())   # heading wraps at 2*pi
    start = np.clip(np.searchsorted(times, timestamps), 0, len(times) - 1)
    end = np.clip(np.searchsorted(times, timestamps + horizon_sec * 1e6), 0, len(times) - 1)
    elapsed = np.maximum(times[end] - times[start], 1) / 1e6
    return np.abs(yaw[end] - yaw[start]) / elapsed


def make_labels(windows: pd.DataFrame, ins_df: Optional[pd.DataFrame] = None,
                name: Optional[str] = None) -> np.ndarray:
    """Labels (0/1) for INS feature windows (a dataframe with 'timestamp' and the window features)."""
    name = name or get_param('label.name', 'turn_next_1s')
    if name == 'turn_next_1s':
        if ins_df is None:
            raise ValueError("the 'turn_next_1s' label needs the INS readings")
        rate = heading_change_rate(ins_df, windows['timestamp'].to_numpy(),
                                   get_param('label.horizon_sec', 1.0))
        return (rate > get_param('label.yaw_rate_threshold', 0.1)).astype(int)
    if name == 'east_velocity':
        return (windows['velocity_east_mean'] > 7.0).astype(int).to_numpy()
    raise ValueError(f"Unknown label: {name}")
