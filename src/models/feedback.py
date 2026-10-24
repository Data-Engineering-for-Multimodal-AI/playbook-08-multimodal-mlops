"""Labeled production windows saved by the monitor (experiments/monitoring/feedback/batch_*.parquet)."""
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_param

FEEDBACK_DIR = os.path.join("experiments", "monitoring", "feedback")


def feedback_batches(feedback_dir: str = FEEDBACK_DIR) -> List[str]:
    return sorted(glob.glob(os.path.join(feedback_dir, "batch_*.parquet")))


def to_arrays(paths: List[str]) -> Optional[Dict[str, Any]]:
    if not paths:
        return None
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    monitor = [c for c in get_param('monitoring.monitor_features', []) if c in df]
    return {'X': np.array(df['features'].tolist(), dtype=float), 'y': df['label'].to_numpy(),
            'monitor': df[['timestamp'] + monitor] if 'timestamp' in df else df[monitor],
            'batches': [os.path.basename(p) for p in paths]}


def split_recent(feedback_dir: str = FEEDBACK_DIR,
                 since: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    (training part, holdout) of the labeled production batches.

    The newest monitoring.holdout_batches batches are held out to compare a retrained model with
    the champion; only batches from `since` onwards (a batch file name) are used for training.
    """
    paths = feedback_batches(feedback_dir)
    holdout = int(get_param('monitoring.holdout_batches', 2))
    if len(paths) <= holdout:
        return None, to_arrays(paths)
    train_paths, holdout_paths = paths[:-holdout], paths[-holdout:]
    if since is not None:
        train_paths = [p for p in train_paths if os.path.basename(p) >= since]
    return to_arrays(train_paths), to_arrays(holdout_paths)
