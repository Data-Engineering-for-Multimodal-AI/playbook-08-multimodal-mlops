import os
from typing import Optional
import pandas as pd
import warnings

from src.utils import get_pipeline_logger, get_param
from src.features.ins_features import process_ins_features

logger = get_pipeline_logger("imu_features")


def process_imu_features(imu_data_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Legacy wrapper for process_ins_features. 
    This function is kept for backward compatibility.
    
    Args:
        imu_data_path: Path to processed INS data CSV
        output_path: Path to save extracted features
        
    Returns:
        DataFrame with extracted features
    """
    warnings.warn(
        "process_imu_features is deprecated and will be removed in a future version. "
        "Use process_ins_features instead.",
        DeprecationWarning, stacklevel=2
    )
    
    logger.info("Using INS feature extraction instead of IMU (legacy wrapper)")
    
    # Check if we have ins_processed.csv, use it in preference to imu_processed.csv
    ins_path = os.path.join(os.path.dirname(imu_data_path), "ins_processed.csv")
    if os.path.exists(ins_path):
        logger.info(f"Using INS data from {ins_path}")
        return process_ins_features(ins_path, output_path)
    else:
        logger.info(f"Using data from {imu_data_path}")
        return process_ins_features(imu_data_path, output_path)


if __name__ == "__main__":
    imu_data_path = os.path.join(get_param('ingest.imu.output_path'), "imu_processed.csv")
    process_imu_features(imu_data_path) 