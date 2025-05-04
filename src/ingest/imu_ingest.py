import os
from typing import Optional
import pandas as pd
import warnings

from src.utils import get_pipeline_logger
from src.ingest.ins_ingest import ingest_ins_data

logger = get_pipeline_logger("imu_ingest")


def ingest_imu_data(output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Legacy wrapper for ingest_ins_data. 
    This function is kept for backward compatibility.
    
    Args:
        output_path: Directory to store processed data
        
    Returns:
        DataFrame with processed INS data
    """
    warnings.warn(
        "ingest_imu_data is deprecated and will be removed in a future version. "
        "Use ingest_ins_data instead.",
        DeprecationWarning, stacklevel=2
    )
    
    logger.info("Using INS data processing instead of IMU (legacy wrapper)")
    return ingest_ins_data(output_path)


if __name__ == "__main__":
    ingest_imu_data() 