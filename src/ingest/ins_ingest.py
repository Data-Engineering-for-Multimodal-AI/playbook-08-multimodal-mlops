import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import json

from src.utils import get_pipeline_logger, load_schema, get_param

logger = get_pipeline_logger("ins_ingest")


def load_ins_data(source_path: str) -> pd.DataFrame:
    """
    Load INS (Inertial Navigation System) data from the Oxford RobotCar dataset.
    
    Args:
        source_path: Path to the INS data CSV file
        
    Returns:
        DataFrame containing INS data with parsed timestamps
    """
    logger.info(f"Loading INS data from {source_path}")
    
    try:
        # Read INS data
        # Oxford RobotCar ins.csv file contains:
        # timestamp, ins_status, latitude, longitude, altitude, northing, easting, down,
        # utm_zone, velocity_north, velocity_east, velocity_down, roll, pitch, yaw
        df = pd.read_csv(source_path)
        
        # We expect the first column to be timestamp, so check if it's there
        if 'timestamp' not in df.columns and len(df.columns) > 0:
            df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
        
        # Extract needed columns if they exist in the dataframe
        needed_cols = [
            'timestamp', 
            'velocity_north', 'velocity_east', 'velocity_down',
            'roll', 'pitch', 'yaw'
        ]
        
        # If we don't have these columns but have data, try to extract by position
        if not all(col in df.columns for col in needed_cols) and len(df.columns) >= 15:
            # Map to Oxford RobotCar dataset format
            column_mapping = {
                df.columns[0]: 'timestamp',
                df.columns[9]: 'velocity_north',
                df.columns[10]: 'velocity_east',
                df.columns[11]: 'velocity_down',
                df.columns[12]: 'roll',
                df.columns[13]: 'pitch',
                df.columns[14]: 'yaw'
            }
            df.rename(columns=column_mapping, inplace=True)
        
        # Check if we have the needed columns now
        missing_cols = [col for col in needed_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in INS data: {missing_cols}")
            # Add missing columns with NaN values
            for col in missing_cols:
                df[col] = np.nan
                
        # Select only the columns we need
        df = df[needed_cols]
        
        logger.info(f"Loaded INS data with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading INS data: {str(e)}")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=needed_cols)


def process_ins_data(df: pd.DataFrame, resample_rate: Optional[float] = None) -> pd.DataFrame:
    """
    Process INS data by filtering and resampling.
    
    Args:
        df: DataFrame containing INS data
        resample_rate: Optional resampling rate in Hz
        
    Returns:
        Processed DataFrame
    """
    # Check if DataFrame is empty
    if df.empty:
        logger.warning("Empty INS dataframe, skipping processing")
        return df
    
    # Convert timestamp to datetime for resampling
    df['timestamp_ns'] = pd.to_numeric(df['timestamp'])
    df['datetime'] = pd.to_datetime(df['timestamp_ns'], unit='us')
    df.set_index('datetime', inplace=True)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Resample if needed
    if resample_rate is not None:
        logger.info(f"Resampling INS data to {resample_rate} Hz")
        # Calculate time period in ns from Hz
        period_ns = int(1e9 / resample_rate)
        period = f"{period_ns}ns"  # pandas period string (ns = nanoseconds)
        
        # Resample using mean
        df = df.resample(period).mean()
        
        # Reset index to get datetime as column
        df.reset_index(inplace=True)
        
        # Convert datetime back to timestamp in microseconds. Ask for microseconds explicitly:
        # pandas 3 keeps the resolution of the input (us here), so the first edition's
        # `astype(np.int64) // 1000` (which assumed nanoseconds) turned microseconds into milliseconds
        df['timestamp'] = df['datetime'].dt.as_unit('us').astype(np.int64)
    else:
        # Reset index if we didn't resample
        df.reset_index(inplace=True)
    
    return df


def ingest_ins_data(output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Ingest INS data, process it, and save to the output directory.
    
    Args:
        output_path: Directory to store processed data
        
    Returns:
        DataFrame with processed INS data
    """
    schema = load_schema()
    source_path = schema['imu']['source_path']
    
    # Get parameters
    if output_path is None:
        output_path = get_param('ingest.imu.output_path')
    resample_rate = get_param('ingest.imu.resample_rate')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load INS data
    df = load_ins_data(source_path)
    
    # Process data
    df_processed = process_ins_data(df, resample_rate)
    
    # Save processed data
    processed_path = os.path.join(output_path, "ins_processed.csv")
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Saved processed INS data to {processed_path}")
    
    # Create and save summary
    if not df_processed.empty:
        summary = {
            'total_samples': int(len(df_processed)),
            'start_timestamp': int(df_processed['timestamp'].min()),
            'end_timestamp': int(df_processed['timestamp'].max()),
            'duration_seconds': float((df_processed['timestamp'].max() - df_processed['timestamp'].min()) / 1e6),
            'sampling_rate_hz': float(resample_rate if resample_rate else \
                len(df_processed) / ((df_processed['timestamp'].max() - df_processed['timestamp'].min()) / 1e6))
        }
    else:
        summary = {'total_samples': 0, 'error': 'No INS data available'}
    
    summary_path = os.path.join(output_path, "ins_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Ingestion complete. Processed {len(df_processed)} INS samples")
    
    # For backward compatibility, also create imu_processed.csv as a copy
    imu_path = os.path.join(output_path, "imu_processed.csv")
    df_processed.to_csv(imu_path, index=False)
    logger.info(f"Created backward compatibility file at {imu_path}")
    
    return df_processed


if __name__ == "__main__":
    ingest_ins_data() 