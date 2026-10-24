import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import glob
import json
from datetime import datetime

from src.utils import get_pipeline_logger, load_schema, get_param

logger = get_pipeline_logger("video_ingest")


def extract_timestamp_from_filename(filename: str) -> int:
    """
    Extract timestamp from Oxford RobotCar dataset image filename.
    
    Args:
        filename: Image filename with timestamp (e.g., '1418381798889068.png')
        
    Returns:
        Timestamp in microseconds as integer
    """
    timestamp = os.path.splitext(os.path.basename(filename))[0]
    return int(timestamp)


def list_video_files(source_path: str, file_format: str = 'png') -> List[str]:
    """
    List all video frame files in the source directory.
    
    Args:
        source_path: Directory containing video frames
        file_format: File format extension (e.g., 'png', 'jpg')
        
    Returns:
        List of file paths sorted by timestamp
    """
    pattern = os.path.join(source_path, f"*.{file_format}")
    files = glob.glob(pattern)
    
    # Sort files by their timestamps
    files.sort(key=extract_timestamp_from_filename)
    
    logger.info(f"Found {len(files)} video frames in {source_path}")
    return files


def load_frame(file_path: str, resize_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Load a single video frame and optionally resize it.
    
    Args:
        file_path: Path to the image file
        resize_shape: Optional tuple of (height, width) to resize the image
        
    Returns:
        Image as a numpy array in RGB format
    """
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {file_path}")
        
    if img.ndim == 2:
        # Oxford RobotCar stores raw Bayer sensor images (one channel). Demosaic them as the
        # dataset's SDK does (the stereo camera's pattern is GBRG, which OpenCV calls BayerGR).
        # The first edition read them as grayscale.
        img = cv2.cvtColor(img, cv2.COLOR_BayerGR2RGB)
    else:
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if resize_shape is not None:
        img = cv2.resize(img, (resize_shape[1], resize_shape[0]))
    
    return img


def ingest_video_frames(output_path: str = None) -> pd.DataFrame:
    """
    Ingest video frames from the source directory and create a metadata DataFrame.
    
    Args:
        output_path: Directory to store processed metadata
        
    Returns:
        DataFrame with video frame metadata
    """
    schema = load_schema()
    source_path = schema['video']['source_path']
    file_format = schema['video']['format']
    
    # Get parameters
    if output_path is None:
        output_path = get_param('ingest.video.output_path')
    resize_shape = get_param('ingest.video.resize_shape')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # List and sort video files
    video_files = list_video_files(source_path, file_format)
    
    # Create metadata list
    metadata = []
    for idx, file_path in enumerate(video_files):
        try:
            timestamp = extract_timestamp_from_filename(file_path)
            metadata.append({
                'frame_id': idx,
                'timestamp': timestamp,
                'timestamp_readable': datetime.fromtimestamp(timestamp / 1e6).isoformat(),
                'file_path': file_path,
                'ingested_path': os.path.join(output_path, os.path.basename(file_path))
            })
            
            # Load and save processed frame (resize if needed)
            if idx % 10 == 0:  # Log only every 10 frames
                logger.info(f"Processing frame {idx}/{len(video_files)}: {file_path}")
                
            frame = load_frame(file_path, resize_shape)
            processed_path = os.path.join(output_path, os.path.basename(file_path))
            cv2.imwrite(processed_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            logger.error(f"Error processing frame {file_path}: {str(e)}")
    
    # Create DataFrame
    metadata_df = pd.DataFrame(metadata)
    
    # Save metadata
    metadata_path = os.path.join(output_path, "video_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved video metadata to {metadata_path}")
    
    # Save JSON summary
    summary = {
        'total_frames': int(len(metadata_df)),
        'start_timestamp': int(metadata_df['timestamp'].min()),
        'end_timestamp': int(metadata_df['timestamp'].max()),
        'duration_seconds': float((metadata_df['timestamp'].max() - metadata_df['timestamp'].min()) / 1e6),
        'fps_avg': float(len(metadata_df) / ((metadata_df['timestamp'].max() - metadata_df['timestamp'].min()) / 1e6))
    }
    
    summary_path = os.path.join(output_path, "video_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Ingestion complete. Processed {len(metadata_df)} frames")
    return metadata_df


if __name__ == "__main__":
    ingest_video_frames() 