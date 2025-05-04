import os
import time
import json
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import shutil

from src.utils import get_pipeline_logger, get_param
from src.ingest import ingest_video_frames, ingest_imu_data
from src.features import process_video_features, process_imu_features
from src.align import align_features, prepare_ml_dataset
from src.models import train_model, batch_predict
from src.drift import check_for_drift, create_reference_dataset, inject_drift

logger = get_pipeline_logger("pipeline")


def run_ingest_stage() -> Tuple[str, str]:
    """
    Run the data ingestion stage.
    
    Returns:
        Tuple of (video_metadata_path, imu_data_path)
    """
    logger.info("Starting data ingestion stage")
    
    # Ingest video frames
    video_df = ingest_video_frames()
    video_metadata_path = os.path.join(get_param('ingest.video.output_path'), "video_metadata.csv")
    
    # Ingest IMU data
    imu_df = ingest_imu_data()
    imu_data_path = os.path.join(get_param('ingest.imu.output_path'), "imu_processed.csv")
    
    logger.info("Data ingestion stage completed")
    return video_metadata_path, imu_data_path


def run_feature_extraction_stage(video_metadata_path: str, imu_data_path: str) -> Tuple[str, str]:
    """
    Run the feature extraction stage.
    
    Args:
        video_metadata_path: Path to video metadata CSV
        imu_data_path: Path to processed IMU data
        
    Returns:
        Tuple of (video_features_path, imu_features_path)
    """
    logger.info("Starting feature extraction stage")
    
    # Extract video features
    video_features_df = process_video_features(video_metadata_path)
    video_features_path = os.path.join(get_param('features.output_path'), 'video', 'video_features.parquet')
    
    # Extract IMU features
    imu_features_df = process_imu_features(imu_data_path)
    imu_features_path = os.path.join(get_param('features.output_path'), 'imu', 'imu_features.parquet')
    
    logger.info("Feature extraction stage completed")
    return video_features_path, imu_features_path


def run_alignment_stage(video_features_path: str, imu_features_path: str) -> Tuple[str, str, str]:
    """
    Run the temporal alignment stage.
    
    Args:
        video_features_path: Path to video features
        imu_features_path: Path to IMU features
        
    Returns:
        Tuple of (aligned_features_path, ml_features_path, ml_labels_path)
    """
    logger.info("Starting temporal alignment stage")
    
    # Align modalities
    aligned_df = align_features(video_features_path, imu_features_path)
    aligned_features_path = os.path.join(get_param('alignment.output_path'), 'aligned_features.parquet')
    
    # Prepare ML dataset
    features_df, labels_df = prepare_ml_dataset(aligned_features_path)
    ml_features_path = os.path.join(get_param('alignment.output_path'), 'ml_dataset', 'features.parquet')
    ml_labels_path = os.path.join(get_param('alignment.output_path'), 'ml_dataset', 'labels.parquet')
    
    logger.info("Temporal alignment stage completed")
    return aligned_features_path, ml_features_path, ml_labels_path


def run_training_stage(ml_features_path: str, ml_labels_path: str) -> str:
    """
    Run the model training stage.
    
    Args:
        ml_features_path: Path to ML features
        ml_labels_path: Path to ML labels
        
    Returns:
        Path to trained model
    """
    logger.info("Starting model training stage")
    
    # Train model
    training_results = train_model(ml_features_path, ml_labels_path)
    model_path = os.path.join(get_param('model.output_path'), 'model.pkl')
    
    logger.info(f"Model training completed. Test accuracy: {training_results['test_metrics']['accuracy']:.4f}")
    return model_path


def run_drift_detection_stage(aligned_features_path: str, imu_data_path: str) -> Dict[str, Any]:
    """
    Run the drift detection stage.
    
    Args:
        aligned_features_path: Path to aligned features
        imu_data_path: Path to IMU data for drift injection
        
    Returns:
        Dictionary with drift detection results
    """
    logger.info("Starting drift detection stage")
    
    # Create reference dataset
    reference_df = create_reference_dataset(aligned_features_path)
    reference_path = os.path.join(get_param('drift.reference_path'), 'reference_dataset.parquet')
    
    # Inject drift for testing
    drifted_df = inject_drift(imu_data_path)
    drifted_path = os.path.join(os.path.dirname(imu_data_path), 'imu_drifted.csv')
    
    # Process the drifted data
    logger.info("Processing drifted IMU data through the pipeline")
    
    # Create temporary directories for drifted data processing
    temp_dir = os.path.join('experiments', 'drift_test')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process drifted IMU features
    drifted_imu_features_path = os.path.join(temp_dir, 'imu_features.parquet')
    drifted_imu_features = process_imu_features(drifted_path, output_path=os.path.dirname(drifted_imu_features_path))
    
    # Load original video features (reused since we're only testing IMU drift)
    video_features_path = os.path.join(get_param('features.output_path'), 'video', 'video_features.parquet')
    
    # Align drifted data
    drifted_aligned_path = os.path.join(temp_dir, 'aligned_features.parquet')
    drifted_aligned = align_features(
        video_features_path, 
        drifted_imu_features_path,
        output_path=os.path.dirname(drifted_aligned_path)
    )
    
    # Check for drift
    drift_results = check_for_drift(reference_path, drifted_aligned_path)
    
    # Log drift detection results
    if drift_results['drift_detected']:
        logger.warning("Drift detected in the IMU data. Retraining may be needed.")
    else:
        logger.info("No significant drift detected in the IMU data.")
    
    logger.info("Drift detection stage completed")
    return drift_results


def run_retraining_if_needed(drift_results: Dict[str, Any],
                           aligned_features_path: str,
                           ml_features_path: str,
                           ml_labels_path: str) -> Optional[str]:
    """
    Run model retraining if drift is detected.
    
    Args:
        drift_results: Results from drift detection
        aligned_features_path: Path to aligned features
        ml_features_path: Path to ML features
        ml_labels_path: Path to ML labels
        
    Returns:
        Path to retrained model if retraining was performed, None otherwise
    """
    if drift_results['drift_detected']:
        logger.info("Drift detected, starting model retraining")
        
        # Create a new experiment directory for retraining
        retrain_dir = os.path.join('experiments', 'retraining', f"retrain_{int(time.time())}")
        os.makedirs(retrain_dir, exist_ok=True)
        
        # Copy the ML dataset to the retraining directory
        retrain_features_path = os.path.join(retrain_dir, 'features.parquet')
        retrain_labels_path = os.path.join(retrain_dir, 'labels.parquet')
        
        shutil.copy(ml_features_path, retrain_features_path)
        shutil.copy(ml_labels_path, retrain_labels_path)
        
        # Retrain the model
        retrain_model_dir = os.path.join(retrain_dir, 'model')
        os.makedirs(retrain_model_dir, exist_ok=True)
        
        training_results = train_model(
            retrain_features_path, 
            retrain_labels_path,
            output_path=retrain_model_dir,
            experiment_name=f"retraining_{int(time.time())}"
        )
        
        retrain_model_path = os.path.join(retrain_model_dir, 'model.pkl')
        
        logger.info(f"Model retraining completed. Test accuracy: {training_results['test_metrics']['accuracy']:.4f}")
        return retrain_model_path
    else:
        logger.info("No drift detected, skipping retraining")
        return None


def run_pipeline(simulate_drift: bool = True) -> Dict[str, Any]:
    """
    Run the complete data pipeline.
    
    Args:
        simulate_drift: Whether to simulate drift for testing
        
    Returns:
        Dictionary with pipeline results
    """
    start_time = time.time()
    logger.info("Starting data pipeline")
    
    try:
        # 1. Data Ingestion
        video_metadata_path, imu_data_path = run_ingest_stage()
        
        # 2. Feature Extraction
        video_features_path, imu_features_path = run_feature_extraction_stage(
            video_metadata_path, imu_data_path
        )
        
        # 3. Temporal Alignment
        aligned_features_path, ml_features_path, ml_labels_path = run_alignment_stage(
            video_features_path, imu_features_path
        )
        
        # 4. Model Training
        model_path = run_training_stage(ml_features_path, ml_labels_path)
        
        # 5. Drift Detection
        if simulate_drift:
            drift_results = run_drift_detection_stage(aligned_features_path, imu_data_path)
        else:
            # Create reference dataset without simulating drift
            reference_df = create_reference_dataset(aligned_features_path)
            drift_results = {'drift_detected': False, 'message': 'Drift simulation disabled'}
        
        # 6. Retraining if needed
        if drift_results.get('drift_detected', False):
            retrained_model_path = run_retraining_if_needed(
                drift_results, aligned_features_path, ml_features_path, ml_labels_path
            )
        else:
            retrained_model_path = None
        
        # Create pipeline summary
        pipeline_summary = {
            'status': 'success',
            'start_time': int(start_time * 1000),
            'end_time': int(time.time() * 1000),
            'duration_seconds': time.time() - start_time,
            'stages': {
                'ingest': {
                    'video_metadata_path': video_metadata_path,
                    'imu_data_path': imu_data_path
                },
                'features': {
                    'video_features_path': video_features_path,
                    'imu_features_path': imu_features_path
                },
                'alignment': {
                    'aligned_features_path': aligned_features_path,
                    'ml_features_path': ml_features_path,
                    'ml_labels_path': ml_labels_path
                },
                'training': {
                    'model_path': model_path
                },
                'drift_detection': drift_results,
                'retraining': {
                    'triggered': drift_results.get('drift_detected', False),
                    'retrained_model_path': retrained_model_path
                }
            }
        }
        
        # Save pipeline summary
        pipeline_summary_path = os.path.join('experiments', f"pipeline_summary_{int(time.time())}.json")
        os.makedirs(os.path.dirname(pipeline_summary_path), exist_ok=True)
        
        with open(pipeline_summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2)
        
        logger.info(f"Pipeline completed successfully in {pipeline_summary['duration_seconds']:.2f} seconds")
        logger.info(f"Pipeline summary saved to {pipeline_summary_path}")
        
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        
        # Create failure summary
        failure_summary = {
            'status': 'failed',
            'start_time': int(start_time * 1000),
            'end_time': int(time.time() * 1000),
            'duration_seconds': time.time() - start_time,
            'error': str(e)
        }
        
        # Save failure summary
        failure_path = os.path.join('experiments', f"pipeline_failure_{int(time.time())}.json")
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        
        with open(failure_path, 'w') as f:
            json.dump(failure_summary, f, indent=2)
        
        logger.info(f"Pipeline failure summary saved to {failure_path}")
        
        return failure_summary


if __name__ == "__main__":
    run_pipeline(simulate_drift=True) 