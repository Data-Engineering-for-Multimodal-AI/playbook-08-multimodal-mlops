import os
import subprocess
import sys
import time
import json
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import shutil

from src.utils import get_pipeline_logger, get_param
from src.ingest.ins_ingest import ingest_ins_data
from src.features.ins_features import process_ins_features
from src.align.temporal_alignment import align_features
from src.models.dataset import build_dataset
from src.models.train import train_model, chronological_split
from src.drift.drift_detector import check_for_drift, create_reference_dataset, inject_drift

logger = get_pipeline_logger("pipeline")


def run_module(module: str) -> None:
    """
    Run a pipeline step in its own Python process.

    The camera steps load PyTorch; PyTorch and XGBoost in the same process crash on macOS, so
    they run separately from training.
    """
    subprocess.run([sys.executable, "-m", module], check=True)


def run_ingest_stage() -> Tuple[str, str]:
    """
    Run the data ingestion stage.
    
    Returns:
        Tuple of (video_metadata_path, imu_data_path)
    """
    logger.info("Starting data ingestion stage")
    
    # Ingest video frames
    run_module("src.ingest.video_ingest")
    video_metadata_path = os.path.join(get_param('ingest.video.output_path'), "video_metadata.csv")
    
    # Ingest INS data
    ingest_ins_data(get_param('ingest.imu.output_path'))
    imu_data_path = os.path.join(get_param('ingest.imu.output_path'), "ins_processed.csv")
    
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
    
    # Extract video features (reads the metadata written by the ingest stage)
    run_module("src.features.video_features")
    video_features_path = os.path.join(get_param('features.output_path'), 'video', 'video_features.parquet')
    
    # Extract INS features
    imu_output = os.path.join(get_param('features.output_path'), 'ins')
    process_ins_features(imu_data_path, output_path=imu_output)
    imu_features_path = os.path.join(imu_output, 'ins_features.parquet')
    
    logger.info("Feature extraction stage completed")
    return video_features_path, imu_features_path


def run_alignment_stage(video_features_path: str, imu_features_path: str) -> Tuple[str, str, str]:
    """
    Run the temporal alignment stage and build the training dataset.
    
    Args:
        video_features_path: Path to video features
        imu_features_path: Path to IMU features
        
    Returns:
        Tuple of (aligned_features_path, ml_features_path, ml_labels_path)
    """
    logger.info("Starting temporal alignment stage")
    
    # Align modalities (the windows that have a camera frame)
    aligned_df = align_features(video_features_path, imu_features_path)
    aligned_features_path = os.path.join(get_param('alignment.output_path'), 'aligned_features.parquet')
    
    # Prepare ML dataset from every INS window (see src/models/dataset.py)
    imu_data_path = os.path.join(get_param('ingest.imu.output_path'), "ins_processed.csv")
    build_dataset(imu_features_path, imu_data_path, video_features_path=video_features_path)
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


def run_drift_detection_stage(imu_features_path: str, imu_data_path: str) -> Dict[str, Any]:
    """
    Run the drift detection stage: inject drift into the INS data and compare its features
    with the training period's.
    
    Args:
        imu_features_path: Path to the INS window features
        imu_data_path: Path to INS data for drift injection
        
    Returns:
        Dictionary with drift detection results
    """
    logger.info("Starting drift detection stage")
    
    # Create reference dataset from the training period of the drive
    features = pd.read_parquet(imu_features_path).dropna().sort_values('timestamp')
    train, _, _ = chronological_split(len(features), get_param('pipeline.test_size'),
                                      get_param('pipeline.validation_size'))
    temp_dir = os.path.join('experiments', 'drift_test')
    os.makedirs(temp_dir, exist_ok=True)
    training_features_path = os.path.join(temp_dir, 'training_features.parquet')
    features.iloc[train].to_parquet(training_features_path, index=False)
    create_reference_dataset(training_features_path)
    reference_path = os.path.join(get_param('drift.reference_path'), 'reference_dataset.parquet')
    
    # Inject drift for testing
    inject_drift(imu_data_path, output_path=temp_dir)
    drifted_path = os.path.join(temp_dir, 'imu_drifted.csv')
    
    # Process the drifted data
    logger.info("Processing drifted INS data through the pipeline")
    process_ins_features(drifted_path, output_path=temp_dir)
    drifted_features_path = os.path.join(temp_dir, 'ins_features.parquet')
    
    # Check for drift
    drift_results = check_for_drift(reference_path, drifted_features_path,
                                    output_path=os.path.join(temp_dir, 'drift_results'))
    drift_results['drifted_features_path'] = drifted_features_path
    drift_results['drifted_data_path'] = drifted_path
    
    # Log drift detection results
    if drift_results['drift_detected']:
        logger.warning("Drift detected in the INS data. Retraining may be needed.")
    else:
        logger.info("No significant drift detected in the INS data.")
    
    logger.info("Drift detection stage completed")
    return drift_results


def run_retraining_if_needed(drift_results: Dict[str, Any]) -> Optional[int]:
    """
    Run model retraining if drift is detected: train on the drifted data and register the result
    as a new model version. `python -m src.models.evaluate_model` decides whether it replaces the
    champion.
    
    Args:
        drift_results: Results from drift detection
        
    Returns:
        The new model version if retraining was performed, None otherwise
    """
    if drift_results['drift_detected']:
        logger.info("Drift detected, starting model retraining")
        
        # Create a new experiment directory for retraining
        retrain_dir = os.path.join('experiments', 'retraining', f"retrain_{int(time.time())}")
        os.makedirs(retrain_dir, exist_ok=True)
        
        # Build the dataset from the drifted features (labels from the recorded heading)
        build_dataset(drift_results['drifted_features_path'],
                      os.path.join(get_param('ingest.imu.output_path'), "ins_processed.csv"),
                      output_path=retrain_dir)
        
        # Retrain the model
        retrain_model_dir = os.path.join(retrain_dir, 'model')
        training_results = train_model(
            os.path.join(retrain_dir, 'features.parquet'),
            os.path.join(retrain_dir, 'labels.parquet'),
            output_path=retrain_model_dir,
            run_name=f"retraining_{int(time.time())}"
        )
        
        logger.info(f"Model retraining completed (version {training_results['model_version']}). "
                    f"Test F1 on drifted data: {training_results['test_metrics']['f1']:.4f}")
        return training_results['model_version']
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
            drift_results = run_drift_detection_stage(imu_features_path, imu_data_path)
        else:
            drift_results = {'drift_detected': False, 'message': 'Drift simulation disabled'}
        
        # 6. Retraining if needed
        retrained_version = run_retraining_if_needed(drift_results)
        
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
                    'retrained_model_version': retrained_version
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