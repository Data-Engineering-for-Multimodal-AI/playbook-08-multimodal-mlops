import os
import json
import time
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging

from src.utils import get_pipeline_logger, get_param

logger = get_pipeline_logger("model_monitoring")

class ModelMonitor:
    """
    Monitoring service for tracking model predictions, data drift, and performance metrics.
    """
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the monitoring service.
        
        Args:
            log_dir: Directory to store monitoring logs
        """
        if log_dir is None:
            log_dir = os.path.join("experiments", "monitoring")
            
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create directories for different monitoring aspects
        self.predictions_dir = os.path.join(self.log_dir, "predictions")
        self.data_drift_dir = os.path.join(self.log_dir, "data_drift")
        self.performance_dir = os.path.join(self.log_dir, "performance")
        
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.data_drift_dir, exist_ok=True)
        os.makedirs(self.performance_dir, exist_ok=True)
        
        self.feedback_dir = os.path.join(self.log_dir, "feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Initialize tracking dataframes
        self.predictions_df = pd.DataFrame()
        self.batch_size = int(get_param('monitoring.batch_size', 100))
        self.alert_count = 0
        
        # Statistics of the data the served model learned from (see set_reference). Without them,
        # drift is measured against earlier batches, as in the first edition.
        self.reference = None
        self.model_version = None
        
        # Load historical feature statistics from disk
        self.feature_stats_df = self._load_historical_feature_stats()
        
        logger.info(f"Model monitoring initialized. Logs will be stored in {self.log_dir}")
        if not self.feature_stats_df.empty:
            logger.info(f"Loaded {len(self.feature_stats_df)} historical feature statistic records")
    
    def set_reference(self, reference_path: Optional[str]) -> None:
        """Compare future batches with the training data of a model version (reference_v<N>.json)."""
        if reference_path is None or not os.path.exists(reference_path):
            self.reference = None
            return
        with open(reference_path) as f:
            self.reference = json.load(f)
        self.model_version = self.reference.get('model_version')
        logger.info(f"Monitoring against the training data of model version {self.model_version}")
    
    def _load_historical_feature_stats(self) -> pd.DataFrame:
        """Load historical feature statistics from disk."""
        feature_stats_files = glob.glob(os.path.join(self.data_drift_dir, "feature_stats_*.json"))
        if not feature_stats_files:
            return pd.DataFrame()
            
        stats_list = []
        for file in sorted(feature_stats_files):
            try:
                with open(file, 'r') as f:
                    stats = json.load(f)
                    stats_list.append(stats)
            except Exception as e:
                logger.error(f"Error loading feature stats from {file}: {e}")
                
        if not stats_list:
            return pd.DataFrame()
            
        return pd.DataFrame(stats_list)
    
    def log_prediction(self, features: Dict[str, float], prediction: int, 
                       probability: float, timestamp: Optional[int] = None) -> None:
        """
        Log a prediction made by the model.
        
        Args:
            features: Input features for the prediction
            prediction: Model prediction (class)
            probability: Prediction probability
            timestamp: Timestamp of the prediction (default: current time)
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)
            
        # Create record
        record = {
            "timestamp": timestamp,
            "prediction": prediction,
            "probability": probability,
            **features
        }
        
        # Add to tracking dataframe
        self.predictions_df = pd.concat([
            self.predictions_df, 
            pd.DataFrame([record])
        ], ignore_index=True)
        
        # Save and check for drift after every batch (monitoring.batch_size, 100 by default)
        if len(self.predictions_df) % self.batch_size == 0:
            self.end_batch()
            
        logger.debug(f"Logged prediction: {prediction} with probability {probability:.4f}")
    
    def log_feedback(self, timestamp: int, actual_label: int) -> None:
        """
        Log feedback (actual label) for a previous prediction.
        
        Args:
            timestamp: Timestamp of the original prediction
            actual_label: Actual ground truth label
        """
        # Find the prediction with the matching timestamp
        if self.predictions_df.empty:
            logger.warning(f"No predictions found for timestamp {timestamp}")
            return
            
        idx = self.predictions_df[self.predictions_df["timestamp"] == timestamp].index
        if len(idx) == 0:
            logger.warning(f"No prediction found for timestamp {timestamp}")
            return
            
        # Update the prediction record with actual label
        self.predictions_df.loc[idx[0], "actual_label"] = actual_label
        
        # Calculate performance if we have enough feedback
        feedback_count = self.predictions_df["actual_label"].notna().sum()
        if feedback_count > 0 and feedback_count % 50 == 0:
            self._calculate_performance()
            
        logger.info(f"Logged feedback for prediction at {timestamp}: actual label {actual_label}")
    
    def end_batch(self) -> None:
        """
        Check the logged batch for drift, then save it.

        The first edition saved first, which also cleared the batch, so the drift check that
        followed always saw an empty batch and never ran.
        """
        self._check_data_drift()
        self._save_predictions()
    
    def _save_predictions(self) -> None:
        """Save predictions to disk."""
        if self.predictions_df.empty:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.predictions_dir, f"predictions_{timestamp}.parquet")
        
        self.predictions_df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(self.predictions_df)} predictions to {filepath}")
        
        # Reset the tracking dataframe to prevent it from growing too large
        self.predictions_df = pd.DataFrame()
    
    def _check_data_drift(self) -> None:
        """Check for data drift in feature distributions."""
        if self.predictions_df.empty:
            return
            
        # Select only the feature columns (exclude timestamp, prediction, etc.)
        feature_cols = [col for col in self.predictions_df.columns 
                       if col not in ["timestamp", "prediction", "probability", "actual_label"]]
        
        if not feature_cols:
            return
            
        # Calculate statistics for each feature
        stats = {}
        for col in feature_cols:
            stats[f"{col}_mean"] = float(self.predictions_df[col].mean())
            stats[f"{col}_std"] = float(self.predictions_df[col].std())
            stats[f"{col}_min"] = float(self.predictions_df[col].min())
            stats[f"{col}_max"] = float(self.predictions_df[col].max())
        
        stats["timestamp"] = datetime.now().isoformat()
        stats["num_records"] = int(len(self.predictions_df))
        
        # Save statistics to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.data_drift_dir, f"feature_stats_{timestamp}.json")
        
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Saved feature statistics to {filepath}")
        
        # Add to the in-memory tracking dataframe
        self.feature_stats_df = pd.concat([
            self.feature_stats_df,
            pd.DataFrame([stats])
        ], ignore_index=True)
        
        # Check for drift regardless of in-memory state
        # This ensures drift detection works across different runs
        self._detect_drift(stats)
    
    def _detect_drift(self, current_stats: Dict[str, Any]) -> None:
        """
        Detect data drift by comparing current stats with historical stats.
        
        Args:
            current_stats: Current feature statistics to compare with history
        """
        if self.reference is not None:
            self._detect_drift_against_reference(current_stats)
            return
        
        # Need at least one previous record to detect drift
        if len(self.feature_stats_df) < 2:
            logger.info("Not enough historical data to detect drift")
            return
            
        # Create a dataframe with just the historical records (excluding the current one)
        historical_df = self.feature_stats_df.iloc[:-1]
        
        # Calculate mean of historical records
        historical_means = {}
        historical_stds = {}
        
        for col in [c for c in historical_df.columns if c.endswith("_mean")]:
            feature = col.replace("_mean", "")
            
            # Skip non-feature columns
            if feature in ["timestamp", "num_records"]:
                continue
                
            historical_means[col] = float(historical_df[col].mean())
            historical_stds[f"{feature}_std"] = float(historical_df[f"{feature}_std"].mean())
        
        drift_detected = False
        drift_metrics = {}
        
        # Check each feature for drift
        for col in [c for c in current_stats.keys() if c.endswith("_mean")]:
            feature = col.replace("_mean", "")
            
            # Skip non-feature columns
            if feature in ["timestamp", "num_records"]:
                continue
            
            if col not in historical_means:
                continue
                
            current_mean = float(current_stats[col])
            historical_mean = historical_means[col]
            historical_std = historical_stds.get(f"{feature}_std", 1.0)
            
            # Calculate z-score to detect drift
            if historical_std > 0:
                z_score = abs(current_mean - historical_mean) / historical_std
                
                # Define drift threshold (e.g., 2 standard deviations)
                if z_score > 2.0:
                    drift_detected = True
                    drift_metrics[feature] = {
                        "z_score": float(z_score),
                        "current_mean": float(current_mean),
                        "historical_mean": float(historical_mean),
                        "historical_std": float(historical_std)
                    }
        
        if drift_detected:
            self._save_drift_alert(drift_metrics)
    
    def _detect_drift_against_reference(self, current_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Z-score of this batch's mean for each monitored feature, in units of the feature's standard
        deviation in the served model's training data. Above monitoring.drift_z_threshold is drift.
        """
        threshold = float(get_param('monitoring.drift_z_threshold', 2.0))
        drift_metrics = {}
        z_scores = {}
        for feature, reference_mean in self.reference['mean'].items():
            column = f"{feature}_mean"
            reference_std = self.reference['std'].get(feature, 0.0)
            if column not in current_stats or not reference_std:
                continue
            z_score = abs(float(current_stats[column]) - reference_mean) / reference_std
            z_scores[feature] = z_score
            if z_score > threshold:
                drift_metrics[feature] = {
                    "z_score": float(z_score),
                    "current_mean": float(current_stats[column]),
                    "historical_mean": float(reference_mean),
                    "historical_std": float(reference_std)
                }
        self.last_drift = {
            "max_z_score": max(z_scores.values(), default=0.0),
            "drift_detected": bool(drift_metrics),
            "features": sorted(drift_metrics, key=lambda f: -drift_metrics[f]["z_score"]),
        }
        if drift_metrics:
            self._save_drift_alert(drift_metrics)
        return self.last_drift
    
    def _save_drift_alert(self, drift_metrics: Dict[str, Any]) -> str:
        """Write a drift_alert_*.json file, which the drift handler and the Airflow DAG read."""
        self.alert_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.data_drift_dir, f"drift_alert_{timestamp}_{self.alert_count:04d}.json")
        
        with open(filepath, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "drift_detected": True,
                "model_version": self.model_version,
                "metrics": drift_metrics
            }, f, indent=2)
            
        logger.warning(f"Data drift detected! Details saved to {filepath}")
        return filepath
    
    def record_labeled_batch(self, batch_id: int, X: np.ndarray, monitor_df: pd.DataFrame,
                             labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Store a batch of served windows once their true labels are known, and score the model on it.

        The stored batches (feedback/batch_*.parquet) are what the drift handler retrains on.
        """
        from sklearn.metrics import f1_score
        batch = monitor_df.reset_index(drop=True).copy()
        batch['features'] = list(np.asarray(X, dtype=float))
        batch['label'] = np.asarray(labels).astype(int)
        batch['prediction'] = np.asarray(predictions).astype(int)
        batch.to_parquet(os.path.join(self.feedback_dir, f"batch_{batch_id:04d}.parquet"), index=False)
        
        correct = int((batch['prediction'] == batch['label']).sum())
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "batch": int(batch_id),
            "model_version": self.model_version,
            "num_samples": int(len(batch)),
            "accuracy": float(correct / len(batch)),
            "f1": float(f1_score(batch['label'], batch['prediction'], zero_division=0)),
            "total_correct": correct,
            "total_incorrect": int(len(batch) - correct),
            "class_distribution": {str(k): int(v) for k, v in batch['label'].value_counts().items()},
        }
        filepath = os.path.join(self.performance_dir, f"performance_batch_{batch_id:04d}.json")
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    
    def _calculate_performance(self) -> None:
        """Calculate model performance metrics based on feedback."""
        # Filter to predictions that have feedback
        df = self.predictions_df[self.predictions_df["actual_label"].notna()].copy()
        
        if len(df) < 10:
            logger.info("Not enough feedback to calculate performance metrics")
            return
            
        # Calculate metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": int(len(df)),
            "accuracy": float((df["prediction"] == df["actual_label"]).mean()),
            "total_correct": int((df["prediction"] == df["actual_label"]).sum()),
            "total_incorrect": int((df["prediction"] != df["actual_label"]).sum())
        }
        
        # Calculate class distribution
        class_counts = df["actual_label"].value_counts().to_dict()
        metrics["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}
        
        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.performance_dir, f"performance_{timestamp}.json")
        
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Performance metrics: accuracy={metrics['accuracy']:.4f}, samples={metrics['num_samples']}")
        logger.info(f"Saved performance metrics to {filepath}")

# Singleton instance for use across the application
_monitor = None

def get_monitor() -> ModelMonitor:
    """Get the global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ModelMonitor()
    return _monitor

if __name__ == "__main__":
    # Simple test of the monitoring service
    monitor = ModelMonitor()
    
    # Simulate some predictions
    for i in range(150):
        features = {
            "velocity_north": np.random.normal(1.0, 0.5),
            "velocity_east": np.random.normal(5.0, 2.0),
            "velocity_down": np.random.normal(0.1, 0.05),
            "roll": np.random.normal(0.01, 0.005),
            "pitch": np.random.normal(0.02, 0.01),
            "yaw": np.random.normal(1.5, 0.2)
        }
        
        prediction = 1 if features["velocity_east"] > 7.0 else 0
        probability = 0.5 + min(abs(features["velocity_east"] - 7.0) / 10.0, 0.49)
        
        timestamp = int(time.time() * 1000) + i
        
        monitor.log_prediction(features, prediction, probability, timestamp)
        
        # Simulate some feedback
        if i % 10 == 0:
            actual_label = 1 if features["velocity_east"] > 7.0 else 0
            # Add some noise to simulate prediction errors
            if np.random.random() < 0.2:
                actual_label = 1 - actual_label
                
            monitor.log_feedback(timestamp, actual_label)
            
    print("Monitoring test complete") 