import os
import json
import time
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging

from src.utils import get_pipeline_logger

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
        
        # Initialize tracking dataframes
        self.predictions_df = pd.DataFrame()
        
        # Load historical feature statistics from disk
        self.feature_stats_df = self._load_historical_feature_stats()
        
        logger.info(f"Model monitoring initialized. Logs will be stored in {self.log_dir}")
        if not self.feature_stats_df.empty:
            logger.info(f"Loaded {len(self.feature_stats_df)} historical feature statistic records")
    
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
        
        # Save periodically (every 100 records)
        if len(self.predictions_df) % 100 == 0:
            self._save_predictions()
            
            # Check for data drift
            self._check_data_drift()
            
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
            # Log the drift
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.data_drift_dir, f"drift_alert_{timestamp}.json")
            
            with open(filepath, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "drift_detected": True,
                    "metrics": drift_metrics
                }, f, indent=2)
                
            logger.warning(f"Data drift detected! Details saved to {filepath}")
    
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