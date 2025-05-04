import os
import json
import glob
import logging
import subprocess
import time
from datetime import datetime
import sys

from src.utils import get_pipeline_logger

# Configure logging
logger = get_pipeline_logger("drift_handler")

class DriftHandler:
    """
    Handles detected data drift by triggering model retraining
    and updating the deployed model if necessary.
    """
    def __init__(self, 
                 drift_dir: str = "experiments/monitoring/data_drift",
                 model_dir: str = "experiments/models",
                 train_script: str = "src.models.train_model",
                 evaluate_script: str = "src.models.evaluate_model",
                 restart_service: bool = True):
        """
        Initialize the drift handler.
        
        Args:
            drift_dir: Directory containing drift alerts
            model_dir: Directory for model artifacts
            train_script: Script to run for model training
            evaluate_script: Script to run for model evaluation
            restart_service: Whether to restart the model service after retraining
        """
        self.drift_dir = drift_dir
        self.model_dir = model_dir
        self.train_script = train_script
        self.evaluate_script = evaluate_script
        self.restart_service = restart_service
        
        # Ensure directories exist
        os.makedirs(drift_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Keep track of processed drift alerts
        self.processed_alerts = set()
        
    def check_for_drift(self) -> bool:
        """
        Check for new drift alerts.
        
        Returns:
            True if significant drift is detected, False otherwise
        """
        drift_files = glob.glob(os.path.join(self.drift_dir, "drift_alert_*.json"))
        
        if not drift_files:
            logger.info("No drift alerts found")
            return False
        
        # Process only new alerts
        new_alerts = [f for f in drift_files if f not in self.processed_alerts]
        
        if not new_alerts:
            logger.info("No new drift alerts since last check")
            return False
        
        # Get the most recent alert
        latest_alert = sorted(new_alerts)[-1]
        
        try:
            with open(latest_alert, 'r') as f:
                drift_data = json.load(f)
                
            # Check if this is a significant drift
            if not drift_data.get('drift_detected', False):
                logger.info(f"Drift alert {latest_alert} does not indicate significant drift")
                self.processed_alerts.add(latest_alert)
                return False
                
            # Check severity of drift
            severe_drift = False
            for feature, metrics in drift_data.get('metrics', {}).items():
                # Consider z-score > 3.0 as severe drift
                if metrics.get('z_score', 0) > 3.0:
                    logger.warning(f"Severe drift detected in feature {feature} with z-score {metrics['z_score']}")
                    severe_drift = True
            
            # Mark this alert as processed
            self.processed_alerts.add(latest_alert)
            
            if not severe_drift:
                logger.info("Drift is not severe enough to trigger retraining")
                return False
                
            logger.warning(f"Significant drift detected in alert {latest_alert}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing drift alert {latest_alert}: {str(e)}")
            return False
    
    def retrain_model(self) -> bool:
        """
        Execute model retraining process.
        
        Returns:
            True if retraining was successful, False otherwise
        """
        try:
            logger.info("Starting model retraining...")
            
            # Run training script
            logger.info(f"Running training script: {self.train_script}")
            result = subprocess.run(
                [sys.executable, "-m", self.train_script],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Training output: {result.stdout}")
            
            # Run evaluation script
            logger.info(f"Running evaluation script: {self.evaluate_script}")
            result = subprocess.run(
                [sys.executable, "-m", self.evaluate_script],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Evaluation output: {result.stdout}")
            
            # Restart service if needed
            if self.restart_service:
                self._restart_service()
                
            logger.info("Model retraining and deployment completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during model retraining: {str(e)}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model retraining: {str(e)}")
            return False
    
    def _restart_service(self):
        """Restart the model serving API."""
        try:
            logger.info("Restarting model service...")
            
            # Kill existing service (if any)
            subprocess.run(
                ["pkill", "-f", "uvicorn src.serve.simple_api:app"],
                check=False  # Don't raise error if nothing to kill
            )
            
            # Wait a moment
            time.sleep(1)
            
            # Start new service
            subprocess.Popen(
                ["uvicorn", "src.serve.simple_api:app", "--reload"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("Model service restarted successfully")
            
        except Exception as e:
            logger.error(f"Error restarting model service: {str(e)}")
    
    def run_continuous_monitoring(self, check_interval: int = 300):
        """
        Continuously monitor for drift and retrain as needed.
        
        Args:
            check_interval: Time between checks in seconds (default: 5 minutes)
        """
        logger.info(f"Starting continuous drift monitoring with {check_interval}s interval")
        
        try:
            while True:
                if self.check_for_drift():
                    self.retrain_model()
                
                logger.debug(f"Sleeping for {check_interval} seconds")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Drift monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in drift monitoring: {str(e)}")

def main():
    """Run the drift handler in continuous monitoring mode."""
    handler = DriftHandler()
    handler.run_continuous_monitoring()

if __name__ == "__main__":
    main() 