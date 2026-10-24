import os
import json
import glob
import logging
import subprocess
import time
from datetime import datetime
import sys

from typing import Optional

from src.utils import get_pipeline_logger, get_param

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
                 restart_service: bool = True,
                 api_url: str = "http://127.0.0.1:8000"):
        """
        Initialize the drift handler.
        
        Args:
            drift_dir: Directory containing drift alerts
            model_dir: Directory for model artifacts
            train_script: Script to run for model training
            evaluate_script: Script to run for model evaluation
            restart_service: Whether to tell the model service to reload after a promotion
            api_url: Base URL of the model API
        """
        self.drift_dir = drift_dir
        self.model_dir = model_dir
        self.train_script = train_script
        self.evaluate_script = evaluate_script
        self.restart_service = restart_service
        self.api_url = api_url
        self.last_evaluation = None
        
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
            severe_threshold = float(get_param('monitoring.severe_z_threshold', 3.0))
            for feature, metrics in drift_data.get('metrics', {}).items():
                # Consider z-score > 3.0 (monitoring.severe_z_threshold) as severe drift
                if metrics.get('z_score', 0) > severe_threshold:
                    logger.warning(f"Severe drift detected in feature {feature} with z-score {metrics['z_score']}")
                    severe_drift = True
            
            # Mark this alert (and older ones) as processed
            self.processed_alerts.update(new_alerts)
            
            if not severe_drift:
                logger.info("Drift is not severe enough to trigger retraining")
                return False
                
            logger.warning(f"Significant drift detected in alert {latest_alert}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing drift alert {latest_alert}: {str(e)}")
            return False
    
    def retrain_model(self, since: Optional[str] = None) -> bool:
        """
        Retrain on the labeled production data, then let evaluate_model decide on promotion.
        
        Args:
            since: Retrain on production batches from this one on (e.g. batch_0007.parquet)
        
        Returns:
            True if a retrained model was promoted to champion, False otherwise
        """
        try:
            logger.info("Starting model retraining...")
            
            # Run training script
            command = [sys.executable, "-m", self.train_script, "--retrain"]
            if since:
                command += ["--since", since]
            logger.info(f"Running training script: {' '.join(command[1:])}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"Training output: {result.stdout.strip()}")
            
            # Run evaluation script (promotes the new version only if it beats the champion)
            logger.info(f"Running evaluation script: {self.evaluate_script}")
            result = subprocess.run(
                [sys.executable, "-m", self.evaluate_script],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Evaluation output: {result.stdout.strip()}")
            with open(os.path.join(self.model_dir, "evaluation_summary.json")) as f:
                self.last_evaluation = json.load(f)
            promoted = self.last_evaluation.get("promoted", False)
            
            # Tell the model service to serve the new champion
            if promoted and self.restart_service:
                self._restart_service()
                
            logger.info("Model retraining completed: " + ("new champion promoted" if promoted else "champion kept"))
            return promoted
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during model retraining: {str(e)}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model retraining: {str(e)}")
            return False
    
    def _restart_service(self):
        """Ask the running model API to load the new champion."""
        try:
            import requests
            logger.info(f"Asking the model service at {self.api_url} to reload...")
            response = requests.post(f"{self.api_url}/reload", timeout=60)
            response.raise_for_status()
            logger.info(f"Model service now serves version {response.json().get('model_version')}")
            
        except Exception as e:
            logger.error(f"Could not reach the model service to reload it: {str(e)}")
    
    def run_continuous_monitoring(self, check_interval: int = 300, once: bool = False):
        """
        Continuously monitor for drift and retrain as needed.
        
        Args:
            check_interval: Time between checks in seconds (default: 5 minutes)
            once: Check once and return (for schedulers such as Airflow or cron)
        """
        logger.info(f"Starting continuous drift monitoring with {check_interval}s interval")
        
        try:
            while True:
                if self.check_for_drift():
                    self.retrain_model()
                if once:
                    break
                
                logger.debug(f"Sleeping for {check_interval} seconds")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Drift monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in drift monitoring: {str(e)}")

def main():
    """Run the drift handler in continuous monitoring mode."""
    import argparse
    parser = argparse.ArgumentParser(description="Retrain and promote when severe drift is detected")
    parser.add_argument("--once", action="store_true", help="check once and exit")
    parser.add_argument("--interval", type=int, default=300, help="seconds between checks")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://127.0.0.1:8000"),
                        help="model API to reload after a promotion")
    args = parser.parse_args()
    handler = DriftHandler(api_url=args.api_url)
    handler.run_continuous_monitoring(check_interval=args.interval, once=args.once)

if __name__ == "__main__":
    main() 