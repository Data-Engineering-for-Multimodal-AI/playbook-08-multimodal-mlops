#!/usr/bin/env python
"""
Run the complete MLOps pipeline from data ingestion to model monitoring.
This script simulates what the Airflow DAG would do, but can be run independently.
"""
import os
import subprocess
import time
import threading
import signal
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("mlops_pipeline")

# Global variable to track running processes
processes = []

def run_command(command, description=None, wait=True):
    """Run a shell command and optionally wait for completion."""
    if description:
        logger.info(f"Starting: {description}")
    
    # Create process
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Add to global list for cleanup
    processes.append(process)
    
    if wait:
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error output: {stderr}")
            return False, stderr
        logger.info(f"Successfully completed: {description or command}")
        return True, stdout
    else:
        logger.info(f"Started background process: {description or command}")
        return True, None

def signal_handler(sig, frame):
    """Handle process termination and cleanup."""
    logger.info("Shutting down pipeline processes...")
    for p in processes:
        try:
            p.terminate()
        except:
            pass
    logger.info("Pipeline shutdown complete")
    sys.exit(0)

def start_model_serving():
    """Start the model serving API."""
    run_command(
        "pkill -f 'uvicorn src.serve.simple_api:app' || true",
        "Killing any existing API servers",
        wait=True
    )
    run_command(
        "nohup uvicorn src.serve.simple_api:app --reload > api.log 2>&1 &",
        "Starting model serving API",
        wait=False
    )
    # Give the server time to start
    time.sleep(2)

def start_drift_monitoring():
    """Start the drift monitoring process."""
    run_command(
        "python -m src.monitoring.drift_handler",
        "Starting drift monitoring",
        wait=False
    )

def main():
    """Run the complete pipeline."""
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Ensure required directories exist
    os.makedirs("experiments/features/ins", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)
    os.makedirs("experiments/monitoring/data_drift", exist_ok=True)
    os.makedirs("experiments/monitoring/performance", exist_ok=True)
    os.makedirs("experiments/monitoring/predictions", exist_ok=True)
    
    logger.info("Starting MLOps pipeline execution")
    
    # 1. Extract features from INS data
    success, _ = run_command(
        "python -m src.features.extract_ins_features",
        "Extracting features from INS data"
    )
    if not success:
        logger.error("Feature extraction failed, aborting pipeline")
        return False
    
    # 2. Train the model
    success, _ = run_command(
        "python -m src.models.train_model",
        "Training model"
    )
    if not success:
        logger.error("Model training failed, aborting pipeline")
        return False
    
    # 3. Evaluate the model
    success, _ = run_command(
        "python -m src.models.evaluate_model",
        "Evaluating model"
    )
    if not success:
        logger.warning("Model evaluation failed, continuing with deployment")
    
    # 4. Start model serving
    start_model_serving()
    
    # 5. Generate some synthetic data for monitoring
    success, _ = run_command(
        "python -m src.monitoring.force_monitoring_data",
        "Generating synthetic monitoring data"
    )
    if not success:
        logger.warning("Failed to generate synthetic monitoring data")
    
    # 6. Start drift monitoring
    start_drift_monitoring()
    
    logger.info("MLOps pipeline successfully initiated!")
    logger.info("Press Ctrl+C to stop all processes")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    
    return True

if __name__ == "__main__":
    main() 