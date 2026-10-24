#!/usr/bin/env python
"""
Run the complete MLOps pipeline from data ingestion to model monitoring.
This script simulates what the Airflow DAG would do, but can be run independently.

    python run_pipeline.py            # run every step once and exit
    python run_pipeline.py --serve    # then keep the API and drift monitoring running (Ctrl+C stops)

Steps:
  1. Extract features from the INS data (half-second windows)
  2. Ingest the camera frames, extract their features and align them with the INS windows
  3. Train a model, log it to MLflow and register it (the first version becomes the champion)
  4. Evaluate the champion
  5. Start the model API (http://127.0.0.1:8000) and send it a window of INS readings
  6. Replay the end of the drive through the model, with a sensor fault part way through: drift
     monitoring, retraining and promotion (src/monitoring/closed_loop.py)
  7. Check which version the API serves now
"""
import argparse
import json
import os
import subprocess
import time
import signal
import sys
import logging
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)   # the steps use paths relative to the repository root
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)

logger = logging.getLogger("mlops_pipeline")

# Child processes started by this script (stopped on exit)
processes = []


def run_command(args, description=None, show_output=False):
    """Run a Python module or script with this interpreter and wait for it."""
    if description:
        logger.info(f"Starting: {description}")
    started = time.time()
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: python {' '.join(args)}")
        logger.error(f"Error output: {result.stderr[-3000:]}")
        return False, result.stderr
    if show_output and result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    logger.info(f"Successfully completed: {description or ' '.join(args)} ({time.time() - started:.0f} s)")
    return True, result.stdout


def stop_processes():
    for p in processes:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def signal_handler(sig, frame):
    """Handle process termination and cleanup."""
    logger.info("Shutting down pipeline processes...")
    stop_processes()
    logger.info("Pipeline shutdown complete")
    sys.exit(0)


def port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_model_serving(port):
    """Start the model serving API and wait until it answers."""
    import requests
    if port_in_use(port):
        # Never talk to (or reload) some other service that happens to use this port
        raise RuntimeError(f"Port {port} is already in use; pass --port with a free port")
    logger.info(f"Starting: model serving API on http://127.0.0.1:{port}")
    log = open(os.path.join("logs", "api.log"), "w")
    processes.append(subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.serve.api:app", "--host", "127.0.0.1", "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT))
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            health = requests.get(f"{url}/health", timeout=5).json()
            logger.info(f"Model API is up: {health}")
            return url
        except requests.RequestException:
            if processes[-1].poll() is not None:
                break
            time.sleep(1)
    raise RuntimeError(f"The model API didn't start; see logs/api.log")


def sample_request(url):
    """Send half a second of INS readings from the drive (minute 36.4, as the car starts a turn) to the API."""
    import pandas as pd
    import requests
    ins = pd.read_csv(os.path.join("experiments", "ingested", "imu", "ins_processed.csv")).dropna()
    start = ins['timestamp'].iloc[0] + 36.4 * 60e6 - 0.25e6
    window = ins[(ins['timestamp'] >= start) & (ins['timestamp'] < start + 0.5e6)]
    readings = window[["timestamp", "velocity_north", "velocity_east", "velocity_down", "roll", "pitch", "yaw"]]
    response = requests.post(f"{url}/predict", json={"readings": readings.to_dict("records")}, timeout=30)
    response.raise_for_status()
    logger.info(f"POST /predict with {len(readings)} readings -> {response.json()}")


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true",
                        help="keep the model API and drift monitoring running after the steps (Ctrl+C stops)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--skip-camera", action="store_true",
                        help="skip step 2 (it downloads ResNet-18 weights, 45 MB, on first use)")
    args = parser.parse_args()

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Ensure required directories exist
    os.makedirs("experiments/features/ins", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)
    for folder in ["data_drift", "performance", "predictions", "feedback"]:
        os.makedirs(os.path.join("experiments", "monitoring", folder), exist_ok=True)

    logger.info("Starting MLOps pipeline execution")
    started = time.time()

    # 1. Extract features from INS data
    success, _ = run_command(["-m", "src.features.extract_ins_features"], "Extracting features from INS data")
    if not success:
        logger.error("Feature extraction failed, aborting pipeline")
        return False

    # 2. Camera frames (a separate process each: PyTorch and XGBoost in one process crash on macOS)
    if not args.skip_camera:
        for module, description in [("src.ingest.video_ingest", "Ingesting camera frames"),
                                    ("src.features.video_features", "Extracting camera features (ResNet-18)"),
                                    ("src.align.temporal_alignment", "Aligning camera frames with INS windows")]:
            success, _ = run_command(["-m", module], description)
            if not success:
                logger.error(f"{description} failed, aborting pipeline")
                return False

    # 3. Train the model
    # A fresh run starts from the model it trains (earlier runs may have promoted a retrained one)
    success, _ = run_command(["-m", "src.models.train_model", "--promote"], "Training model", show_output=True)
    if not success:
        logger.error("Model training failed, aborting pipeline")
        return False

    # 4. Evaluate the model
    success, _ = run_command(["-m", "src.models.evaluate_model"], "Evaluating model", show_output=True)
    if not success:
        logger.error("Model evaluation failed, aborting pipeline")
        return False

    try:
        # 5. Start model serving
        url = start_model_serving(args.port)
        sample_request(url)

        # 6. Production replay with a sensor fault: monitoring, retraining, promotion
        success, _ = run_command(["-m", "src.monitoring.closed_loop", "--api-url", url],
                                 "Replaying the end of the drive with a sensor fault", show_output=True)
        if not success:
            logger.error("Closed-loop replay failed")
            return False

        # 7. The API now serves the current champion
        import requests
        logger.info(f"Model API after the replay: {requests.get(f'{url}/health', timeout=10).json()}")
        sample_request(url)

        logger.info(f"MLOps pipeline completed in {time.time() - started:.0f} s. "
                    "MLflow runs and models: `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`")

        if args.serve:
            # Keep serving and check for drift every 5 minutes until stopped
            processes.append(subprocess.Popen([sys.executable, "-m", "src.monitoring.drift_handler",
                                               "--api-url", url]))
            logger.info(f"Serving at {url} with drift monitoring. Press Ctrl+C to stop all processes")
            while True:
                time.sleep(1)
    finally:
        stop_processes()

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
