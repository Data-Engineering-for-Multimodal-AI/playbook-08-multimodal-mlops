import os
import sys
from datetime import datetime, timedelta
import json

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import BranchPythonOperator

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 5, 3),
}

# Create the DAG
dag = DAG(
    'ins_mlops_pipeline',
    default_args=default_args,
    description='End-to-end MLOps pipeline for INS data processing',
    schedule=timedelta(days=1),  # Run daily
    catchup=False,
)

# The tasks run in the playbook folder (two levels up from this file, or PLAYBOOK_DIR) with the
# playbook's Python (PLAYBOOK_PYTHON). Airflow usually lives in its own virtual environment, so
# set PLAYBOOK_PYTHON to the playbook environment's python (see the README).
PROJECT_DIR = os.environ.get("PLAYBOOK_DIR",
                             os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PYTHON = os.environ.get("PLAYBOOK_PYTHON", sys.executable)
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Define the data directories
DATA_DIR = os.path.join(PROJECT_DIR, "oxfordrobocar_sample_small/gps")  # Updated to use the original INS data
FEATURE_DIR = os.path.join(PROJECT_DIR, "experiments/features/ins")
MODEL_DIR = os.path.join(PROJECT_DIR, "experiments/models")
MONITORING_DIR = os.path.join(PROJECT_DIR, "experiments/monitoring")

# Define the tasks

# 1. Check for new data
check_new_data = BashOperator(
    task_id='check_new_data',
    bash_command=f'test -s "{DATA_DIR}/ins.csv" && wc -l < "{DATA_DIR}/ins.csv"',
    dag=dag,
)

# 2. Extract features from INS data
extract_features = BashOperator(
    task_id='extract_features',
    bash_command=f'"{PYTHON}" -m src.features.extract_ins_features',
    cwd=PROJECT_DIR,
    dag=dag,
)

# 3. Decide whether to train: no model yet, or severe drift in the latest alert
def _check_drift_alerts():
    """Check if there are drift alerts that require model retraining."""
    if not os.path.exists(os.path.join(MODEL_DIR, "training_summary.json")):
        return 'train_model'

    drift_dir = f"{MONITORING_DIR}/data_drift"

    # Get all drift alert files
    drift_files = [f for f in os.listdir(drift_dir) if f.startswith("drift_alert_")] \
        if os.path.isdir(drift_dir) else []

    if not drift_files:
        return 'skip_training'

    # Check the most recent drift alert
    latest_drift = sorted(drift_files)[-1]
    with open(os.path.join(drift_dir, latest_drift), 'r') as f:
        drift_data = json.load(f)

    # If significant drift is detected (defined by a severe drift in any feature)
    if drift_data.get('drift_detected', False):
        # Check severity by looking at z-scores
        for feature, metrics in drift_data.get('metrics', {}).items():
            if metrics.get('z_score', 0) > 3.0:  # Severe drift threshold
                return 'retrain_model'

    return 'skip_training'

check_drift = BranchPythonOperator(
    task_id='check_drift',
    python_callable=_check_drift_alerts,
    dag=dag,
)

# 4. Train model (first model: becomes the champion)
train_model = BashOperator(
    task_id='train_model',
    bash_command=f'"{PYTHON}" -m src.models.train_model',
    cwd=PROJECT_DIR,
    dag=dag,
)

# 4b. Retrain on the labeled production data after severe drift
retrain_model = BashOperator(
    task_id='retrain_model',
    bash_command=f'"{PYTHON}" -m src.models.train_model --retrain',
    cwd=PROJECT_DIR,
    dag=dag,
)

# 5. Skip training
skip_training = BashOperator(
    task_id='skip_training',
    bash_command='echo "Skipping model training as no significant drift detected"',
    dag=dag,
)

# 6. Evaluate model (promotes a new version only if it beats the champion)
evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command=f'"{PYTHON}" -m src.models.evaluate_model',
    cwd=PROJECT_DIR,
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# 7. Deploy model: tell the running API to serve the current champion
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command=(
        f'curl -fsS -X POST {API_URL}/reload || '
        f'echo "Model API not running at {API_URL}; start it with: uvicorn src.serve.api:app"'
    ),
    trigger_rule='none_failed_min_one_success',  # runs after evaluation or after skip_training
    dag=dag,
)

# Define the task dependencies
check_new_data >> extract_features >> check_drift
check_drift >> [train_model, retrain_model, skip_training]
[train_model, retrain_model] >> evaluate_model
skip_training >> deploy_model
evaluate_model >> deploy_model

# The first edition also had a FileSensor waiting up to 24 hours for drift alerts and a task
# running `airflow dags trigger`. Neither was connected to the tasks above, the sensor kept every
# run open for a day, and Airflow 3 tasks can't run Airflow CLI commands against the metadata
# database. Drift is checked by check_drift in each run; for a faster response, shorten the
# schedule or run `python -m src.monitoring.drift_handler`.
