import os
from datetime import datetime, timedelta
import json

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.models import Variable
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

# Define the data directories
DATA_DIR = "oxfordrobocar_sample_small/gps"  # Updated to use the original INS data
FEATURE_DIR = "experiments/features/ins"
MODEL_DIR = "experiments/models"
MONITORING_DIR = "experiments/monitoring"

# Define the tasks

# 1. Check for new data
check_new_data = BashOperator(
    task_id='check_new_data',
    bash_command=f'ls -la {DATA_DIR}/ins.csv | wc -l > /tmp/data_count.txt',
    dag=dag,
)

# 2. Extract features from INS data
extract_features = BashOperator(
    task_id='extract_features',
    bash_command='python -m src.features.extract_ins_features',
    dag=dag,
)

# 3. Check if retraining is needed by examining drift alerts
def _check_drift_alerts():
    """Check if there are drift alerts that require model retraining."""
    drift_dir = f"{MONITORING_DIR}/data_drift"
    
    # Get all drift alert files
    drift_files = [f for f in os.listdir(drift_dir) if f.startswith("drift_alert_")]
    
    if not drift_files:
        return 'skip_training'
    
    # Check the most recent drift alert
    latest_drift = sorted(drift_files)[-1]
    
    with open(os.path.join(drift_dir, latest_drift), 'r') as f:
        drift_data = json.load(f)
    
    # If significant drift is detected (defined by a severe drift in any feature)
    if drift_data.get('drift_detected', False):
        # Check severity by looking at z-scores
        severe_drift = False
        for feature, metrics in drift_data.get('metrics', {}).items():
            if metrics.get('z_score', 0) > 3.0:  # Severe drift threshold
                severe_drift = True
                break
        
        if severe_drift:
            return 'train_model'
    
    return 'skip_training'

check_drift = BranchPythonOperator(
    task_id='check_drift',
    python_callable=_check_drift_alerts,
    dag=dag,
)

# 4. Train model (if needed)
train_model = BashOperator(
    task_id='train_model',
    bash_command='python -m src.models.train_model',
    dag=dag,
)

# 5. Skip training
skip_training = BashOperator(
    task_id='skip_training',
    bash_command='echo "Skipping model training as no significant drift detected"',
    dag=dag,
)

# 6. Evaluate model
evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command='python -m src.models.evaluate_model',
    dag=dag,
)

# 7. Deploy model
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command=(
        'pkill -f "uvicorn src.serve.simple_api:app" || true && '
        'nohup uvicorn src.serve.simple_api:app --reload > api.log 2>&1 &'
    ),
    dag=dag,
)

# Define the task dependencies
check_new_data >> extract_features >> check_drift
check_drift >> [train_model, skip_training]
train_model >> evaluate_model
skip_training >> deploy_model
evaluate_model >> deploy_model

# Sensor to detect drift alerts during pipeline operation
drift_alert_sensor = FileSensor(
    task_id='drift_alert_sensor',
    filepath=f"{MONITORING_DIR}/data_drift/drift_alert_*.json",
    poke_interval=300,  # Check every 5 minutes
    timeout=60 * 60 * 24,  # Timeout after 24 hours
    mode='reschedule',  # Don't block a worker
    dag=dag,
)

# This special task triggers a new DAG run when a drift alert is detected
trigger_new_run = BashOperator(
    task_id='trigger_new_run',
    bash_command='airflow dags trigger ins_mlops_pipeline',
    dag=dag,
)

drift_alert_sensor >> trigger_new_run 