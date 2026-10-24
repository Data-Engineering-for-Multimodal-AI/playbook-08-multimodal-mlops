# Implementing an end-to-end MLOps pipeline

**Chapter 8 playbook** from *Data Engineering for Multimodal AI* (O'Reilly, 2026). Page numbers to be confirmed.
Book: https://oreil.ly/data-engineering-multimodal

> Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## What you'll build

An MLOps pipeline for a self-driving car's sensor data, on a real 42-minute drive through Oxford from the Oxford RobotCar dataset: inertial navigation (INS) readings at 50 Hz and forward camera frames. The pipeline keeps a model useful after deployment: when a sensor starts to degrade, monitoring notices, the model is retrained on fresh labeled data, and the new version replaces the old one only if it does better.

- **Feature engineering**: half-second windows of INS readings (velocity, roll, pitch, heading) with statistical and frequency features; camera frames demosaiced from the raw sensor and described by color histograms and ResNet-18 embeddings
- **Temporal alignment** of camera frames with INS windows
- **Model**: an XGBoost classifier that predicts, from the last half second of motion, whether the car turns in the next second
- **Experiment tracking and Model Registry with MLflow**: every model is a registered version; the one in service has the alias `champion`
- **Serving with FastAPI**: send a window of INS readings, get a prediction from the champion
- **Monitoring**: each batch of served windows is compared with the model's training data; labeled batches are scored (F1)
- **Drift-driven retraining**: a severe drift alert retrains on the newest labeled data; the candidate is promoted only if it beats the champion on data neither model trained on
- **Orchestration**: a standalone runner, an Airflow 3 DAG, and an Argo Workflows definition
- **Dashboard** (Streamlit) for predictions, drift and performance over time

Everything runs on a laptop CPU with the bundled sample data.

## Quick start

You need Python 3.12 or newer and about 2 GB of disk space.

```bash
# Clone the repository
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-08-multimodal-mlops.git
cd playbook-08-multimodal-mlops

# Create and activate a virtual environment, then install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r pipeline/requirements.txt
pip install -r enhancements_modules/requirements-enhancements.txt
pip install -e .
```

(`pip install -r requirements.txt` installs the same pinned packages. With [uv](https://docs.astral.sh/uv/): `uv sync`, then prefix commands with `uv run`.)

**Run the complete pipeline:**

```bash
python pipeline/run_pipeline.py      # or: python run_pipeline.py
```

It runs every step once and exits (about 5 minutes on a laptop; the first run downloads the ResNet-18 weights, 45 MB):

1. Extract features from the INS data
2. Ingest the camera frames, extract their features and align them with the INS windows
3. Train the model, log it to MLflow and register it as the champion
4. Evaluate the champion on the end of the drive
5. Start the model API on http://127.0.0.1:8000 and send it half a second of readings
6. Replay the end of the drive through the model with a sensor fault part way through: drift alerts, retraining, promotion
7. Show which model version the API serves now

`python run_pipeline.py --serve` keeps the API and drift monitoring running afterwards (Ctrl+C stops them). `--port` picks another port; the script refuses to use a port that is already taken.

**Look around:**

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db   # runs, metrics and model versions: http://127.0.0.1:5000
streamlit run src/monitoring/dashboard.py                        # monitoring dashboard: http://localhost:8501
uvicorn src.serve.api:app                                        # the model API on its own: http://127.0.0.1:8000/docs
```

**Run the steps one at a time:**

```bash
python -m src.features.extract_ins_features   # INS windows -> experiments/features/ins/
python -m src.models.train_model              # dataset, training, MLflow run, new registered version
python -m src.models.evaluate_model           # newest version vs champion; promote if better
python -m src.monitoring.closed_loop          # replay with a sensor fault: monitoring, retraining, promotion
python -m src.monitoring.drift_handler --once # check the latest drift alert and retrain if it is severe
python -m src.pipeline.pipeline               # the other orchestration: ingest to drift-triggered retraining (make run-pipeline)
python enhancements_modules/demo.py           # schema validation and the Parquet feature store
```

`make smoke` runs the tests (unit tests and the whole pipeline; a few minutes).

### Orchestration with Airflow

The DAG `airflow/dags/ins_mlops_pipeline.py` is written for Airflow 3 (tested with 3.3.2). Install Airflow in its own environment, using the constraints file for your Python version, and point the DAG at the playbook's Python:

```bash
python -m venv airflow-venv
airflow-venv/bin/pip install "apache-airflow==3.3.2" \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt

export AIRFLOW_HOME=$(pwd)/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
export PLAYBOOK_PYTHON=$(pwd)/venv/bin/python   # the environment from the Quick start

airflow-venv/bin/airflow dags test ins_mlops_pipeline   # one run, no scheduler needed
airflow-venv/bin/airflow standalone                     # or: the scheduler, API server and UI at http://localhost:8080
```

`airflow standalone` creates an admin user and prints its password. The DAG checks for data, extracts features, then trains (no model yet), retrains (a severe drift alert) or skips, evaluates, and asks the API to reload.

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `PORT`, `HOST` | `8000`, `127.0.0.1` | the APIs (`HOST=0.0.0.0` accepts connections from other machines) |
| `API_URL` | `http://127.0.0.1:8000` | the drift handler and the DAG's deploy step |
| `MLFLOW_TRACKING_URI` | `sqlite:///experiments/mlflow.db` | training, evaluation and serving |
| `PLAYBOOK_PYTHON`, `PLAYBOOK_DIR` | Airflow's Python, the repository | the Airflow DAG |

No keys or accounts are needed.

## Tested on

| | |
|---|---|
| Date | 2026-10-24 |
| OS | macOS 26.5.1 (arm64); Ubuntu 24.04 (x86-64) in CI |
| Python | 3.12.15 (macOS); 3.12 and 3.13 (CI) |
| Key packages | pandas 3.1.0, scikit-learn 1.9.1, XGBoost 3.4.1, MLflow 3.17.2, PyTorch 2.14.1 (CPU), FastAPI 0.162.0, Streamlit 1.66.1; Airflow 3.3.2 for the DAG (macOS and CI) |

On the MacBook (fresh clone, Python 3.12.15): the three printed `pip install` lines took 87 s; `python pipeline/run_pipeline.py` took 126 s (INS features 41 s, camera ingest 24 s, ResNet-18 features 17 s with the weights already downloaded, training 14 s, the closed-loop replay 20 s). `make smoke` runs the tests, including the whole pipeline, in about 2 minutes.

## Differences from the printed book

| Printed book | This version | Why |
|---|---|---|
| Python 3.8+ (conda, Python 3.9) | Python 3.12+ | Current libraries need it |
| Unpinned `>=` requirements | Pinned versions (`requirements.txt`, `uv.lock`); `pipeline/requirements.txt` includes them | Reproducible installs |
| `python pipeline/run_pipeline.py` stopped at step 2 and never exited | Runs every step once and exits; `--serve` keeps serving | Steps 2 and 3 called modules that weren't in the repository; now they are |
| Model trained on the 15 INS windows that have a camera frame; label: `velocity_east_mean > 7` (an input) | All 10,103 INS windows; label: a turn in the next second; split by time | 15 samples can't train or test a model; a label that is an input is learned trivially; overlapping windows leak across a random split. The old label: `label.name: east_velocity` |
| Camera features always part of the model | Optional (`model.use_video_features`) | The sample has 3.6 s of camera frames in a 42-minute drive |
| Camera frames read as grayscale | Demosaiced to color | They are raw Bayer sensor images |
| MLflow `./experiments/mlruns` | `sqlite:///experiments/mlflow.db`, Model Registry with the `champion` alias | MLflow 3 no longer accepts new file stores; stages were replaced by aliases |
| `model.pkl` and `scaler.pkl` (StandardScaler + XGBoost) | XGBoost alone, registered in XGBoost's format; `model.pkl` still written | Trees don't need scaling; MLflow's scikit-learn loader imports PyTorch, which crashes XGBoost on macOS |
| Served model: a fixed rule on `velocity_east` | The registered champion; `/predict` also accepts a window of readings | Serve the trained model |
| Synthetic random monitoring data; drift checked against earlier batches | Replay of the end of the drive with a simulated sensor fault; drift checked against the model's training data; F1 per labeled batch | Shows the loop working on real data. `src/monitoring/force_monitoring_data.py` is still there |
| Drift handler restarted the API with `pkill`/`uvicorn --reload` | Calls `/reload` on the API after a promotion | Don't kill other processes |
| Airflow commands: `airflow db init`, `airflow users create`, `airflow webserver` | `airflow dags test`, `airflow standalone` | Removed in Airflow 3 |
| DAG: FileSensor and `airflow dags trigger` task | Removed | They weren't connected to the pipeline, kept each run open for 24 hours, and Airflow 3 tasks can't run CLI commands against the database |
| Code in two copies (`pipeline/` and the root) | One copy at the root; `pipeline/` keeps the two printed entry points | The commands ran the root copy |
| `python -m pipeline.src.serve.simple_model_service` (troubleshooting) | `python -m src.serve.simple_model_service` | The copy is gone |
| APIs listen on `0.0.0.0` | `127.0.0.1` unless `HOST` is set | Not reachable from the network by default |

The full list of fixes is in [CHANGELOG.md](CHANGELOG.md).

## Data and licenses

| Data | Source | License |
|---|---|---|
| `oxfordrobocar_sample_small/` (INS, GPS, camera frames, LiDAR scans from one drive on 2014-12-12) | [Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/), W. Maddern, G. Pascoe, C. Linegar and P. Newman, "1 Year, 1000km: The Oxford RobotCar Dataset", IJRR 2017 | **CC BY-NC-SA 4.0** (non-commercial, attribution, share-alike). Not covered by this repository's MIT license |
| ResNet-18 ImageNet weights (downloaded by torchvision on first use) | [torchvision](https://pytorch.org/vision/stable/models.html) | BSD-3-Clause (torchvision); weights trained on ImageNet |

[`oxfordrobocar_sample_small/README.md`](oxfordrobocar_sample_small/README.md) lists every file and which ones the code uses. The code is MIT-licensed (see [LICENSE](LICENSE)).

## Expected output

From `python pipeline/run_pipeline.py --port 59599` (a free port; the default is 8000) in a fresh clone after the printed `pip install` lines (2026-10-24, macOS 26.5.1 arm64, Python 3.12.15). The log lines of the individual steps are left out; times are local. On Linux (CI) the scores can differ in the last digit.

```
12:27:59  Starting MLOps pipeline execution
12:27:59  Starting: Extracting features from INS data
12:28:40  Successfully completed: Extracting features from INS data (41 s)
12:28:40  Starting: Ingesting camera frames
12:29:04  Successfully completed: Ingesting camera frames (24 s)
12:29:04  Starting: Extracting camera features (ResNet-18)
12:29:21  Successfully completed: Extracting camera features (ResNet-18) (17 s)
12:29:21  Starting: Aligning camera frames with INS windows
12:29:24  Successfully completed: Aligning camera frames with INS windows (3 s)
12:29:24  Starting: Training model
Registered ins-turn-classifier version 1: test F1 0.880, accuracy 0.960 (2021 windows from the end of the drive)
12:29:39  Successfully completed: Training model (14 s)
12:29:39  Starting: Evaluating model
Evaluated on test period (end of the drive): 2021 windows, 352 positive
  champion  v1: F1 0.880, accuracy 0.960
12:29:42  Successfully completed: Evaluating model (3 s)
12:29:42  Starting: model serving API on http://127.0.0.1:59599
12:29:45  Model API is up: {'status': 'healthy', 'message': 'Model is loaded and ready', 'model_version': 1}
12:29:46  POST /predict with 25 readings -> {'prediction': 1, 'probability': 0.9957363605499268, 'prediction_text': 'Turn ahead', 'model_version': 1, 'timestamp': 1792870186017}
12:29:46  Starting: Replaying the end of the drive with a sensor fault
Replaying minutes 33.7-42.1 of the drive: 2021 windows in batches of 100; the INS readings get noisy from minute 36.5
Serving ins-turn-classifier v1

batch  minute sensor drift z served turns    F1 F1 v1 only  event
    0    33.7     ok     0.6     v1     0     -          -  
    1    34.1     ok     0.5     v1     0     -          -  
    2    34.5     ok     0.5     v1     0     -          -  
    3    35.0     ok     0.8     v1     0     -          -  
    4    35.4     ok     0.8     v1     0     -          -  
    5    35.8     ok     0.5     v1     0     -          -  
    6    36.2  noisy     2.5     v1    26  0.60       0.61  drift alert (velocity_east_std, velocity_north_std)
    7    36.6  noisy     6.1     v1    39  0.56       0.56  drift alert (velocity_north_std, velocity_east_std)
    8    37.0  noisy     6.2     v1    18  0.32       0.32  drift alert (velocity_east_std, velocity_north_std); retrained v2: F1 0.49 vs 0.45 -> promoted
    9    37.5  noisy     1.5     v2    20  0.69       0.34  
   10    37.9  noisy     2.1     v2    70  0.72       0.82  drift alert (yaw_std)
   11    38.3  noisy     1.5     v2    38  0.61       0.55  
   12    38.7  noisy     1.5     v2    14  0.30       0.25  
   13    39.1  noisy     1.9     v2    16  0.33       0.34  
   14    39.5  noisy     2.7     v2     0     -       0.00  drift alert (yaw_std)
   15    40.0  noisy     2.2     v2     0  0.00       0.00  drift alert (velocity_down_std)
   16    40.4  noisy     1.5     v2    21  0.76       0.38  
   17    40.8  noisy     1.5     v2    20  0.52       0.34  
   18    41.2  noisy     1.5     v2    21  0.37       0.35  
   19    41.6  noisy     1.5     v2    28  0.35       0.44  
   20    42.0  noisy     1.8     v2    21  0.83       1.00  

F1 before the fault: 0.878
F1 while the sensor is noisy: 0.503 with the loop, 0.439 if v1 had stayed in service
Champion at the end: v2
12:30:05  Successfully completed: Replaying the end of the drive with a sensor fault (20 s)
12:30:05  Model API after the replay: {'status': 'healthy', 'message': 'Model is loaded and ready', 'model_version': 2}
12:30:05  POST /predict with 25 readings -> {'prediction': 1, 'probability': 0.9977954626083374, 'prediction_text': 'Turn ahead', 'model_version': 2, 'timestamp': 1792870205573}
12:30:05  MLOps pipeline completed in 126 s. MLflow runs and models: `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`
```

How to read the replay:

- Batches 0-5 (minutes 33.7-36.2) have no turns, so F1 is undefined (`-`). The sensor is fine and the drift scores stay under 1.
- From minute 36.5 the INS readings are noisy. Batch 7 raises a severe drift alert (z 6.1). At batch 8 there are enough labeled noisy batches: v2 is trained with batch 6, beats v1 on batches 7-8 (F1 0.49 vs 0.45) and is promoted, and the API reloads it.
- From batch 9 the monitor compares batches with v2's training data, so the noisy readings no longer count as drift. The remaining alerts (z 2.1-2.7) are below the retraining threshold of 3.
- Over the noisy period F1 is 0.503 with the loop and 0.439 if v1 had stayed in service. Retraining helps, but noisy readings make turns harder to anticipate: before the fault, F1 was 0.88.

## How the closed loop works

```
 drive minutes 0-25: train   25-34: validate   34-42: test, then replayed as production
                                                         |
     served by the champion -> monitor (batch of 100 windows vs training data, z-score)
                                                         |  severe drift (z > 3)
                labeled batches since the drift started -> retrain -> candidate version
                                                         |
     candidate vs champion on the 2 newest labeled batches -> promote if F1 is higher -> API reloads
```

- **Sensor fault**: from minute 36.5, Gaussian noise (10% of each channel's standard deviation) is added to the INS readings. Labels come from the recorded, noise-free heading, standing in for ground truth that arrives later.
- **Drift**: the monitored features are the within-window standard deviations of each channel. A batch is flagged when its mean moves more than 2 training standard deviations (alert) or 3 (severe, retrain).
- **After a promotion** the monitor compares batches with the new champion's training data, so the alerts stop.
- **What to expect**: retraining helps but can't undo a noisy sensor. Turns are harder to anticipate from noisy readings, so F1 recovers only part of the way. The settings are in `config/params.yaml` (`monitoring`, `simulation`).

## Troubleshooting

- **`Port 8000 is already in use`**: pass `--port 8010` (or another free port) to `run_pipeline.py`.
- **macOS: segmentation fault** when your own code trains XGBoost in a process that also ran PyTorch: the two bundle different OpenMP runtimes. Run the PyTorch steps in a separate process, as `run_pipeline.py` does.
- **Start over**: `make clean-data` removes everything under `experiments/` (runs, models, monitoring) and `logs/`.
- **Rule-based model**: the first edition's rule-based service is still available: `python -m src.serve.simple_model_service`, `uvicorn src.serve.simple_api:app`.
- **Import errors** when running a file directly: run modules from the repository root with `python -m`, as above.

## Repository structure

```
├── run_pipeline.py                 # Runs the complete pipeline (python run_pipeline.py)
├── pipeline/                       # The printed entry points: run_pipeline.py, requirements.txt
├── src/
│   ├── ingest/                     # INS and camera ingestion
│   ├── features/                   # INS window features, camera features
│   ├── align/                      # Temporal alignment
│   ├── models/                     # Dataset, labels, training, evaluation and promotion
│   ├── drift/                      # Statistical drift tests and drift injection
│   ├── monitoring/                 # Monitor, drift handler, closed-loop replay, dashboard
│   ├── serve/                      # FastAPI model service (and the rule-based one)
│   ├── utils/                      # Configuration, logging, MLflow registry helpers
│   └── pipeline/                   # Single-process pipeline (python -m src.pipeline.pipeline)
├── airflow/dags/                   # Airflow 3 DAG
├── pipeline.yaml                   # Argo Workflows definition (needs a container image of this repository)
├── config/                         # params.yaml (pipeline settings), schema.yaml (data sources)
├── enhancements_modules/           # Pydantic schemas, Parquet feature store, demo
├── tests/                          # Unit tests and smoke tests
├── oxfordrobocar_sample_small/     # The sample drive (CC BY-NC-SA 4.0)
├── experiments/                    # Outputs: features, models, MLflow database, monitoring
├── notebooks/                      # Exploration notebook
└── docs/                           # Architecture diagram
```

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## License

The code is licensed under the MIT License; see the [LICENSE](LICENSE) file. The sample data in `oxfordrobocar_sample_small/` is licensed separately under CC BY-NC-SA 4.0 (see [Data and licenses](#data-and-licenses)).

## Acknowledgments

- Oxford Robotics Institute for providing the RobotCar dataset
- Apache Airflow team for their excellent workflow orchestration framework
- The MLOps community for best practices and inspiration
