# Playbook 08: Build an MLOps pipeline that catches drift and retrains
**How to build an end-to-end MLOps pipeline for multimodal sensor data.** Chapter 8 of [*Data Engineering for Multimodal AI*](https://oreil.ly/data-engineering-multimodal) (O'Reilly, 2026), pages 294–311.

Models trained on sensor data drift as conditions change, and catching that takes a pipeline, not a one-off script. In this repo, we explore this with ForgeSense, a fictional robotics AI company, and a subset of the Oxford RobotCar dataset. Follow along to build ingestion, alignment, feature extraction, drift detection and automated retraining, with schema checks, tests and CI.

**The problem.** A model that worked at launch slowly gets worse as the data changes, and someone has to notice and fix it.

**What you'll work with.** You'll build the MLOps lifecycle as code: validate schemas, detect drift, retrain automatically, and test every change in CI.

> Following the book? Its steps are in [Run it the way the book does](#run-it-the-way-the-book-does) and still work as printed. Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## Run it

You need [uv](https://docs.astral.sh/uv/) and `make`. No Docker, no GPU, no API keys; the first run downloads the ResNet-18 weights (45 MB). Or open the repo in a GitHub Codespace, where uv and `make` are ready. On Windows without `make`, use a Codespace or the book's steps below.

**1. Get the code.** The clone is about 376 MB, most of it the sample drive.

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-08-multimodal-mlops.git
cd playbook-08-multimodal-mlops
```

**2. Run the whole pipeline.** Installs the locked dependencies (`uv sync`) and runs every step once (`python pipeline/run_pipeline.py`): INS features, camera ingest and features, alignment, training with MLflow, evaluation, the model API, and the replay of the end of the drive with a sensor fault, where monitoring raises drift alerts and the model is retrained and promoted. On a Mac this took 6 min 44 s while other jobs were running; on the same Mac with nothing else running it took 126 s (see [Tested on](#tested-on)).

```bash
make run
```

You should see, among the log lines:

```
Registered ins-turn-classifier version 1: test F1 0.880, accuracy 0.960 (2021 windows from the end of the drive)
    7    36.6  noisy     6.1     v1    39  0.56       0.56  drift alert (velocity_north_std, velocity_east_std)
    8    37.0  noisy     6.2     v1    18  0.32       0.32  drift alert (velocity_east_std, velocity_north_std); retrained v2: F1 0.49 vs 0.45 -> promoted
F1 while the sensor is noisy: 0.503 with the loop, 0.439 if v1 had stayed in service
Champion at the end: v2
```

[Expected output](#expected-output) shows the whole replay table and how to read it.

**3. See the model versions.** Lists what the MLflow Model Registry holds after step 2 and which version carries the `champion` alias.

```bash
make models
```

```
ins-turn-classifier: 2 version(s), champion v2

version  trained on                         test F1  run
     v1  the drive                            0.880  training_1795325546
     v2  the drive + 100 production windows   0.882  retraining_1795325673     <- champion
```

The run names end in a timestamp, so yours differ. To browse the runs, parameters and metrics: `uv run mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`, then http://127.0.0.1:5000.

**4. Ask the model.** Starts the model API on port 8000 (`uvicorn src.serve.api:app`). Leave it running.

```bash
make api
```

You should see `Application startup complete.` after about 20 s. In a second terminal, send it half a second of the drive (25 readings, as `POST /predict` with `{"readings": [...]}`) and compare its answer with what the car did next:

```bash
make predict MINUTE=36.4
```

```
POST /predict with 25 readings from minute 36.40 (speed 4.6 m/s)
  model v2: Turn ahead (probability of a turn 0.997)
  what happened: Turn ahead (heading changed 0.334 rad/s in the next 1 s; a turn is over 0.1)
```

Not every answer is right. At `MINUTE=36.65` the model says `Driving straight` (probability 0.034) and the car turned: a gentle turn, 0.125 rad/s, just over the threshold. Try other minutes between 0 and 42. The API answers with the champion, v2 after step 2. Stop it with Ctrl+C.

**5. Clean up.**

```bash
make clean
```

This deletes everything the pipeline wrote under `experiments/` (features, models, the MLflow database, monitoring, your own drives' folders) and `logs/`. The sample data and `.venv/` stay.

## How much time do you have?

| Time | What to do |
|---|---|
| 15 minutes | [Run it](#run-it), steps 1 to 4. On a Mac, step 2 took between 2 and 7 minutes depending on what else was running. |
| 1 hour | The three [Try this](#try-this) experiments (5 minutes together on a busy Mac), then the MLflow UI and the dashboard (`uv run streamlit run src/monitoring/dashboard.py`). |
| 1 day | [Use your own data](#use-your-own-data): run the loop on your own INS log, then work through [Before production](#before-production) and the Airflow DAG. |

## Try this

Each experiment changes one thing and measures the effect. They read what step 2 wrote and change nothing: no new model versions, no promotions. They are in `try_this/` because `make clean` empties `experiments/`. The numbers below are from a Mac on 2026-11-21.

**1. How big a sensor fault does monitoring catch?** The replay adds noise of 0.1 times each channel's standard deviation from minute 36.5. This rebuilds the replayed stream with other noise levels and reports, without retraining, what monitoring sees and how the first model does.

```bash
make try-noise-level
```

```
noise noisy batches with alert severe  max z F1 of v1 while noisy
 0.00            15          0      0    1.4              0.879
 0.02            15          0      0    1.9              0.514
 0.05            15         14      6    3.4              0.445
 0.10            15         15     14    6.7              0.439
 0.20            15         15     15   13.5              0.436
```

At 0.02, F1 drops from 0.879 to 0.514 and monitoring raises no alert: the drift score never passes 2. A small fault costs most of the accuracy before the input statistics move. Watching the F1 of labeled batches, which the monitor already computes, would have caught it. This took 3 min 25 s on a busy Mac.

**2. How many windows should monitoring compare at a time?** The monitor compares batches of 100 windows (25 s of driving) with the training data. This counts false alerts on the fault-free part of the drive (the validation period and the replay before the fault) and how long the alerts take after the fault, for other batch sizes.

```bash
make try-batch-size
```

```
batch size seconds   false alerts  false severe  first alert after fault  first severe after fault
        25       6       5 of 107       0 of 107                       1 s                       1 s
        50      12       1 of 54        0 of 54                        8 s                       8 s
       100      25       0 of 27        0 of 27                        8 s                      33 s
       200      50       0 of 14        0 of 14                       33 s                      33 s
       400     100       0 of 7         0 of 7                        33 s                     133 s
```

Small batches react within seconds but raise false alerts; large ones are quiet and slow. With 100, a retraining alert comes 33 s after the fault starts. Change `monitoring.batch_size` or the z thresholds in `config/params.yaml`.

**3. What does a random train/test split hide?** The pipeline splits the windows by time; the first edition split them at random. This trains the same XGBoost model both ways on the first 80% of the drive and compares the test F1 with the F1 on the last 20%, the minutes the model faces next.

```bash
make try-split
```

```
split of the first 80% of the drive       test F1  F1 on the future
random (first edition)                      0.943             0.882
by time (this version)                      0.828             0.880

Random split: 94% of the test windows have a training window that overlaps them by half (the next or previous window).
```

The random split promises 0.943 and delivers 0.882: almost every test window has an overlapping neighbor in the training set. The split by time gives a lower estimate (0.828) here, because minutes 25 to 34 are harder than the end of the drive, but it never tests on windows the model has seen. This took 9 s.

## Use your own data

Put an INS log in a folder:

```
my_drive/
  ins.csv      (or gps/ins.csv, as in the Oxford RobotCar download)
```

One row per reading, with a header row. The pipeline needs these columns; others are ignored:

| Column | Unit |
|---|---|
| `timestamp` | UNIX time in microseconds |
| `velocity_north`, `velocity_east`, `velocity_down` | m/s |
| `roll`, `pitch`, `yaw` | radians |

Check it first. This reads the file and writes nothing:

```bash
make check-data DIR=my_drive
```

It reports every problem it finds and stops if there is any. On a test file (two minutes of the sample drive, thinned to 10 Hz, yaw converted to degrees, one row repeated at the end):

```
4 problem(s) in bad_drive; nothing was written:
  ins.csv: timestamp: 1 duplicate value(s), first 1418381453659714
  ins.csv: timestamp: goes backwards 1 time(s); sort the rows by time
  ins.csv: 10.0 readings per second; needs at least 25 (the pipeline resamples to 50 Hz)
  ins.csv: yaw: values up to 359.1; needs radians (degrees?)
```

It also checks for missing columns (the loader would fill them with empty values without a message), values that aren't numbers, columns more than 5% empty, timestamps in seconds or milliseconds, roll or pitch in degrees, speeds over 100 m/s, Git LFS pointers, and whether the drive is long enough and has both turns and straight driving in each of the training, validation and test periods (by time, 60/20/20). When everything passes, run the loop on it:

```bash
make load DIR=my_drive
```

This runs the pipeline in its own folder, `experiments/own/my_drive/`, with its own copy of `config/` and its own MLflow database, so the sample's models are not touched: features, training (`python -m src.models.train_model`), evaluation, and the replay with a sensor fault (`python -m src.monitoring.closed_loop`) on the last 20% of your drive. On a test drive of 15 minutes cut from the sample (minutes 5 to 20), this took 3 min 14 s on a busy Mac:

```
my_drive: 44999 readings, 15.0 minutes at 50 Hz. Turn windows by period: training 225 of 2156, validation 77 of 719, test 94 of 719. All checks passed.
Working folder: experiments/own/my_drive/ (its own config/, experiments/ and MLflow database)
Registered ins-turn-classifier version 1: test F1 0.837, accuracy 0.958 (720 windows from the end of the drive)
...
    4    13.7  noisy     5.8     v1    42  0.59       0.59  drift alert (velocity_east_std, velocity_north_std); retrained v2: F1 0.48 vs 0.37 -> promoted
...
F1 while the sensor is noisy: 0.269 with the loop, 0.269 if v1 had stayed in service
MLflow runs and models: mlflow ui --backend-store-uri sqlite:///experiments/own/my_drive/experiments/mlflow.db
```

On this shorter drive the loop catches the fault and promotes a retrained v2, but the F1 while the sensor is noisy is the same with and without it: the replay is 3 minutes, and only 3 batches follow the promotion, two of them without a turn. The minutes in the replay table count from the start of your file.

`make clean` deletes these folders too.

## What you end up with

A model lifecycle you can rerun on your own sensor log: features, a registered and evaluated model, a served API, and a monitoring loop that retrains and promotes, with measurements of when it notices a fault and when it doesn't.

Skills you'll practice:
- Building time-windowed features and labels from a sensor stream
- Splitting time series by time, and measuring what a random split hides
- Tracking runs and model versions in MLflow, with an alias for the served version
- Detecting drift against a model's training data, and choosing batch sizes and thresholds
- Retraining on new labeled data and promoting a model only when it does better
- Validating a sensor log before it enters the pipeline

## Before production

This playbook runs on one machine with one recorded drive. Before you run something like it for real:

1. **Alert on the model's errors, not only on its inputs.** With noise of 0.02 times each channel's standard deviation, F1 fell from 0.879 to 0.514 and no drift alert fired ([Try this 1](#try-this)). The monitor already scores every labeled batch; alert on that score too. *Chapter 10, "Detecting Drift" (page 384); Chapter 8, "Observability for Multimodal Systems" (page 287).*
2. **Choose the batch size from your alert budget.** Batches of 25 windows raised 5 false alerts in 107 fault-free batches; batches of 400 waited 133 s before a retraining alert ([Try this 2](#try-this)). *Chapter 7, "Drift Types, Detection, and Engineering Response" (page 229).*
3. **Split by time, and keep the future out of training.** A random split scored 0.943 on its test set and 0.882 on the minutes that followed ([Try this 3](#try-this)). *Chapter 8, "Testing and Validation for Multimodal Models" (page 286).*
4. **Log metrics on data the model hasn't seen.** v2's logged test F1 (0.882, `make models`) is measured on the end of the drive, which includes the 100 production windows it was retrained on. The promotion decision uses held-out batches, but the registry's number doesn't. *Chapter 8, "Data Versioning and Experiment Tracking for Multimodal AI" (page 276).*
5. **Promote on more evidence.** `evaluate_model` promotes a candidate whose F1 is higher on the 2 newest labeled batches (200 windows); v2 won 0.49 to 0.45. Run a candidate in shadow or as a canary first, and keep a rollback. *Chapter 10, "Canary Deployment" (page 397); Chapter 10, "Rollback" (page 400).*
6. **Plan for labels that arrive late.** The replay labels each batch from the recorded heading right after serving it. Real ground truth comes later, and retraining has to wait for it. *Chapter 8, "Automated Retraining Pipelines for Multimodal AI" (page 279); Chapter 10, "Feedback Loops" (page 385).*
7. **Validate data at ingestion.** `load_ins_data` fills missing columns with empty values, renames columns by position when the names don't match, and returns an empty table on any error. `make check-data` catches these; the pipeline itself doesn't stop. *Chapter 8, "Validation Architecture" (page 286).*
8. **Lock down the API.** `POST /reload` and `POST /feedback` have no authentication: anyone who reaches the port can make the API reload its model or add labels. *Chapter 11, "Protecting Data Across the Pipeline" (page 421).*

## What it costs

| Where you run it | Cost |
|---|---|
| Your machine | $0. Everything runs locally: no API keys, no paid services. The environment is about 1.3 GB, the clone about 376 MB. |
| A GitHub Codespace (2 cores, 8 GB, the default size; this repo asks for no larger one) | Estimated $0.18 per hour at GitHub's list price (November 2026). A free personal account includes 120 core-hours a month, about 60 hours on this machine. In CI, `make run` took 59 s in a freshly built devcontainer. |

## How it works

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

## Run it the way the book does

These are the book's steps (pages 294–311), as printed apart from the fixes listed in [Differences from the printed book](#differences-from-the-printed-book). [Run it](#run-it) above runs the same pipeline with `make`.

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

`uv run pytest -q tests/test_own_data.py` checks the file checks behind `make check-data` (under a minute). `make models`, `make api`, `make predict`, `make check-data`, `make load` and `make try-*`: see [Run it](#run-it), [Try this](#try-this) and [Use your own data](#use-your-own-data).

**The book's step-by-step commands (pages 295–301)** still work, in this order:

```bash
python -m src.ingest.video_ingest
python -m src.ingest.ins_ingest
python -m src.features.ins_features
python -m src.features.video_features
python -m src.align.temporal_alignment
python -m src.models.train          # now builds the dataset from every INS window, then trains and registers
python -m src.models.predict
python -m src.serve.model_service
python -m src.serve.client
```

`python -m src.serve.http_client` and `python -m src.monitoring.simulate_traffic` talk to the API, so start it first in another terminal: `uvicorn src.serve.api:app`.

### Orchestration with Airflow

The DAG `airflow/dags/ins_mlops_pipeline.py` is written for Airflow 3 (tested with 3.3.2). Install Airflow in its own environment, using the constraints file for your Python version, and point the DAG at the playbook's Python:

```bash
python -m venv airflow-venv
airflow-venv/bin/pip install "apache-airflow==3.3.2" \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt

export AIRFLOW_HOME=$(pwd)/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
export PLAYBOOK_PYTHON=$(pwd)/venv/bin/python   # the environment from the book's steps above

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
| Python | 3.12.15 (macOS); 3.12.3 and 3.13.17 (CI) |
| Key packages | pandas 3.1.0, scikit-learn 1.9.1, XGBoost 3.4.1, MLflow 3.17.2, PyTorch 2.14.1 (CPU), FastAPI 0.162.0, Streamlit 1.66.1; Airflow 3.3.2 for the DAG (macOS and CI) |

On the MacBook (fresh clone, Python 3.12.15): the three `pip install` lines above took 87 s; `python pipeline/run_pipeline.py` took 126 s (INS features 41 s, camera ingest 24 s, ResNet-18 features 17 s with the weights already downloaded, training 14 s, the closed-loop replay 20 s). `make smoke` runs the tests, including the whole pipeline, in about 2 minutes.

## Differences from the printed book

The left column is the first edition: the code at the `first-edition` tag, which the book describes. Commands that changed still work unless the table says otherwise.

| First edition | This version | Why |
|---|---|---|
| Python 3.8+ (conda, Python 3.9) | Python 3.12+ | Current libraries need it |
| Unpinned `>=` requirements | Pinned versions (`requirements.txt`, `uv.lock`); `pipeline/requirements.txt` includes them | Reproducible installs |
| `python pipeline/run_pipeline.py` stopped at step 2 and never exited | Runs every step once and exits; `--serve` keeps serving | Steps 2 and 3 called modules that weren't in the repository; now they are |
| `python -m src.models.train` (p. 298) trained on the aligned dataset | Builds the dataset from every INS window if needed, then trains and registers the model (as `python -m src.models.train_model` does) | The training data changed (next row) |
| Model trained on the 15 INS windows that have a camera frame; label: `velocity_east_mean > 7` (an input) | All 10,103 INS windows; label: a turn in the next second; split by time | 15 samples can't train or test a model; a label that is an input is learned trivially; overlapping windows leak across a random split. The old label: `label.name: east_velocity` |
| Camera features always part of the model | Optional (`model.use_video_features`) | The sample has 3.6 s of camera frames in a 42-minute drive |
| Camera frames read as grayscale | Demosaiced to color | They are raw Bayer sensor images |
| MLflow `./experiments/mlruns` | `sqlite:///experiments/mlflow.db`, Model Registry with the `champion` alias | MLflow 3 no longer accepts new file stores; stages were replaced by aliases |
| `model.pkl` and `scaler.pkl` (StandardScaler + XGBoost) | XGBoost alone, registered in XGBoost's format; `model.pkl` still written | Trees don't need scaling; MLflow's scikit-learn loader imports PyTorch, which crashes XGBoost on macOS |
| Served model: a fixed rule on `velocity_east` | The registered champion; `/predict` also accepts a window of readings | Serve the trained model |
| Synthetic random monitoring data; drift checked against earlier batches | Replay of the end of the drive with a simulated sensor fault; drift checked against the model's training data; F1 per labeled batch | Shows the loop working on real data. `src/monitoring/force_monitoring_data.py` is still there |
| Drift handler restarted the API with `pkill`/`uvicorn --reload` | Calls `/reload` on the API after a promotion | Don't kill other processes |
| Airflow commands (p. 307): `airflow db reset -y`, `db migrate`, `users create`, `api-server --port 8081`, `scheduler`, `dag-processor`, `triggerer`; log in as admin/admin | `airflow dags test`, `airflow standalone` | Airflow changed: in Airflow 3.3.2 `airflow users create` is not a command unless the FAB auth manager is installed ("invalid choice: 'users'", checked 2026-10-27), so there is no admin/admin user. `airflow standalone` creates an admin user and prints its password. The other printed commands still exist |
| DAG: FileSensor and `airflow dags trigger` task | Removed | They weren't connected to the pipeline, kept each run open for 24 hours, and Airflow 3 tasks can't run CLI commands against the database |
| Code in two copies (`pipeline/` and the root) | One copy at the root; `pipeline/` keeps the two first-edition entry points | The commands ran the root copy |
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

From `python pipeline/run_pipeline.py --port 59599` (a free port; the default is 8000) in a fresh clone after the `pip install` lines above (2026-10-24, macOS 26.5.1 arm64, Python 3.12.15). The log lines of the individual steps are left out; times are local. This is the output of this version; it may differ from the output printed in the book, and that's okay.

XGBoost's results differ slightly between platforms, and the loop can take a different path. In CI on Linux (Ubuntu 24.04, Python 3.12.3) the test F1 is 0.871, the retrained v2 and v3 lose to v1 on the newest batches and are kept out of service, v4 is promoted at batch 12, and F1 over the noisy period is 0.508 with the loop and 0.396 without.

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
├── pipeline/                       # The first-edition entry points: run_pipeline.py, requirements.txt
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
├── scripts/                        # models.py, predict.py (make models, make predict), own_drive.py (make check-data, make load)
├── try_this/                       # make try-*: the three Try this experiments
├── tests/                          # Unit tests, smoke tests, test_own_data.py
├── oxfordrobocar_sample_small/     # The sample drive (CC BY-NC-SA 4.0)
├── experiments/                    # Outputs: features, models, MLflow database, monitoring; own/ for make load
├── notebooks/                      # Exploration notebook
└── docs/                           # Architecture diagram
```

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

If this playbook helped you, a star helps other readers find it.

## License

The code is licensed under the MIT License; see the [LICENSE](LICENSE) file. The sample data in `oxfordrobocar_sample_small/` is licensed separately under CC BY-NC-SA 4.0 (see [Data and licenses](#data-and-licenses)).

## Acknowledgments

- Oxford Robotics Institute for providing the RobotCar dataset
- Apache Airflow team for their excellent workflow orchestration framework
- The MLOps community for best practices and inspiration
