# Changelog

## 2026.10 (2026-10-24)

Maintenance update for current Python and libraries. The code as printed in the book is tagged
`first-edition`.

### Fixed
- `python pipeline/run_pipeline.py` called `src.models.train_model` and `src.models.evaluate_model`,
  which didn't exist, so it stopped after the first step; it also never exited. Both modules now
  exist (the Airflow DAG and the drift handler call them too), and the script runs every step once and
  exits (`--serve` keeps the API and monitoring running).
- INS timestamps under pandas 3: the conversion back from datetimes assumed nanoseconds, which turned
  microseconds into milliseconds. The windows became 1000 times too long (9 instead of 10,107) and
  alignment produced nothing.
- MLflow 3 no longer accepts new `./mlruns` file stores; tracking now uses a local SQLite file
  (`sqlite:///experiments/mlflow.db`).
- `python -m src.pipeline.pipeline` crashed (segmentation fault) on macOS: PyTorch and XGBoost in one
  process. The camera steps run in their own process, and importing `src.features` no longer loads
  PyTorch.
- The camera frames are raw Bayer sensor images; they were read as grayscale. They are now demosaiced
  as the dataset's SDK does.
- The monitor saved (and cleared) each batch before checking it for drift, so drift was never checked.
- `inject_drift` added noise to accelerometer/gyroscope columns the INS data doesn't have.
- `find_nearest_frame` could pick the wrong camera frame after sorting (`.iloc` with an index label).
- The drift handler restarted the API with `pkill`/`nohup` and called the missing modules.
- `enhancements_modules`: imports still used the old folder name `MLOps_Enhancements`; validators
  moved to pydantic 2; headings from 0 to 2π (as RobotCar reports them) are accepted.
- Tests: 5 of 10 failed (they described raw IMU data from before the switch to INS); 2 test files didn't
  import. All pass.
- Airflow DAG: tasks had no working directory, the deploy step never ran after a branch
  (`trigger_rule`), and the first run trained nothing. The unconnected FileSensor (open for 24 hours)
  and `airflow dags trigger` task were removed.

### Changed
- The model trains on all 10,103 INS windows of the drive, split by time (60/20/20), to predict a turn
  in the next second from the motion so far. The first edition trained on the 15 windows with a camera
  frame, with a label that is one of the inputs (still available: `label.name: east_velocity`).
  Camera features are optional (`model.use_video_features`).
- Models are registered in the MLflow Model Registry in XGBoost's own format (no pickle); the served
  version has the alias `champion`. The StandardScaler is gone: tree splits don't depend on feature
  scale, and MLflow's scikit-learn loader imports PyTorch, which crashed the API on macOS.
  `experiments/models/model.pkl` is still written; there is no `scaler.pkl`.
  A retrained version is promoted only if it scores a higher F1 than the champion on the newest
  labeled data.
- The API serves that model (the first edition's model service used a fixed rule on `velocity_east`).
  `/predict` accepts a window of INS readings as well as a single reading. Both APIs listen on
  127.0.0.1 unless `HOST` is set.
- `run_pipeline.py` replays the end of the drive with a simulated sensor fault instead of generating
  random monitoring data (`src/monitoring/force_monitoring_data.py` is still there).
- Python 3.12+, `pyproject.toml`, `uv.lock`, pinned `requirements.txt`; `pipeline/requirements.txt`
  includes it. `opencv-python-headless` replaces `opencv-python`; on Linux, `xgboost-cpu` replaces
  `xgboost` and torch comes from the CPU index.
- INS features about 6 times faster (same values).

### Added
- `src/monitoring/closed_loop.py`: drift alerts, retraining, promotion and the F1 without retraining,
  batch by batch.
- `oxfordrobocar_sample_small/README.md`: source, license (CC BY-NC-SA 4.0) and use of each file.
- Smoke tests (`make smoke`) and CI on Python 3.12 and 3.13, including the printed pip install and
  an Airflow 3 run of the DAG.

### Removed
- `pipeline/`, a second copy of the code (the commands ran the root copy). `pipeline/requirements.txt`
  and `pipeline/run_pipeline.py` remain so the printed commands work.
- `setup.py` (replaced by `pyproject.toml`), the committed Airflow SQLite side files, and three unused
  requirements (duckdb, fastparquet, tqdm).
- The Makefile target `create-pipeline-dag` (its module never existed).
