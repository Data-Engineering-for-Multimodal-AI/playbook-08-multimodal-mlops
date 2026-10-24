#!/usr/bin/env python
"""
Replay the end of the drive through the served model, with a sensor fault part way through, and
let monitoring, retraining and promotion respond:

    python -m src.monitoring.closed_loop

1. Production stream: the test period (the last part of the drive, which no model trained on),
   cut into half-second windows. After simulation.fault_start of it, Gaussian noise is added to the
   INS readings (simulation.noise_scale of each channel's standard deviation), as a degrading
   sensor would. Labels come from the recorded, noise-free heading: they stand in for ground truth
   that arrives after the fact.
2. Each batch of monitoring.batch_size windows is served by the champion (INSPredictionService)
   and logged by the monitor (ModelMonitor), which compares the batch with the champion's
   training data and writes a drift alert when a feature moves more than drift_z_threshold
   standard deviations.
3. When the labels arrive, the monitor stores the labeled batch and scores the model on it.
4. On a severe alert (severe_z_threshold), the drift handler retrains on the labeled production
   batches since the drift started, evaluate_model compares the new version with the champion on
   the newest batches, and the service switches to the new version only if it was promoted.

The replay starts from the newest model trained on the drive alone (it moves the 'champion'
alias back if an earlier replay promoted a retrained version). That model is also run on every
batch without ever being replaced, to show what the loop changed. Results: experiments/monitoring/closed_loop_summary.json.
"""
import argparse
import json
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.features.ins_features import create_sliding_windows, extract_window_features
from src.models.dataset import INS_DATA_PATH
from src.models.labels import make_labels
from src.models.train import chronological_split
from src.monitoring.drift_handler import DriftHandler
from src.monitoring.monitoring_service import ModelMonitor
from src.serve.model_service import INSPredictionService
from src.utils import get_param, get_pipeline_logger
from src.utils.registry import champion_version, latest_base_version, load_model, set_champion, setup_mlflow

logger = get_pipeline_logger("closed_loop")

CHANNELS = ['velocity_north', 'velocity_east', 'velocity_down', 'roll', 'pitch', 'yaw']
MONITOR_DIR = os.path.join("experiments", "monitoring")


def production_stream() -> pd.DataFrame:
    """Windows of the test period with the simulated sensor fault, their labels and a fault flag."""
    ins = pd.read_csv(INS_DATA_PATH).dropna(subset=CHANNELS).sort_values('timestamp').reset_index(drop=True)
    windows = pd.read_parquet(os.path.join(get_param('alignment.output_path'), 'ml_dataset', 'labels.parquet'))
    _, _, test = chronological_split(len(windows), get_param('pipeline.test_size'), get_param('pipeline.validation_size'))
    start = int(windows['timestamp'].iloc[test.start])
    end = int(ins['timestamp'].iloc[-1])
    fault_time = start + get_param('simulation.fault_start') * (end - start)

    window_us = get_param('features.imu.window_size') * 1e6
    readings = ins[ins['timestamp'] >= start - window_us / 2].copy()
    rng = np.random.default_rng(get_param('simulation.seed'))
    faulty = readings['timestamp'] >= fault_time
    for channel in CHANNELS:
        noise = rng.normal(0, get_param('simulation.noise_scale') * ins[channel].std(), int(faulty.sum()))
        readings.loc[faulty, channel] += noise

    rows = [extract_window_features(w, get_param('features.imu.fft_components'))
            for w in create_sliding_windows(readings, get_param('features.imu.window_size'),
                                            get_param('features.imu.overlap'))]
    stream = pd.DataFrame(rows)
    stream = stream[stream['timestamp'] >= start].reset_index(drop=True)
    stream['label'] = make_labels(stream, ins)
    stream['fault'] = stream['timestamp'] >= fault_time
    stream['drive_minute'] = (stream['timestamp'] - ins['timestamp'].iloc[0]) / 60e6
    return stream


def reset_monitoring() -> None:
    """Start a new monitoring period: remove the files earlier runs of the loop wrote."""
    for folder, prefixes in [("feedback", ("batch_",)), ("data_drift", ("drift_alert_", "feature_stats_")),
                             ("performance", ("performance_",)), ("predictions", ("predictions_",))]:
        path = os.path.join(MONITOR_DIR, folder)
        if os.path.isdir(path):
            for name in os.listdir(path):
                if name.startswith(prefixes):
                    os.remove(os.path.join(path, name))


def quiet_component_logs() -> None:
    """The table below reports what happens; keep the components' own logs to errors."""
    for component in ["model_monitoring", "drift_handler", "model_service", "ins_features"]:
        logging.getLogger(f"forge_pipeline.{component}").setLevel(logging.ERROR)


def run(api_url: str = None) -> Dict[str, Any]:
    quiet_component_logs()
    reset_monitoring()
    stream = production_stream()
    monitored = get_param('monitoring.monitor_features')
    batch_size = int(get_param('monitoring.batch_size'))
    cooldown = int(get_param('monitoring.retrain_cooldown_batches'))
    holdout = int(get_param('monitoring.holdout_batches'))

    # Start from the newest model trained on the drive alone: a model an earlier replay promoted
    # has already learned from these windows
    client = setup_mlflow()
    base = latest_base_version(client)
    if base is None:
        raise SystemExit("No registered model: run `python -m src.models.train_model` first")
    if champion_version(client) != base:
        set_champion(client, base)
        logger.warning(f"Champion reset to v{base}, the newest model trained on the drive alone")
    service = INSPredictionService()
    service.load()
    monitor = ModelMonitor(MONITOR_DIR)
    monitor.set_reference(os.path.join(MONITOR_DIR, "reference", f"reference_v{service.version}.json"))
    handler = DriftHandler(restart_service=api_url is not None, api_url=api_url or "")
    first_version = service.version
    first_model = load_model(first_version)   # the model as first deployed, never replaced

    print(f"Replaying minutes {stream['drive_minute'].iloc[0]:.1f}-{stream['drive_minute'].iloc[-1]:.1f} "
          f"of the drive: {len(stream)} windows in batches of {batch_size}; the INS readings get noisy "
          f"from minute {stream.loc[stream['fault'], 'drive_minute'].iloc[0]:.1f}")
    print(f"Serving {get_param('model.registered_name')} v{service.version}\n")
    print(f"{'batch':>5} {'minute':>7} {'sensor':>6} {'drift z':>7} {'served':>6} {'turns':>5} "
          f"{'F1':>5} {'F1 v' + str(first_version) + ' only':>10}  event")

    rows, drift_start, last_retrain = [], None, -cooldown
    for batch_id, begin in enumerate(range(0, len(stream), batch_size)):
        batch = stream.iloc[begin:begin + batch_size]
        X = np.array(batch['statistical_features'].tolist(), dtype=float)
        predictions, probabilities = service.predict_windows(X)
        served = service.version

        # Serve and log each window; the monitor checks for drift when the batch is complete
        monitor.last_drift = {"max_z_score": 0.0, "drift_detected": False, "features": []}
        for i, (_, window) in enumerate(batch.iterrows()):
            monitor.log_prediction({f: float(window[f]) for f in monitored}, int(predictions[i]),
                                   float(probabilities[i]), int(window['timestamp']))
        if len(batch) < batch_size:
            monitor.end_batch()                  # the last, partial batch
        drift = monitor.last_drift

        # Labels arrive: store the labeled batch and score the served model and the first model
        metrics = monitor.record_labeled_batch(batch_id, X, batch[['timestamp'] + monitored],
                                               batch['label'].to_numpy(), predictions)
        first_f1 = f1_score(batch['label'], first_model.predict(X), zero_division=0)

        event = ""
        if drift['drift_detected']:
            drift_start = batch_id if drift_start is None else drift_start
            event = "drift alert (" + ", ".join(drift['features'][:2]) + ")"
        # Retrain once there are labeled batches since the drift started beyond the ones held out
        # for the comparison, and not more often than every retrain_cooldown_batches batches
        enough_data = drift_start is not None and batch_id - drift_start + 1 > holdout
        if handler.check_for_drift() and enough_data and batch_id - last_retrain >= cooldown:
            last_retrain = batch_id
            promoted = handler.retrain_model(since=f"batch_{drift_start:04d}.parquet")
            evaluation = handler.last_evaluation or {}
            if 'candidate' in evaluation:
                event += (f"; retrained v{evaluation['candidate']}: F1 "
                          f"{evaluation['candidate_metrics']['f1']:.2f} vs "
                          f"{evaluation['champion_metrics']['f1']:.2f} -> "
                          f"{'promoted' if promoted else 'kept v' + str(evaluation['champion_before'])}")
            if promoted and service.reload_if_changed():
                monitor.set_reference(os.path.join(MONITOR_DIR, "reference", f"reference_v{service.version}.json"))
                drift_start = None

        row = {'batch': batch_id, 'drive_minute': round(float(batch['drive_minute'].iloc[0]), 2),
               'sensor_fault': bool(batch['fault'].any()), 'max_z_score': round(drift['max_z_score'], 2),
               'served_version': served, 'turn_windows': int(batch['label'].sum()),
               'f1': round(metrics['f1'], 3), f'f1_first_model': round(float(first_f1), 3), 'event': event}
        rows.append(row)
        # F1 is undefined for a batch without turns (and with no turn predicted)
        def shown(f1, preds):
            return "-" if row['turn_windows'] == 0 and not np.any(preds) else f"{f1:.2f}"
        print(f"{batch_id:>5} {row['drive_minute']:>7.1f} {'noisy' if row['sensor_fault'] else 'ok':>6} "
              f"{row['max_z_score']:>7.1f} {'v' + str(served):>6} {row['turn_windows']:>5} "
              f"{shown(row['f1'], predictions):>5} {shown(first_f1, first_model.predict(X)):>10}  {event}",
              flush=True)

    fault = stream['fault'].to_numpy()
    served_predictions = np.concatenate([
        pd.read_parquet(os.path.join(MONITOR_DIR, "feedback", f"batch_{r['batch']:04d}.parquet"))['prediction']
        for r in rows])
    X_all = np.array(stream['statistical_features'].tolist(), dtype=float)
    summary = {
        'windows': int(len(stream)),
        'fault_windows': int(fault.sum()),
        'first_version': first_version,
        'final_version': service.version,
        'promotions': [r['event'] for r in rows if 'promoted' in r['event']],
        'f1_before_fault': float(f1_score(stream['label'][~fault], served_predictions[~fault], zero_division=0)),
        'f1_during_fault_with_loop': float(f1_score(stream['label'][fault], served_predictions[fault], zero_division=0)),
        'f1_during_fault_first_model_only': float(f1_score(stream['label'][fault],
                                                           first_model.predict(X_all[fault]), zero_division=0)),
        'batches': rows,
    }
    with open(os.path.join(MONITOR_DIR, "closed_loop_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nF1 before the fault: {summary['f1_before_fault']:.3f}")
    print(f"F1 while the sensor is noisy: {summary['f1_during_fault_with_loop']:.3f} with the loop, "
          f"{summary['f1_during_fault_first_model_only']:.3f} if v{first_version} had stayed in service")
    print(f"Champion at the end: v{service.version}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", default=None,
                        help="also tell a running model API (e.g. http://127.0.0.1:8000) to reload after a promotion")
    run(parser.parse_args().api_url)
