#!/usr/bin/env python
"""
Train a model and register it in MLflow (the step run_pipeline.py, the Airflow DAG and the drift
handler call as `python -m src.models.train_model`; the first edition called this module but didn't
include it).

    python -m src.models.train_model            # train on the drive's training period
    python -m src.models.train_model --retrain  # also learn from labeled production windows

The first model registered becomes the champion. Later versions are candidates until
`python -m src.models.evaluate_model` promotes them (or --promote makes them champion directly).
"""
import argparse
import os
import time

from src.models.dataset import INS_FEATURES_PATH, build_dataset
from src.models.feedback import split_recent
from src.models.train import train_model
from src.utils import get_param, get_pipeline_logger

logger = get_pipeline_logger("train_model")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrain", action="store_true",
                        help="add the labeled production batches (except the newest, kept for evaluation)")
    parser.add_argument("--promote", action="store_true",
                        help="make the new version the champion right away (run_pipeline.py does this for a fresh run)")
    parser.add_argument("--since", default=None,
                        help="with --retrain: use production batches from this one on (e.g. batch_0007.parquet)")
    args = parser.parse_args(argv)

    dataset_dir = os.path.join(get_param('alignment.output_path'), 'ml_dataset')
    if not os.path.exists(os.path.join(dataset_dir, 'features.parquet')):
        if not os.path.exists(INS_FEATURES_PATH):
            from src.features.extract_ins_features import main as extract_ins_features
            extract_ins_features()
        build_dataset()

    recent = None
    if args.retrain:
        recent, _ = split_recent(since=args.since)
        if recent is None:
            raise SystemExit("No labeled production batches to retrain on yet")
        logger.info(f"Retraining with {len(recent['y'])} labeled production windows "
                    f"({recent['batches'][0]} to {recent['batches'][-1]})")

    summary = train_model(recent_data=recent,
                          run_name=f"{'retraining' if args.retrain else 'training'}_{int(time.time())}")
    if args.promote:
        from src.utils.registry import set_champion, setup_mlflow
        set_champion(setup_mlflow(), summary['model_version'])
        logger.info(f"Version {summary['model_version']} is now the champion")
    print(f"Registered {summary['model_name']} version {summary['model_version']}: "
          f"test F1 {summary['test_metrics']['f1']:.3f}, accuracy {summary['test_metrics']['accuracy']:.3f} "
          f"({summary['test_size']} windows from the end of the drive)")
    return summary


if __name__ == "__main__":
    main()
