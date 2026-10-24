#!/usr/bin/env python
"""
Evaluate the newest registered model against the champion and promote it if it is better
(the step run_pipeline.py, the Airflow DAG and the drift handler call as
`python -m src.models.evaluate_model`; the first edition called this module but didn't include it).

The models are compared on the newest labeled production batches (monitoring.holdout_batches),
which no model trained on, or on the test period at the end of the drive before any production
data exists. Without a newer version, the champion is scored on the test period. The candidate
gets the 'champion' alias only if its F1 is higher.
"""
import json
import os
import time

import numpy as np

from src.models.feedback import split_recent
from src.models.train import chronological_split, classification_metrics, load_dataset
from src.utils import get_param, get_pipeline_logger
from src.utils.registry import champion_version, latest_version, load_model, model_name, set_champion, setup_mlflow

logger = get_pipeline_logger("evaluate_model")


def evaluation_set(compare: bool = True):
    """(X, y, description): the newest labeled production batches when comparing, else the test period."""
    _, holdout = split_recent() if compare else (None, None)
    if holdout is not None and len(holdout['y']):
        return holdout['X'], holdout['y'], f"newest production batches ({', '.join(holdout['batches'])})"
    dataset_dir = os.path.join(get_param('alignment.output_path'), 'ml_dataset')
    X, y = load_dataset(os.path.join(dataset_dir, 'features.parquet'), os.path.join(dataset_dir, 'labels.parquet'))
    _, _, test = chronological_split(len(y), get_param('pipeline.test_size'), get_param('pipeline.validation_size'))
    return X[test], y[test], "test period (end of the drive)"


def main():
    client = setup_mlflow()
    champion, newest = champion_version(client), latest_version(client)
    if champion is None:
        raise SystemExit("No registered model yet: run `python -m src.models.train_model` first")
    X, y, described = evaluation_set(compare=newest is not None and newest != champion)

    result = {'timestamp': int(time.time()), 'model_name': model_name(), 'evaluated_on': described,
              'num_windows': int(len(y)), 'positives': int(np.sum(y)), 'champion_before': champion}
    result['champion_metrics'] = classification_metrics(load_model(champion), X, y)
    promoted = False
    if newest is not None and newest != champion:
        result['candidate'] = newest
        result['candidate_metrics'] = classification_metrics(load_model(newest), X, y)
        promoted = result['candidate_metrics']['f1'] > result['champion_metrics']['f1']
        if promoted:
            set_champion(client, newest)
        for version, metrics in [(champion, result['champion_metrics']), (newest, result['candidate_metrics'])]:
            client.set_model_version_tag(model_name(), str(version), f"eval_f1_{result['timestamp']}",
                                         f"{metrics['f1']:.4f}")
    result['promoted'] = promoted
    result['champion_after'] = newest if promoted else champion

    os.makedirs(get_param('model.output_path'), exist_ok=True)
    with open(os.path.join(get_param('model.output_path'), "evaluation_summary.json"), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Evaluated on {described}: {len(y)} windows, {int(np.sum(y))} positive")
    print(f"  champion  v{champion}: F1 {result['champion_metrics']['f1']:.3f}, "
          f"accuracy {result['champion_metrics']['accuracy']:.3f}")
    if 'candidate' in result:
        print(f"  candidate v{newest}: F1 {result['candidate_metrics']['f1']:.3f}, "
              f"accuracy {result['candidate_metrics']['accuracy']:.3f}")
        print(f"  -> {'promoted' if promoted else 'kept'}: champion is now v{result['champion_after']}")
    return result


if __name__ == "__main__":
    main()
