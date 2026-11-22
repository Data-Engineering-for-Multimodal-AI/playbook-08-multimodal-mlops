"""Shared helpers for the Try this experiments. They read what `make run` wrote and change nothing:
no new model versions, no promotions, no monitoring files."""

import contextlib
import copy
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # the pipeline reads config/ and experiments/ from the repository root
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.utils import config  # noqa: E402

DATASET = os.path.join("experiments", "aligned", "ml_dataset")
REFERENCE = os.path.join("experiments", "monitoring", "reference")


def require_run():
    if not os.path.exists(os.path.join(DATASET, "labels.parquet")) or not os.path.isdir(REFERENCE):
        sys.exit("No pipeline outputs in experiments/. Run `make run` first.")
    logging.getLogger("playbook_08_pipeline").setLevel(logging.ERROR)
    logging.disable(logging.WARNING)


@contextlib.contextmanager
def params(**overrides):
    """Run with some config/params.yaml values changed, e.g. params(**{"simulation.noise_scale": 0.2})."""
    original = config.load_params

    def patched():
        p = copy.deepcopy(original())
        for path, value in overrides.items():
            node = p
            *parents, last = path.split(".")
            for key in parents:
                node = node[key]
            node[last] = value
        return p

    config.load_params = patched
    try:
        yield
    finally:
        config.load_params = original


def reference(version=1):
    with open(os.path.join(REFERENCE, f"reference_v{version}.json")) as f:
        return json.load(f)


def batch_z(stream, ref, batch_size, features):
    """Max z-score per batch, as ModelMonitor computes it: |batch mean - training mean| / training std."""
    out = []
    for begin in range(0, len(stream), batch_size):
        batch = stream.iloc[begin:begin + batch_size]
        z = [abs(batch[f].mean() - ref["mean"][f]) / ref["std"][f] for f in features if ref["std"].get(f)]
        out.append((begin, max(z)))
    return out


def features(stream):
    return np.array(stream["statistical_features"].tolist(), dtype=float)
