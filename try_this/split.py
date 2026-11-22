"""Try this 3: what does a random train/test split hide? (make try-split)

The pipeline splits the 10,103 half-second INS windows by time: the first 60% of the drive to
train, the next 20% to validate, the last 20% to test (src/models/train.py). The first edition
split at random. Windows overlap by 0.25 s, so with a random split most test windows have a
neighbor in the training set. This trains the same XGBoost model both ways and compares the
test F1 with the F1 on the minutes the model will actually face next. Needs `make run` first;
writes nothing.

Change the split fractions or the model in config/params.yaml and run it again.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from _common import DATASET, require_run


def main():
    require_run()
    from src.models.train import chronological_split, make_model
    from src.utils import get_param

    X = np.array(pd.read_parquet(f"{DATASET}/features.parquet")["features"].tolist(), dtype=float)
    y = pd.read_parquet(f"{DATASET}/labels.parquet")["label"].to_numpy()
    train, val, test = chronological_split(len(y), get_param("pipeline.test_size"), get_param("pipeline.validation_size"))
    seed = get_param("pipeline.random_seed")

    def fit(idx):
        return make_model(get_param("model.params"), seed).fit(X[idx], y[idx])  # as src/models/train.py

    early = np.arange(train.start, val.stop)            # minutes 0-34
    future = np.arange(test.start, test.stop)           # minutes 34-42
    print(f"{len(y)} windows, {y.mean():.1%} turns. 'Future' = the last {len(future)} windows of the drive.\n")
    print(f"{'split of the first 80% of the drive':40} {'test F1':>8} {'F1 on the future':>17}")

    tr, te = train_test_split(early, test_size=len(future), random_state=seed)
    model = fit(tr)
    print(f"{'random (first edition)':40} {f1_score(y[te], model.predict(X[te])):>8.3f} "
          f"{f1_score(y[future], model.predict(X[future])):>17.3f}")

    cut = early[: int(len(early) * 0.75)]
    model = fit(cut)
    held = early[len(cut):]
    print(f"{'by time (this version)':40} {f1_score(y[held], model.predict(X[held])):>8.3f} "
          f"{f1_score(y[future], model.predict(X[future])):>17.3f}")

    # How close is each random test window to the training set?
    train_set = set(tr.tolist())
    neighbor = np.array([(i - 1 in train_set) or (i + 1 in train_set) for i in te])
    print(f"\nRandom split: {neighbor.mean():.0%} of the test windows have a training window that overlaps "
          f"them by half (the next or previous window).")


if __name__ == "__main__":
    main()
