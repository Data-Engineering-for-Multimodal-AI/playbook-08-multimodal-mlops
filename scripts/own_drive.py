"""Run the MLOps pipeline on your own INS log: make check-data DIR=... and make load DIR=...

Expected folder layout (the Oxford RobotCar layout works as it is):

    my_drive/
      ins.csv        or gps/ins.csv. One row per reading, with a header. Required columns:
                     timestamp (UNIX time in microseconds), velocity_north, velocity_east,
                     velocity_down (m/s), roll, pitch, yaw (radians). Other columns are ignored.

--check-only reads the file and writes nothing. Without it, once every check passes, it runs the
pipeline in its own folder, experiments/own/<folder name>/, with its own MLflow database, so the
sample's models and features are not touched: INS features, training and registration, evaluation,
and the closed-loop replay with a simulated sensor fault.
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUIRED = ["timestamp", "velocity_north", "velocity_east", "velocity_down", "roll", "pitch", "yaw"]
CHANNELS = REQUIRED[1:]
US_MIN, US_MAX = 9.5e14, 4.1e15   # 2000-02 to 2099-12 in microseconds
MIN_RATE_HZ = 25                  # the pipeline resamples to 50 Hz; below this most bins are empty
MAX_MISSING = 0.05
MAX_SPEED = 100.0                 # m/s


def params():
    with open(os.path.join(ROOT, "config", "params.yaml")) as f:
        return yaml.safe_load(f)


def find_csv(folder):
    for rel in ("ins.csv", os.path.join("gps", "ins.csv")):
        if os.path.isfile(os.path.join(folder, rel)):
            return rel
    return None


def chronological_split(n, test_size, val_size):
    """The same split as src/models/train.py (importing that module would start its log file)."""
    train_end, val_end = int(n * (1 - test_size - val_size)), int(n * (1 - test_size))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def check_splits(df, p, problems, rel):
    """The model needs turns and straight driving in each of the train, validation and test periods."""
    from src.models.labels import heading_change_rate

    win = p["features"]["imu"]["window_size"]
    step = win - p["features"]["imu"]["overlap"]
    t = df["timestamp"].to_numpy()
    starts = np.arange(t[0], t[-1] - (win + p["label"]["horizon_sec"]) * 1e6, step * 1e6)
    if len(starts) < 100:
        problems.append(f"{rel}: too short: {len(starts)} half-second windows; needs at least 100 "
                        f"(about {100 * step + win:.0f} s of driving)")
        return None
    rate = heading_change_rate(df, starts + win * 1e6, p["label"]["horizon_sec"])
    turns = rate > p["label"]["yaw_rate_threshold"]
    parts = chronological_split(len(turns), p["pipeline"]["test_size"], p["pipeline"]["validation_size"])
    counts = {}
    for name, part in zip(("training", "validation", "test"), parts):
        n, k = part.stop - part.start, int(turns[part].sum())
        counts[name] = (k, n)
        if k == 0 or k == n:
            what = "no turns" if k == 0 else "only turns"
            problems.append(f"{rel}: the {name} period (by time, {n} windows) has {what}; "
                            f"the model needs both to learn and to be scored")
    return counts


def scan(folder):
    """Check a folder. Returns (csv path, summary, problems). Writes nothing."""
    if not os.path.isdir(folder):
        return None, {}, [f"{folder}: not a folder"]
    rel = find_csv(folder)
    if rel is None:
        return None, {}, ["ins.csv: missing (looked for ins.csv and gps/ins.csv)"]
    path = os.path.join(folder, rel)
    with open(path, "rb") as f:
        head = f.read(200)
    if head.startswith(b"version https://git-lfs"):
        return path, {}, [f"{rel}: a Git LFS pointer, not the data (run git lfs pull)"]
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return path, {}, [f"{rel}: not UTF-8 text"]
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        return path, {}, [f"{rel}: not a readable CSV ({str(e).splitlines()[0]})"]

    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f"{rel}: missing column(s) {', '.join(missing)} (the loader would fill them with empty values)")
        return path, {}, problems
    if len(df) < 2:
        return path, {}, problems + [f"{rel}: {len(df)} rows"]

    for col in REQUIRED:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = int(values.isna().sum() - df[col].isna().sum())
        if bad:
            first = df.loc[values.isna() & df[col].notna(), col].iloc[0]
            problems.append(f"{rel}: {col}: {bad} value(s) that aren't numbers, first {first!r} "
                            f"(row {int(values.isna().idxmax()) + 2})")
        empty = values.isna().mean()
        if empty > MAX_MISSING:
            problems.append(f"{rel}: {col}: {empty:.0%} of the rows are empty (limit {MAX_MISSING:.0%})")
        df[col] = values
    if problems:
        return path, {}, problems

    t = df["timestamp"].dropna()
    if (t % 1 != 0).any():
        problems.append(f"{rel}: timestamp: not whole numbers; needs UNIX time in microseconds")
    elif not t.between(US_MIN, US_MAX).all():
        median = float(t.median())
        unit = "seconds" if median < 1e11 else "milliseconds" if median < 1e14 else "nanoseconds" if median > 1e17 else "an unknown unit"
        problems.append(f"{rel}: timestamp: looks like {unit} (median {median:.0f}); needs UNIX time in microseconds")
    else:
        dups = int(t.duplicated().sum())
        if dups:
            problems.append(f"{rel}: timestamp: {dups} duplicate value(s), first {int(t[t.duplicated()].iloc[0])}")
        backwards = int((t.diff() < 0).sum())
        if backwards:
            problems.append(f"{rel}: timestamp: goes backwards {backwards} time(s); sort the rows by time")
        dt = np.diff(np.sort(t.unique()))
        if len(dt):
            rate = 1e6 / np.median(dt)
            if rate < MIN_RATE_HZ:
                problems.append(f"{rel}: {rate:.1f} readings per second; needs at least {MIN_RATE_HZ} "
                                f"(the pipeline resamples to 50 Hz)")
    for col, limit, unit in (("yaw", 2 * np.pi + 0.5, "radians"), ("roll", np.pi, "radians"),
                             ("pitch", np.pi, "radians")):
        biggest = float(df[col].abs().max())
        if biggest > limit:
            problems.append(f"{rel}: {col}: values up to {biggest:.1f}; needs {unit} (degrees?)")
    for col in ("velocity_north", "velocity_east", "velocity_down"):
        biggest = float(df[col].abs().max())
        if biggest > MAX_SPEED:
            problems.append(f"{rel}: {col}: values up to {biggest:.1f}; needs m/s (at most {MAX_SPEED:.0f})")
    if problems:
        return path, {}, problems

    df = df.dropna(subset=REQUIRED).sort_values("timestamp")
    counts = check_splits(df, params(), problems, rel)
    t = df["timestamp"].to_numpy()
    summary = {"rows": len(df), "minutes": (t[-1] - t[0]) / 60e6,
               "rate": 1e6 / np.median(np.diff(np.unique(t))), "splits": counts}
    return path, summary, problems


def workdir_for(folder):
    name = os.path.basename(os.path.normpath(folder)) or "drive"
    return os.path.join(ROOT, "experiments", "own", name)


def load(folder, csv_path):
    work = workdir_for(folder)
    if os.path.isdir(work):
        shutil.rmtree(work)                  # rerunning starts over
    os.makedirs(os.path.join(work, "config"))
    shutil.copy(os.path.join(ROOT, "config", "params.yaml"), os.path.join(work, "config"))
    with open(os.path.join(ROOT, "config", "schema.yaml")) as f:
        schema = yaml.safe_load(f)
    schema["imu"]["source_path"] = os.path.abspath(csv_path)
    with open(os.path.join(work, "config", "schema.yaml"), "w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("MLFLOW_TRACKING_URI", None)    # experiments/mlflow.db inside the work folder
    print(f"Working folder: {os.path.relpath(work, ROOT)}/ (its own config/, experiments/ and MLflow database)",
          flush=True)
    for step in ("src.models.train_model", "src.models.evaluate_model", "src.monitoring.closed_loop"):
        print(f"\n$ python -m {step}", flush=True)
        subprocess.run([sys.executable, "-m", step], cwd=work, env=env, check=True)
    print(f"\nMLflow runs and models: mlflow ui --backend-store-uri "
          f"sqlite:///{os.path.relpath(work, ROOT)}/experiments/mlflow.db")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="folder with ins.csv (or gps/ins.csv)")
    parser.add_argument("--check-only", action="store_true", help="check the file and write nothing")
    args = parser.parse_args()
    folder = os.path.abspath(args.dir)
    csv_path, summary, problems = scan(folder)
    if problems:
        print(f"{len(problems)} problem(s) in {args.dir}; nothing was written:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    splits = ", ".join(f"{k} {v[0]} of {v[1]}" for k, v in summary["splits"].items())
    print(f"{args.dir}: {summary['rows']} readings, {summary['minutes']:.1f} minutes at "
          f"{summary['rate']:.0f} Hz. Turn windows by period: {splits}. All checks passed.", flush=True)
    if not args.check_only:
        load(folder, csv_path)


if __name__ == "__main__":
    main()
