"""Checks for the "Use your own data" adapter (scripts/own_drive.py). No training, a few seconds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import own_drive  # noqa: E402
from own_drive import scan  # noqa: E402

SAMPLE = os.path.join(ROOT, "oxfordrobocar_sample_small", "gps", "ins.csv")


@pytest.fixture(scope="module")
def drive():
    """Ten minutes of the sample drive (minutes 5-15), which has turns in every period."""
    df = pd.read_csv(SAMPLE)
    minute = (df["timestamp"] - df["timestamp"].iloc[0]) / 60e6
    return df[(minute >= 5) & (minute < 15)].reset_index(drop=True)


def write(folder, df, rel="ins.csv"):
    path = os.path.join(folder, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return str(folder)


def test_good_drive_passes_and_writes_nothing(tmp_path, drive):
    folder = write(tmp_path, drive)
    before = sorted(os.listdir(ROOT))
    path, summary, problems = scan(folder)
    assert problems == []
    assert summary["rows"] == len(drive) and round(summary["rate"]) == 50
    assert all(k > 0 for k, _ in summary["splits"].values())
    assert sorted(os.listdir(ROOT)) == before


def test_robotcar_layout_and_extra_columns(tmp_path, drive):
    folder = write(tmp_path, drive.assign(extra="x"), rel=os.path.join("gps", "ins.csv"))
    assert scan(folder)[2] == []


def test_split_matches_training():
    from src.models.train import chronological_split
    for n in (100, 2021, 10103):
        assert own_drive.chronological_split(n, 0.2, 0.2) == chronological_split(n, 0.2, 0.2)


@pytest.mark.parametrize("change,expected", [
    (lambda d: d.drop(columns=["pitch"]), "missing column(s) pitch"),
    (lambda d: d.assign(roll=d["roll"].astype(object).where(d.index != 5, "0.03 rad")), "roll: 1 value(s) that aren't numbers, first '0.03 rad' (row 7)"),
    (lambda d: d.assign(yaw=d["yaw"].where(d.index % 10 != 0)), "yaw: 10% of the rows are empty"),
    (lambda d: d.assign(timestamp=d["timestamp"] // 1000), "timestamp: looks like milliseconds"),
    (lambda d: d.assign(timestamp=d["timestamp"] / 1e6), "timestamp: not whole numbers"),
    (lambda d: d.iloc[::5], "10.0 readings per second; needs at least 25"),
    (lambda d: d.assign(yaw=np.degrees(d["yaw"])), "yaw: values up to"),
    (lambda d: d.assign(velocity_east=d["velocity_east"] * 3.6 * 10), "velocity_east: values up to"),
    (lambda d: pd.concat([d, d.iloc[[3]]]), "goes backwards 1 time(s)"),
    (lambda d: pd.concat([d.iloc[:10], d.iloc[[5]], d.iloc[10:]]), "1 duplicate value(s)"),
    (lambda d: d.iloc[:500], "too short"),
])
def test_bad_drives_are_reported(tmp_path, drive, change, expected):
    folder = write(tmp_path, change(drive.copy()))
    _, _, problems = scan(folder)
    assert any(expected in p for p in problems), problems


def test_straight_drive_has_no_turns(tmp_path, drive):
    # Hold the heading still: no window counts as a turn, so the model has nothing to learn
    folder = write(tmp_path, drive.assign(yaw=1.0))
    _, _, problems = scan(folder)
    assert len(problems) == 3 and all("has no turns" in p for p in problems)


def test_unreadable_files(tmp_path):
    assert scan(str(tmp_path))[2] == ["ins.csv: missing (looked for ins.csv and gps/ins.csv)"]
    (tmp_path / "ins.csv").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    assert "Git LFS pointer" in scan(str(tmp_path))[2][0]
    (tmp_path / "ins.csv").write_bytes("timestamp,yaw\n1,caf\xe9\n".encode("latin-1"))
    assert scan(str(tmp_path))[2] == ["ins.csv: not UTF-8 text"]
    (tmp_path / "ins.csv").write_bytes(b"")
    assert "not a readable CSV" in scan(str(tmp_path))[2][0]
