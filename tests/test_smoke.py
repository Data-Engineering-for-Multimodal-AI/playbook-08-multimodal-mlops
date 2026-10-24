"""Smoke tests: the sample data, the printed commands, the full pipeline, the API and the dashboard.

Run with `make smoke` (or `uv run pytest -q`). CPU only; the full pipeline takes a few minutes and
downloads the ResNet-18 weights (45 MB) on first use. Everything is written under experiments/
and logs/ (both gitignored). The model API runs on a free port, never on 8000.
"""
import json
import os
import re
import socket
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "oxfordrobocar_sample_small")
sys.path.insert(0, ROOT)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(*args, timeout=1800):
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result.stdout


# ------------------------------------------------------------------ data and printed commands

def test_sample_data_present_and_documented():
    with open(os.path.join(DATA, "README.md")) as fh:
        readme = fh.read()
    assert "CC BY-NC-SA 4.0" in readme
    for name in os.listdir(DATA):
        if name != "README.md":
            assert f"`{name}" in readme, name
    with open(os.path.join(DATA, "gps", "ins.csv")) as fh:
        assert sum(1 for _ in fh) == 126362          # header + 126,361 readings
    assert len([f for f in os.listdir(os.path.join(DATA, "stereo", "centre")) if f.endswith(".png")]) == 49


def test_printed_commands_still_exist():
    with open(os.path.join(ROOT, "pipeline", "requirements.txt")) as fh:
        assert "-r ../requirements.txt" in fh.read()
    assert "--serve" in run("pipeline/run_pipeline.py", "--help", timeout=120)
    # run_pipeline.py, the Airflow DAG and the drift handler call these; the first edition lacked them
    assert "--retrain" in run("-m", "src.models.train_model", "--help", timeout=300)
    run("-c", "import src.models.evaluate_model", timeout=300)


def test_camera_frames_are_demosaiced():
    from src.ingest.video_ingest import load_frame
    frame = load_frame(os.path.join(DATA, "stereo", "centre", "1418381798076682.png"), (224, 224))
    assert frame.shape == (224, 224, 3)
    assert frame[..., 0].mean() != pytest.approx(frame[..., 2].mean(), abs=1)   # color, not gray


# ------------------------------------------------------------------ the full pipeline

@pytest.fixture(scope="module")
def pipeline_output():
    return run("run_pipeline.py", "--port", str(free_port()))


def test_run_pipeline(pipeline_output):
    out = pipeline_output
    with open(os.path.join(ROOT, "experiments", "features", "ins", "ins_features_summary.json")) as fh:
        assert json.load(fh)["total_windows"] == 10107
    match = re.search(r"Registered ins-turn-classifier version (\d+): test F1 ([0-9.]+)", out)
    assert match and float(match.group(2)) > 0.8
    with open(os.path.join(ROOT, "experiments", "aligned", "alignment_summary.json")) as fh:
        assert json.load(fh)["total_aligned_samples"] == 15     # 3 s of camera frames
    with open(os.path.join(ROOT, "experiments", "monitoring", "closed_loop_summary.json")) as fh:
        loop = json.load(fh)
    assert loop["windows"] == 2021
    assert loop["promotions"], "the sensor fault should lead to a promoted retrained model"
    assert loop["f1_during_fault_with_loop"] > loop["f1_during_fault_first_model_only"]
    assert "Champion at the end: v" in out
    for row in loop["batches"]:
        assert row["sensor_fault"] or row["max_z_score"] < 3      # no severe alert before the fault


def test_api(pipeline_output):
    """The model API in its own process (as run_pipeline.py runs it) on a free port."""
    import time
    import pandas as pd
    import requests
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.serve.api:app", "--host", "127.0.0.1",
                               "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                health = requests.get(f"{url}/health", timeout=120).json()
                break
            except requests.ConnectionError:
                assert server.poll() is None, "the API process exited"
                time.sleep(1)
        assert health["status"] == "healthy" and health["model_version"]
        ins = pd.read_csv(os.path.join(ROOT, "experiments", "ingested", "imu", "ins_processed.csv")).dropna()
        start = ins["timestamp"].iloc[0] + 36.4 * 60e6 - 0.25e6          # the car starts a turn
        window = ins[(ins["timestamp"] >= start) & (ins["timestamp"] < start + 0.5e6)][
            ["timestamp", "velocity_north", "velocity_east", "velocity_down", "roll", "pitch", "yaw"]]
        response = requests.post(f"{url}/predict", json={"readings": window.to_dict("records")}, timeout=60)
        assert response.status_code == 200 and response.json()["prediction_text"] == "Turn ahead"
        reading = window.iloc[0].to_dict()
        reading["timestamp"] = int(reading["timestamp"])
        assert requests.post(f"{url}/predict", json=reading, timeout=60).status_code == 200   # first-edition format
        batch = requests.post(f"{url}/batch_predict", json=[{"readings": window.to_dict("records")}, reading], timeout=60)
        assert len(batch.json()) == 2
        assert requests.post(f"{url}/reload", timeout=120).json()["model_version"] == health["model_version"]
    finally:
        server.terminate()
        server.wait(timeout=30)


def test_simple_api_still_rule_based():
    from fastapi.testclient import TestClient
    from src.serve.simple_api import app
    client = TestClient(app)
    reading = {"velocity_north": 1.0, "velocity_east": 7.5, "velocity_down": 0.0}
    assert client.post("/predict", json=reading).json()["prediction"] == 1


def test_dashboard(pipeline_output):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "src", "monitoring", "dashboard.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Predictions", "Data Drift", "Performance"]


def test_makefile_pipeline_module(pipeline_output):
    """`make run-pipeline` (python -m src.pipeline.pipeline): ingest to drift-triggered retraining."""
    run("-m", "src.pipeline.pipeline")
    summaries = sorted(f for f in os.listdir(os.path.join(ROOT, "experiments")) if f.startswith("pipeline_summary_"))
    with open(os.path.join(ROOT, "experiments", summaries[-1])) as fh:
        summary = json.load(fh)
    assert summary["status"] == "success"
    assert summary["stages"]["drift_detection"]["drift_detected"]
    assert summary["stages"]["retraining"]["retrained_model_version"]


def test_notebook_runs():
    import matplotlib
    matplotlib.use("Agg")
    with open(os.path.join(ROOT, "notebooks", "01_eda.ipynb")) as fh:
        notebook = json.load(fh)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        namespace = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)
    finally:
        os.chdir(cwd)
    assert 0.05 < namespace["ins"]["turning"].mean() < 0.2
