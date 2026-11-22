"""Send half a second of the recorded drive to the running model API (make predict MINUTE=...).

Takes the 25 INS readings (0.5 s at 50 Hz) that start at MINUTE of the drive, posts them to
POST /predict as {"readings": [...]}, and prints the answer next to what the car really did in
the following second (the label the model was trained on). Start the API first: make api.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
INS = os.path.join(ROOT, "oxfordrobocar_sample_small", "gps", "ins.csv")
CHANNELS = ["velocity_north", "velocity_east", "velocity_down", "roll", "pitch", "yaw"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--minute", type=float, required=True, help="minute of the 42-minute drive")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()
    from src.models.labels import heading_change_rate
    from src.utils.config import load_params

    p = load_params()
    ins = pd.read_csv(INS)
    t0 = ins["timestamp"].iloc[0]
    start = t0 + args.minute * 60e6
    window = ins[ins["timestamp"] >= start].head(int(p["features"]["imu"]["window_size"] * 50))
    if len(window) < 2:
        sys.exit(f"Minute {args.minute} is past the end of the drive ({(ins['timestamp'].iloc[-1] - t0) / 60e6:.1f} min)")
    readings = window[["timestamp"] + CHANNELS].to_dict("records")
    try:
        r = requests.post(f"http://127.0.0.1:{args.port}/predict", json={"readings": readings}, timeout=30)
    except requests.ConnectionError:
        sys.exit(f"No model API on port {args.port}. Start it in another terminal with: make api")
    r.raise_for_status()
    answer = r.json()
    # Labels are taken from the middle of the window, as in src/features/ins_features.py
    middle = np.array([(window["timestamp"].min() + window["timestamp"].max()) // 2])
    rate = heading_change_rate(ins, middle, p["label"]["horizon_sec"])[0]
    actual = "Turn ahead" if rate > p["label"]["yaw_rate_threshold"] else "No turn"
    speed = float(np.hypot(window["velocity_north"], window["velocity_east"]).mean())
    print(f"POST /predict with {len(readings)} readings from minute {args.minute:.2f} (speed {speed:.1f} m/s)")
    print(f"  model v{answer['model_version']}: {answer['prediction_text']} (probability of a turn {answer['probability']:.3f})")
    print(f"  what happened: {actual} (heading changed {rate:.3f} rad/s in the next "
          f"{p['label']['horizon_sec']:.0f} s; a turn is over {p['label']['yaw_rate_threshold']})")


if __name__ == "__main__":
    main()
