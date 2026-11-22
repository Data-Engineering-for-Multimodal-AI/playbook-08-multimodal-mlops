"""Try this 2: how many windows should monitoring compare at a time? (make try-batch-size)

The monitor compares each batch of monitoring.batch_size = 100 served windows (25 s of driving)
with the champion's training data. Smaller batches can react sooner but their means are noisier.
This replays the same stream (noise 0.1 from minute 36.5, as step 6) with other batch sizes and
counts false alerts and how long after the fault the first alert and the first
severe alert come. False alerts are counted on minutes 25-34 of the drive (the validation
period: no fault, and no model trained on it) and on the replay before the fault. Needs `make run` first; writes nothing.

Change SIZES (or the thresholds in config/params.yaml) and run it again.
"""

import pandas as pd

from _common import DATASET, batch_z, require_run

SIZES = [25, 50, 100, 200, 400]


def main():
    require_run()
    from src.monitoring.closed_loop import production_stream
    from src.models.train import chronological_split
    from src.utils import get_param
    from src.utils.registry import latest_base_version, setup_mlflow
    from _common import reference

    ref = reference(latest_base_version(setup_mlflow()))
    monitored = get_param("monitoring.monitor_features")
    alert, severe = get_param("monitoring.drift_z_threshold"), get_param("monitoring.severe_z_threshold")
    stream = production_stream()
    fault = stream["fault"].to_numpy()
    minute = stream["drive_minute"].to_numpy()
    fault_minute = minute[fault][0]
    clean = pd.read_parquet(f"{DATASET}/monitor_features.parquet").sort_values("timestamp").reset_index(drop=True)
    _, val, _ = chronological_split(len(clean), get_param("pipeline.test_size"), get_param("pipeline.validation_size"))
    clean = clean.iloc[val].reset_index(drop=True)
    print(f"Fault from minute {fault_minute:.1f}; alert at z > {alert}, severe (retrain) at z > {severe}.\n")
    print(f"{'batch size':>10} {'seconds':>7} {'false alerts':>14} {'false severe':>13} "
          f"{'first alert after fault':>24} {'first severe after fault':>25}")
    for size in SIZES:
        z = batch_z(stream, ref, size, monitored)
        # a batch counts as "before the fault" only if all its windows are
        before = [(b, s) for b, s in z if not fault[b:b + size].any()] + batch_z(clean, ref, size, monitored)
        after = [(b, s) for b, s in z if fault[b:b + size].any()]

        def delay(limit):
            hit = next((b for b, s in after if s > limit), None)
            # the alert is raised when the batch is complete
            return "never" if hit is None else f"{(minute[min(hit + size, len(minute)) - 1] - fault_minute) * 60:.0f} s"

        false = sum(s > alert for _, s in before)
        false_severe = sum(s > severe for _, s in before)
        print(f"{size:>10} {size * 0.25:>7.0f} {false:>7} of {len(before):<4} {false_severe:>6} of {len(before):<4}"
              f" {delay(alert):>24} {delay(severe):>25}")
    print("\nWindows start every 0.25 s. Delays count to the end of the batch that raised the alert.")


if __name__ == "__main__":
    main()
