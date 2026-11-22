"""Try this 1: how big a sensor fault does monitoring catch? (make try-noise-level)

The closed-loop replay (step 6) adds Gaussian noise to the INS readings from minute 36.5,
simulation.noise_scale = 0.1 of each channel's standard deviation. This rebuilds the replayed
stream with other noise levels and, without retraining anything, reports what monitoring sees
(the batch drift z-score against the champion's training data) and what the champion's F1 does.
Needs `make run` first; writes nothing.

Change NOISE and run it again.
"""

from sklearn.metrics import f1_score

from _common import batch_z, features, params, reference, require_run

NOISE = [0.0, 0.02, 0.05, 0.1, 0.2]


def main():
    require_run()
    from src.monitoring.closed_loop import production_stream
    from src.utils import get_param
    from src.utils.registry import champion_version, latest_base_version, load_model, setup_mlflow

    client = setup_mlflow()
    version = latest_base_version(client)
    model = load_model(version)
    ref = reference(version)
    monitored = get_param("monitoring.monitor_features")
    batch_size = int(get_param("monitoring.batch_size"))
    alert, severe = get_param("monitoring.drift_z_threshold"), get_param("monitoring.severe_z_threshold")
    print(f"Model v{version} (champion now: v{champion_version(client)}), batches of {batch_size} windows, "
          f"alert at z > {alert}, retrain at z > {severe}.\n")
    print(f"{'noise':>5} {'noisy batches':>13} {'with alert':>10} {'severe':>6} {'max z':>6} "
          f"{'F1 of v' + str(version) + ' while noisy':>18}")
    for noise in NOISE:
        with params(**{"simulation.noise_scale": noise}):
            stream = production_stream()
        fault = stream["fault"].to_numpy()
        z = batch_z(stream, ref, batch_size, monitored)
        noisy = [(b, s) for b, s in z if fault[b:b + batch_size].any()]
        pred = model.predict(features(stream))
        f1_noisy = f1_score(stream["label"][fault], pred[fault], zero_division=0)
        print(f"{noise:>5.2f} {len(noisy):>13} {sum(s > alert for _, s in noisy):>10} "
              f"{sum(s > severe for _, s in noisy):>6} {max(s for _, s in noisy):>6.1f} "
              f"{f1_noisy:>18.3f}")
    print(f"\nNoisy batches: the {len(noisy)} batches from the fault on. "
          "noise: standard deviation of the added noise, as a fraction of each channel's standard deviation.")


if __name__ == "__main__":
    main()
