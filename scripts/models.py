"""List the registered model versions and which one is the champion (make models).

The same information as the MLflow UI's Models page:
    mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.utils.registry import champion_version, model_name, setup_mlflow  # noqa: E402


def main():
    if not os.path.exists(os.path.join("experiments", "mlflow.db")) and not os.environ.get("MLFLOW_TRACKING_URI"):
        sys.exit("No MLflow database yet. Run `make run` first.")
    client = setup_mlflow()
    versions = sorted(client.search_model_versions(f"name='{model_name()}'"), key=lambda v: int(v.version))
    champion = champion_version(client)
    print(f"{model_name()}: {len(versions)} version(s), champion v{champion}\n")
    print(f"{'version':>7}  {'trained on':34} {'test F1':>7}  {'run':24}")
    for v in versions:
        run = client.get_run(v.run_id)
        recent = int(float(run.data.params.get("num_recent_samples", 0)))
        source = "the drive" if not recent else f"the drive + {recent} production windows"
        f1 = run.data.metrics.get("test_f1")
        mark = "  <- champion" if int(v.version) == champion else ""
        print(f"{'v' + str(v.version):>7}  {source:34} {f1:>7.3f}  {run.info.run_name:24}{mark}")


if __name__ == "__main__":
    main()
