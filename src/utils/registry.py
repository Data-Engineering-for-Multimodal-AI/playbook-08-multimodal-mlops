"""MLflow tracking and Model Registry helpers.

Every trained model is registered as a new version of model.registered_name. The version being
served carries the alias 'champion' (MLflow 3 uses aliases; the old stages were removed). A new
version only gets the alias when evaluate_model decides it is better.
"""
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # MLflow 3.x prints a hint on import otherwise

import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

from src.utils.config import get_param  # noqa: E402

CHAMPION = "champion"


def setup_mlflow() -> MlflowClient:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or get_param('mlflow.tracking_uri')
    if uri.startswith("sqlite:///"):
        folder = os.path.dirname(uri[len("sqlite:///"):])
        if folder:
            os.makedirs(folder, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def model_name() -> str:
    return get_param('model.registered_name', 'ins-turn-classifier')


def champion_version(client: MlflowClient):
    """The version number with the 'champion' alias, or None."""
    try:
        return int(client.get_model_version_by_alias(model_name(), CHAMPION).version)
    except MlflowException:
        return None


def latest_version(client: MlflowClient):
    versions = client.search_model_versions(f"name='{model_name()}'")
    return max((int(v.version) for v in versions), default=None)


def set_champion(client: MlflowClient, version: int) -> None:
    client.set_registered_model_alias(model_name(), CHAMPION, str(version))


def load_model(version=None):
    """Load a registered version (default: the champion) as an XGBClassifier."""
    reference = f"@{CHAMPION}" if version is None else f"/{version}"
    return mlflow.xgboost.load_model(f"models:/{model_name()}{reference}")


def latest_base_version(client: MlflowClient):
    """Newest version trained on the drive alone (not retrained on production data), or None."""
    versions = client.search_model_versions(f"name='{model_name()}'")
    return max((int(v.version) for v in versions if v.tags.get("trained_on") == "drive"), default=None)
