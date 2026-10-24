import os
import numpy as np
import pandas as pd
import json
import pickle
import time
from typing import Dict, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

from src.utils import get_pipeline_logger, get_param
from src.utils.registry import mlflow, setup_mlflow, model_name, champion_version, set_champion

logger = get_pipeline_logger("model_training")

REFERENCE_DIR = os.path.join("experiments", "monitoring", "reference")


def load_dataset(features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels for model training, in time order.

    Args:
        features_path: Path to features parquet file
        labels_path: Path to labels parquet file

    Returns:
        Tuple of (X, y) for model training
    """
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_parquet(features_path)

    logger.info(f"Loading labels from {labels_path}")
    labels_df = pd.read_parquet(labels_path)

    # Ensure features and labels are aligned by timestamp, oldest first
    merged_df = pd.merge(features_df, labels_df, on='timestamp').sort_values('timestamp')

    # Extract feature vectors and labels
    X = np.array(merged_df['features'].tolist(), dtype=float)
    y = np.array(merged_df['label'])

    logger.info(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    return X, y


def chronological_split(n: int, test_size: float, val_size: float) -> Tuple[slice, slice, slice]:
    """
    Train on the start of the drive, validate on the next part and test on the end.

    The windows overlap (0.5 s windows every 0.25 s), so the first edition's random split put
    near-copies of test windows in the training set. Splitting by time avoids that and matches
    how the model is used: trained on the past, applied to what comes next.
    """
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def make_model(params: Dict[str, Any], random_seed: int) -> Pipeline:
    """Scaler and XGBoost classifier, saved and served as one scikit-learn pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=random_seed,
            eval_metric='logloss',
            n_jobs=1,  # deterministic and stable when torch is loaded in the same process
        )),
    ])


def classification_metrics(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(y) == 0:
        return {}
    pred = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
    }
    if len(np.unique(y)) == 2:
        metrics['roc_auc'] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return {k: float(v) for k, v in metrics.items()}


def save_reference_stats(monitor_df: pd.DataFrame, version: int) -> str:
    """Mean and standard deviation of the monitored features in the data a model version learned from."""
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    columns = [c for c in get_param('monitoring.monitor_features', []) if c in monitor_df]
    stats = {
        'model_version': version,
        'num_windows': int(len(monitor_df)),
        'mean': {c: float(monitor_df[c].mean()) for c in columns},
        'std': {c: float(monitor_df[c].std()) for c in columns},
    }
    path = os.path.join(REFERENCE_DIR, f"reference_v{version}.json")
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    return path


def train_model(features_path: Optional[str] = None,
               labels_path: Optional[str] = None,
               output_path: Optional[str] = None,
               experiment_name: Optional[str] = None,
               recent_data: Optional[Dict[str, Any]] = None,
               run_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Train a model on the INS window features, log it to MLflow and register a new version.

    Args:
        features_path: Path to features parquet file
        labels_path: Path to labels parquet file
        output_path: Path to save trained model and artifacts
        experiment_name: MLflow experiment name
        recent_data: Labeled production windows for retraining: dict with X, y and monitor
            (dataframe of monitored features). They are added to the training and validation data
            with weight monitoring.sample_weight_recent.
        run_name: MLflow run name

    Returns:
        Dictionary with training results
    """
    # Get default paths if not provided
    if features_path is None:
        features_path = os.path.join(get_param('alignment.output_path'), 'ml_dataset', 'features.parquet')

    if labels_path is None:
        labels_path = os.path.join(get_param('alignment.output_path'), 'ml_dataset', 'labels.parquet')

    if output_path is None:
        output_path = get_param('model.output_path')

    if experiment_name is None:
        experiment_name = get_param('mlflow.experiment_name')

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Set up MLflow tracking (a local SQLite database by default)
    client = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    params = get_param('model.params')
    model_type = get_param('model.type')
    if model_type != 'xgboost':
        raise ValueError(f"Unsupported model type: {model_type}")
    random_seed = get_param('pipeline.random_seed')

    # Load dataset and split by time
    X, y = load_dataset(features_path, labels_path)
    train, val, test = chronological_split(len(y), get_param('pipeline.test_size'),
                                           get_param('pipeline.validation_size'))
    X_train, y_train = X[train], y[train]
    weights = np.ones(len(y_train))
    monitor_path = os.path.join(os.path.dirname(features_path), 'monitor_features.parquet')
    reference_df = pd.read_parquet(monitor_path).sort_values('timestamp').iloc[train] \
        if os.path.exists(monitor_path) else pd.DataFrame()

    if recent_data is not None:
        # Retraining: learn from everything before the test period plus the labeled production
        # windows, which get a higher weight because they show the sensor as it is now
        X_train = np.vstack([X[train], X[val], recent_data['X']])
        y_train = np.concatenate([y[train], y[val], recent_data['y']])
        weights = np.concatenate([np.ones(val.stop),
                                  np.full(len(recent_data['y']), float(get_param('monitoring.sample_weight_recent', 1)))])
        reference_df = recent_data['monitor']

    with mlflow.start_run(run_name=run_name or f"training_{int(time.time())}") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

        mlflow.log_params(params)
        mlflow.log_params({
            'model_type': model_type,
            'label': get_param('label.name'),
            'num_features': X.shape[1],
            'num_train_samples': len(y_train),
            'num_recent_samples': 0 if recent_data is None else len(recent_data['y']),
            'split': 'chronological',
        })

        # Train model
        model = make_model(params, random_seed)
        model.fit(X_train, y_train, model__sample_weight=weights)

        train_metrics = classification_metrics(model, X_train, y_train)
        val_metrics = classification_metrics(model, X[val], y[val])
        test_metrics = classification_metrics(model, X[test], y[test])
        for prefix, metrics in [('train', train_metrics), ('val', val_metrics), ('test', test_metrics)]:
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()})

        # Save model artifacts (model.pkl and scaler.pkl as in the first edition, plus the pipeline)
        model_path = os.path.join(output_path, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model.named_steps['model'], f)
        scaler_path = os.path.join(output_path, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(model.named_steps['scaler'], f)
        importances_path = os.path.join(output_path, "feature_importances.json")
        with open(importances_path, 'w') as f:
            json.dump({'feature_importances': model.named_steps['model'].feature_importances_.tolist()}, f, indent=2)
        mlflow.log_artifact(importances_path)

        # Log the pipeline and register it as a new version
        # MLflow 3 saves scikit-learn models with skops, which loads only types declared safe
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name(),
                                        input_example=X_train[:2],
                                        skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"])
        version = int(info.registered_model_version)
        client.set_model_version_tag(model_name(), str(version), "trained_on",
                                     "drive" if recent_data is None else "drive+production")
        if champion_version(client) is None:
            set_champion(client, version)      # the first model goes straight into service
            logger.info(f"Registered {model_name()} version {version} as the champion")
        else:
            logger.info(f"Registered {model_name()} version {version} (candidate; evaluate_model decides on promotion)")

        if not reference_df.empty:
            mlflow.log_artifact(save_reference_stats(reference_df, version))

        summary = {
            'model_type': model_type,
            'model_name': model_name(),
            'model_version': version,
            'run_id': run_id,
            'label': get_param('label.name'),
            'params': params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'train_size': int(len(y_train)),
            'val_size': int(val.stop - val.start),
            'test_size': int(test.stop - test.start),
            'feature_size': int(X.shape[1]),
            'positive_rate': float(y.mean()),
            'timestamp': int(time.time() * 1000)
        }

        summary_path = os.path.join(output_path, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

        logger.info(f"Model training complete. Test accuracy: {test_metrics['accuracy']:.4f}, "
                    f"F1: {test_metrics['f1']:.4f}")
        logger.info(f"Model saved to {model_path} and registered as version {version}")

        return summary


if __name__ == "__main__":
    train_model()
