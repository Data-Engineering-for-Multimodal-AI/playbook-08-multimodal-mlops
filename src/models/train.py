import os
import numpy as np
import pandas as pd
import json
import pickle
import mlflow
import time
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

from src.utils import get_pipeline_logger, get_param, load_schema

logger = get_pipeline_logger("model_training")


def load_dataset(features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels for model training.
    
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
    
    # Ensure features and labels are aligned by timestamp
    merged_df = pd.merge(features_df, labels_df, on='timestamp')
    
    # Extract feature vectors and labels
    X = np.array(merged_df['features'].tolist())
    y = np.array(merged_df['label'])
    
    logger.info(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    return X, y


def train_model(features_path: Optional[str] = None, 
               labels_path: Optional[str] = None,
               output_path: Optional[str] = None,
               experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Train a model on the aligned multimodal features.
    
    Args:
        features_path: Path to features parquet file
        labels_path: Path to labels parquet file
        output_path: Path to save trained model and artifacts
        experiment_name: MLflow experiment name
        
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
    
    # Set up MLflow tracking
    tracking_uri = get_param('mlflow.tracking_uri')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{int(time.time())}") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")
        
        # Log parameters
        params = get_param('model.params')
        model_type = get_param('model.type')
        
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        mlflow.log_param('model_type', model_type)
        
        # Load dataset
        X, y = load_dataset(features_path, labels_path)
        
        # Split data
        random_seed = get_param('pipeline.random_seed')
        test_size = get_param('pipeline.test_size')
        val_size = get_param('pipeline.validation_size')
        
        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        
        # Then split train+val into train and val
        # Adjust validation size to account for the test split
        adjusted_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=adjusted_val_size, 
            random_state=random_seed, stratify=y_train_val
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Log dataset information
        mlflow.log_param('num_samples', X.shape[0])
        mlflow.log_param('num_features', X.shape[1])
        mlflow.log_param('num_train_samples', X_train.shape[0])
        mlflow.log_param('num_val_samples', X_val.shape[0])
        mlflow.log_param('num_test_samples', X_test.shape[0])
        
        # Train model
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=random_seed,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # Train with early stopping - Fix for incompatible parameters
            eval_set = [(X_val_scaled, y_val)]
            model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=100
            )
            
            # Get feature importances
            feature_importances = model.feature_importances_
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred)
        }
        
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred)
        }
        
        # Get probability predictions for ROC-AUC
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        train_metrics['roc_auc'] = roc_auc_score(y_train, y_train_proba)
        val_metrics['roc_auc'] = roc_auc_score(y_val, y_val_proba)
        test_metrics['roc_auc'] = roc_auc_score(y_test, y_test_proba)
        
        # Log metrics
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)
            
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value)
            
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)
        
        # Save model artifacts
        model_path = os.path.join(output_path, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        scaler_path = os.path.join(output_path, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        
        # Save feature importances
        if model_type == 'xgboost':
            importances_path = os.path.join(output_path, "feature_importances.json")
            with open(importances_path, 'w') as f:
                json.dump({
                    'feature_importances': feature_importances.tolist()
                }, f, indent=2)
            
            mlflow.log_artifact(importances_path)
        
        # Create training summary
        summary = {
            'model_type': model_type,
            'run_id': run_id,
            'params': params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'train_size': X_train.shape[0],
            'val_size': X_val.shape[0],
            'test_size': X_test.shape[0],
            'feature_size': X.shape[1],
            'timestamp': int(time.time() * 1000)
        }
        
        summary_path = os.path.join(output_path, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        mlflow.log_artifact(summary_path)
        
        logger.info(f"Model training complete. Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Model saved to {model_path}")
        
        return summary


if __name__ == "__main__":
    train_model() 