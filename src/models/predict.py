import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple

from src.utils import get_pipeline_logger, get_param

logger = get_pipeline_logger("model_predict")


def load_model(model_path: str, scaler_path: str) -> Tuple[Any, Any]:
    """
    Load a trained model and scaler.
    
    Args:
        model_path: Path to the trained model pickle file
        scaler_path: Path to the scaler pickle file
        
    Returns:
        Tuple of (model, scaler)
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loading scaler from {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def predict(features: Union[np.ndarray, List[List[float]]],
           model_dir: Optional[str] = None) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        features: Feature vectors to make predictions on
        model_dir: Directory containing the model and scaler
        
    Returns:
        Numpy array of predictions
    """
    if model_dir is None:
        model_dir = get_param('model.output_path')
    
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    # Load model and scaler
    model, scaler = load_model(model_path, scaler_path)
    
    # Convert features to numpy array if necessary
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    logger.info(f"Made predictions for {len(features)} samples")
    
    return predictions, probabilities


def predict_from_dataframe(features_df: pd.DataFrame,
                         model_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Make predictions from a DataFrame containing feature vectors.
    
    Args:
        features_df: DataFrame with 'features' column containing feature vectors
        model_dir: Directory containing the model and scaler
        
    Returns:
        DataFrame with original data plus predictions
    """
    # Extract feature vectors
    feature_vectors = np.array(features_df['features'].tolist())
    
    # Make predictions
    predictions, probabilities = predict(feature_vectors, model_dir)
    
    # Add predictions to DataFrame
    result_df = features_df.copy()
    result_df['prediction'] = predictions
    result_df['probability'] = probabilities[:, 1]  # Probability of class 1
    
    return result_df


def batch_predict(features_path: str, 
                 model_dir: Optional[str] = None,
                 output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Make batch predictions on a features dataset.
    
    Args:
        features_path: Path to features parquet file
        model_dir: Directory containing the model and scaler
        output_path: Path to save prediction results
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_parquet(features_path)
    
    # Get default paths if not provided
    if model_dir is None:
        model_dir = get_param('model.output_path')
    
    if output_path is None:
        output_path = os.path.join(model_dir, "predictions")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Make predictions
    results_df = predict_from_dataframe(features_df, model_dir)
    
    # Save predictions
    predictions_path = os.path.join(output_path, "predictions.parquet")
    results_df.to_parquet(predictions_path, index=False)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Calculate summary statistics
    summary = {
        'total_samples': len(results_df),
        'positive_predictions': int(results_df['prediction'].sum()),
        'negative_predictions': int(len(results_df) - results_df['prediction'].sum()),
        'positive_ratio': float(results_df['prediction'].mean()),
        'avg_probability': float(results_df['probability'].mean())
    }
    
    summary_path = os.path.join(output_path, "predictions_summary.json")
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    logger.info(f"Batch prediction complete. Processed {len(results_df)} samples")
    
    return results_df


if __name__ == "__main__":
    features_path = os.path.join(get_param('alignment.output_path'), 'ml_dataset', 'features.parquet')
    batch_predict(features_path) 