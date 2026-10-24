import os
import numpy as np
import pandas as pd
import json
from typing import Dict, Any, Optional, List, Union, Tuple
import time

from src.features.ins_features import extract_window_features
from src.utils import get_pipeline_logger, get_param

logger = get_pipeline_logger("model_service")

INS_FIELDS = ["velocity_north", "velocity_east", "velocity_down", "roll", "pitch", "yaw"]
LABEL_TEXT = {
    'turn_next_1s': ("Driving straight", "Turn ahead"),
    'east_velocity': ("Lower East Velocity", "Higher East Velocity"),
}


class INSPredictionService:
    """
    Service for loading the trained model and making predictions on INS data.

    It serves the registered model version with the MLflow alias 'champion' (see
    src/utils/registry.py). The first edition created an untrained mock model here and
    predicted with a fixed rule; that rule-based service is still in simple_model_service.py.
    """
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the prediction service.

        Args:
            model_dir: Directory with model.pkl, used if the registry has no champion
        """
        if model_dir is None:
            model_dir = get_param('model.output_path')

        self.model_dir = model_dir
        self.model = None
        self.version = None
        self.is_loaded = False
        self.label_text = LABEL_TEXT.get(get_param('label.name'), ("Class 0", "Class 1"))

        logger.info(f"Prediction service initialized with model directory: {self.model_dir}")

    def load(self) -> bool:
        """
        Load the champion model from the MLflow Model Registry (or model.pkl).

        Returns:
            True if loading was successful, False otherwise
        """
        try:
            from src.utils.registry import setup_mlflow, champion_version, load_model
            client = setup_mlflow()
            version = champion_version(client)
            if version is not None:
                self.model = load_model()
                self.version = version
            else:
                import pickle
                with open(os.path.join(self.model_dir, "model.pkl"), 'rb') as f:
                    self.model = pickle.load(f)
                self.version = None
            self.is_loaded = True
            logger.info(f"Loaded model version {self.version}" if self.version
                        else f"Loaded model from {self.model_dir}")
            return True

        except Exception as e:
            logger.error(f"Could not load a trained model (run training first): {str(e)}")
            self.is_loaded = False
            return False

    def reload_if_changed(self) -> bool:
        """Load the champion again if a different version has been promoted. True if it changed."""
        from src.utils.registry import setup_mlflow, champion_version
        if champion_version(setup_mlflow()) != self.version:
            previous = self.version
            self.load()
            logger.info(f"Now serving model version {self.version} (was {previous})")
            return True
        return False

    def predict_windows(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predictions and class-1 probabilities for a matrix of window features."""
        if not self.is_loaded and not self.load():
            raise RuntimeError("No trained model available")
        X = np.asarray(X, dtype=float)
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]

    def window_features(self, readings: List[Dict[str, float]]) -> Dict[str, Any]:
        """Features of a window of INS readings (dicts with the INS_FIELDS and optionally 'timestamp')."""
        window = pd.DataFrame(readings)
        for field in INS_FIELDS:
            if field not in window:
                window[field] = 0.0
        if 'timestamp' not in window:
            window['timestamp'] = np.arange(len(window)) * 20000   # 50 Hz
        return extract_window_features(window, get_param('features.imu.fft_components', 5))

    def predict(self, features: Union[Dict[str, float], List[Dict[str, float]]]) -> Tuple[int, float]:
        """
        Make a prediction for one INS reading or a window of readings.

        The model was trained on half-second windows (25 readings at 50 Hz). A single reading is
        treated as a window of one, which works but carries no motion information.

        Args:
            features: One reading (dict of INS values) or a list of readings

        Returns:
            Tuple of (prediction, probability of class 1)
        """
        readings = features if isinstance(features, list) else [features]
        window = self.window_features(readings)
        prediction, probability = self.predict_windows([window['statistical_features']])
        return int(prediction[0]), float(probability[0])

    def batch_predict(self, feature_list: List[Union[List[float], Dict[str, float]]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple examples.

        Args:
            feature_list: List of readings, or of feature vectors as produced by the pipeline

        Returns:
            List of prediction results
        """
        results = []
        for features in feature_list:
            if isinstance(features, dict):
                results.append(self.predict(features))
            else:
                prediction, probability = self.predict_windows([features])
                results.append((int(prediction[0]), float(probability[0])))
        return results

    def predict_from_ins_data(self, ins_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Predict from INS data and return a response dictionary.

        Args:
            ins_data: One reading or a list of readings with INS data fields

        Returns:
            Dictionary with prediction results
        """
        try:
            prediction, probability = self.predict(ins_data)
            return {
                "prediction": prediction,
                "probability": probability,
                "prediction_text": self.label_text[prediction],
                "model_version": self.version,
                "timestamp": int(time.time() * 1000)
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": str(e)}


# Simple example usage
if __name__ == "__main__":
    # Initialize service
    service = INSPredictionService()

    # Load model
    if service.load():
        # Example: half a second of INS readings while the heading swings left
        example_window = [
            {"velocity_north": 4.0, "velocity_east": 2.0 + 0.05 * i, "velocity_down": 0.0,
             "roll": 0.01, "pitch": 0.02, "yaw": 1.2 + 0.006 * i, "timestamp": i * 20000}
            for i in range(25)
        ]

        # Make prediction
        result = service.predict_from_ins_data(example_window)
        print(json.dumps(result, indent=2))
    else:
        print("Failed to load model")
