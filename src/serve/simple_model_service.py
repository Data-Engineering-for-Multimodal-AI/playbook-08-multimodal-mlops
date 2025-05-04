import os
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Optional, List, Union, Tuple

from src.utils import get_pipeline_logger, get_param

logger = get_pipeline_logger("simple_model_service")

class SimpleINSPredictionService:
    """
    A simplified prediction service for demonstration purposes.
    
    This service uses rule-based predictions instead of the actual trained model,
    which allows us to work directly with raw INS data without complex feature engineering.
    """
    def __init__(self):
        """Initialize the simplified prediction service."""
        self.is_loaded = True
        logger.info("Simple INS prediction service initialized")
        
    def load(self) -> bool:
        """
        Simplified load method. No actual model to load.
        
        Returns:
            Always returns True
        """
        self.is_loaded = True
        logger.info("Simple prediction service ready")
        return True
    
    def predict(self, features: Dict[str, float]) -> Tuple[int, float]:
        """
        Make a prediction using a simple rule based on velocity_east.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (prediction, probability)
        """
        try:
            # Use a simple rule: If velocity_east > 7.0, predict class 1, otherwise class 0
            # This matches our synthetic label generation logic from training
            velocity_east = features.get("velocity_east", 0.0)
            
            prediction = 1 if velocity_east > 7.0 else 0
            
            # Calculate a fake probability based on how far the value is from the threshold
            distance = abs(velocity_east - 7.0)
            base_prob = 0.5
            prob_adjustment = min(distance / 10.0, 0.49)  # Cap adjustment at 0.49
            
            # For class 1 prediction, probability should be > 0.5
            # For class 0 prediction, probability should be < 0.5
            if prediction == 1:
                probability = base_prob + prob_adjustment
            else:
                probability = base_prob - prob_adjustment
                
            logger.debug(f"Made prediction {prediction} with probability {probability:.4f} " +
                        f"based on velocity_east={velocity_east:.2f}")
                
            return prediction, probability
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return 0, 0.5  # Default prediction in case of error
    
    def predict_from_ins_data(self, ins_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction directly from INS data.
        
        Args:
            ins_data: Dictionary with INS data fields
            
        Returns:
            Dictionary with prediction results
        """
        try:
            features = {
                "velocity_north": ins_data.get("velocity_north", 0.0),
                "velocity_east": ins_data.get("velocity_east", 0.0),
                "velocity_down": ins_data.get("velocity_down", 0.0),
                "roll": ins_data.get("roll", 0.0),
                "pitch": ins_data.get("pitch", 0.0),
                "yaw": ins_data.get("yaw", 0.0)
            }
            
            prediction, probability = self.predict(features)
            
            return {
                "prediction": prediction,
                "probability": probability,
                "prediction_text": "Higher East Velocity" if prediction == 1 else "Lower East Velocity",
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": str(e)}
    
    def batch_predict(self, feature_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple examples.
        
        Args:
            feature_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for features in feature_list:
            result = self.predict_from_ins_data(features)
            results.append(result)
            
        return results


# Simple example usage 
if __name__ == "__main__":
    # Initialize service
    service = SimpleINSPredictionService()
    
    # Example INS data
    example_ins_data = {
        "velocity_north": 1.45,
        "velocity_east": 7.2,  # Above threshold, should be class 1
        "velocity_down": 0.1,
        "roll": 0.02,
        "pitch": 0.01,
        "yaw": 1.2
    }
    
    # Make prediction
    result = service.predict_from_ins_data(example_ins_data)
    print("Prediction for velocity_east=7.2:")
    print(result)
    
    # Another example
    example_ins_data2 = {
        "velocity_north": 1.45,
        "velocity_east": 5.1,  # Below threshold, should be class 0
        "velocity_down": 0.1,
        "roll": 0.02,
        "pitch": 0.01,
        "yaw": 1.2
    }
    
    # Make prediction
    result2 = service.predict_from_ins_data(example_ins_data2)
    print("\nPrediction for velocity_east=5.1:")
    print(result2) 