#!/usr/bin/env python
"""
Test script for validating INS schemas.
"""
import numpy as np
from enhancements_modules.schemas.ins_schemas import (
    INSDataPoint,
    INSFeatures,
    PredictionRequest,
    PredictionResponse
)

def test_ins_data_point():
    print("Testing INSDataPoint schema:")
    
    # Valid data point
    valid_data = {
        "timestamp": 1586346000000000,  # In microseconds
        "velocity_north": 0.5,
        "velocity_east": 1.2,
        "velocity_down": 0.1,
        "roll": 0.01,
        "pitch": 0.02,
        "yaw": 0.5
    }
    
    try:
        validated = INSDataPoint(**valid_data)
        print(f"✅ Valid data point validated: {validated.model_dump()}")
    except Exception as e:
        print(f"❌ Validation failed for valid data: {e}")
    
    # Invalid data point (timestamp out of range)
    invalid_data = valid_data.copy()
    invalid_data["timestamp"] = 9999999999999999  # Out of valid range
    
    try:
        validated = INSDataPoint(**invalid_data)
        print(f"❌ Invalid data validated without error")
    except Exception as e:
        print(f"✅ Correctly rejected invalid timestamp: {e}")
    
    # Invalid data point (angle out of range)
    invalid_data = valid_data.copy()
    invalid_data["roll"] = 5.0  # Out of valid range (> pi)
    
    try:
        validated = INSDataPoint(**invalid_data)
        print(f"❌ Invalid angle validated without error")
    except Exception as e:
        print(f"✅ Correctly rejected invalid angle: {e}")

def test_ins_features():
    print("\nTesting INSFeatures schema:")
    
    # Create a valid feature vector (24 values)
    valid_features = {
        "statistical_features": [0.1, 0.2, 0.3, 0.4] * 6,  # 24 values (6 channels × 4 stats)
        "window_id": 123,
        "timestamp": 1586346000000000
    }
    
    try:
        validated = INSFeatures(**valid_features)
        print(f"✅ Valid features validated: {validated.model_dump()}")
    except Exception as e:
        print(f"❌ Validation failed for valid features: {e}")
    
    # Invalid features (wrong length)
    invalid_features = valid_features.copy()
    invalid_features["statistical_features"] = [0.1, 0.2, 0.3]  # Too few values
    
    try:
        validated = INSFeatures(**invalid_features)
        print(f"❌ Invalid features validated without error")
    except Exception as e:
        print(f"✅ Correctly rejected invalid feature length: {e}")

def test_prediction_api():
    print("\nTesting Prediction API schemas:")
    
    # Valid prediction request
    valid_request = {
        "features": [0.1, 0.2, 0.3, 0.4] * 6  # 24 features
    }
    
    try:
        validated = PredictionRequest(**valid_request)
        print(f"✅ Valid request validated: {validated.model_dump()}")
    except Exception as e:
        print(f"❌ Validation failed for valid request: {e}")
    
    # Valid prediction response
    valid_response = {
        "prediction": 1,
        "probability": 0.95,
        "prediction_text": "Anomaly detected",
        "timestamp": 1586346000000000
    }
    
    try:
        validated = PredictionResponse(**valid_response)
        print(f"✅ Valid response validated: {validated.model_dump()}")
    except Exception as e:
        print(f"❌ Validation failed for valid response: {e}")
    
    # Invalid response (probability out of range)
    invalid_response = valid_response.copy()
    invalid_response["probability"] = 1.5  # Out of valid range (> 1.0)
    
    try:
        validated = PredictionResponse(**invalid_response)
        print(f"❌ Invalid response validated without error")
    except Exception as e:
        print(f"✅ Correctly rejected invalid probability: {e}")

if __name__ == "__main__":
    test_ins_data_point()
    test_ins_features()
    test_prediction_api()
    print("\nAll schema validation tests completed.") 