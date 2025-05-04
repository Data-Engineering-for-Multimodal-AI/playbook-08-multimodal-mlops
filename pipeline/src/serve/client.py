import json
from src.serve.model_service import INSPredictionService

def test_model_service():
    """
    Test the model service directly without HTTP.
    """
    print("Testing INS Prediction Service")
    
    # Create service
    service = INSPredictionService()
    
    # Load model
    success = service.load()
    if not success:
        print("Failed to load model")
        return
    
    print("Model loaded successfully")
    
    # Create test cases
    test_cases = [
        # High east velocity
        {
            "velocity_north": 1.42, 
            "velocity_east": 7.5,  # Above 7.0 threshold
            "velocity_down": 0.3,
            "roll": 0.01,
            "pitch": 0.02,
            "yaw": 1.5
        },
        # Low east velocity
        {
            "velocity_north": 1.39,
            "velocity_east": 6.5,  # Below 7.0 threshold
            "velocity_down": 0.2,
            "roll": 0.03,
            "pitch": 0.01,
            "yaw": 1.3
        }
    ]
    
    # Run predictions
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Input: {json.dumps(test_case, indent=2)}")
        
        # Make prediction
        result = service.predict_from_ins_data(test_case)
        
        print(f"Output: {json.dumps(result, indent=2)}")
        
    print("\nTesting complete!")


if __name__ == "__main__":
    test_model_service() 