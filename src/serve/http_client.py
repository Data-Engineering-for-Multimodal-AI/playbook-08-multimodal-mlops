import requests
import json
import time

def test_api():
    """
    Test the FastAPI endpoints using real HTTP requests.
    """
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test prediction endpoint
    print("\n2. Testing prediction endpoint...")
    
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
            "velocity_north": 0.5, 
            "velocity_east": 2.1,  # Below 7.0 threshold
            "velocity_down": 0.1,
            "roll": 0.02,
            "pitch": 0.01,
            "yaw": 1.2
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Input: {json.dumps(test_case, indent=2)}")
        response = requests.post(
            f"{base_url}/predict",
            json=test_case
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("Waiting 2 seconds for server to start...")
    time.sleep(2)  # Give time for the server to start
    test_api() 