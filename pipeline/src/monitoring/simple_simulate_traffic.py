import os
import time
import json
import random
import numpy as np
import requests
import argparse
from datetime import datetime
from typing import Dict, List

def generate_random_ins_data(drift_factor: float = 0.0) -> Dict[str, float]:
    """
    Generate random INS data.
    
    Args:
        drift_factor: Factor to introduce data drift (0.0 = no drift)
    
    Returns:
        Dictionary with random INS data values
    """
    # Base means
    means = {
        "velocity_north": 1.0 + drift_factor * 2.0,  # Drift: increase north velocity
        "velocity_east": 5.0,  # No drift
        "velocity_down": 0.1,  # No drift
        "roll": 0.01,  # No drift
        "pitch": 0.02,  # No drift
        "yaw": 1.5 + drift_factor * 1.0  # Drift: increase yaw
    }
    
    # Base standard deviations
    stds = {
        "velocity_north": 0.5,
        "velocity_east": 2.0,
        "velocity_down": 0.05,
        "roll": 0.005,
        "pitch": 0.01,
        "yaw": 0.2
    }
    
    # Generate random values
    data = {}
    for key in means:
        data[key] = random.normalvariate(means[key], stds[key])
    
    return data

def simulate_traffic(api_url: str, num_requests: int, 
                    feedback_ratio: float = 0.2, 
                    introduce_drift: bool = False,
                    request_delay: float = 0.1) -> List[Dict]:
    """
    Simulate traffic to the model API.
    
    Args:
        api_url: Base URL of the API
        num_requests: Number of requests to send
        feedback_ratio: Proportion of requests to send feedback for
        introduce_drift: Whether to introduce data drift over time
        request_delay: Delay between requests in seconds
    
    Returns:
        List of response dictionaries
    """
    responses = []
    timestamps = []
    
    print(f"Sending {num_requests} requests to {api_url}...")
    for i in range(num_requests):
        # Calculate drift factor (0.0 to 1.0) if requested
        drift_factor = 0.0
        if introduce_drift and i > num_requests // 2:
            drift_factor = min(1.0, (i - num_requests // 2) / (num_requests // 2))
            
        # Generate data
        data = generate_random_ins_data(drift_factor)
        
        # Print progress
        if i % 10 == 0:
            print(f"Request {i}/{num_requests} - Drift factor: {drift_factor:.2f}")
            print(f"Data: {json.dumps(data, indent=2)}")
            
        try:
            # Send prediction request
            start_time = time.time()
            response = requests.post(f"{api_url}/predict", json=data)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                timestamps.append(result["timestamp"])
                responses.append({
                    "timestamp": result["timestamp"],
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "response_time": response_time
                })
                
                # Send feedback for some predictions
                if random.random() < feedback_ratio:
                    # For demonstration: use the prediction as the "actual" label
                    # but introduce some noise (20% chance of being different)
                    actual_label = result["prediction"]
                    if random.random() < 0.2:
                        actual_label = 1 - actual_label
                        
                    feedback_response = requests.post(
                        f"{api_url}/feedback",
                        params={"timestamp": result["timestamp"], "actual_label": actual_label}
                    )
                    
                    if feedback_response.status_code != 200:
                        print(f"Error sending feedback: {feedback_response.text}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"Request error: {str(e)}")
            
        # Add delay between requests
        time.sleep(request_delay)
    
    return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate traffic to simple model API")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--feedback", type=float, default=0.3, help="Feedback ratio (0.0 to 1.0)")
    parser.add_argument("--drift", action="store_true", help="Introduce data drift")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests (seconds)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = "experiments/simulation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation
    start_time = time.time()
    responses = simulate_traffic(
        api_url=args.url,
        num_requests=args.requests,
        feedback_ratio=args.feedback,
        introduce_drift=args.drift,
        request_delay=args.delay
    )
    total_time = time.time() - start_time
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"simple_simulation_{timestamp}.json")
    
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "total_time": total_time,
            "responses": responses,
            "avg_response_time": sum(r["response_time"] for r in responses) / len(responses) if responses else 0,
            "num_responses": len(responses)
        }, f, indent=2)
    
    print(f"\nSimulation complete!")
    print(f"Total time: {total_time:.2f} seconds")
    if responses:
        print(f"Average response time: {sum(r['response_time'] for r in responses) / len(responses):.4f} seconds")
        print(f"Predictions: {len([r for r in responses if r['prediction'] == 1])} class 1, {len([r for r in responses if r['prediction'] == 0])} class 0")
    else:
        print("No successful responses received")
    print(f"Results saved to: {result_file}") 