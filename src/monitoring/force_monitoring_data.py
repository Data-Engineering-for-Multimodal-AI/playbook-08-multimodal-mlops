import os
import numpy as np
import pandas as pd
import time
import json
import traceback
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any

from src.monitoring.monitoring_service import get_monitor

# Get the monitoring service
monitor = get_monitor()

def generate_batch_with_drift(num_samples: int, 
                              drift_factor: float = 0.0, 
                              feedback_ratio: float = 0.5,
                              error_rate: float = 0.2) -> None:
    """
    Generate a batch of synthetic prediction data with controlled drift.
    
    Args:
        num_samples: Number of samples to generate
        drift_factor: Factor to introduce drift (0.0 to 2.0)
        feedback_ratio: Ratio of samples to provide feedback for
        error_rate: Rate of errors in feedback
    """
    print(f"Generating batch with drift_factor={drift_factor:.2f}")
    
    # Base statistics for our features (no drift)
    base_means = {
        "velocity_north": 1.0,
        "velocity_east": 5.0,
        "velocity_down": 0.1,
        "roll": 0.01,
        "pitch": 0.02,
        "yaw": 1.5
    }
    
    base_stds = {
        "velocity_north": 0.5,
        "velocity_east": 2.0,
        "velocity_down": 0.05,
        "roll": 0.005,
        "pitch": 0.01,
        "yaw": 0.2
    }
    
    # Apply drift to means
    drifted_means = {}
    for key, value in base_means.items():
        # Apply different drift amounts to different features
        if key == "velocity_north":
            # Strong drift on velocity_north
            drifted_means[key] = value + (drift_factor * 2.0)
        elif key == "yaw":
            # Medium drift on yaw
            drifted_means[key] = value + (drift_factor * 1.0)
        elif key == "velocity_east":
            # Slight drift on velocity_east
            drifted_means[key] = value + (drift_factor * 0.5)
        else:
            # No drift on other features
            drifted_means[key] = value
    
    timestamps = []
    batch_features = []
    
    # Generate samples with the drifted distribution
    for i in range(num_samples):
        # Generate feature values
        features = {}
        for key in base_means.keys():
            features[key] = random.normalvariate(drifted_means[key], base_stds[key])
        
        batch_features.append(features)
        
        # Determine prediction based on velocity_east
        velocity_east = features["velocity_east"]
        prediction = 1 if velocity_east > 7.0 else 0
        
        # Calculate probability
        distance = abs(velocity_east - 7.0)
        base_prob = 0.5
        prob_adjustment = min(distance / 10.0, 0.49)
        if prediction == 1:
            probability = base_prob + prob_adjustment
        else:
            probability = base_prob - prob_adjustment
        
        # Create timestamp with small random increment
        timestamp = int(time.time() * 1000) + i
        timestamps.append(timestamp)
        
        # Log the prediction
        monitor.log_prediction(features, prediction, probability, timestamp)
        
        # Randomly provide feedback for some predictions
        if random.random() < feedback_ratio:
            # Determine actual label (with some error rate)
            if random.random() < error_rate:
                actual_label = 1 - prediction  # Wrong prediction
            else:
                actual_label = prediction  # Correct prediction
                
            # Log feedback
            monitor.log_feedback(timestamp, actual_label)
    
    print(f"Generated {num_samples} samples with drift_factor={drift_factor:.2f}")
    print(f"Provided feedback for approximately {int(num_samples * feedback_ratio)} samples")
    
    # Force save predictions if there are any in the buffer
    if hasattr(monitor, 'predictions_df') and not monitor.predictions_df.empty:
        print(f"Calling _save_predictions() with {len(monitor.predictions_df)} records")
        monitor._save_predictions()
    
    # Manually calculate and save feature statistics
    print("Manually calculating feature statistics for this batch")
    stats = {}
    batch_df = pd.DataFrame(batch_features)
    
    for col in batch_df.columns:
        stats[f"{col}_mean"] = float(batch_df[col].mean())
        stats[f"{col}_std"] = float(batch_df[col].std())
        stats[f"{col}_min"] = float(batch_df[col].min())
        stats[f"{col}_max"] = float(batch_df[col].max())
    
    stats["timestamp"] = datetime.now().isoformat()
    stats["num_records"] = int(len(batch_df))
    
    # Save statistics to disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("experiments", "monitoring", "data_drift", f"feature_stats_{timestamp}.json")
    
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved feature statistics to {filepath}")
    
    # Add to feature_stats_df in monitor
    if hasattr(monitor, 'feature_stats_df'):
        monitor.feature_stats_df = pd.concat([
            monitor.feature_stats_df,
            pd.DataFrame([stats])
        ], ignore_index=True)
    
    # If we have at least two batches, detect drift 
    if hasattr(monitor, 'feature_stats_df') and len(monitor.feature_stats_df) >= 2:
        print(f"Detecting drift (feature_stats_df has {len(monitor.feature_stats_df)} records)")
        monitor._detect_drift(stats)
    
    # Only calculate performance if we actually have feedback data
    feedback_samples = [t for t in timestamps if random.random() < feedback_ratio]
    if len(feedback_samples) > 10 and hasattr(monitor, 'predictions_df'):
        try:
            print("Calculating performance metrics")
            monitor._calculate_performance()
        except Exception as e:
            print(f"Failed to calculate performance: {str(e)}")
            traceback.print_exc()

def simulate_drift_progression():
    """
    Simulate a progression of drift over several batches.
    """
    print("Starting monitoring data generation with drift progression...")
    
    # Check data_drift directory
    data_drift_dir = os.path.join("experiments", "monitoring", "data_drift")
    print(f"Data drift directory: {data_drift_dir}")
    print(f"Directory exists: {os.path.exists(data_drift_dir)}")
    print(f"Directory is writable: {os.access(data_drift_dir, os.W_OK)}")
    
    # Clean up existing data for a fresh run
    for f in os.listdir(data_drift_dir):
        if f.startswith("feature_stats_") or f.startswith("drift_alert_"):
            os.remove(os.path.join(data_drift_dir, f))
    print("Cleaned up existing feature stats and drift alerts")
    
    # Batch 1: No drift
    generate_batch_with_drift(120, drift_factor=0.0, feedback_ratio=0.6)
    time.sleep(1)  # Pause between batches
    
    # Batch 2: Slight drift
    generate_batch_with_drift(120, drift_factor=0.4, feedback_ratio=0.6)
    time.sleep(1)
    
    # Batch 3: Medium drift
    generate_batch_with_drift(120, drift_factor=0.8, feedback_ratio=0.6)
    time.sleep(1)
    
    # Batch 4: Strong drift
    generate_batch_with_drift(120, drift_factor=1.2, feedback_ratio=0.6)
    time.sleep(1)
    
    # Batch 5: Very strong drift
    generate_batch_with_drift(120, drift_factor=1.6, feedback_ratio=0.6)
    time.sleep(1)
    
    # Batch 6: Extreme drift
    generate_batch_with_drift(120, drift_factor=2.0, feedback_ratio=0.6)
    
    # Final check of data_drift directory
    try:
        files = os.listdir(data_drift_dir)
        print(f"\nFinal check - Files in data_drift directory: {files}")
        
        # Count drift alerts
        drift_alerts = [f for f in files if f.startswith("drift_alert_")]
        print(f"Drift alerts detected: {len(drift_alerts)}")
    except Exception as e:
        print(f"Error checking data_drift directory: {str(e)}")
    
    print("\nDrift progression simulation complete!")
    print("Generated a total of 720 samples with varying degrees of drift")
    print("Monitoring data should now show drift detection and performance metrics")

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs("experiments/monitoring/predictions", exist_ok=True)
    os.makedirs("experiments/monitoring/data_drift", exist_ok=True)
    os.makedirs("experiments/monitoring/performance", exist_ok=True)
    
    # Run the simulation
    simulate_drift_progression()
    
    print("\nMonitoring data generation complete. You can now run the dashboard to view the results:")
    print("streamlit run src/monitoring/dashboard.py") 