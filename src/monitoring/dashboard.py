import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st

def load_monitoring_data(monitoring_dir="experiments/monitoring"):
    """Load all monitoring data for display."""
    data = {
        "predictions": [],
        "data_drift": [],
        "performance": []
    }
    
    # Load predictions
    pred_files = glob.glob(os.path.join(monitoring_dir, "predictions", "*.parquet"))
    if pred_files:
        pred_dfs = [pd.read_parquet(f) for f in pred_files]
        if pred_dfs:
            data["predictions"] = pd.concat(pred_dfs, ignore_index=True)
    
    # Load data drift
    drift_files = glob.glob(os.path.join(monitoring_dir, "data_drift", "feature_stats_*.json"))
    for f in sorted(drift_files):
        try:
            with open(f, 'r') as file:
                stats = json.load(file)
                data["data_drift"].append(stats)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    # Load performance metrics
    perf_files = glob.glob(os.path.join(monitoring_dir, "performance", "performance_*.json"))
    for f in sorted(perf_files, key=os.path.getmtime):
        try:
            with open(f, 'r') as file:
                metrics = json.load(file)
                data["performance"].append(metrics)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    # Load drift alerts
    drift_alert_files = glob.glob(os.path.join(monitoring_dir, "data_drift", "drift_alert_*.json"))
    data["drift_alerts"] = []
    for f in sorted(drift_alert_files):
        try:
            with open(f, 'r') as file:
                alert = json.load(file)
                data["drift_alerts"].append(alert)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return data

def display_prediction_stats(predictions_df):
    """Display prediction statistics."""
    st.subheader("Prediction Statistics")
    
    if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
        # Basic stats
        st.write(f"Total predictions: {len(predictions_df)}")
        
        # Class distribution
        if "prediction" in predictions_df.columns:
            class_dist = predictions_df["prediction"].value_counts().to_dict()
            st.write("Class distribution:")
            st.write(class_dist)
            
            # Plot class distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(["Class 0", "Class 1"], 
                   [class_dist.get(0, 0), class_dist.get(1, 0)])
            ax.set_ylabel("Count")
            ax.set_title("Prediction Class Distribution")
            st.pyplot(fig)
        
        # Probability distribution
        if "probability" in predictions_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(predictions_df["probability"], bins=20)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Count")
            ax.set_title("Prediction Probability Distribution")
            st.pyplot(fig)
    else:
        st.write("No prediction data available")

def display_data_drift(data_drift_list, drift_alerts=None):
    """Display data drift information."""
    st.subheader("Data Drift Monitoring")
    
    if not data_drift_list:
        st.write("No data drift information available")
        return
    if drift_alerts is None:
        drift_alerts = []
    
    # Convert to DataFrame for easier analysis
    drift_df = pd.DataFrame(data_drift_list)
    
    # Display feature trends over time
    if len(drift_df) > 1:
        # Get feature columns (those ending with _mean)
        feature_cols = [col for col in drift_df.columns if col.endswith("_mean") 
                       and not col.startswith("timestamp")]
        
        if feature_cols:
            st.write("Feature trends over time:")
            
            # Convert timestamp strings to datetime objects
            timestamps = [datetime.fromisoformat(ts) if isinstance(ts, str) else "N/A" 
                         for ts in drift_df["timestamp"]]
            
            # Plot feature means over time
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in feature_cols:
                feature_name = col.replace("_mean", "")
                ax.plot(range(len(drift_df)), drift_df[col], marker='o', label=feature_name)
            
            ax.set_xlabel("Data point")
            ax.set_ylabel("Value")
            ax.set_title("Feature Means Over Time")
            ax.legend()
            st.pyplot(fig)
    
    # Display drift alerts if any
    st.write("### Drift Alerts")
    if drift_alerts:
        for alert in drift_alerts:
            alert_time = datetime.fromisoformat(alert["timestamp"])
            st.write(f"⚠️ Drift detected at {alert_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display drift metrics
            if "metrics" in alert:
                for feature, metrics in alert["metrics"].items():
                    st.write(f"- **{feature}**: Z-score = {metrics['z_score']:.2f}")
                    st.write(f"  Current mean: {metrics['current_mean']:.4f}, Historical mean: {metrics['historical_mean']:.4f}")
    else:
        st.write("No drift alerts detected")

def display_performance_metrics(performance_list):
    """Display model performance metrics."""
    st.subheader("Model Performance")
    
    if not performance_list:
        st.write("No performance data available")
        return
    
    # Create DataFrame for easier analysis
    perf_df = pd.DataFrame(performance_list)
    
    # Convert timestamps to datetime
    timestamps = []
    for ts in perf_df["timestamp"]:
        try:
            timestamps.append(datetime.fromisoformat(ts))
        except:
            timestamps.append(None)
    
    # Plot accuracy (and F1, per labeled batch) over time if we have multiple data points
    if len(perf_df) > 1 and "accuracy" in perf_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(perf_df)), perf_df["accuracy"], marker='o', label="Accuracy")
        if "f1" in perf_df.columns:
            ax.plot(range(len(perf_df)), perf_df["f1"], marker='s', label="F1")
        if "model_version" in perf_df.columns:
            changes = perf_df.index[perf_df["model_version"].ne(perf_df["model_version"].shift())][1:]
            for i in changes:
                ax.axvline(i, color="gray", linestyle="--")
                ax.annotate(f"v{perf_df['model_version'][i]}", (i, 1.0), xytext=(3, -12), textcoords="offset points")
        ax.set_xlabel("Labeled batch")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Over Time (dashed line: a new model version)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    # Display latest performance metrics
    if len(perf_df) > 0:
        latest = perf_df.iloc[-1]
        st.write("### Latest Performance Metrics")
        st.write(f"- Timestamp: {timestamps[-1].strftime('%Y-%m-%d %H:%M:%S') if timestamps[-1] else 'N/A'}")
        st.write(f"- Accuracy: {latest.get('accuracy', 'N/A'):.4f}")
        st.write(f"- Number of samples: {latest.get('num_samples', 'N/A')}")
        st.write(f"- Correct predictions: {latest.get('total_correct', 'N/A')}")
        st.write(f"- Incorrect predictions: {latest.get('total_incorrect', 'N/A')}")
        
        # Class distribution
        if "class_distribution" in latest and latest["class_distribution"]:
            st.write("- Class distribution:")
            class_dist = latest["class_distribution"]
            for cls, count in class_dist.items():
                st.write(f"  - Class {cls}: {count}")

# Streamlit app
st.title("INS Model Monitoring Dashboard")

# Load data
data = load_monitoring_data()

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Predictions", "Data Drift", "Performance"])

with tab1:
    display_prediction_stats(data["predictions"])

with tab2:
    display_data_drift(data["data_drift"], data["drift_alerts"])

with tab3:
    display_performance_metrics(data["performance"])

# Display timestamp of dashboard refresh
st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Add refresh button
if st.sidebar.button("Refresh Data"):
    st.rerun() 