#!/usr/bin/env python
"""
Extract features from INS data for the MLOps pipeline.
This script is called by the Airflow DAG to process INS data.
"""
import os
import pandas as pd
import logging

from src.ingest.ins_ingest import ingest_ins_data
from src.features.ins_features import process_ins_features
from src.utils import get_pipeline_logger

# Set up logging
logger = get_pipeline_logger("extract_ins_features")

def main():
    """Extract features from INS data."""
    logger.info("Starting INS feature extraction")
    
    # Step 1: Ingest INS data if it hasn't been ingested yet
    ingest_output_path = os.path.join("experiments", "ingested", "imu")
    if not os.path.exists(os.path.join(ingest_output_path, "ins_processed.csv")):
        logger.info("INS data not yet ingested, performing ingestion...")
        ingest_ins_data(output_path=ingest_output_path)
    else:
        logger.info("Using previously ingested INS data")
    
    # Step 2: Process INS features
    ins_data_path = os.path.join(ingest_output_path, "ins_processed.csv")
    features_output_path = os.path.join("experiments", "features", "ins")
    
    # Create output directory if it doesn't exist
    os.makedirs(features_output_path, exist_ok=True)
    
    # Process features
    logger.info("Extracting INS features...")
    features_df = process_ins_features(ins_data_path, output_path=features_output_path)
    
    num_features = len(features_df) if features_df is not None else 0
    logger.info(f"Feature extraction complete. Generated {num_features} feature windows")
    
    return features_df

if __name__ == "__main__":
    main() 