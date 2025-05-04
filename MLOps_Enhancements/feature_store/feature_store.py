import os
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feature_store")

class FeatureStore:
    """
    A simple feature store for storing and retrieving INS features.
    
    This implementation uses Parquet files as the storage backend with
    an Apache Arrow dataset to support efficient querying.
    """
    def __init__(self, 
                 base_path: str = "experiments/feature_store",
                 partition_by: Optional[List[str]] = None):
        """
        Initialize the feature store.
        
        Args:
            base_path: Base directory for feature storage
            partition_by: Columns to use for partitioning data
        """
        self.base_path = base_path
        self.partition_by = partition_by or ["date"]
        
        # Create directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)
        
        # Create subdirectories for different feature types
        self.ins_features_path = os.path.join(self.base_path, "ins_features")
        os.makedirs(self.ins_features_path, exist_ok=True)
        
        # Create metadata directory
        self.metadata_path = os.path.join(self.base_path, "metadata")
        os.makedirs(self.metadata_path, exist_ok=True)
        
        # Create a registry file if it doesn't exist
        self.registry_file = os.path.join(self.metadata_path, "feature_registry.json")
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, 'w') as f:
                json.dump({"features": {}, "last_updated": datetime.now().isoformat()}, f)
        
        logger.info(f"Initialized feature store at {self.base_path}")
    
    def get_dataset(self, feature_type: str = "ins_features") -> Optional[ds.Dataset]:
        """
        Get the Arrow dataset for a specific feature type.
        
        Args:
            feature_type: Type of features (e.g., 'ins_features')
            
        Returns:
            Arrow dataset or None if not found
        """
        if feature_type == "ins_features":
            path = self.ins_features_path
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        try:
            # Check if the directory has any parquet files (including in subdirectories)
            has_parquet = False
            for root, _, files in os.walk(path):
                if any(f.endswith('.parquet') for f in files):
                    has_parquet = True
                    break
                    
            if has_parquet:
                # Use absolute path to make sure dataset is created correctly
                abs_path = os.path.abspath(path)
                logger.info(f"Creating dataset from path: {abs_path}")
                dataset = ds.dataset(abs_path, format="parquet")
                logger.info(f"Dataset created with schema: {dataset.schema}")
                return dataset
            else:
                logger.warning(f"No parquet files found in {path}")
                return None
        except Exception as e:
            logger.error(f"Error accessing dataset at {path}: {str(e)}")
            return None
    
    def store_features(self, 
                      features_df: pd.DataFrame, 
                      feature_type: str = "ins_features",
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store features in the feature store.
        
        Args:
            features_df: DataFrame with features to store
            feature_type: Type of features (e.g., 'ins_features')
            metadata: Additional metadata about these features
            
        Returns:
            Feature ID
        """
        if features_df.empty:
            logger.warning("Empty feature DataFrame, not storing")
            return ""
        
        # Add date column for partitioning if needed
        if "timestamp" in features_df.columns and "date" not in features_df.columns:
            features_df["date"] = pd.to_datetime(features_df["timestamp"], unit='us').dt.date.astype(str)
        
        # Add unique feature ID if not present
        if "feature_id" not in features_df.columns:
            feature_id = str(uuid4())
            features_df["feature_id"] = feature_id
        else:
            feature_id = features_df["feature_id"].iloc[0]
        
        # Select target path based on feature type
        if feature_type == "ins_features":
            path = self.ins_features_path
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        try:
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(features_df)
            
            # Write to parquet with partitioning
            pq.write_to_dataset(
                table,
                root_path=path,
                partition_cols=self.partition_by
            )
            
            # Update registry
            self._update_registry(feature_id, feature_type, metadata)
            
            logger.info(f"Stored {len(features_df)} rows of {feature_type} with ID {feature_id}")
            return feature_id
            
        except Exception as e:
            logger.error(f"Error storing features: {str(e)}")
            return ""
    
    def retrieve_features(self, 
                         feature_id: Optional[str] = None,
                         filters: Optional[List[Tuple]] = None,
                         feature_type: str = "ins_features") -> pd.DataFrame:
        """
        Retrieve features from the feature store.
        
        Args:
            feature_id: Optional ID to retrieve specific features
            filters: Optional list of filter conditions [(column, op, value), ...]
            feature_type: Type of features to retrieve
            
        Returns:
            DataFrame with retrieved features
        """
        dataset = self.get_dataset(feature_type)
        if dataset is None:
            logger.warning(f"No dataset found for {feature_type}")
            return pd.DataFrame()
        
        try:
            # Build filter expression
            filter_expressions = []
            
            if feature_id:
                filter_expressions.append(("feature_id", "==", feature_id))
                
            if filters:
                filter_expressions.extend(filters)
            
            # Apply filters if any
            if filter_expressions:
                # Convert to PyArrow filter expressions
                arrow_filters = []
                for col, op, val in filter_expressions:
                    if op == "==":
                        arrow_filters.append((ds.field(col) == val))
                    elif op == "!=":
                        arrow_filters.append((ds.field(col) != val))
                    elif op == ">":
                        arrow_filters.append((ds.field(col) > val))
                    elif op == ">=":
                        arrow_filters.append((ds.field(col) >= val))
                    elif op == "<":
                        arrow_filters.append((ds.field(col) < val))
                    elif op == "<=":
                        arrow_filters.append((ds.field(col) <= val))
                
                # Combine with AND
                final_filter = arrow_filters[0]
                for f in arrow_filters[1:]:
                    final_filter = final_filter & f
                
                # Read with filter
                result = dataset.to_table(filter=final_filter).to_pandas()
            else:
                # Read all
                result = dataset.to_table().to_pandas()
            
            logger.info(f"Retrieved {len(result)} rows of {feature_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving features: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_features(self, 
                           limit: int = 100,
                           feature_type: str = "ins_features") -> pd.DataFrame:
        """
        Get the latest features from the store.
        
        Args:
            limit: Maximum number of rows to return
            feature_type: Type of features to retrieve
            
        Returns:
            DataFrame with latest features
        """
        dataset = self.get_dataset(feature_type)
        if dataset is None:
            logger.warning(f"No dataset found for latest features in {feature_type}")
            return pd.DataFrame()
        
        try:
            # Read all data and sort in memory (we could use a database for better performance)
            df = dataset.to_table().to_pandas()
            
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp", ascending=False).head(limit)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving latest features: {str(e)}")
            return pd.DataFrame()
    
    def _update_registry(self, 
                        feature_id: str, 
                        feature_type: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update the feature registry with new feature metadata."""
        try:
            # Read current registry
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
            
            # Update registry
            if feature_type not in registry["features"]:
                registry["features"][feature_type] = {}
            
            registry["features"][feature_type][feature_id] = {
                "id": feature_id,
                "type": feature_type,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            registry["last_updated"] = datetime.now().isoformat()
            
            # Write updated registry
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating feature registry: {str(e)}")
    
    def list_features(self, feature_type: Optional[str] = None) -> Dict[str, Any]:
        """
        List available features in the store.
        
        Args:
            feature_type: Optional type filter
            
        Returns:
            Dictionary with feature metadata
        """
        try:
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
            
            if feature_type:
                return {
                    feature_type: registry["features"].get(feature_type, {})
                }
            else:
                return registry["features"]
                
        except Exception as e:
            logger.error(f"Error listing features: {str(e)}")
            return {} 