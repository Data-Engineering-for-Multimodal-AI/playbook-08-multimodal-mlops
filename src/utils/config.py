import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_schema() -> Dict[str, Any]:
    """
    Load the schema configuration for data processing.
    
    Returns:
        Dictionary containing schema configuration
    """
    schema_path = os.path.join("config", "schema.yaml")
    return load_config(schema_path)


def load_params() -> Dict[str, Any]:
    """
    Load the parameters configuration for the pipeline.
    
    Returns:
        Dictionary containing parameters configuration
    """
    params_path = os.path.join("config", "params.yaml")
    return load_config(params_path)


def get_param(param_path: str, default: Optional[Any] = None) -> Any:
    """
    Get a specific parameter from the params.yaml file using dot notation.
    
    Args:
        param_path: Parameter path using dot notation (e.g., "model.params.n_estimators")
        default: Default value to return if the parameter is not found
        
    Returns:
        Parameter value or default if not found
    """
    params = load_params()
    
    keys = param_path.split('.')
    value = params
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default 