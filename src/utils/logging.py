import logging
import os
import sys
from typing import Optional


def setup_logger(name: str, log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and optionally file handlers.
    
    Args:
        name: Name of the logger
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to a log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_pipeline_logger(component: str) -> logging.Logger:
    """
    Get a logger configured for a specific pipeline component.
    
    Args:
        component: Pipeline component name (e.g., 'ingest', 'align', 'features')
        
    Returns:
        Configured logger for the component
    """
    log_dir = os.path.join("experiments", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{component}.log")
    logger = setup_logger(f"forge_pipeline.{component}", log_file=log_file)
    
    return logger 