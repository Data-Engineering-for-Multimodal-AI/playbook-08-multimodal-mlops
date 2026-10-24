from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Dict, List, Optional, Union, Any
import numpy as np
from datetime import datetime

class INSDataPoint(BaseModel):
    """
    Schema for a single INS (Inertial Navigation System) data point.
    Validates the structure and types of incoming INS data.
    """
    timestamp: int = Field(..., description="Timestamp in microseconds")
    velocity_north: float = Field(..., description="North component of velocity in m/s")
    velocity_east: float = Field(..., description="East component of velocity in m/s")
    velocity_down: float = Field(..., description="Down component of velocity in m/s")
    roll: float = Field(..., description="Roll angle in radians")
    pitch: float = Field(..., description="Pitch angle in radians")
    yaw: float = Field(..., description="Yaw angle (heading) in radians")
    
    model_config = ConfigDict(extra="ignore")  # Allow extra fields for backward compatibility
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Ensure timestamp is reasonable (between 2000 and 2100)"""
        # Convert microseconds to seconds
        seconds = v / 1_000_000
        
        # Check if timestamp is in reasonable range (between years 2000 and 2100)
        year_2000_timestamp = 946684800  # January 1, 2000
        year_2100_timestamp = 4102444800  # January 1, 2100
        
        if seconds < year_2000_timestamp or seconds > year_2100_timestamp:
            raise ValueError(f"Timestamp {v} is out of valid range")
        return v
    
    @field_validator('roll', 'pitch')
    @classmethod
    def validate_angle(cls, v):
        """Ensure angles are in valid range for radians (-π to π)"""
        if v < -np.pi or v > np.pi:
            raise ValueError(f"Angle {v} is out of valid range (-π to π)")
        return v
    
    @field_validator('yaw')
    @classmethod
    def validate_heading(cls, v):
        """Ensure the heading is in -π to 2π (Oxford RobotCar reports it from 0 to 2π)"""
        if v < -np.pi or v > 2 * np.pi:
            raise ValueError(f"Heading {v} is out of valid range (-π to 2π)")
        return v


class INSFeatures(BaseModel):
    """
    Schema for processed INS features, used to validate feature format
    before prediction or storage.
    """
    statistical_features: List[float] = Field(..., description="Statistical features from INS signals")
    window_id: Optional[int] = Field(None, description="Window identifier for the features")
    timestamp: Optional[int] = Field(None, description="Timestamp of the feature window")
    
    @field_validator('statistical_features')
    @classmethod
    def validate_features(cls, v):
        """Ensure feature vector has the expected length"""
        expected_length = 24  # Mean, std, min, max for 6 channels
        if len(v) != expected_length:
            raise ValueError(f"Feature vector must have {expected_length} elements, got {len(v)}")
        return v


class PredictionRequest(BaseModel):
    """
    Schema for prediction request payload.
    """
    features: Union[List[float], Dict[str, float]] = Field(
        ..., 
        description="INS features as list or dictionary"
    )
    
    @field_validator('features')
    @classmethod
    def validate_features_format(cls, v):
        """Ensure features are in expected format"""
        if isinstance(v, list) and not v:
            raise ValueError("Features list cannot be empty")
        
        if isinstance(v, dict) and not v:
            raise ValueError("Features dictionary cannot be empty")
            
        return v


class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    """
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    prediction_text: str = Field(..., description="Human-readable prediction description")
    timestamp: int = Field(..., description="Timestamp of prediction")
    
    @field_validator('prediction')
    @classmethod
    def validate_prediction(cls, v):
        """Ensure prediction is 0 or 1"""
        if v not in [0, 1]:
            raise ValueError(f"Prediction must be 0 or 1, got {v}")
        return v
    
    @field_validator('probability')
    @classmethod
    def validate_probability(cls, v):
        """Ensure probability is between 0 and 1"""
        if v < 0 or v > 1:
            raise ValueError(f"Probability must be between 0 and 1, got {v}")
        return v 