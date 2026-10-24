import os
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union

from src.serve.simple_model_service import SimpleINSPredictionService
from src.monitoring.monitoring_service import get_monitor

# Create FastAPI app
app = FastAPI(
    title="Simple INS Prediction API",
    description="API for making predictions using INS data with a simplified rule-based model",
    version="1.0.0"
)

# Initialize prediction service
service = SimpleINSPredictionService()

# Initialize monitoring
monitor = get_monitor()

# Models for request and response
class INSData(BaseModel):
    velocity_north: float
    velocity_east: float
    velocity_down: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    prediction_text: str
    timestamp: int

class ErrorResponse(BaseModel):
    error: str
    
# Routes
@app.get("/")
async def root():
    return {"message": "Simple INS Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Simple model is ready"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: INSData):
    """
    Make a prediction using the INS data.
    """
    # Prepare features
    features = {
        "velocity_north": data.velocity_north,
        "velocity_east": data.velocity_east,
        "velocity_down": data.velocity_down,
        "roll": data.roll,
        "pitch": data.pitch,
        "yaw": data.yaw
    }
    
    # Get prediction
    prediction, probability = service.predict(features)
    
    # Get text label
    prediction_text = "Higher East Velocity" if prediction == 1 else "Lower East Velocity"
    
    # Create timestamp
    timestamp = int(time.time() * 1000)
    
    # Log the prediction for monitoring
    monitor.log_prediction(features, prediction, probability, timestamp)
    
    return {
        "prediction": prediction,
        "probability": probability,
        "prediction_text": prediction_text,
        "timestamp": timestamp
    }

@app.post("/feedback")
async def feedback(timestamp: int, actual_label: int):
    """
    Log feedback for a previous prediction.
    
    Args:
        timestamp: The timestamp of the prediction
        actual_label: The actual ground truth label
    """
    monitor.log_feedback(timestamp, actual_label)
    
    return {"status": "success", "message": "Feedback recorded"}

@app.post("/batch_predict")
async def batch_predict(data_list: List[INSData]):
    results = []
    for data in data_list:
        features = data.model_dump()
        result = service.predict_from_ins_data(features)
        results.append(result)
    
    return results


# Run with: uvicorn src.serve.simple_api:app --reload
if __name__ == "__main__":
    import uvicorn
    # Local only by default; set HOST=0.0.0.0 to accept connections from other machines
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000"))) 