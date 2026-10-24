import os
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union

from src.serve.model_service import INSPredictionService
from src.monitoring.monitoring_service import get_monitor
from src.utils import get_param

# Create FastAPI app
app = FastAPI(
    title="INS Prediction API",
    description="Serves the champion model from the MLflow Model Registry on INS data",
    version="2.0.0"
)

# Initialize prediction service
service = INSPredictionService()

# Initialize monitoring
monitor = get_monitor()
MONITORED = get_param('monitoring.monitor_features', [])


def load_champion() -> bool:
    """Load the champion model and monitor against the data it was trained on."""
    if not service.load():
        return False
    reference = os.path.join("experiments", "monitoring", "reference", f"reference_v{service.version}.json")
    monitor.set_reference(reference)
    return True


# Models for request and response
class INSData(BaseModel):
    velocity_north: float
    velocity_east: float
    velocity_down: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    timestamp: Optional[int] = Field(None, description="Microseconds; readings are assumed 50 Hz without it")


class INSWindow(BaseModel):
    readings: List[INSData] = Field(..., min_length=1,
                                    description="Consecutive INS readings; the model was trained on 0.5 s (25 readings)")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    prediction_text: str
    model_version: Optional[int] = None
    timestamp: int


class ErrorResponse(BaseModel):
    error: str


def readings_of(data: Union[INSWindow, INSData]) -> List[Dict[str, Any]]:
    readings = data.readings if isinstance(data, INSWindow) else [data]
    return [r.model_dump(exclude_none=True) for r in readings]


# Routes
@app.get("/")
async def root():
    return {"message": "INS Prediction API", "version": "2.0.0"}


@app.get("/health")
async def health():
    # Check if model is loaded or can be loaded
    if not service.is_loaded:
        if not load_champion():
            return {"status": "error", "message": "No trained model (run the training step first)"}

    return {"status": "healthy", "message": "Model is loaded and ready", "model_version": service.version}


@app.post("/reload")
async def reload():
    """Serve the current champion (call after a new version is promoted)."""
    previous = service.version
    if not load_champion():
        raise HTTPException(status_code=500, detail="Failed to load model")
    return {"previous_version": previous, "model_version": service.version}


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: Union[INSWindow, INSData]):
    """
    Make a prediction from a window of INS readings ({"readings": [...]}) or a single reading.
    """
    if not service.is_loaded:
        if not load_champion():
            raise HTTPException(status_code=500, detail="Failed to load model")

    window = service.window_features(readings_of(data))
    prediction, probability = service.predict_windows([window['statistical_features']])
    prediction, probability = int(prediction[0]), float(probability[0])

    # Create timestamp
    timestamp = int(time.time() * 1000)

    # Log the prediction and the monitored window features for drift detection
    monitor.log_prediction({f: float(window[f]) for f in MONITORED if f in window},
                           prediction, probability, timestamp)

    return {
        "prediction": prediction,
        "probability": probability,
        "prediction_text": service.label_text[prediction],
        "model_version": service.version,
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
async def batch_predict(data_list: List[Union[INSWindow, INSData]]):
    if not service.is_loaded:
        if not load_champion():
            raise HTTPException(status_code=500, detail="Failed to load model")
    return [service.predict_from_ins_data(readings_of(data)) for data in data_list]


# Run with: uvicorn src.serve.api:app
if __name__ == "__main__":
    import uvicorn
    # Local only by default; set HOST=0.0.0.0 to accept connections from other machines
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
