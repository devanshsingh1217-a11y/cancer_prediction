from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import numpy as np

# Dictionary to hold our machine learning artifacts globally
ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load artifacts exactly once during application startup using clean relative paths
    ml_artifacts["model"] = joblib.load("model_xgb_new.pkl")
    ml_artifacts["label_encoder"] = joblib.load("label_encoder.pkl")
    ml_artifacts["feature_names"] = joblib.load("feature_names.pkl")
    yield
    # Clean up artifacts when the application shuts down
    ml_artifacts.clear()

# Initialize FastAPI with the lifespan hook
app = FastAPI(title="Cancer Risk Prediction API", version="1.0", lifespan=lifespan)

# Define the expected input schema with strict boundaries
class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid") # Rejects unknown fields instead of ignoring them
    
    Age: int = Field(ge=0, le=120)
    Gender: int = Field(ge=0, le=1)
    BMI: float = Field(ge=10.0, le=70.0)
    Smoking: int = Field(ge=0, le=10)
    Alcohol_Use: int = Field(ge=0, le=10)
    Obesity: int = Field(ge=0, le=10)
    Diet_Red_Meat: int = Field(ge=0, le=10)
    Diet_Salted_Processed: int = Field(ge=0, le=10)
    Fruit_Veg_Intake: int = Field(ge=0, le=10)
    Physical_Activity: int = Field(ge=0, le=10)
    Physical_Activity_Level: int = Field(ge=0, le=10)
    Air_Pollution: int = Field(ge=0, le=10)
    Occupational_Hazards: int = Field(ge=0, le=10)
    Calcium_Intake: int = Field(ge=0, le=10)
    Family_History: int = Field(ge=0, le=1)
    BRCA_Mutation: int = Field(ge=0, le=1)
    H_Pylori_Infection: int = Field(ge=0, le=1)

class PredictionResponse(BaseModel):
    risk_level: str
    probabilities: dict[str, float]
    model_version: str

@app.get("/health")
async def health_check():
    """Liveness probe for orchestration tools like Kubernetes or Docker Compose."""
    if not ml_artifacts.get("model"):
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {"status": "healthy"}

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(patient: PatientFeatures):
    """Single-patient prediction endpoint."""
    try:
        input_data = pd.DataFrame([patient.model_dump()])
        # Enforce column order to exactly match training data
        input_data = input_data[ml_artifacts["feature_names"]]
        
        # Run prediction
        model = ml_artifacts["model"]
        encoder = ml_artifacts["label_encoder"]
        
        prediction_idx = model.predict(input_data)[0]
        prediction_probs = model.predict_proba(input_data)[0]
        
        # Decode the predicted integer
        raw_risk_label = encoder.inverse_transform([prediction_idx])[0]
        
        # Map numeric output or labels to clean strings safely
        risk_map = {"0": "Low", "1": "Medium", "2": "High", 0: "Low", 1: "Medium", 2: "High"}
        risk_label = risk_map.get(raw_risk_label, str(raw_risk_label))
        
        # Map probabilities ensuring dictionary keys are strings and values are native floats
        class_names = encoder.classes_
        prob_dict = {
            risk_map.get(cls, str(cls)): float(prediction_probs[i]) 
            for i, cls in enumerate(class_names)
        }
        
        return PredictionResponse(
            risk_level=str(risk_label),
            probabilities=prob_dict,
            model_version="v1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))