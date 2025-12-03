import os
import mlflow
import pandas as pd
from typing import Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException, status
import uvicorn

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

app = FastAPI(title="MLflow Model Serving", version="1.0.0")

# Global model cache
MODEL_CACHE = {}

def get_model(model_name: str, version: str):
    """Load and cache model"""
    cache_key = f"{model_name}:{version}"
    
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = mlflow.pyfunc.load_model(
            f"models:/{model_name}/{version}"
        )
    
    return MODEL_CACHE[cache_key]

@app.post("/predict")
async def predict(
    model_input: Dict[str, Any],
    serve_multiplexed_model_id: Optional[str] = Header(None)
):
    """Prediction endpoint with version multiplexing"""
    model_name = os.getenv("MODEL_NAME", "translation_model")
    version = serve_multiplexed_model_id or os.getenv("MODEL_VERSION", "1")
    
    try:
        model = get_model(model_name, version)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )
    
    # Convert input to DataFrame
    df = pd.DataFrame({k: [v] for k, v in model_input.items()})
    
    # Get prediction
    try:
        result = model.predict(df)
        return {
            "prediction": result, 
            "version": version, 
            "model": model_name
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model": os.getenv("MODEL_NAME", "translation_model"),
        "default_version": os.getenv("MODEL_VERSION", "1")
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MLflow Model Serving",
        "model": os.getenv("MODEL_NAME", "translation_model"),
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
