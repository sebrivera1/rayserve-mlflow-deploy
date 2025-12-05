import os
import mlflow
import pandas as pd
import uvicorn
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel

# Global state
MODEL_NAME = os.getenv("MODEL_NAME", "Demo-DummyModel")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
model_cache = {}
default_signature = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global default_signature

    # Initialize MLflow with proper URI scheme
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    # Ensure proper URI scheme (http/https)
    if not tracking_uri.startswith(("http://", "https://")):
        tracking_uri = f"http://{tracking_uri}"

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")

    # Load default model on startup
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        model_cache[MODEL_VERSION] = loaded_model

        # Safely get signature
        if loaded_model.metadata and hasattr(loaded_model.metadata, 'signature'):
            default_signature = loaded_model.metadata.signature

        print(f"Loaded default model: {MODEL_NAME} version {MODEL_VERSION}")
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")

    yield

    # Cleanup on shutdown (if needed)
    model_cache.clear()

# FastAPI app with lifespan
app = FastAPI(title="MLflow Model Serving", version="1.0.0", lifespan=lifespan)

class PredictRequest(BaseModel):
    model_input: Dict[str, Any]
    version: Optional[str] = None

def load_model(version: str):
    """Load model with caching"""
    if version not in model_cache:
        model_uri = f"models:/{MODEL_NAME}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        if default_signature and model.metadata.signature != default_signature:
            raise ValueError(f"Model version {version} has incompatible signature")
        
        model_cache[version] = model
    
    return model_cache[version]

@app.post("/predict")
async def predict(
    request: PredictRequest,
    serve_multiplexed_model_id: Optional[str] = Header(None)
):
    """Make predictions using specified model version"""
    version = request.version or serve_multiplexed_model_id or MODEL_VERSION
    
    try:
        model = load_model(version)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )
    
    df = pd.DataFrame({k: [v] for k, v in request.model_input.items()})
    
    try:
        result = model.predict(df)
        return {
            "prediction": result.tolist() if hasattr(result, 'tolist') else result,
            "version": version,
            "model": MODEL_NAME
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
        "model": MODEL_NAME,
        "default_version": MODEL_VERSION,
        "cached_versions": list(model_cache.keys())
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "MLflow Model Serving",
        "model": MODEL_NAME,
        "default_version": MODEL_VERSION,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
