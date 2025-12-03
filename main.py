import os
import mlflow
import pandas as pd
from typing import Optional, Dict, Any
from ray import serve
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel

# Configure MLflow tracking URI from environment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

app = FastAPI(title="MLflow Model Serving", version="1.0.0")

@serve.deployment
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, model_name: str, default_version: str = "1"):
        self.model_name = model_name
        self.default_version = default_version
        
        # Load default model
        self.default_model = mlflow.pyfunc.load_model(
            f"models:/{self.model_name}/{self.default_version}"
        )
        self.default_signature = self.default_model.metadata.signature
        
    def load_model(self, version: str):
        """Load model with signature validation"""
        model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{version}")
        
        # Validate signature compatibility
        if model.metadata.signature != self.default_signature:
            raise ValueError(f"Model version {version} has incompatible signature")
        
        return model
    
    @app.post("/predict")
    async def predict(
        self, 
        model_input: Dict[str, Any],
        serve_multiplexed_model_id: Optional[str] = Header(None)
    ):
        """Prediction endpoint with version multiplexing"""
        version = serve_multiplexed_model_id or self.default_version
        
        try:
            model = self.load_model(version)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        
        # Convert input to DataFrame
        df = pd.DataFrame({k: [v] for k, v in model_input.items()})
        
        # Get prediction
        result = model.predict(df)
        
        return {"prediction": result, "version": version, "model": self.model_name}
    
    @app.get("/health")
    async def health(self):
        """Health check endpoint"""
        return {
            "status": "healthy", 
            "model": self.model_name,
            "default_version": self.default_version
        }
    
    @app.get("/")
    async def root(self):
        """Root endpoint"""
        return {
            "service": "MLflow Model Serving",
            "model": self.model_name,
            "endpoints": {
                "predict": "/predict",
                "health": "/health",
                "docs": "/docs"
            }
        }

if __name__ == "__main__":
    # Get configuration from environment
    MODEL_NAME = os.getenv("MODEL_NAME", "translation_model")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
    
    # Create and run deployment
    deployment = ModelDeployment.bind(model_name=MODEL_NAME, default_version=MODEL_VERSION)
    serve.run(deployment, host="0.0.0.0", port=8000)
