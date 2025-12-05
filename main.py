import os
import json

# Set thread limits FIRST before any other imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ray
import mlflow
import pandas as pd
from typing import Optional, Dict, Any
from ray import serve
from fastapi import FastAPI, Header, HTTPException, status

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

app = FastAPI(title="MLflow Model Serving", version="1.0.0")

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, model_name: str = "translation_model", default_version: str = "1"):
        self.model_name = model_name
        self.default_version = default_version
        self.default_model = mlflow.pyfunc.load_model(
            f"models:/{self.model_name}/{self.default_version}"
        )
        self.default_signature = self.default_model.metadata.signature
        
    def load_model(self, version: str):
        model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{version}")
        if model.metadata.signature != self.default_signature:
            raise ValueError(f"Model version {version} has incompatible signature")
        return model
    
    @app.post("/predict")
    async def predict(
        self, 
        model_input: Dict[str, Any],
        serve_multiplexed_model_id: Optional[str] = Header(None)
    ):
        version = serve_multiplexed_model_id or self.default_version
        try:
            model = self.load_model(version)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error loading model: {str(e)}")
        
        df = pd.DataFrame({k: [v] for k, v in model_input.items()})
        try:
            result = model.predict(df)
            return {"prediction": result, "version": version, "model": self.model_name}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")
    
    @app.get("/health")
    async def health(self):
        return {"status": "healthy", "model": self.model_name, "default_version": self.default_version}
    
    @app.get("/")
    async def root(self):
        return {
            "service": "MLflow Model Serving",
            "model": self.model_name,
            "endpoints": {"predict": "/predict", "health": "/health", "docs": "/docs"}
        }

if __name__ == "__main__":
    MODEL_NAME = os.getenv("MODEL_NAME", "translation_model")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
    PORT = int(os.getenv("PORT", 8000))
    
    ray.init(
        num_cpus=2,
        object_store_memory=1 * 1024 * 1024 * 1024,
        _plasma_directory="/tmp",
        _system_config={
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}
            ),
            "max_io_workers": 2
        }
    )
    
    serve.start(http_options={"host": "0.0.0.0", "port": PORT})
    
    deployment = ModelDeployment.bind(model_name=MODEL_NAME, default_version=MODEL_VERSION)
    serve.run(deployment, blocking=True)
