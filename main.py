import os
import mlflow
import pandas as pd
import numpy as np
import uvicorn
import warnings
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

def transform_payload_to_features(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform incoming payload to percentage-based features for clustering.

    Expected input payload keys:
    - Best3SquatKg (or best3squatkg, etc.)
    - Best3BenchKg (or best3benchkg, etc.)
    - Best3DeadliftKg (or best3deadliftkg, etc.)

    Returns DataFrame with percentage-based features: SquatPct, BenchPct, DeadliftPct
    """
    # Create a case-insensitive mapping for the payload
    payload_lower = {k.lower(): v for k, v in payload.items()}

    # Extract the three lift values
    squat_kg = float(payload_lower.get('best3squatkg', 0.0))
    bench_kg = float(payload_lower.get('best3benchkg', 0.0))
    deadlift_kg = float(payload_lower.get('best3deadliftkg', 0.0))

    # Calculate clustering total
    clustering_total = squat_kg + bench_kg + deadlift_kg

    # Create temporary DataFrame with the lift values
    df = pd.DataFrame([{
        'Best3SquatKg': squat_kg,
        'Best3BenchKg': bench_kg,
        'Best3DeadliftKg': deadlift_kg,
        'ClustingTotal': clustering_total
    }])

    # Calculate percentage-based features
    df['SquatPct'] = df['Best3SquatKg'] / df['ClustingTotal']
    df['BenchPct'] = df['Best3BenchKg'] / df['ClustingTotal']
    df['DeadliftPct'] = df['Best3DeadliftKg'] / df['ClustingTotal']

    # Return only the percentage features as array values
    c_array = df[['SquatPct', 'BenchPct', 'DeadliftPct']].values

    # Convert back to DataFrame for consistency with predict function
    result_df = pd.DataFrame(c_array, columns=['SquatPct', 'BenchPct', 'DeadliftPct'])

    return result_df

def transform_payload_for_second_model(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform incoming payload for the second model (after clustering).

    Expected input payload keys:
    - Squat1Kg (or squat1kg, etc.)
    - BodyweightKg (or bodyweightkg, etc.)
    - Sex ('F' or 'M')
    - Cluster (0-14, output from first model)
    - long_distance_travel (boolean)

    Returns DataFrame with all required features for second model.
    """
    # Define all expected features in the order the model expects
    expected_features = [
        'Squat1Kg', 'BodyweightKg', 'Sex_F', 'Sex_M',
        'Cluster_0.0', 'Cluster_1.0', 'Cluster_2.0', 'Cluster_3.0', 'Cluster_4.0',
        'Cluster_5.0', 'Cluster_6.0', 'Cluster_7.0', 'Cluster_8.0', 'Cluster_9.0',
        'Cluster_10.0', 'Cluster_11.0', 'Cluster_12.0', 'Cluster_13.0', 'Cluster_14.0',
        'long_distance_travel'
    ]

    # Create a case-insensitive mapping for the payload
    payload_lower = {k.lower(): v for k, v in payload.items()}

    # Initialize result dictionary
    result = {}

    # Extract numeric features
    result['Squat1Kg'] = float(payload_lower.get('squat1kg', 0.0))
    result['BodyweightKg'] = float(payload_lower.get('bodyweightkg', 0.0))

    # One-hot encode Sex
    sex_value = payload_lower.get('sex', '').upper()
    result['Sex_F'] = sex_value == 'F'
    result['Sex_M'] = sex_value == 'M'

    # One-hot encode Cluster (supports both int and float formats)
    cluster_value = payload_lower.get('cluster', None)
    if cluster_value is not None:
        cluster_value = float(cluster_value)

    for i in range(15):  # Clusters 0 to 14
        result[f'Cluster_{float(i):.1f}'] = (cluster_value == float(i)) if cluster_value is not None else False

    # Boolean feature
    travel_value = payload_lower.get('long_distance_travel', False)
    if isinstance(travel_value, str):
        travel_value = travel_value.lower() in ('true', '1', 'yes')
    result['long_distance_travel'] = bool(travel_value)

    # Create DataFrame with features in the correct order
    df = pd.DataFrame([result])[expected_features]

    # Ensure proper types
    df['Squat1Kg'] = df['Squat1Kg'].astype('float64')
    df['BodyweightKg'] = df['BodyweightKg'].astype('float64')

    # Convert boolean columns to bool type
    bool_columns = [col for col in expected_features if col not in ['Squat1Kg', 'BodyweightKg']]
    for col in bool_columns:
        df[col] = df[col].astype('bool')

    return df

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

    try:
        # Transform payload to match model's feature set
        df = transform_payload_to_features(request.model_input)

        print(f"Transformed input shape: {df.shape}")
        print(f"Transformed columns: {df.columns.tolist()}")
        print(f"Transformed data types:\n{df.dtypes}")
        print(f"Transformed data:\n{df.to_dict(orient='records')}")

        # Suppress sklearn feature names warning for models trained without feature names
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X has feature names")
            result = model.predict(df)

        return {
            "prediction": result.tolist() if hasattr(result, 'tolist') else result,
            "version": version,
            "model": MODEL_NAME
        }
    except Exception as e:
        # Log detailed error information
        print(f"Prediction error details:")
        print(f"  Error: {str(e)}")
        print(f"  Input payload: {request.model_input}")

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
