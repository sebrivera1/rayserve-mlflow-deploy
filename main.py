import os
import sys
import subprocess
import mlflow
import pandas as pd
import numpy as np
import uvicorn
import warnings
import traceback
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
MODEL_NAME = os.getenv("MODEL_NAME", "lifter-kmeans")
MODEL_VERSION = os.getenv("MODEL_VERSION", "2")
MODEL_2_NAME = os.getenv("MODEL_2_NAME", "model_a_squat")
MODEL_2_VERSION = os.getenv("MODEL_2_VERSION", "1")
model_cache = {}
default_signature = None

def install_model_dependencies(model_name: str, model_version: str) -> bool:
    """
    Install model-specific dependencies from MLflow.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"Checking dependencies for {model_name} v{model_version}...")

        # Get model dependencies from MLflow
        model_uri = f"models:/{model_name}/{model_version}"
        deps_file = mlflow.pyfunc.get_model_dependencies(model_uri)

        if deps_file:
            print(f"Found dependencies file: {deps_file}")

            # Read and display what's being installed
            try:
                with open(deps_file, 'r') as f:
                    deps_content = f.read()
                    print(f"Dependencies to install:\n{deps_content}")
            except Exception as e:
                print(f"Could not read dependencies file: {e}")

            print(f"Installing dependencies for {model_name}...")

            # Install using pip with verbose output
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", deps_file],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for h2o
            )

            if result.returncode == 0:
                print(f"✓ Successfully installed dependencies for {model_name}")
                if result.stdout:
                    print(f"Install output: {result.stdout[-500:]}")  # Last 500 chars

                # CRITICAL FIX: Force downgrade scipy to compatible version
                # MLflow may install scipy 1.16.3 which is incompatible with scikit-learn 1.6.1
                print(f"Forcing scipy==1.14.1 for compatibility with scikit-learn 1.6.1...")
                fix_result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "scipy==1.14.1"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if fix_result.returncode == 0:
                    print(f"✓ scipy 1.14.1 installed successfully")
                else:
                    print(f"⚠ Warning: Could not force scipy version: {fix_result.stderr}")

                return True
            else:
                print(f"⚠ Warning: Failed to install some dependencies for {model_name}")
                print(f"Error: {result.stderr}")
                print(f"Output: {result.stdout}")
                return False
        else:
            print(f"No dependencies file found for {model_name}, skipping...")
            return True

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout installing dependencies for {model_name}")
        return False
    except Exception as e:
        print(f"⚠ Warning: Could not install dependencies for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global default_signatures
    # Clear model cache on startup
    #
    # Initialize MLflow with proper URI scheme
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    # Ensure proper URI scheme (http/https)
    if not tracking_uri.startswith(("http://", "https://")):
        tracking_uri = f"http://{tracking_uri}"

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")

    # Install model dependencies before loading models
    print("\n" + "="*80)
    print("Installing Model Dependencies")
    print("="*80)

    # Install dependencies for both models
    install_model_dependencies(MODEL_NAME, MODEL_VERSION)
    install_model_dependencies(MODEL_2_NAME, MODEL_2_VERSION)

    print("="*80)
    print("Loading Models")
    print("="*80 + "\n")

    # Load default models on startup
    try:
        # Load model 1 (clustering)
        logger.info(f"Loading model 1: {MODEL_NAME} version {MODEL_VERSION}")
        model_uri_1 = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        loaded_model_1 = mlflow.pyfunc.load_model(model_uri_1)
        cache_key_1 = f"{MODEL_NAME}:{MODEL_VERSION}"
        model_cache[cache_key_1] = loaded_model_1

        # Safely get signature
        if loaded_model_1.metadata and hasattr(loaded_model_1.metadata, 'signature'):
            default_signature = loaded_model_1.metadata.signature

        print(f"✓ Loaded model 1 (clustering): {MODEL_NAME} version {MODEL_VERSION}")
        logger.info(f"✓ Model 1 loaded successfully")
    except Exception as e:
        print(f"✗ Warning: Could not load model 1 (clustering): {e}")
        logger.error(f"Failed to load model 1: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

    try:
        # Load model 2 (total predictor)
        logger.info(f"Loading model 2: {MODEL_2_NAME} version {MODEL_2_VERSION}")
        model_uri_2 = f"models:/{MODEL_2_NAME}/{MODEL_2_VERSION}"
        loaded_model_2 = mlflow.pyfunc.load_model(model_uri_2)
        cache_key_2 = f"{MODEL_2_NAME}:{MODEL_2_VERSION}"
        model_cache[cache_key_2] = loaded_model_2

        print(f"✓ Loaded model 2 (total predictor): {MODEL_2_NAME} version {MODEL_2_VERSION}")
        logger.info(f"✓ Model 2 loaded successfully")
    except Exception as e:
        print(f"✗ Warning: Could not load model 2 (total predictor): {e}")
        logger.error(f"Failed to load model 2: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

    print("\n" + "="*80)
    print("Server Ready")
    print("="*80 + "\n")

    yield

    # Cleanup on shutdown (if needed)
    model_cache.clear()

# FastAPI app with lifespan
app = FastAPI(title="MLflow Model Serving", version="1.0.0", lifespan=lifespan)

class PredictRequest(BaseModel):
    model_input: Dict[str, Any]
    version: Optional[str] = None

class FullPredictRequest(BaseModel):
    model_input: Dict[str, Any]
    version: Optional[str] = None
    model_2_version: Optional[str] = None

def load_model(version: str, model_name: str = None):
    """Load model with caching"""
    model_name = model_name or MODEL_NAME
    cache_key = f"{model_name}:{version}"

    if cache_key not in model_cache:
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)

        if model_name == MODEL_NAME and default_signature and model.metadata.signature != default_signature:
            raise ValueError(f"Model version {version} has incompatible signature")

        model_cache[cache_key] = model

    return model_cache[cache_key]

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

        # Handle different result types (DataFrame, numpy array, etc.)
        if isinstance(result, pd.DataFrame):
            if 'predict' in result.columns:
                prediction = result['predict'].tolist()
            else:
                prediction = result.iloc[:, 0].tolist()
        elif hasattr(result, 'tolist'):
            prediction = result.tolist()
        else:
            prediction = result

        return {
            "prediction": prediction,
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

@app.post("/predict_full")
async def predict_full(
    request: FullPredictRequest,
    serve_multiplexed_model_id: Optional[str] = Header(None)
):
    """
    Two-stage prediction: First model predicts cluster, then second model predicts total.

    Expected input payload:
    - name: str
    - long_distance: bool
    - weight: float (bodyweight in kg)
    - squat: float (best squat in kg)
    - bench: float (best bench in kg)
    - deadlift: float (best deadlift in kg)
    - sex: str ('M' or 'F')
    - squat_first_attempt: float (first squat attempt at meet)
    - total: float (squat + bench + deadlift)
    """
    logger.info("="*80)
    logger.info("/predict_full endpoint called")
    logger.info(f"Request payload: {request.model_input}")

    version_1 = request.version or serve_multiplexed_model_id or MODEL_VERSION
    version_2 = request.model_2_version or MODEL_2_VERSION

    logger.info(f"Model versions - clustering: {version_1}, total predictor: {version_2}")

    try:
        # Load both models
        logger.info(f"Loading models from cache...")
        logger.info(f"Cache keys available: {list(model_cache.keys())}")

        # Check if model 1 exists in cache
        cache_key_1 = f"{MODEL_NAME}:{version_1}"
        if cache_key_1 not in model_cache:
            error_msg = f"Model 1 (clustering) not loaded in cache. Available models: {list(model_cache.keys())}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Clustering model not available. Server may still be starting up or model failed to load during startup."
            )

        model_1 = load_model(version_1, MODEL_NAME)
        logger.info(f"✓ Model 1 loaded from cache")

        model_2 = load_model(version_2, MODEL_2_NAME)
        logger.info(f"✓ Model 2 loaded from cache")

    except ValueError as e:
        logger.error(f"ValueError loading models: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading models: {str(e)}"
        )

    try:
        payload_input = request.model_input

        # Stage 1: Predict cluster using model_1
        logger.info("STAGE 1: Clustering prediction")
        payload_lower = {k.lower(): v for k, v in payload_input.items()}

        # Extract values for model 1
        squat_kg = float(payload_lower.get('squat', 0.0))
        bench_kg = float(payload_lower.get('bench', 0.0))
        deadlift_kg = float(payload_lower.get('deadlift', 0.0))

        logger.info(f"Extracted lifts - squat: {squat_kg}, bench: {bench_kg}, deadlift: {deadlift_kg}")

        # Transform for model 1 (clustering)
        cluster_payload = {
            'Best3SquatKg': squat_kg,
            'Best3BenchKg': bench_kg,
            'Best3DeadliftKg': deadlift_kg
        }
        df_cluster = transform_payload_to_features(cluster_payload)

        logger.info(f"[Model 1] Transformed input shape: {df_cluster.shape}")
        logger.info(f"[Model 1] Transformed columns: {df_cluster.columns.tolist()}")
        logger.info(f"[Model 1] Transformed data:\n{df_cluster.to_dict(orient='records')}")

        # Predict cluster
        logger.info("[Model 1] Starting prediction...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X has feature names")
            cluster_result = model_1.predict(df_cluster)

        logger.info(f"[Model 1] Raw prediction result: {cluster_result}")
        logger.info(f"[Model 1] Result type: {type(cluster_result)}")

        # Extract cluster value - handle both DataFrame and array results
        if isinstance(cluster_result, pd.DataFrame):
            # For DataFrame results (like from H2O models)
            if 'predict' in cluster_result.columns:
                cluster_value = cluster_result['predict'].iloc[0]
            else:
                cluster_value = cluster_result.iloc[0, 0]
        elif isinstance(cluster_result, np.ndarray):
            # For numpy array results
            cluster_value = cluster_result[0]
        elif hasattr(cluster_result, '__getitem__') and not isinstance(cluster_result, str):
            # For list or other indexable types
            cluster_value = cluster_result[0]
        else:
            # For scalar results
            cluster_value = cluster_result

        logger.info(f"[Model 1] ✓ Cluster prediction: {cluster_value}")

        # Stage 2: Predict total using model_2 with cluster output
        logger.info("STAGE 2: Total prediction")

        # Extract squat_first_attempt, defaulting to squat if not provided or invalid
        squat_first_attempt_raw = payload_lower.get('squat_first_attempt')
        if squat_first_attempt_raw is None or squat_first_attempt_raw is False or squat_first_attempt_raw == '':
            squat_first_attempt = squat_kg  # Use best squat as default
        else:
            squat_first_attempt = float(squat_first_attempt_raw)

        # Extract long_distance, handle as boolean
        long_distance_raw = payload_lower.get('long_distance', False)
        if isinstance(long_distance_raw, bool):
            long_distance = long_distance_raw
        elif isinstance(long_distance_raw, str):
            long_distance = long_distance_raw.lower() in ('true', '1', 'yes')
        elif isinstance(long_distance_raw, (int, float)):
            # Frontend is sending slider value instead of checkbox, treat non-zero as True
            long_distance = bool(long_distance_raw)
        else:
            long_distance = False

        model_2_payload = {
            'Squat1Kg': squat_first_attempt,
            'BodyweightKg': float(payload_lower.get('weight', 0.0)),
            'Sex': payload_lower.get('sex', 'M').upper(),
            'Cluster': float(cluster_value),
            'long_distance_travel': long_distance
        }

        logger.info(f"[Model 2] Extracted squat_first_attempt: {squat_first_attempt}")
        logger.info(f"[Model 2] Extracted bodyweight: {payload_lower.get('weight', 0.0)}")
        logger.info(f"[Model 2] Extracted sex: {payload_lower.get('sex', 'M').upper()}")
        logger.info(f"[Model 2] Extracted long_distance: {long_distance}")

        df_total = transform_payload_for_second_model(model_2_payload)

        logger.info(f"[Model 2] Transformed input shape: {df_total.shape}")
        logger.info(f"[Model 2] Transformed columns: {df_total.columns.tolist()}")
        logger.info(f"[Model 2] Transformed data:\n{df_total.to_dict(orient='records')}")

        # Predict total
        logger.info("[Model 2] Starting prediction...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X has feature names")
            total_result = model_2.predict(df_total)

        logger.info(f"[Model 2] Raw prediction result: {total_result}")

        # H2O models return a DataFrame with a 'predict' column
        if isinstance(total_result, pd.DataFrame):
            if 'predict' in total_result.columns:
                total_prediction = total_result['predict'].iloc[0]
            else:
                # Fallback: get the first column's first value
                total_prediction = total_result.iloc[0, 0]
        elif hasattr(total_result, '__getitem__') and not isinstance(total_result, str):
            total_prediction = total_result[0]
        else:
            total_prediction = total_result

        logger.info(f"[Model 2] ✓ Total prediction: {total_prediction}")

        # Convert numpy types to native Python types for JSON serialization
        cluster_pred_value = int(cluster_value) if isinstance(cluster_value, (int, float, np.integer, np.floating)) else cluster_value
        total_pred_value = float(total_prediction) if isinstance(total_prediction, (int, float, np.integer, np.floating)) else total_prediction

        response = {
            "cluster_prediction": cluster_pred_value,
            "total_prediction": total_pred_value,
            "model_1_version": version_1,
            "model_2_version": version_2,
            "model_1_name": MODEL_NAME,
            "model_2_name": MODEL_2_NAME,
            "input_summary": {
                "name": payload_lower.get('name', 'Unknown'),
                "bodyweight_kg": float(payload_lower.get('weight', 0.0)),
                "sex": payload_lower.get('sex', 'M').upper(),
                "squat_kg": float(squat_kg),
                "bench_kg": float(bench_kg),
                "deadlift_kg": float(deadlift_kg),
                "squat_first_attempt": float(payload_lower.get('squat_first_attempt', squat_kg)),
                "long_distance": bool(payload_lower.get('long_distance', False))
            }
        }

        logger.info("✓ Prediction successful!")
        logger.info(f"Response: {response}")
        logger.info("="*80)

        return response

    except Exception as e:
        logger.error("="*80)
        logger.error("PREDICTION ERROR")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Input payload: {request.model_input}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        logger.error("="*80)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {type(e).__name__}: {str(e)}"
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "default_version": MODEL_VERSION,
        "model_2": MODEL_2_NAME,
        "model_2_version": MODEL_2_VERSION,
        "cached_versions": list(model_cache.keys())
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "MLflow Model Serving",
        "model": MODEL_NAME,
        "default_version": MODEL_VERSION,
        "model_2": MODEL_2_NAME,
        "model_2_version": MODEL_2_VERSION,
        "endpoints": {
            "predict": "/predict (single model - clustering)",
            "predict_full": "/predict_full (two-stage: clustering + total prediction)",
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
