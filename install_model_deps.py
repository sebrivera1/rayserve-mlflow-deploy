#!/usr/bin/env python3
"""
Quick fix: Install model dependencies directly
Run this if you're getting dependency mismatch errors
"""
import os
import sys
import subprocess
import mlflow

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model configurations
MODEL_1_NAME = os.getenv("MODEL_NAME", "lifter-kmeans")
MODEL_1_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_2_NAME = os.getenv("MODEL_2_NAME", "model_a_squat")
MODEL_2_VERSION = os.getenv("MODEL_2_VERSION", "1")

def install_model_dependencies(model_name, model_version):
    """Install dependencies for a specific model"""
    print(f"\n{'='*80}")
    print(f"Installing dependencies for {model_name} v{model_version}")
    print('='*80)

    try:
        model_uri = f"models:/{model_name}/{model_version}"
        deps_file = mlflow.pyfunc.get_model_dependencies(model_uri)

        if deps_file:
            print(f"Found dependencies file: {deps_file}")
            print(f"\nInstalling from {deps_file}...")

            # Install using pip
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", deps_file],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"✓ Successfully installed dependencies for {model_name}")
            else:
                print(f"✗ Error installing dependencies:")
                print(result.stderr)
                return False
        else:
            print(f"⚠ No dependencies file found for {model_name}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("MLflow Model Dependency Installer")
    print("="*80)

    success = True

    # Install Model 1 dependencies
    if not install_model_dependencies(MODEL_1_NAME, MODEL_1_VERSION):
        success = False

    # Install Model 2 dependencies
    if not install_model_dependencies(MODEL_2_NAME, MODEL_2_VERSION):
        success = False

    print("\n" + "="*80)
    if success:
        print("✓ All model dependencies installed successfully!")
    else:
        print("✗ Some dependencies failed to install. Check errors above.")
    print("="*80)

    sys.exit(0 if success else 1)
