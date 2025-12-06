#!/usr/bin/env python3
"""
Fetch MLflow model dependencies and generate requirements.txt
Run this script to get the exact dependencies needed by your models
"""
import os
import mlflow

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model configurations
MODEL_1_NAME = os.getenv("MODEL_NAME", "lifter-kmeans")
MODEL_1_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_2_NAME = os.getenv("MODEL_2_NAME", "model_a_squat")
MODEL_2_VERSION = os.getenv("MODEL_2_VERSION", "1")

print("=" * 80)
print("Fetching Model Dependencies from MLflow")
print("=" * 80)

# Fetch dependencies for Model 1 (clustering)
print(f"\n[Model 1] {MODEL_1_NAME} v{MODEL_1_VERSION}")
print("-" * 80)
try:
    model_uri_1 = f"models:/{MODEL_1_NAME}/{MODEL_1_VERSION}"
    deps_1 = mlflow.pyfunc.get_model_dependencies(model_uri_1)
    print(f"Dependencies file: {deps_1}")

    if deps_1:
        print("\nModel 1 Requirements:")
        with open(deps_1, 'r') as f:
            model_1_reqs = f.read()
            print(model_1_reqs)
except Exception as e:
    print(f"Error fetching Model 1 dependencies: {e}")
    model_1_reqs = None

# Fetch dependencies for Model 2 (total predictor)
print(f"\n[Model 2] {MODEL_2_NAME} v{MODEL_2_VERSION}")
print("-" * 80)
try:
    model_uri_2 = f"models:/{MODEL_2_NAME}/{MODEL_2_VERSION}"
    deps_2 = mlflow.pyfunc.get_model_dependencies(model_uri_2)
    print(f"Dependencies file: {deps_2}")

    if deps_2:
        print("\nModel 2 Requirements:")
        with open(deps_2, 'r') as f:
            model_2_reqs = f.read()
            print(model_2_reqs)
except Exception as e:
    print(f"Error fetching Model 2 dependencies: {e}")
    model_2_reqs = None

# Merge and deduplicate requirements
print("\n" + "=" * 80)
print("Generating Combined Requirements")
print("=" * 80)

all_requirements = set()

# Add FastAPI/serving dependencies
serving_deps = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "requests",
    "python-multipart"
]

for dep in serving_deps:
    all_requirements.add(dep)

# Parse model requirements
def parse_requirements(req_text):
    if not req_text:
        return []
    lines = req_text.strip().split('\n')
    reqs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            reqs.append(line)
    return reqs

if model_1_reqs:
    all_requirements.update(parse_requirements(model_1_reqs))

if model_2_reqs:
    all_requirements.update(parse_requirements(model_2_reqs))

# Write combined requirements
output_file = "requirements_combined.txt"
with open(output_file, 'w') as f:
    for req in sorted(all_requirements):
        f.write(f"{req}\n")

print(f"\nCombined requirements written to: {output_file}")
print("\nContents:")
print("-" * 80)
with open(output_file, 'r') as f:
    print(f.read())

print("\n" + "=" * 80)
print("Next Steps:")
print("=" * 80)
print(f"1. Review {output_file}")
print(f"2. Replace your current requirements.txt with the combined version")
print(f"3. Run: pip install -r requirements.txt")
print("=" * 80)
