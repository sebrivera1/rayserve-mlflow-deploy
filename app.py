import os
import gradio as gr
import requests

# Backend configuration from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# In-memory storage for full feature payloads
stored_payloads = []

# Fix Railway internal URL if needed
if BACKEND_URL and not BACKEND_URL.startswith(("http://", "https://")):
    # Add http:// scheme if missing
    BACKEND_URL = f"http://{BACKEND_URL}"

# If Railway internal URL has no port or wrong port, fix it
if "railway.internal" in BACKEND_URL:
    # Check if it has :80 and replace with :8080
    if ":80/" in BACKEND_URL or BACKEND_URL.endswith(":80"):
        BACKEND_URL = BACKEND_URL.replace(":80", ":8080")
    # Check if it has no port at all
    elif not any(f":{port}" in BACKEND_URL for port in ["8080", "443", "8000"]):
        # Add port 8080 right after railway.internal
        BACKEND_URL = BACKEND_URL.replace("railway.internal", "railway.internal:8080")

#debug print
#print(f"[INFO] Backend URL configured as: {BACKEND_URL}")

def predict(name, weight, squat, bench, deadlift, sex):
    """Send prediction request to backend"""

    # Prepare request
    url = f"{BACKEND_URL}/predict"
    headers = {}

    # Calculate total from the three lifts
    total = squat + bench + deadlift

    # Payload 1: Current backend prediction (only 3 lift features)
    payload = {
        "model_input": {
            "Best3SquatKg": squat,
            "Best3BenchKg": bench,
            "Best3DeadliftKg": deadlift
        }
    }

    # Payload 2: Full feature set for future predictor (stored in memory)
    full_payload = {
        "model_input": {
            "name": name,
            "long_distance": long_distance,
            "weight": weight,
            "squat": squat,
            "bench": bench,
            "deadlift": deadlift,
            "sex": sex,
            "squat_first_attempt": squat_first_attempt,
            "total": total
        }
    }

    # Store full payload in memory
    stored_payloads.append(full_payload)
    print(f"[INFO] Stored payload #{len(stored_payloads)} in memory for {name}")

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "No prediction returned")
            return f"Cluster Prediction for {name}:\n{prediction}"
        else:
            # Fallback if backend unavailable
            result = f"""
Cluster Prediction for {name}:
- Weight: {weight} kg
- Squat: {squat} kg
- Bench: {bench} kg
- Deadlift: {deadlift} kg
- Sex: {sex}
- Total: {total} kg

Note: Backend unavailable (status {response.status_code}), showing input summary only.
"""
            return result

    except Exception as e:
        # Fallback if backend unavailable
        result = f"""
Cluster Prediction for {name}:
- Weight: {weight} kg
- Squat: {squat} kg
- Bench: {bench} kg
- Deadlift: {deadlift} kg
- Sex: {sex}
- Total: {total} kg

Note: Cannot connect to backend ({str(e)}), showing input summary only.
"""
        return result

def check_health():
    """Check backend health status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            data = response.json()
            return f"✓ Backend healthy - Model: {data.get('model', 'unknown')}"
        else:
            return f"✗ Backend unhealthy (status {response.status_code})"
    except Exception as e:
        return f"✗ Cannot connect to backend: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Power Lifting SBD Predictor") as demo:
    gr.Markdown("# Power Lifting SBD Predictor")
    #debugging print
    # gr.Markdown(f"Backend: `{BACKEND_URL}`")

    with gr.Row():
        health_output = gr.Textbox(label="Backend Status", interactive=False)
        health_btn = gr.Button("Check Health")

    gr.Markdown("## Enter Your Personal Records")

    with gr.Row():
        with gr.Column():
            name_input = gr.Textbox(label="Name", placeholder="Enter your name")
            weight_input = gr.Slider(minimum=30, maximum=2000, value=75, label="Weight (kg)")
            sex_input = gr.Radio(choices=["M", "F"], value="M", label="Sex")
            long_distance = gr.Checkbox(label="Long Distance")

        with gr.Column():
            squat_input = gr.Slider(minimum=0, maximum=1000, value=150, label="Squat Max (kg)")
            bench_input = gr.Slider(minimum=0, maximum=700, value=100, label="Bench Press Max (kg)")
            deadlift_input = gr.Slider(minimum=0, maximum=1000, value=180, label="Deadlift Max (kg)")
            squat_first_attempt = gr.Slider(minimum=0, maximum=1000, value=150, label="Squat Attempt 1 (kg)")

    submit_btn = gr.Button("Predict Cluster", variant="primary")

    output = gr.Textbox(
        label="Prediction Result",
        lines=10,
        interactive=False
    )

    # Wire up events
    health_btn.click(fn=check_health, outputs=health_output)
    submit_btn.click(
        fn=predict,
        inputs=[name_input, weight_input, squat_input, bench_input, deadlift_input, sex_input],
        outputs=output
    )

    # Check health on load
    demo.load(fn=check_health, outputs=health_output)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
