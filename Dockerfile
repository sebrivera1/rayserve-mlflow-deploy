# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Copy uv binary from official image (faster than installing)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with uv (10x faster than pip)
# --system flag installs to system Python (no venv needed in container)
# --no-cache keeps image small
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY serve_backend.py .

# Railway provides PORT env var
ENV PORT=8000

# Start Ray Serve
CMD ["python", "serve_backend.py"]
