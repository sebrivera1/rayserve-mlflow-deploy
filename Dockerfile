# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Copy uv from official image (no installation needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install with uv 
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application
COPY main.py .

ENV PORT=8000

CMD ["python", "main.py"]
