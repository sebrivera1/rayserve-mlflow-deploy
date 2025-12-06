# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install Java 21 (required for H2O, supports Java 8-21)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

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
