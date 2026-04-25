# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860 \
    LLM_MODEL_NAME="google/gemma-2-2b-it" \
    LLM_TIMEOUT_SECONDS=90 \
    CHAT_LOG_DIR="/app/chats" \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies with retries
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=100 --prefer-binary \
        -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY app.py .
COPY src/ ./src/
COPY XENO_Uganda_KnowledgeBase_Advisory.json ./

# Create necessary directories
RUN mkdir -p /tmp/xeno_db /app/chats

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# Run the application
CMD ["python", "app.py"]
