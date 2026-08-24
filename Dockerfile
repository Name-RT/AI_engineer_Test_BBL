# ==============================================================================
# Dockerfile for Bangkok Bank Agentic RAG Policy Assistant
# Lightweight Python 3.11 Slim image with pre-configured ChromaDB & Gradio
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system utilities & C++ compiler for native dependencies (chromadb/hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Ensure storage directory for ChromaDB exists
RUN mkdir -p /app/chroma_db

# Expose ports: 7861 (Gradio), 8501 (Streamlit)
EXPOSE 7861 8501

# Healthcheck for Gradio server
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7861/ || exit 1

# Default entrypoint: Run Gradio Dark Edition
CMD ["python", "gradio_app.py"]
