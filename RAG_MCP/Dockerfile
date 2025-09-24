# TA_V8/RAG_MCP/Dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Install Python and essentials
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install UV
RUN pip install uv --break-system-packages

# Set RTX 5090 environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# Copy application code first
COPY . .

# Create empty __init__.py for module imports
RUN touch /app/__init__.py /app/shared/__init__.py

# Install dependencies with UV
RUN uv sync --no-dev

# Set UV environment
ENV UV_PYTHON_INSTALL_DIR=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install bash for activation script
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*

# Default command (can be overridden in docker-compose)
CMD ["bash", "-c", "source /app/.venv/bin/activate && uvicorn unified_mcp_server:app --host 0.0.0.0 --port 8000"]
