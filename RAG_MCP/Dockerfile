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

# Copy required files for dependency installation
COPY pyproject.toml uv.lock* README.md ./

# Create virtual environment and install dependencies
RUN uv sync --no-dev

# Copy the rest of the application code
COPY . .

# Create empty __init__.py for module imports
RUN touch /app/__init__.py /app/shared/__init__.py

# Set environment for UV virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install uvicorn globally as fallback
RUN pip install uvicorn[standard] --break-system-packages

# Default command with fallback to system uvicorn
CMD ["bash", "-c", "cd /app && (/app/.venv/bin/uvicorn rag_agent_team:app --host 0.0.0.0 --port 8000 || uvicorn rag_agent_team:app --host 0.0.0.0 --port 8000)"]
