#!/bin/bash
# TA_V8/RAG_MCP/deploy.sh
# Quick deployment script for RAG MCP with UV

set -e

echo "=========================================="
echo "RAG MCP Quick Deployment for TA_V8 (UV)"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "unified_mcp_server.py" ]; then
    echo "Error: Please run this script from the TA_V8/RAG_MCP directory"
    exit 1
fi

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Error: UV is not installed. Please install UV first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Initialize UV project if .venv doesn't exist
if [ ! -d ".venv" ]; then
    echo "1. Initializing UV virtual environment..."
    uv venv
    echo "   ✓ Virtual environment created"
else
    echo "1. Using existing UV virtual environment..."
fi

# Install dependencies
echo "2. Installing dependencies with UV..."
uv sync

# Create __init__ files for Python modules
echo "3. Setting up Python modules..."
touch __init__.py
mkdir -p shared && touch shared/__init__.py

# Check if ta_v8_default network exists
echo "4. Checking Docker network..."
if ! docker network ls | grep -q "ta_v8_default"; then
    echo "   Creating ta_v8_default network..."
    docker network create ta_v8_default
else
    echo "   ✓ Network ta_v8_default exists"
fi

# Build and start services
echo "5. Building Docker images..."
docker-compose build

echo "6. Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "7. Waiting for services to initialize..."
sleep 10

# Check health endpoints
echo "8. Checking service health..."
echo -n "   MCP Server: "
curl -s http://localhost:8005/health | grep -q "ok" && echo "✓ OK" || echo "✗ Failed"

echo -n "   Agent Team: "
curl -s http://localhost:8006/health | grep -q "ok" && echo "✓ OK" || echo "✗ Failed"

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Services running at:"
echo "  - MCP Server: http://localhost:8005"
echo "  - Agent Team: http://localhost:8006"
echo ""
echo "To test the pipeline (with UV):"
echo "  uv run python test_rag_pipeline.py"
echo ""
echo "To run services locally (development):"
echo "  uv run uvicorn unified_mcp_server:app --host 0.0.0.0 --port 8005 --reload"
echo "  uv run uvicorn rag_agent_team:app --host 0.0.0.0 --port 8006 --reload"
echo ""
echo "To view Docker logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop Docker services:"
echo "  docker-compose down"
echo ""
