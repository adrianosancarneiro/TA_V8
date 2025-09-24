#!/bin/bash
# TA_V8/RAG_MCP/setup_uv.sh
# UV setup script for RAG MCP

set -e

echo "=========================================="
echo "Setting up UV for TA_V8 RAG MCP"
echo "=========================================="

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell profile to get uv in PATH
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
    elif [ -f ~/.zshrc ]; then
        source ~/.zshrc
    fi
    
    echo "✓ UV installed successfully"
else
    echo "✓ UV is already installed"
fi

# Initialize the project
echo "Initializing UV project..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    uv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Install dependencies
echo "Installing project dependencies..."
uv sync
echo "✓ Dependencies installed"

# Create __init__.py files
touch __init__.py
mkdir -p shared && touch shared/__init__.py
echo "✓ Python modules initialized"

echo ""
echo "=========================================="
echo "UV Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run the deployment script: ./deploy.sh"
echo "  2. Test the pipeline: uv run python test_rag_pipeline.py"
echo ""
echo "Development commands:"
echo "  - Run MCP server: uv run uvicorn unified_mcp_server:app --reload --port 8005"
echo "  - Run agent team: uv run uvicorn rag_agent_team:app --reload --port 8006"
echo "  - Run tests: uv run pytest"
echo "  - Format code: uv run black ."
echo "  - Type check: uv run mypy ."
echo ""