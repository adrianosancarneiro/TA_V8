#!/bin/bash
# TA_V8/RAG_MCP/deploy.sh
# Quick deployment script for RAG MCP with systemd services

set -e

echo "=========================================="
echo "RAG MCP Systemd Deployment for TA_V8"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "rag_agent_team.py" ]; then
    echo "Error: Please run this script from the TA_V8/RAG_MCP directory"
    exit 1
fi

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed. Please install Python 3.12+ first."
    exit 1
fi

# Check if running as root or with sudo access for systemd
if [[ $EUID -ne 0 ]]; then
    echo "Note: Some operations may require sudo access for systemd service installation"
fi

# Initialize virtual environment if .venv doesn't exist
if [ ! -d ".venv" ]; then
    echo "1. Creating virtual environment..."
    python -m venv .venv
    echo "   ✓ Virtual environment created"
else
    echo "1. Using existing virtual environment..."
fi

# Install dependencies
echo "2. Installing dependencies with pip..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Create __init__ files for Python modules
echo "3. Setting up Python modules..."
touch __init__.py
mkdir -p shared && touch shared/__init__.py

echo "4. Installing systemd services..."

# Check if systemd directory exists
if [ ! -d "systemd" ]; then
    echo "Error: systemd directory not found. Please ensure systemd service files exist."
    exit 1
fi

# Function to install systemd service
install_systemd_service() {
    local service_file="$1"
    local service_name=$(basename "$service_file")
    
    echo "   Installing $service_name..."
    
    # Copy service file to systemd directory
    sudo cp "systemd/$service_file" "/etc/systemd/system/"
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable the service
    sudo systemctl enable "$service_name"
    
    echo "   ✓ $service_name installed and enabled"
}

# Install all systemd services
for service_file in systemd/*.service systemd/*.target; do
    if [ -f "$service_file" ]; then
        install_systemd_service $(basename "$service_file")
    fi
done

echo "5. Starting TA_V8 RAG MCP services..."

# Start the target which will start all services
sudo systemctl start ta-v8-rag.target

echo "6. Checking service status..."
sleep 3

# Check status of all services
services=("chunking-mcp" "embedding-mcp" "retrieval-mcp" "rag-agent-team" "tab-mcp-client")

for service in "${services[@]}"; do
    if systemctl is-active --quiet "${service}.service"; then
        echo "   ✓ $service: RUNNING"
    else
        echo "   ✗ $service: FAILED"
        echo "     Check logs: sudo journalctl -u ${service}.service -f"
    fi
done

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
echo "=========================================="
echo "✅ TA_V8 RAG MCP Deployment Complete!"
echo "=========================================="

echo ""
echo "Services are managed by systemd:"
echo "  - chunking-mcp.service"
echo "  - embedding-mcp.service" 
echo "  - retrieval-mcp.service"
echo "  - rag-agent-team.service"
echo "  - tab-mcp-client.service"
echo ""
echo "Service URLs (if web UI enabled):"
echo "  - TAB MCP Client: http://localhost:8005"
echo ""
echo "Useful commands:"
echo "  Start all services: sudo systemctl start ta-v8-rag.target"
echo "  Stop all services:  sudo systemctl stop ta-v8-rag.target"
echo "  Check status:       sudo systemctl status ta-v8-rag.target"
echo "  View logs:          sudo journalctl -u <service-name> -f"
echo ""
echo "To test individual services:"
echo "  .venv/bin/python rag_agent_team.py --help"
echo "  .venv/bin/python mcp_services/chunking/server.py --help"
echo ""
