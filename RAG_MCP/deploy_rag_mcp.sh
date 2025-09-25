#!/bin/bash
# =============================================================================
# RAG MCP DEPLOYMENT SCRIPT
# =============================================================================
# This script starts the complete TA_V8 AI services stack with RAG MCP services
# 
# Usage: ./deploy_rag_mcp.sh [start|stop|restart|status]
#
# Prerequisites:
# - Docker and Docker Compose installed
# - NVIDIA Docker runtime (for GPU services)
# - At least 32GB RAM and RTX 5090 (or compatible GPU)
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
AI_SERVICES_DIR="/home/mentorius/AI_Services/TA_V8/AI_Support_Services_Containers"
RAG_MCP_DIR="/home/mentorius/AI_Services/TA_V8/RAG_MCP"
MASTER_COMPOSE="${AI_SERVICES_DIR}/docker-compose-master.yml"
MCP_COMPOSE="${RAG_MCP_DIR}/deployment/docker-compose-mcp.yml"

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "\n${CYAN}${WHITE}$1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"
}

check_prerequisites() {
    log_header "CHECKING PREREQUISITES"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_success "Docker is installed"
    
    # Check if Docker Compose is available
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    log_success "Docker Compose is available"
    
    # Check if NVIDIA Docker runtime is available (for GPU services)
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_success "NVIDIA Docker runtime is available"
    else
        log_warning "NVIDIA Docker runtime not available - GPU services may not work"
    fi
    
    # Check if compose files exist
    if [[ ! -f "$MASTER_COMPOSE" ]]; then
        log_error "Master compose file not found: $MASTER_COMPOSE"
        exit 1
    fi
    log_success "Master compose file found"
    
    if [[ ! -f "$MCP_COMPOSE" ]]; then
        log_error "MCP compose file not found: $MCP_COMPOSE"
        exit 1
    fi
    log_success "MCP compose file found"
}

create_network() {
    log_header "SETTING UP DOCKER NETWORK"
    
    # Create the shared network if it doesn't exist
    if ! docker network ls | grep -q "ta_v8_network"; then
        log_info "Creating ta_v8_network..."
        docker network create ta_v8_network
        log_success "Network created: ta_v8_network"
    else
        log_success "Network already exists: ta_v8_network"
    fi
}

start_ai_services() {
    log_header "STARTING AI SERVICES INFRASTRUCTURE"
    
    cd "$AI_SERVICES_DIR"
    
    log_info "Starting PostgreSQL, Neo4j, Qdrant, MinIO, Ollama, and Embedding services..."
    docker compose -f docker-compose-master.yml up -d
    
    log_info "Waiting for services to be healthy..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker compose -f docker-compose-master.yml ps --status running | grep -q "healthy\|Up"; then
            log_success "AI services are starting up"
            break
        fi
        
        attempt=$((attempt + 1))
        log_info "Waiting for services... (attempt $attempt/$max_attempts)"
        sleep 10
    done
    
    log_success "AI services infrastructure started"
}

start_mcp_services() {
    log_header "STARTING RAG MCP SERVICES"
    
    cd "$RAG_MCP_DIR"
    
    log_info "Building and starting RAG MCP services..."
    docker compose -f deployment/docker-compose-mcp.yml up -d --build
    
    log_info "Waiting for MCP services to be ready..."
    sleep 30
    
    # Check health of each MCP service
    local services=("chunking-mcp" "embedding-mcp" "retrieval-mcp")
    local ports=("8001" "8002" "8003")
    
    for i in "${!services[@]}"; do
        local service="${services[$i]}"
        local port="${ports[$i]}"
        
        log_info "Checking health of $service on port $port..."
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            log_success "$service is healthy"
        else
            log_warning "$service may not be ready yet"
        fi
    done
    
    log_success "RAG MCP services started"
}

stop_services() {
    log_header "STOPPING ALL SERVICES"
    
    # Stop MCP services first
    if [[ -f "$MCP_COMPOSE" ]]; then
        cd "$RAG_MCP_DIR"
        log_info "Stopping RAG MCP services..."
        docker compose -f deployment/docker-compose-mcp.yml down
        log_success "MCP services stopped"
    fi
    
    # Stop AI services
    if [[ -f "$MASTER_COMPOSE" ]]; then
        cd "$AI_SERVICES_DIR"
        log_info "Stopping AI services infrastructure..."
        docker compose -f docker-compose-master.yml down
        log_success "AI services stopped"
    fi
}

show_status() {
    log_header "SERVICE STATUS"
    
    echo -e "\n${CYAN}AI Services Infrastructure:${NC}"
    cd "$AI_SERVICES_DIR"
    docker compose -f docker-compose-master.yml ps
    
    echo -e "\n${CYAN}RAG MCP Services:${NC}"
    cd "$RAG_MCP_DIR"
    docker compose -f deployment/docker-compose-mcp.yml ps
    
    echo -e "\n${CYAN}Service URLs:${NC}"
    echo "🔗 PostgreSQL:      localhost:5432"
    echo "🔗 Neo4j Browser:   http://localhost:7474"
    echo "🔗 Qdrant:          http://localhost:6333"
    echo "🔗 MinIO Console:   http://localhost:9001"
    echo "🔗 Ollama:          http://localhost:11434"
    echo "🔗 Embedding:       http://localhost:8080"
    echo "🔗 Chunking MCP:    http://localhost:8001"
    echo "🔗 Embedding MCP:   http://localhost:8002"  
    echo "🔗 Retrieval MCP:   http://localhost:8003"
    echo "🔗 RAG Agent Team:  http://localhost:8006"
}

run_tests() {
    log_header "RUNNING MCP INTEGRATION TESTS"
    
    cd "$RAG_MCP_DIR"
    
    # Activate virtual environment and run tests
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
        log_info "Running MCP compliance tests..."
        python testing/mcp_compliance/test_mcp_servers.py
        log_success "Tests completed"
    else
        log_warning "Virtual environment not found - install dependencies first"
    fi
}

# Main script logic
case "${1:-start}" in
    "start")
        log_header "🚀 STARTING COMPLETE TA_V8 RAG MCP STACK"
        check_prerequisites
        create_network
        start_ai_services
        start_mcp_services
        show_status
        log_success "🎉 Complete stack is running!"
        log_info "Run './deploy_rag_mcp.sh test' to validate the deployment"
        ;;
    
    "stop")
        stop_services
        log_success "🛑 All services stopped"
        ;;
    
    "restart")
        log_header "🔄 RESTARTING COMPLETE STACK"
        stop_services
        sleep 5
        start_ai_services
        start_mcp_services
        show_status
        log_success "🎉 Complete stack restarted!"
        ;;
    
    "status")
        show_status
        ;;
    
    "test")
        run_tests
        ;;
    
    *)
        echo "Usage: $0 [start|stop|restart|status|test]"
        echo ""
        echo "Commands:"
        echo "  start    - Start complete TA_V8 AI stack with RAG MCP services"
        echo "  stop     - Stop all services"  
        echo "  restart  - Restart all services"
        echo "  status   - Show status of all services"
        echo "  test     - Run MCP integration tests"
        exit 1
        ;;
esac