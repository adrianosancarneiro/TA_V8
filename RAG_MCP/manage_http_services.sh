#!/bin/bash

# =============================================================================
# HTTP + SSE MCP Services Management Script
# =============================================================================
# Purpose: Start, stop, and manage all MCP services in HTTP + SSE transport mode
# Author: TA_V8 Migration Team
# Date: December 2024
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
export MCP_TRANSPORT="http"
export VLLM_URL="http://localhost:8000"
export CHUNKING_MCP_URL="http://localhost:8001"
export EMBEDDING_MCP_URL="http://localhost:8004"
export RETRIEVAL_MCP_URL="http://localhost:8003"
export RAG_AGENT_TEAM_URL="http://localhost:8006"

# Database environment variables
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DATABASE="ta_v8"
export POSTGRES_USER="postgres_user"
export POSTGRES_PASSWORD="postgres_pass"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="pJnssz3khcLtn6T"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export MINIO_ENDPOINT="localhost:9000"

# Service definitions with HTTP + SSE transport
declare -A SERVICES=(
    ["chunking-mcp"]="mcp_services/chunking/server.py --transport http --host 0.0.0.0 --port 8001"
    ["embedding-mcp"]="mcp_services/embedding/server.py --transport http --host 0.0.0.0 --port 8004"
    ["retrieval-mcp"]="mcp_services/retrieval/server.py --transport http --host 0.0.0.0 --port 8003"
    ["rag-agent-team"]="rag_agent_team.py"
)

# PID file directory
PID_DIR="$SCRIPT_DIR/pids"
mkdir -p "$PID_DIR"

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if virtual environment exists and is activated
check_venv() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        if [[ -f ".venv/bin/activate" ]]; then
            log "Activating virtual environment..."
            source .venv/bin/activate
        else
            error "Virtual environment not found. Please create .venv first."
            exit 1
        fi
    fi
}

# Start a service
start_service() {
    local service_name="$1"
    local service_cmd="${SERVICES[$service_name]}"
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        warning "$service_name is already running (PID: $(cat "$pid_file"))"
        return 0
    fi
    
    log "Starting $service_name..."
    
    # Set environment variables for HTTP + SSE transport
    export MCP_TRANSPORT="http"
    
    # Start service in background
    if [[ "$service_name" == "rag-agent-team" ]]; then
        # RAG Agent Team runs on port 8006 by default
        nohup python "$service_cmd" > "logs/${service_name}.log" 2>&1 &
    else
        # MCP services with specific transport and port settings
        nohup python $service_cmd > "logs/${service_name}.log" 2>&1 &
    fi
    
    local pid=$!
    echo "$pid" > "$pid_file"
    
    # Wait a moment and check if service started successfully
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        success "$service_name started (PID: $pid)"
    else
        error "$service_name failed to start"
        rm -f "$pid_file"
        return 1
    fi
}

# Stop a service
stop_service() {
    local service_name="$1"
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [[ ! -f "$pid_file" ]]; then
        warning "$service_name is not running (no PID file)"
        return 0
    fi
    
    local pid
    pid=$(cat "$pid_file")
    
    if ! kill -0 "$pid" 2>/dev/null; then
        warning "$service_name is not running (stale PID file)"
        rm -f "$pid_file"
        return 0
    fi
    
    log "Stopping $service_name (PID: $pid)..."
    
    # Try graceful shutdown first
    kill -TERM "$pid" 2>/dev/null || true
    
    # Wait up to 10 seconds for graceful shutdown
    local count=0
    while kill -0 "$pid" 2>/dev/null && [[ $count -lt 10 ]]; do
        sleep 1
        ((count++))
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        warning "Forcing shutdown of $service_name..."
        kill -KILL "$pid" 2>/dev/null || true
    fi
    
    rm -f "$pid_file"
    success "$service_name stopped"
}

# Get service status
service_status() {
    local service_name="$1"
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [[ ! -f "$pid_file" ]]; then
        echo "stopped"
        return 1
    fi
    
    local pid
    pid=$(cat "$pid_file")
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "running (PID: $pid)"
        return 0
    else
        echo "stopped (stale PID)"
        rm -f "$pid_file"
        return 1
    fi
}

# Show status of all services
status_all() {
    log "Service Status Report:"
    echo
    
    for service_name in "${!SERVICES[@]}"; do
        local status
        status=$(service_status "$service_name")
        local exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            echo -e "  ${GREEN}●${NC} $service_name: $status"
        else
            echo -e "  ${RED}●${NC} $service_name: $status"
        fi
    done
    echo
}

# Start all services
start_all() {
    log "Starting all MCP services in HTTP + SSE transport mode..."
    echo
    
    # Create logs directory
    mkdir -p logs
    
    # Start services in order (dependencies first)
    local service_order=("embedding-mcp" "retrieval-mcp" "chunking-mcp" "rag-agent-team")
    
    for service_name in "${service_order[@]}"; do
        start_service "$service_name"
        sleep 1  # Brief pause between service starts
    done
    
    echo
    success "All services startup completed"
    
    # Show final status
    sleep 2
    status_all
}

# Stop all services
stop_all() {
    log "Stopping all MCP services..."
    echo
    
    # Stop in reverse order
    local service_order=("rag-agent-team" "chunking-mcp" "retrieval-mcp" "embedding-mcp")
    
    for service_name in "${service_order[@]}"; do
        stop_service "$service_name"
    done
    
    echo
    success "All services stopped"
}

# Restart all services
restart_all() {
    log "Restarting all MCP services..."
    stop_all
    sleep 2
    start_all
}

# Test service endpoints
test_services() {
    log "Testing service endpoints..."
    echo
    
    # Check if test script exists
    if [[ -f "test_http_mcp_services.py" ]]; then
        python test_http_mcp_services.py
    else
        # Basic curl tests
        local endpoints=(
            "http://localhost:8001/health:Chunking MCP"
            "http://localhost:8004/health:Embedding MCP"
            "http://localhost:8003/health:Retrieval MCP"
            "http://localhost:8006/health:RAG Agent Team"
        )
        
        for endpoint_info in "${endpoints[@]}"; do
            local endpoint="${endpoint_info%%:*}"
            local name="${endpoint_info##*:}"
            
            if curl -s --max-time 5 "$endpoint" > /dev/null; then
                echo -e "  ${GREEN}✓${NC} $name: healthy"
            else
                echo -e "  ${RED}✗${NC} $name: unreachable"
            fi
        done
    fi
    echo
}

# Show service logs
show_logs() {
    local service_name="${1:-all}"
    
    if [[ "$service_name" == "all" ]]; then
        log "Showing logs for all services (last 20 lines each):"
        echo
        
        for service in "${!SERVICES[@]}"; do
            echo -e "${BLUE}=== $service ===${NC}"
            if [[ -f "logs/${service}.log" ]]; then
                tail -20 "logs/${service}.log"
            else
                echo "No log file found"
            fi
            echo
        done
    else
        if [[ -f "logs/${service_name}.log" ]]; then
            tail -f "logs/${service_name}.log"
        else
            error "Log file not found for $service_name"
            exit 1
        fi
    fi
}

# Help function
show_help() {
    echo "HTTP + SSE MCP Services Management Script"
    echo
    echo "Usage: $0 {start|stop|restart|status|test|logs} [service_name]"
    echo
    echo "Commands:"
    echo "  start [service]   Start service(s)"
    echo "  stop [service]    Stop service(s)" 
    echo "  restart [service] Restart service(s)"
    echo "  status           Show status of all services"
    echo "  test             Test service endpoints"
    echo "  logs [service]   Show logs (default: all services)"
    echo "  help             Show this help message"
    echo
    echo "Available services:"
    for service in "${!SERVICES[@]}"; do
        echo "  - $service"
    done
    echo
    echo "Environment:"
    echo "  MCP_TRANSPORT=http"
    echo "  Services use HTTP + SSE transport mode"
    echo "  Ports: chunking(8001), embedding(8004), retrieval(8003), rag-agent-team(8006)"
}

# Main script logic
main() {
    local command="${1:-}"
    local service_name="${2:-}"
    
    # Check virtual environment
    check_venv
    
    case "$command" in
        "start")
            if [[ -n "$service_name" ]]; then
                if [[ -n "${SERVICES[$service_name]:-}" ]]; then
                    start_service "$service_name"
                else
                    error "Unknown service: $service_name"
                    exit 1
                fi
            else
                start_all
            fi
            ;;
        "stop")
            if [[ -n "$service_name" ]]; then
                if [[ -n "${SERVICES[$service_name]:-}" ]]; then
                    stop_service "$service_name"
                else
                    error "Unknown service: $service_name"
                    exit 1
                fi
            else
                stop_all
            fi
            ;;
        "restart")
            if [[ -n "$service_name" ]]; then
                if [[ -n "${SERVICES[$service_name]:-}" ]]; then
                    stop_service "$service_name"
                    sleep 1
                    start_service "$service_name"
                else
                    error "Unknown service: $service_name"
                    exit 1
                fi
            else
                restart_all
            fi
            ;;
        "status")
            status_all
            ;;
        "test")
            test_services
            ;;
        "logs")
            show_logs "$service_name"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        "")
            error "No command specified"
            show_help
            exit 1
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
