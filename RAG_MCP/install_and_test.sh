#!/bin/bash
# =============================================================================
# TA_V8 RAG MCP Systemd Installation and Validation Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="${SCRIPT_DIR}/systemd"
ENV_FILE="${SCRIPT_DIR}/config.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
}

log_header() {
    echo -e "\n${BLUE}$1${NC}"
    echo "$(printf '=%.0s' {1..80})"
}

# Check if running as root for installation
check_sudo() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root!"
        log_info "Run it as regular user. It will request sudo when needed."
        exit 1
    fi
}

# Validate environment
validate_environment() {
    log_header "🔍 Environment Validation"
    
    # Check Python virtual environment
    if [[ ! -d ".venv" ]]; then
        log_error "Python virtual environment not found! Please create .venv first."
        exit 1
    fi
    
    log_info "Python virtual environment found: $(realpath .venv)"
    
    # Activate virtual environment and check dependencies
    source .venv/bin/activate
    
    # Check critical packages
    local missing_packages=()
    
    if ! python -c "import mcp" 2>/dev/null; then
        missing_packages+=("mcp")
    fi
    
    if ! python -c "import fastapi" 2>/dev/null; then
        missing_packages+=("fastapi")
    fi
    
    if ! python -c "import uvicorn" 2>/dev/null; then
        missing_packages+=("uvicorn")
    fi
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_error "Missing required packages: ${missing_packages[*]}"
        log_info "Installing missing packages..."
        pip install "${missing_packages[@]}"
    fi
    
    log_success "All required Python packages available"
}

# Test MCP services functionality
test_mcp_services() {
    log_header "🧪 Testing MCP Services"
    
    source .venv/bin/activate
    export MCP_TRANSPORT=stdio
    
    # Test RAG Agent Team import
    log_info "Testing RAGAgentTeam import..."
    if python -c "
import sys
sys.path.append('.')
from rag_agent_team import RAGAgentTeam
print('✅ RAGAgentTeam imported successfully')
" 2>/dev/null; then
        log_success "RAGAgentTeam service ready"
    else
        log_error "Failed to import RAGAgentTeam"
        return 1
    fi
    
    # Test if vLLM service is available
    log_info "Testing vLLM service availability..."
    if curl -s -f "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        log_success "vLLM service is accessible at http://127.0.0.1:8000"
    else
        log_warning "vLLM service not accessible. Please ensure vllm-gpt-oss.service is running"
    fi
}

# Install systemd services
install_systemd_services() {
    log_header "⚙️  Installing Systemd Services"
    
    if [[ ! -d "$SERVICE_DIR" ]]; then
        log_error "Systemd service directory not found: $SERVICE_DIR"
        exit 1
    fi
    
    log_info "Installing service files from $SERVICE_DIR"
    
    # Copy service files
    sudo cp "$SERVICE_DIR"/*.service /etc/systemd/system/
    sudo cp "$SERVICE_DIR"/*.target /etc/systemd/system/
    
    # Reload systemd
    log_info "Reloading systemd daemon..."
    sudo systemctl daemon-reload
    
    log_success "Systemd services installed successfully"
}

# Enable and start services
start_services() {
    log_header "🚀 Starting TA_V8 RAG Services"
    
    # Enable target
    log_info "Enabling ta-v8-rag.target..."
    sudo systemctl enable ta-v8-rag.target
    
    # Start all services via target
    log_info "Starting ta-v8-rag.target..."
    sudo systemctl start ta-v8-rag.target
    
    # Wait a moment for services to start
    sleep 3
    
    log_success "Services startup initiated"
}

# Check service status
check_services() {
    log_header "📊 Service Status Check"
    
    ./manage_services.sh status
    
    # Check individual services
    local services=("chunking-mcp" "embedding-mcp" "retrieval-mcp" "rag-agent-team" "tab-mcp-client")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log_success "All services are running successfully"
    else
        log_warning "Some services failed to start: ${failed_services[*]}"
        
        for service in "${failed_services[@]}"; do
            log_info "Checking logs for $service:"
            journalctl -u "$service" --no-pager -n 10
        done
    fi
}

# Show service logs
show_logs() {
    log_header "📜 Recent Service Logs"
    
    local services=("chunking-mcp" "embedding-mcp" "retrieval-mcp" "rag-agent-team" "tab-mcp-client")
    
    for service in "${services[@]}"; do
        log_info "Last 5 lines from $service:"
        journalctl -u "$service" --no-pager -n 5 2>/dev/null || log_warning "No logs found for $service"
        echo
    done
}

# Cleanup function
cleanup() {
    log_header "🧹 Cleanup"
    
    log_info "Stopping all TA_V8 RAG services..."
    sudo systemctl stop ta-v8-rag.target 2>/dev/null || true
    
    log_info "Disabling services..."
    sudo systemctl disable ta-v8-rag.target 2>/dev/null || true
    
    local services=("chunking-mcp" "embedding-mcp" "retrieval-mcp" "rag-agent-team" "tab-mcp-client")
    for service in "${services[@]}"; do
        sudo systemctl disable "$service" 2>/dev/null || true
    done
    
    log_success "Services stopped and disabled"
}

# Show help
show_help() {
    echo "TA_V8 RAG MCP Systemd Management"
    echo "================================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install     - Install and start all systemd services"
    echo "  test        - Test MCP services without installing"
    echo "  start       - Start all services"
    echo "  status      - Check service status"
    echo "  logs        - Show recent service logs"
    echo "  cleanup     - Stop and disable all services"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install    # Full installation and startup"
    echo "  $0 test       # Test services functionality"
    echo "  $0 status     # Check if services are running"
}

# Main execution
main() {
    cd "$SCRIPT_DIR"
    
    case "${1:-install}" in
        "install")
            check_sudo
            validate_environment
            test_mcp_services
            install_systemd_services
            start_services
            check_services
            ;;
        "test")
            validate_environment
            test_mcp_services
            ;;
        "start")
            start_services
            check_services
            ;;
        "status")
            check_services
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
