#!/bin/bash
# TA_V8/RAG_MCP/manage_services.sh
# Systemd service management script for TA_V8 RAG MCP

set -e

SERVICES=("chunking-mcp" "embedding-mcp" "retrieval-mcp" "rag-agent-team" "tab-mcp-client")
TARGET="ta-v8-rag.target"

show_help() {
    echo "TA_V8 RAG MCP Service Manager"
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start all TA_V8 RAG services"
    echo "  stop      Stop all TA_V8 RAG services"
    echo "  restart   Restart all TA_V8 RAG services"
    echo "  status    Show status of all services"
    echo "  logs      Show logs for all services"
    echo "  install   Install systemd service files"
    echo "  uninstall Remove systemd service files"
    echo "  help      Show this help message"
    echo ""
    echo "Individual service commands:"
    echo "  start-<service>   Start individual service"
    echo "  stop-<service>    Stop individual service"
    echo "  logs-<service>    Show logs for individual service"
    echo ""
    echo "Available services: ${SERVICES[@]}"
}

check_sudo() {
    if [[ $EUID -ne 0 && "$1" != "status" && "$1" != "logs"* ]]; then
        echo "Error: This command requires sudo privileges"
        echo "Usage: sudo $0 $1"
        exit 1
    fi
}

start_services() {
    echo "🚀 Starting TA_V8 RAG MCP services..."
    systemctl start $TARGET
    echo "✅ Services started"
    show_status
}

stop_services() {
    echo "🛑 Stopping TA_V8 RAG MCP services..."
    systemctl stop $TARGET
    echo "✅ Services stopped"
}

restart_services() {
    echo "🔄 Restarting TA_V8 RAG MCP services..."
    systemctl restart $TARGET
    echo "✅ Services restarted"
    show_status
}

show_status() {
    echo "📊 TA_V8 RAG MCP Service Status:"
    echo "================================"
    
    # Check target status
    if systemctl is-active --quiet $TARGET; then
        echo "🎯 Target ($TARGET): ✅ ACTIVE"
    else
        echo "🎯 Target ($TARGET): ❌ INACTIVE"
    fi
    
    echo ""
    
    # Check individual services
    for service in "${SERVICES[@]}"; do
        service_name="${service}.service"
        if systemctl is-active --quiet $service_name; then
            uptime=$(systemctl show -p ActiveEnterTimestamp $service_name | cut -d= -f2)
            echo "🔹 $service: ✅ RUNNING (since $uptime)"
        else
            echo "🔹 $service: ❌ STOPPED"
        fi
    done
}

show_logs() {
    if [ -n "$2" ]; then
        # Individual service logs
        service_name="${2}.service"
        echo "📋 Showing logs for $2..."
        journalctl -u $service_name -f --no-pager
    else
        # All service logs
        echo "📋 Showing logs for all TA_V8 RAG services..."
        journalctl -u chunking-mcp.service -u embedding-mcp.service -u retrieval-mcp.service -u rag-agent-team.service -u tab-mcp-client.service -f --no-pager
    fi
}

install_services() {
    echo "📦 Installing TA_V8 RAG MCP systemd services..."
    
    if [ ! -d "systemd" ]; then
        echo "❌ Error: systemd directory not found"
        exit 1
    fi
    
    # Copy service files
    cp systemd/*.service systemd/*.target /etc/systemd/system/
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable services
    for service in "${SERVICES[@]}"; do
        systemctl enable "${service}.service"
        echo "✅ Enabled ${service}.service"
    done
    
    systemctl enable $TARGET
    echo "✅ Enabled $TARGET"
    
    echo "✅ Installation complete"
}

uninstall_services() {
    echo "🗑️ Uninstalling TA_V8 RAG MCP systemd services..."
    
    # Stop services first
    systemctl stop $TARGET 2>/dev/null || true
    
    # Disable and remove services
    for service in "${SERVICES[@]}"; do
        systemctl disable "${service}.service" 2>/dev/null || true
        rm -f "/etc/systemd/system/${service}.service"
        echo "✅ Removed ${service}.service"
    done
    
    systemctl disable $TARGET 2>/dev/null || true
    rm -f "/etc/systemd/system/$TARGET"
    echo "✅ Removed $TARGET"
    
    # Reload systemd
    systemctl daemon-reload
    
    echo "✅ Uninstallation complete"
}

# Handle individual service commands
handle_individual_service() {
    local action=$(echo "$1" | cut -d'-' -f1)
    local service=$(echo "$1" | cut -d'-' -f2)
    
    if [[ " ${SERVICES[@]} " =~ " ${service} " ]]; then
        case $action in
            "start")
                echo "🚀 Starting ${service}..."
                systemctl start "${service}.service"
                echo "✅ ${service} started"
                ;;
            "stop")
                echo "🛑 Stopping ${service}..."
                systemctl stop "${service}.service"
                echo "✅ ${service} stopped"
                ;;
            "logs")
                show_logs "$1" "$service"
                ;;
            *)
                echo "❌ Unknown action: $action"
                exit 1
                ;;
        esac
    else
        echo "❌ Unknown service: $service"
        echo "Available services: ${SERVICES[@]}"
        exit 1
    fi
}

# Main command handling
case "${1:-help}" in
    "start")
        check_sudo $1
        start_services
        ;;
    "stop")
        check_sudo $1
        stop_services
        ;;
    "restart")
        check_sudo $1
        restart_services
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "install")
        check_sudo $1
        install_services
        ;;
    "uninstall")
        check_sudo $1
        uninstall_services
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    start-*|stop-*|logs-*)
        if [[ "$1" == logs-* ]]; then
            handle_individual_service $1
        else
            check_sudo $1
            handle_individual_service $1
        fi
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_help
        exit 1
        ;;
esac
