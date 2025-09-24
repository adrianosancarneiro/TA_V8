#!/bin/bash

# health-monitor.sh
# Health monitoring script for AI Services Stack

set -e

# Configuration
COMPOSE_FILE="/home/mentorius/AI_Services/AI_Services_Containers/docker-compose-master.yml"
LOG_FILE="/home/mentorius/AI_Services/AI_Services_Containers/logs/health-monitor.log"
SERVICES=("postgres" "qdrant" "neo4j" "minio" "ollama")
MAX_RETRIES=3
RETRY_DELAY=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Check if container is running
is_container_running() {
    local service=$1
    docker-compose -f "$COMPOSE_FILE" ps -q "$service" | xargs docker inspect -f '{{.State.Running}}' 2>/dev/null | grep -q true
}

# Check if container is healthy
is_container_healthy() {
    local service=$1
    local health_status=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service" | xargs docker inspect -f '{{.State.Health.Status}}' 2>/dev/null)
    
    # If no health check defined, consider running as healthy
    if [[ "$health_status" == "" ]] || [[ "$health_status" == "<no value>" ]]; then
        is_container_running "$service"
        return $?
    fi
    
    [[ "$health_status" == "healthy" ]]
}

# Restart unhealthy service
restart_service() {
    local service=$1
    log "WARN" "Restarting unhealthy service: $service"
    
    docker-compose -f "$COMPOSE_FILE" restart "$service"
    
    # Wait for service to start
    sleep $RETRY_DELAY
    
    if is_container_healthy "$service"; then
        log "INFO" "Service $service successfully restarted and healthy"
        return 0
    else
        log "ERROR" "Service $service failed to become healthy after restart"
        return 1
    fi
}

# Check service status
check_service() {
    local service=$1
    local retry_count=0
    
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        if is_container_running "$service"; then
            if is_container_healthy "$service"; then
                echo -e "${GREEN}✓${NC} $service: healthy"
                log "INFO" "Service $service is healthy"
                return 0
            else
                echo -e "${YELLOW}⚠${NC} $service: unhealthy"
                log "WARN" "Service $service is unhealthy (attempt $((retry_count + 1))/$MAX_RETRIES)"
                
                if [[ $retry_count -eq $((MAX_RETRIES - 1)) ]]; then
                    restart_service "$service"
                    return $?
                fi
            fi
        else
            echo -e "${RED}✗${NC} $service: not running"
            log "ERROR" "Service $service is not running (attempt $((retry_count + 1))/$MAX_RETRIES)"
            
            if [[ $retry_count -eq $((MAX_RETRIES - 1)) ]]; then
                restart_service "$service"
                return $?
            fi
        fi
        
        ((retry_count++))
        sleep $RETRY_DELAY
    done
    
    return 1
}

# Main health check
main() {
    log "INFO" "Starting health check for AI Services Stack"
    echo "🏥 AI Services Health Monitor"
    echo "=============================="
    
    local failed_services=()
    local all_healthy=true
    
    # Change to the compose directory
    cd "$(dirname "$COMPOSE_FILE")"
    
    for service in "${SERVICES[@]}"; do
        if ! check_service "$service"; then
            failed_services+=("$service")
            all_healthy=false
        fi
    done
    
    echo "=============================="
    
    if $all_healthy; then
        echo -e "${GREEN}🎉 All services are healthy!${NC}"
        log "INFO" "All services are healthy"
        exit 0
    else
        echo -e "${RED}💥 Failed services: ${failed_services[*]}${NC}"
        log "ERROR" "Health check failed for services: ${failed_services[*]}"
        
        # Send notification (optional)
        if command -v notify-send &> /dev/null; then
            notify-send "AI Services Alert" "Some services are unhealthy: ${failed_services[*]}" -u critical
        fi
        
        exit 1
    fi
}

# Run main function
main "$@"
