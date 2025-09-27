#!/bin/bash

# TAB_MCP_Client Startup Script - Systemd Version
# ================================================
# This script sets up and starts the TAB_MCP_Client system with systemd services

set -e

echo "================================================"
echo "TAB_MCP_Client - Systemd Startup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_status "Docker and Docker Compose are installed"

# Navigate to RAG_MCP directory
cd "$(dirname "$0")/.."
RAGMCP_DIR=$(pwd)
print_status "Working directory: $RAGMCP_DIR"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p TAB_MCP_Client/data/uploads
mkdir -p TAB_MCP_Client/data/logs

# Check if .env file exists, create if not
if [ ! -f TAB_MCP_Client/.env ]; then
    print_warning ".env file not found. Creating from defaults..."
    cat > TAB_MCP_Client/.env << EOF
# TAB_MCP_Client Environment Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ta_v8_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

NEO4J_URI=neo4j://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

QDRANT_HOST=qdrant
QDRANT_PORT=6333

MCP_CHUNKING_URL=http://mcp-chunking:8001
MCP_EMBEDDING_URL=http://mcp-embedding:8002
MCP_RETRIEVAL_URL=http://mcp-retrieval:8003

RAG_AGENT_TEAM_URL=http://rag-agent-team:8004
EOF
    print_status ".env file created"
fi

# Function to wait for a service to be ready
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Waiting for $service to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo " Ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo " Timeout!"
    return 1
}

# Start services based on command argument
case "${1:-all}" in
    all)
        print_status "Starting all services..."
        docker-compose -f docker-compose-tab-client.yml up -d
        
        # Wait for critical services
        print_status "Waiting for services to be ready..."
        wait_for_service "PostgreSQL" 5432
        wait_for_service "Neo4j" 7687
        wait_for_service "MinIO" 9000
        wait_for_service "Qdrant" 6333
        wait_for_service "TAB_MCP_Client" 8005
        
        print_status "All services started successfully!"
        echo ""
        echo "================================================"
        echo "Service URLs:"
        echo "------------------------------------------------"
        echo "TAB_MCP_Client UI: http://localhost:8005"
        echo "Neo4j Browser:     http://localhost:7474"
        echo "MinIO Console:     http://localhost:9001"
        echo "Qdrant Dashboard:  http://localhost:6333/dashboard"
        echo "================================================"
        ;;
        
    client)
        print_status "Starting TAB_MCP_Client only..."
        docker-compose -f docker-compose-tab-client.yml up -d tab-mcp-client
        wait_for_service "TAB_MCP_Client" 8005
        print_status "TAB_MCP_Client started at http://localhost:8005"
        ;;
        
    databases)
        print_status "Starting database services..."
        docker-compose -f docker-compose-tab-client.yml up -d postgres neo4j minio qdrant
        wait_for_service "PostgreSQL" 5432
        wait_for_service "Neo4j" 7687
        wait_for_service "MinIO" 9000
        wait_for_service "Qdrant" 6333
        print_status "Database services started!"
        ;;
        
    mcp)
        print_status "Starting MCP services..."
        docker-compose -f docker-compose-tab-client.yml up -d mcp-chunking mcp-embedding mcp-retrieval
        print_status "MCP services started!"
        ;;
        
    stop)
        print_status "Stopping all services..."
        docker-compose -f docker-compose-tab-client.yml down
        print_status "All services stopped!"
        ;;
        
    restart)
        print_status "Restarting all services..."
        docker-compose -f docker-compose-tab-client.yml restart
        print_status "All services restarted!"
        ;;
        
    logs)
        docker-compose -f docker-compose-tab-client.yml logs -f ${2:-tab-mcp-client}
        ;;
        
    status)
        print_status "Service Status:"
        docker-compose -f docker-compose-tab-client.yml ps
        ;;
        
    init-db)
        print_status "Initializing databases..."
        
        # Apply PostgreSQL schema
        docker exec -i ta-v8-postgres psql -U postgres -d ta_v8_rag < infrastructure/databases/postgresql/schema.sql
        print_status "PostgreSQL schema applied"
        
        # Apply Neo4j schema
        docker exec -i ta-v8-neo4j cypher-shell -u neo4j -p password < infrastructure/databases/neo4j/schema.cypher
        print_status "Neo4j schema applied"
        
        print_status "Database initialization complete!"
        ;;
        
    test)
        print_status "Running system tests..."
        
        # Test database connections
        docker exec tab-mcp-client python -c "
import asyncio
import asyncpg
from neo4j import GraphDatabase

async def test_postgres():
    conn = await asyncpg.connect('postgresql://postgres:postgres@postgres/ta_v8_rag')
    result = await conn.fetchval('SELECT 1')
    await conn.close()
    return result == 1

def test_neo4j():
    driver = GraphDatabase.driver('neo4j://neo4j:7687', auth=('neo4j', 'password'))
    with driver.session() as session:
        result = session.run('RETURN 1 as num')
        return result.single()['num'] == 1

async def main():
    pg_ok = await test_postgres()
    neo4j_ok = test_neo4j()
    print(f'PostgreSQL: {'✓' if pg_ok else '✗'}')
    print(f'Neo4j: {'✓' if neo4j_ok else '✗'}')
    return pg_ok and neo4j_ok

if asyncio.run(main()):
    print('All database tests passed!')
else:
    print('Some tests failed!')
    exit(1)
"
        
        # Test API endpoint
        curl -s http://localhost:8005/health | grep -q "healthy" && \
            print_status "API health check passed!" || \
            print_error "API health check failed!"
        ;;
        
    clean)
        print_warning "This will remove all containers and volumes. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            docker-compose -f docker-compose-tab-client.yml down -v
            print_status "All containers and volumes removed!"
        else
            print_status "Cleanup cancelled"
        fi
        ;;
        
    help|*)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  all        - Start all services (default)"
        echo "  client     - Start only TAB_MCP_Client"
        echo "  databases  - Start only database services"
        echo "  mcp        - Start only MCP services"
        echo "  stop       - Stop all services"
        echo "  restart    - Restart all services"
        echo "  logs [svc] - Show logs (optionally for specific service)"
        echo "  status     - Show service status"
        echo "  init-db    - Initialize database schemas"
        echo "  test       - Run system tests"
        echo "  clean      - Remove all containers and volumes"
        echo "  help       - Show this help message"
        ;;
esac

# Check for common issues
if [ "${1:-all}" = "all" ]; then
    echo ""
    print_warning "Troubleshooting Tips:"
    echo "  - If services fail to start, check port availability"
    echo "  - Ensure Docker daemon is running"
    echo "  - Check logs with: $0 logs [service-name]"
    echo "  - Initialize databases with: $0 init-db"
fi
