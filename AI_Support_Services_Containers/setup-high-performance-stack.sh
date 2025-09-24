#!/bin/bash

# setup-high-performance-stack.sh
# Complete setup script for RTX 5090 optimized Docker stack

set -e

echo "🚀 Setting up High-Performance Docker Stack for RTX 5090"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check for NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:12.3.0-base nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Installing NVIDIA Container Toolkit...${NC}"
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi

# Create directory structure
echo -e "${GREEN}Creating directory structure...${NC}"
BASE_DIR="/home/mentorius/AI_Services"
mkdir -p $BASE_DIR/{data,models,backups,logs}
mkdir -p $BASE_DIR/data/{postgres,qdrant,neo4j,minio}
mkdir -p $BASE_DIR/models/ollama
mkdir -p $BASE_DIR/{multilingual-e5-large,ollama,neo4j}

# Copy Dockerfiles to appropriate directories
echo -e "${GREEN}Setting up custom Dockerfiles...${NC}"
cp multilingual-e5-large/Dockerfile.multilingual-e5-large $BASE_DIR/multilingual-e5-large/Dockerfile.multilingual-e5-large
cp ollama/Dockerfile.ollama $BASE_DIR/ollama/Dockerfile.ollama  
cp neo4j/Dockerfile.neo4j $BASE_DIR/neo4j/Dockerfile.neo4j

# Copy master docker-compose
cp docker-compose-master.yml $BASE_DIR/docker-compose.yml

cd $BASE_DIR

# Build custom images
echo -e "${YELLOW}Building custom Docker images...${NC}"

echo "Building Multilingual E5 Large image (this will take 5-10 minutes)..."
docker build -f multilingual-e5-large/Dockerfile.multilingual-e5-large -t multilingual-e5-large-gpu:latest multilingual-e5-large/

echo "Building Ollama image..."
docker build -f ollama/Dockerfile.ollama -t ollama-gpu:latest ollama/

echo "Building Neo4j image..."
docker build -f neo4j/Dockerfile.neo4j -t neo4j-optimized:latest neo4j/

# Create .env file with configurations
echo -e "${GREEN}Creating environment configuration...${NC}"
cat > .env << EOF
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=ta_v8
POSTGRES_USER=postgres_user
POSTGRES_PASSWORD=postgres_pass
DATABASE_URL=postgresql://postgres_user:postgres_pass@localhost:5432/ta_v8

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=ta_v8
QDRANT_URL=http://localhost:6333

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=pJnssz3khcLtn6T
NEO4J_HTTP_URL=http://localhost:7474

# MinIO Configuration
MINIO_ENDPOINT=http://localhost:9000
MINIO_CONSOLE_URL=http://localhost:9001
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_DEFAULT_BUCKET=ta-v8-bucket

# Ollama/LLM Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
LLM_BASE_URL=http://localhost:11434

# Multilingual E5 Large Embedding Configuration
EMBEDDING_URL=http://localhost:8080
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EOF

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Check service health
echo -e "${GREEN}Checking service health...${NC}"
docker-compose ps

# Test GPU access
echo -e "${GREEN}Testing GPU access...${NC}"
docker exec multilingual-e5-large nvidia-smi || echo "Multilingual E5 Large GPU test"
docker exec ta_v8_ollama nvidia-smi || echo "Ollama GPU test"

# Create MinIO bucket
echo -e "${GREEN}Creating MinIO default bucket...${NC}"
docker run --rm --network host \
    minio/mc alias set myminio http://localhost:9000 minioadmin minioadmin && \
    docker run --rm --network host \
    minio/mc mb myminio/ta-v8-bucket || true

# Performance test script
cat > test-performance.py << 'EOF'
#!/usr/bin/env python3
import time
import requests
import numpy as np

def test_embedding_performance():
    """Test Multilingual E5 Large embedding performance"""
    print("Testing Multilingual E5 Large Embedding Service (FP16 on RTX 5090)...")
    
    times = []
    for i in range(100):
        start = time.perf_counter()
        response = requests.post(
            'http://localhost:8080/embeddings',
            json={'texts': ['Test text for embedding']}
        )
        if response.status_code == 200:
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        if i == 0:
            print(f"First request status: {response.status_code}")
    
    if times:
        print(f"\nPerformance Results:")
        print(f"  Average latency: {np.mean(times):.2f}ms")
        print(f"  P50 latency: {np.percentile(times, 50):.2f}ms")
        print(f"  P95 latency: {np.percentile(times, 95):.2f}ms")
        print(f"  P99 latency: {np.percentile(times, 99):.2f}ms")
        print(f"  Throughput: {1000/np.mean(times):.0f} req/s")
    else:
        print("Service not ready yet. Please wait and try again.")

def test_services():
    """Test all services are accessible"""
    services = [
        ("PostgreSQL", "http://localhost:5432", False),
        ("Qdrant", "http://localhost:6333/health", True),
        ("Neo4j", "http://localhost:7474", True),
        ("MinIO", "http://localhost:9000/minio/health/live", True),
        ("Ollama", "http://localhost:11434/api/version", True),
        ("Multilingual E5 Large", "http://localhost:8080/health", True),
    ]
    
    print("\nService Health Check:")
    for name, url, check in services:
        if check:
            try:
                response = requests.get(url, timeout=2)
                status = "✅ ONLINE" if response.status_code == 200 else f"⚠️  Status: {response.status_code}"
            except:
                status = "❌ OFFLINE"
        else:
            status = "🔍 Check manually"
        print(f"  {name:15} {status}")

if __name__ == "__main__":
    test_services()
    print("\n" + "="*50)
    test_embedding_performance()
EOF

chmod +x test-performance.py

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Services are starting up. It may take 1-2 minutes for all services to be ready."
echo ""
echo "Service URLs:"
echo "  - PostgreSQL: localhost:5432"
echo "  - Qdrant: http://localhost:6333"
echo "  - Neo4j: http://localhost:7474"
echo "  - MinIO Console: http://localhost:9001"
echo "  - Ollama: http://localhost:11434"
echo "  - Multilingual E5 Large Embeddings: http://localhost:8080"
echo ""
echo "To test performance: python3 test-performance.py"
echo "To view logs: docker-compose logs -f [service-name]"
echo "To stop all services: docker-compose down"
echo "To restart all services: docker-compose restart"
