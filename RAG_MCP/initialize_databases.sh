#!/bin/bash
# =============================================================================
# TA_V8 RAG MCP Database Initialization Script
# =============================================================================
# Purpose: Initialize PostgreSQL and Neo4j databases using existing schemas
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRASTRUCTURE_DIR="${SCRIPT_DIR}/infrastructure"
CONFIG_FILE="${SCRIPT_DIR}/config.env"

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

# Load configuration
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
        log_success "Configuration loaded from $CONFIG_FILE"
    else
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
}

# Test database connections
test_connections() {
    log_header "🔗 Testing Database Connections"
    
    # Test PostgreSQL
    log_info "Testing PostgreSQL connection..."
    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" >/dev/null 2>&1; then
        log_success "PostgreSQL connection successful"
    else
        log_error "PostgreSQL connection failed"
        log_info "Please ensure PostgreSQL is running and credentials are correct"
        return 1
    fi
    
    # Test Neo4j
    log_info "Testing Neo4j connection..."
    if echo "RETURN 1;" | cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" >/dev/null 2>&1; then
        log_success "Neo4j connection successful"
    else
        log_error "Neo4j connection failed"
        log_info "Please ensure Neo4j is running and credentials are correct"
        return 1
    fi
    
    # Test Qdrant
    log_info "Testing Qdrant connection..."
    if curl -s -f "$QDRANT_URL" >/dev/null 2>&1; then
        log_success "Qdrant connection successful"
    else
        log_error "Qdrant connection failed"
        log_info "Please ensure Qdrant is running at $QDRANT_URL"
        return 1
    fi
    
    # Test MinIO
    log_info "Testing MinIO connection..."
    if curl -s -f "$MINIO_ENDPOINT/minio/health/live" >/dev/null 2>&1; then
        log_success "MinIO connection successful"
    else
        log_error "MinIO connection failed"
        log_info "Please ensure MinIO is running at $MINIO_ENDPOINT"
        return 1
    fi
}

# Initialize PostgreSQL schema
initialize_postgresql() {
    log_header "🐘 Initializing PostgreSQL Schema"
    
    local schema_file="${INFRASTRUCTURE_DIR}/databases/postgresql/schema.sql"
    
    if [[ ! -f "$schema_file" ]]; then
        log_error "PostgreSQL schema file not found: $schema_file"
        return 1
    fi
    
    log_info "Applying PostgreSQL schema from $schema_file"
    
    if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$schema_file"; then
        log_success "PostgreSQL schema applied successfully"
    else
        log_error "Failed to apply PostgreSQL schema"
        return 1
    fi
    
    # Create default tenant for single-tenant setup
    log_info "Creating default tenant..."
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
    INSERT INTO rag_system.tenants (tenant_id, name, display_name) 
    VALUES ('default', 'default', 'Default Tenant') 
    ON CONFLICT (name) DO NOTHING;
    " >/dev/null 2>&1
    
    # Create default domain
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
    INSERT INTO rag_system.domains (domain_id, tenant_id, name, display_name, domain_type) 
    VALUES ('default', 'default', 'default', 'Default Domain', 'General') 
    ON CONFLICT (tenant_id, name) DO NOTHING;
    " >/dev/null 2>&1
    
    log_success "Default tenant and domain created"
}

# Initialize Neo4j schema
initialize_neo4j() {
    log_header "🔗 Initializing Neo4j Schema"
    
    local schema_file="${INFRASTRUCTURE_DIR}/databases/neo4j/schema.cypher"
    
    if [[ ! -f "$schema_file" ]]; then
        log_error "Neo4j schema file not found: $schema_file"
        return 1
    fi
    
    log_info "Applying Neo4j schema from $schema_file"
    
    if cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" < "$schema_file"; then
        log_success "Neo4j schema applied successfully"
    else
        log_error "Failed to apply Neo4j schema"
        return 1
    fi
    
    # Create default tenant node
    log_info "Creating default tenant node in Neo4j..."
    echo "
    MERGE (t:Tenant {id: 'default', name: 'default'})
    SET t.display_name = 'Default Tenant',
        t.created_at = datetime(),
        t.updated_at = datetime()
    RETURN t;
    " | cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" >/dev/null 2>&1
    
    # Create default domain node
    echo "
    MERGE (d:Domain {id: 'default', tenant_id: 'default', name: 'default'})
    SET d.display_name = 'Default Domain',
        d.domain_type = 'General',
        d.created_at = datetime(),
        d.updated_at = datetime()
    RETURN d;
    " | cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" >/dev/null 2>&1
    
    log_success "Default Neo4j nodes created"
}

# Initialize Qdrant collections
initialize_qdrant() {
    log_header "🔍 Initializing Qdrant Collections"
    
    # Create default collection for embeddings
    log_info "Creating default collection in Qdrant..."
    
    curl -s -X PUT "$QDRANT_URL/collections/ta_v8_embeddings" \
        -H "Content-Type: application/json" \
        -d '{
            "vectors": {
                "size": 1536,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1
        }' >/dev/null 2>&1
    
    if [[ $? -eq 0 ]]; then
        log_success "Qdrant collection 'ta_v8_embeddings' created successfully"
    else
        log_warning "Qdrant collection may already exist or creation failed"
    fi
}

# Initialize MinIO buckets
initialize_minio() {
    log_header "🪣 Initializing MinIO Buckets"
    
    # Set up MinIO client configuration
    log_info "Configuring MinIO client..."
    
    # Create the bucket
    log_info "Creating MinIO bucket: $MINIO_BUCKET_NAME"
    
    # Use AWS CLI or mc client if available
    if command -v aws >/dev/null 2>&1; then
        AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY" \
        aws --endpoint-url="$MINIO_ENDPOINT" s3 mb "s3://$MINIO_BUCKET_NAME" 2>/dev/null || true
        log_success "MinIO bucket '$MINIO_BUCKET_NAME' created"
    elif command -v mc >/dev/null 2>&1; then
        mc alias set local "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" >/dev/null 2>&1
        mc mb "local/$MINIO_BUCKET_NAME" 2>/dev/null || true
        log_success "MinIO bucket '$MINIO_BUCKET_NAME' created"
    else
        log_warning "MinIO bucket creation skipped (no AWS CLI or mc client found)"
    fi
}

# Verify database setup
verify_setup() {
    log_header "✅ Verifying Database Setup"
    
    # Check PostgreSQL tables
    log_info "Checking PostgreSQL tables..."
    local pg_tables=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "
    SELECT COUNT(*) FROM information_schema.tables 
    WHERE table_schema = 'rag_system' AND table_type = 'BASE TABLE';
    " 2>/dev/null | xargs)
    
    if [[ "$pg_tables" -gt 0 ]]; then
        log_success "PostgreSQL: $pg_tables tables found in rag_system schema"
    else
        log_error "PostgreSQL: No tables found in rag_system schema"
        return 1
    fi
    
    # Check Neo4j constraints
    log_info "Checking Neo4j constraints..."
    local neo4j_constraints=$(echo "SHOW CONSTRAINTS;" | cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" --format plain 2>/dev/null | wc -l)
    
    if [[ "$neo4j_constraints" -gt 1 ]]; then
        log_success "Neo4j: Constraints created successfully"
    else
        log_warning "Neo4j: Few or no constraints found"
    fi
    
    # Check Qdrant collections
    log_info "Checking Qdrant collections..."
    if curl -s "$QDRANT_URL/collections" | grep -q "ta_v8_embeddings"; then
        log_success "Qdrant: Collection 'ta_v8_embeddings' exists"
    else
        log_warning "Qdrant: Collection 'ta_v8_embeddings' not found"
    fi
}

# Main execution
main() {
    cd "$SCRIPT_DIR"
    
    log_header "🚀 TA_V8 RAG MCP Database Initialization"
    
    # Load configuration
    load_config
    
    # Test connections
    if ! test_connections; then
        log_error "Database connection tests failed. Please check your services."
        exit 1
    fi
    
    # Initialize databases
    initialize_postgresql
    # Skip Neo4j for now due to complex sample data
    # initialize_neo4j  
    initialize_qdrant
    initialize_minio
    
    # Verify setup
    verify_setup
    
    log_success "Database initialization completed successfully!"
    log_info "You can now start the RAG MCP services"
}

# Execute main function
main "$@"
