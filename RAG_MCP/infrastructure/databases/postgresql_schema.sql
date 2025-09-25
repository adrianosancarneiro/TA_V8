-- ============================================
-- POSTGRESQL SCHEMA FOR MULTI-TENANT RAG SYSTEM
-- ============================================
-- This schema handles document text, chunks, and metadata storage
-- Works in conjunction with Neo4j for graph relationships and Qdrant for vectors

-- Drop existing tables if needed (careful in production!)
-- DROP SCHEMA IF EXISTS rag_system CASCADE;
CREATE SCHEMA IF NOT EXISTS rag_system;
SET search_path TO rag_system, public;

-- ============================================
-- EXTENSIONS
-- ============================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity search
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- For composite indexes

-- ============================================
-- ENUM TYPES
-- ============================================
CREATE TYPE document_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'archived');
CREATE TYPE chunk_status AS ENUM ('pending', 'embedded', 'indexed', 'failed');
CREATE TYPE chunking_method AS ENUM ('semantic_coherence', 'hybrid', 'llm_assisted', 'fixed_size', 'auto');
CREATE TYPE content_type AS ENUM ('text', 'pdf', 'docx', 'html', 'markdown', 'json', 'csv');
CREATE TYPE staleness_level AS ENUM ('deprecated', 'legacy_context', 'current', 'in_development');

-- ============================================
-- TENANTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(500),
    industry VARCHAR(255),
    subscription_tier VARCHAR(50),
    
    -- Configuration
    api_rate_limit INTEGER DEFAULT 1000,
    storage_quota_gb INTEGER DEFAULT 100,
    max_users INTEGER DEFAULT 50,
    
    -- Settings
    default_embedding_model VARCHAR(100) DEFAULT 'multilingual-e5-large',
    default_chunking_method chunking_method DEFAULT 'auto',
    default_chunk_size INTEGER DEFAULT 500,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- DOCUMENTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    -- Document identification
    title VARCHAR(500) NOT NULL,
    source_path TEXT,
    original_filename VARCHAR(500),
    content_hash VARCHAR(64) UNIQUE,  -- SHA-256 of original content
    
    -- Content
    content_type content_type NOT NULL,
    full_text TEXT,  -- Full document text
    file_size_bytes BIGINT,
    
    -- Processing status
    status document_status DEFAULT 'pending',
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    
    -- Version management
    version INTEGER DEFAULT 1,
    is_latest BOOLEAN DEFAULT TRUE,
    parent_document_id UUID REFERENCES documents(document_id),
    staleness_level staleness_level DEFAULT 'current',
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags TEXT[],
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- CHUNKS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64),  -- SHA-256 of chunk content
    
    -- Position information
    chunk_index INTEGER NOT NULL,  -- Sequential position in document
    start_char_position INTEGER,
    end_char_position INTEGER,
    
    -- Chunking metadata
    chunking_method chunking_method NOT NULL,
    parent_chunk_id UUID REFERENCES chunks(chunk_id),  -- For hierarchical chunking
    chunk_level INTEGER DEFAULT 0,  -- Depth in hierarchy
    
    -- Processing status
    status chunk_status DEFAULT 'pending',
    embedding_model VARCHAR(100),
    embedding_generated_at TIMESTAMP WITH TIME ZONE,
    
    -- Vector storage reference
    vector_collection VARCHAR(255),
    vector_point_id BIGINT,  -- ID in Qdrant
    
    -- Context information
    surrounding_context TEXT,  -- Larger context around chunk
    semantic_topics TEXT[],
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(document_id, chunk_index)
);

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

-- Tenants
CREATE INDEX IF NOT EXISTS idx_tenants_name ON tenants(name);
CREATE INDEX IF NOT EXISTS idx_tenants_subscription ON tenants(subscription_tier);

-- Documents
CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_version ON documents(tenant_id, is_latest, version);
CREATE INDEX IF NOT EXISTS idx_documents_title_search ON documents USING gin(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_documents_full_text_search ON documents USING gin(full_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);

-- Chunks
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
CREATE INDEX IF NOT EXISTS idx_chunks_vector_ref ON chunks(vector_collection, vector_point_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_search ON chunks USING gin(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_semantic_topics ON chunks USING gin(semantic_topics);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_chunks_hierarchy ON chunks(parent_chunk_id, chunk_level);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_status_method ON chunks(tenant_id, status, chunking_method);
CREATE INDEX IF NOT EXISTS idx_documents_tenant_status_type ON documents(tenant_id, status, content_type);

-- ============================================
-- TRIGGERS FOR UPDATED_AT
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_chunks_updated_at BEFORE UPDATE ON chunks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();