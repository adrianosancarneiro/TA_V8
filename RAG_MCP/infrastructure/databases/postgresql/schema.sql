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

CREATE INDEX idx_tenants_name ON tenants(name);
CREATE INDEX idx_tenants_created ON tenants(created_at DESC);

-- ============================================
-- DOMAINS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS domains (
    domain_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(500),
    domain_type VARCHAR(100) NOT NULL, -- Application, BusinessArea, Department, ProductLine, etc.
    domain_category VARCHAR(100),
    
    -- Hierarchical structure
    path VARCHAR(1000),
    level INTEGER DEFAULT 0,
    parent_domain_id VARCHAR(255) REFERENCES domains(domain_id),
    
    -- Configuration
    metadata_template JSONB DEFAULT '{}',
    knowledge_sources TEXT[] DEFAULT '{}',
    indexing_frequency VARCHAR(50) DEFAULT 'daily',
    retention_days INTEGER DEFAULT 365,
    
    -- Metadata
    attributes JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    
    UNIQUE(tenant_id, name)
);

CREATE INDEX idx_domains_tenant ON domains(tenant_id);
CREATE INDEX idx_domains_type ON domains(domain_type);
CREATE INDEX idx_domains_parent ON domains(parent_domain_id);
CREATE INDEX idx_domains_path ON domains USING gin(path gin_trgm_ops);

-- ============================================
-- DOCUMENTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    domain_id VARCHAR(255) REFERENCES domains(domain_id) ON DELETE SET NULL,
    
    -- Document identification
    title TEXT NOT NULL,
    original_filename VARCHAR(500),
    content_hash VARCHAR(64),
    
    -- Storage references
    minio_path TEXT,
    minio_bucket VARCHAR(255) DEFAULT 'ta-v8-documents',
    
    -- Document properties
    content_type content_type,
    file_size_bytes BIGINT,
    page_count INTEGER,
    word_count INTEGER,
    
    -- Content
    full_text TEXT, -- Store full document text for reference
    summary TEXT,   -- LLM-generated summary
    
    -- Processing info
    chunking_method chunking_method,
    total_chunks INTEGER DEFAULT 0,
    chunk_size_target INTEGER,
    chunk_overlap_tokens INTEGER,
    
    -- Document-level tags (generated during initial processing)
    document_tags JSONB DEFAULT '{}',
    main_topics TEXT[] DEFAULT '{}',
    detected_entities VARCHAR(255)[] DEFAULT '{}', -- References to knowledge entities
    
    -- Quality and freshness
    quality_score NUMERIC(3,2),
    staleness staleness_level DEFAULT 'current',
    staleness_score NUMERIC(3,2),
    
    -- Versioning
    version VARCHAR(50),
    previous_version_id VARCHAR(255) REFERENCES documents(document_id),
    
    -- Status tracking
    status document_status DEFAULT 'pending',
    processing_errors JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP WITH TIME ZONE,
    last_accessed TIMESTAMP WITH TIME ZONE,
    
    -- User tracking
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

-- Indexes for documents
CREATE INDEX idx_documents_tenant ON documents(tenant_id);
CREATE INDEX idx_documents_domain ON documents(tenant_id, domain_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_content_type ON documents(content_type);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_documents_tags ON documents USING gin(document_tags);
CREATE INDEX idx_documents_topics ON documents USING gin(main_topics);
CREATE INDEX idx_documents_fulltext ON documents USING gin(to_tsvector('english', full_text));

-- ============================================
-- CHUNKS TABLE - Core table for chunk storage
-- ============================================
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    domain_id VARCHAR(255) REFERENCES domains(domain_id) ON DELETE SET NULL,
    
    -- Chunk identification
    chunk_index INTEGER NOT NULL,
    chunk_hash VARCHAR(64),
    
    -- Text content
    chunk_text TEXT NOT NULL,                    -- Clean chunk text
    chunk_text_with_overlap TEXT NOT NULL,       -- Text including overlap regions
    overlap_previous TEXT,                       -- Overlap text from previous chunk
    overlap_next TEXT,                          -- Overlap text from next chunk
    
    -- Chunk properties
    token_count INTEGER NOT NULL,
    char_count INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    start_page INTEGER,
    end_page INTEGER,
    
    -- Processing metadata
    method chunking_method NOT NULL,
    overlap_tokens INTEGER DEFAULT 50,
    
    -- Semantic tags (generated by LLM during embedding)
    chunk_tags JSONB DEFAULT '{}',
    
    -- The chunk_tags structure includes:
    -- {
    --   "primary_topic": "string",
    --   "subtopics": ["array", "of", "strings"],
    --   "technical_concepts": ["array"],
    --   "business_concepts": ["array"],
    --   "related_features": ["array"],
    --   "related_entities": ["array of entity_ids"],
    --   "context_type": "string",
    --   "importance": "high/medium/low",
    --   "complexity": "high/medium/low",
    --   "audience": "string",
    --   "keywords": ["array"]
    -- }
    
    -- Knowledge graph references
    related_entities VARCHAR(255)[] DEFAULT '{}',  -- Neo4j KnowledgeEntity IDs
    
    -- Vector store references
    qdrant_point_id VARCHAR(255),
    qdrant_collection VARCHAR(255) DEFAULT 'ta_v8_embeddings',
    embedding_model VARCHAR(100),
    embedding_dimension INTEGER,
    embedding_version VARCHAR(50),
    
    -- Quality metrics
    quality_score NUMERIC(3,2),
    relevance_score NUMERIC(3,2),
    coherence_score NUMERIC(3,2),
    
    -- Status tracking
    embedding_status chunk_status DEFAULT 'pending',
    indexing_status chunk_status DEFAULT 'pending',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedded_at TIMESTAMP WITH TIME ZONE,
    indexed_at TIMESTAMP WITH TIME ZONE,
    last_retrieved TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(document_id, chunk_index)
);

-- Indexes for chunks
CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_tenant_domain ON chunks(tenant_id, domain_id);
CREATE INDEX idx_chunks_qdrant ON chunks(qdrant_point_id) WHERE qdrant_point_id IS NOT NULL;
CREATE INDEX idx_chunks_status ON chunks(embedding_status, indexing_status);
CREATE INDEX idx_chunks_tags ON chunks USING gin(chunk_tags);
CREATE INDEX idx_chunks_entities ON chunks USING gin(related_entities);
CREATE INDEX idx_chunks_fulltext ON chunks USING gin(to_tsvector('english', chunk_text));
CREATE INDEX idx_chunks_overlap ON chunks USING gin(to_tsvector('english', chunk_text_with_overlap));

-- ============================================
-- KNOWLEDGE_ENTITIES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS knowledge_entities (
    entity_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    domain_id VARCHAR(255) NOT NULL REFERENCES domains(domain_id) ON DELETE CASCADE,
    
    -- Entity identification
    entity_type VARCHAR(100) NOT NULL,  -- Feature, Module, Process, Policy, etc.
    entity_subtype VARCHAR(100),
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(500),
    description TEXT,
    
    -- Hierarchical structure
    path VARCHAR(1000),
    hierarchy_level INTEGER DEFAULT 0,
    parent_entity_id VARCHAR(255) REFERENCES knowledge_entities(entity_id),
    
    -- Properties (flexible JSON)
    properties JSONB DEFAULT '{}',
    
    -- Categorization
    tags TEXT[] DEFAULT '{}',
    keywords TEXT[] DEFAULT '{}',
    
    -- Staleness tracking
    staleness staleness_level DEFAULT 'current',
    staleness_score NUMERIC(3,2),
    last_validated TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255),
    
    UNIQUE(tenant_id, domain_id, name)
);

CREATE INDEX idx_knowledge_entities_domain ON knowledge_entities(tenant_id, domain_id);
CREATE INDEX idx_knowledge_entities_type ON knowledge_entities(entity_type, entity_subtype);
CREATE INDEX idx_knowledge_entities_parent ON knowledge_entities(parent_entity_id);
CREATE INDEX idx_knowledge_entities_tags ON knowledge_entities USING gin(tags);
CREATE INDEX idx_knowledge_entities_keywords ON knowledge_entities USING gin(keywords);

-- ============================================
-- EMBEDDING_QUEUE TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS embedding_queue (
    queue_id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    priority INTEGER DEFAULT 5,
    status VARCHAR(50) DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_embedding_queue_status ON embedding_queue(status, priority DESC, scheduled_at);
CREATE INDEX idx_embedding_queue_chunk ON embedding_queue(chunk_id);

-- ============================================
-- QUERIES TABLE - Track user queries
-- ============================================
CREATE TABLE IF NOT EXISTS queries (
    query_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    domain_id VARCHAR(255) REFERENCES domains(domain_id) ON DELETE SET NULL,
    
    -- Query content
    query_text TEXT NOT NULL,
    query_embedding_id VARCHAR(255),
    
    -- Context
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    team_id VARCHAR(255),
    
    -- Results
    chunks_retrieved VARCHAR(255)[] DEFAULT '{}',
    relevance_scores NUMERIC[] DEFAULT '{}',
    response_generated BOOLEAN DEFAULT false,
    response_text TEXT,
    
    -- Performance metrics
    latency_ms INTEGER,
    tokens_used INTEGER,
    embedding_time_ms INTEGER,
    retrieval_time_ms INTEGER,
    generation_time_ms INTEGER,
    
    -- Feedback
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    user_feedback TEXT,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_queries_tenant ON queries(tenant_id);
CREATE INDEX idx_queries_domain ON queries(tenant_id, domain_id);
CREATE INDEX idx_queries_user ON queries(user_id);
CREATE INDEX idx_queries_session ON queries(session_id);
CREATE INDEX idx_queries_created ON queries(created_at DESC);
CREATE INDEX idx_queries_fulltext ON queries USING gin(to_tsvector('english', query_text));

-- ============================================
-- CHUNK_RETRIEVALS TABLE - Track chunk usage
-- ============================================
CREATE TABLE IF NOT EXISTS chunk_retrievals (
    retrieval_id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    query_id VARCHAR(255) REFERENCES queries(query_id) ON DELETE CASCADE,
    
    relevance_score NUMERIC(4,3),
    rank INTEGER,
    
    retrieved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunk_retrievals_chunk ON chunk_retrievals(chunk_id);
CREATE INDEX idx_chunk_retrievals_query ON chunk_retrievals(query_id);
CREATE INDEX idx_chunk_retrievals_time ON chunk_retrievals(retrieved_at DESC);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to all tables with updated_at
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_domains_updated_at BEFORE UPDATE ON domains
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_entities_updated_at BEFORE UPDATE ON knowledge_entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically update chunk retrieval tracking
CREATE OR REPLACE FUNCTION track_chunk_retrieval()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chunks 
    SET last_retrieved = CURRENT_TIMESTAMP
    WHERE chunk_id = NEW.chunk_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER track_chunk_usage AFTER INSERT ON chunk_retrievals
    FOR EACH ROW EXECUTE FUNCTION track_chunk_retrieval();

-- Function to update document statistics after chunk changes
CREATE OR REPLACE FUNCTION update_document_chunk_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE documents 
    SET total_chunks = (
        SELECT COUNT(*) FROM chunks WHERE document_id = NEW.document_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE document_id = NEW.document_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_document_stats AFTER INSERT OR DELETE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_document_chunk_stats();

-- ============================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- ============================================

-- View for chunk search with all necessary joins
CREATE MATERIALIZED VIEW chunk_search_view AS
SELECT 
    c.chunk_id,
    c.chunk_text,
    c.chunk_text_with_overlap,
    c.chunk_tags,
    c.qdrant_point_id,
    c.quality_score,
    c.relevance_score,
    d.document_id,
    d.title as document_title,
    d.document_tags,
    d.main_topics,
    dom.domain_id,
    dom.name as domain_name,
    dom.domain_type,
    t.tenant_id,
    t.name as tenant_name
FROM rag_system.chunks c
JOIN rag_system.documents d ON c.document_id = d.document_id
LEFT JOIN rag_system.domains dom ON c.domain_id = dom.domain_id
JOIN rag_system.tenants t ON c.tenant_id = t.tenant_id
WHERE c.embedding_status = 'embedded';

CREATE INDEX idx_chunk_search_tenant ON chunk_search_view(tenant_id);
CREATE INDEX idx_chunk_search_domain ON chunk_search_view(domain_id);
CREATE INDEX idx_chunk_search_qdrant ON chunk_search_view(qdrant_point_id);

-- Refresh materialized view periodically
-- This would typically be done via a scheduled job
-- REFRESH MATERIALIZED VIEW CONCURRENTLY chunk_search_view;

-- ============================================
-- EXAMPLE QUERIES
-- ============================================

-- 1. Get all chunks for a document with their tags
/*
SELECT 
    chunk_id,
    chunk_index,
    chunk_text,
    chunk_tags,
    related_entities
FROM chunks
WHERE document_id = 'doc_001'
ORDER BY chunk_index;
*/

-- 2. Find chunks by knowledge entity references
/*
SELECT 
    c.chunk_id,
    c.chunk_text,
    d.title,
    c.relevance_score
FROM chunks c
JOIN documents d ON c.document_id = d.document_id
WHERE 'ke_savanta_pricing_001' = ANY(c.related_entities)
ORDER BY c.relevance_score DESC;
*/

-- 3. Get retrieval statistics for chunks
/*
SELECT 
    c.chunk_id,
    COUNT(cr.retrieval_id) as retrieval_count,
    AVG(cr.relevance_score) as avg_relevance,
    MAX(cr.retrieved_at) as last_retrieved
FROM chunks c
LEFT JOIN chunk_retrievals cr ON c.chunk_id = cr.chunk_id
WHERE c.tenant_id = 'tenant_questel_001'
GROUP BY c.chunk_id
ORDER BY retrieval_count DESC
LIMIT 20;
*/

-- 4. Find stale documents needing re-indexing
/*
SELECT 
    document_id,
    title,
    staleness,
    staleness_score,
    updated_at
FROM documents
WHERE staleness IN ('deprecated', 'legacy_context')
   OR staleness_score < 0.5
   OR (CURRENT_TIMESTAMP - updated_at) > INTERVAL '90 days'
ORDER BY staleness_score ASC;
*/

-- ============================================
-- EMBEDDING METADATA TABLES (used by Embedding MCP)
-- ============================================

-- Batch tracking for embedding operations
CREATE TABLE IF NOT EXISTS embedding_batches (
    batch_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    collection_name VARCHAR(255) NOT NULL,
    item_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embedding_batches_tenant ON embedding_batches(tenant_id);

-- Individual item embedding metadata
CREATE TABLE IF NOT EXISTS embeddings (
    item_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    collection_name VARCHAR(255) NOT NULL,
    batch_id VARCHAR(255) REFERENCES embedding_batches(batch_id) ON DELETE SET NULL,
    vector_dim INTEGER,
    item_text TEXT,
    status VARCHAR(50) DEFAULT 'stored',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (item_id, collection_name)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON embeddings(tenant_id);
