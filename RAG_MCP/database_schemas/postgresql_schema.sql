-- ============================================================================
-- TA_V8 RAG MCP - PostgreSQL Database Schema
-- ============================================================================
-- Purpose: Create all necessary tables for document storage, chunking, 
--          embeddings, and metadata management in the RAG system
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector" WITH VERSION '0.4.4';

-- Documents table - stores original documents and metadata
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'text/plain',
    file_size BIGINT NOT NULL DEFAULT 0,
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    hash TEXT UNIQUE NOT NULL,
    metadata JSONB DEFAULT '{}',
    
    -- Indexing for performance
    CONSTRAINT documents_filename_check CHECK (length(filename) > 0),
    CONSTRAINT documents_content_check CHECK (length(content) > 0)
);

-- Document chunks table - stores processed document chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'text',
    token_count INTEGER NOT NULL DEFAULT 0,
    character_count INTEGER NOT NULL DEFAULT 0,
    start_position INTEGER NOT NULL DEFAULT 0,
    end_position INTEGER NOT NULL DEFAULT 0,
    overlap_before INTEGER DEFAULT 0,
    overlap_after INTEGER DEFAULT 0,
    created_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    
    -- Unique constraint to prevent duplicate chunks
    UNIQUE (document_id, chunk_index),
    
    -- Performance constraints
    CONSTRAINT chunks_content_check CHECK (length(content) > 0),
    CONSTRAINT chunks_token_count_check CHECK (token_count >= 0),
    CONSTRAINT chunks_positions_check CHECK (end_position > start_position)
);

-- Embeddings table - stores vector embeddings for chunks
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(1536) NOT NULL,  -- Adjust dimension as needed
    model_name TEXT NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    
    -- Unique constraint for chunk-model combination
    UNIQUE (chunk_id, model_name)
);

-- Collections table - organize documents into collections
CREATE TABLE IF NOT EXISTS collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT collections_name_check CHECK (length(name) > 0)
);

-- Document-Collection junction table
CREATE TABLE IF NOT EXISTS document_collections (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    added_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (document_id, collection_id)
);

-- Query history table - track RAG queries and results
CREATE TABLE IF NOT EXISTS query_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_embedding vector(1536),
    response_text TEXT,
    retrieved_chunks UUID[] DEFAULT '{}',
    execution_time_ms INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT query_history_query_check CHECK (length(query_text) > 0)
);

-- User sessions table - track user interactions
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT NOT NULL UNIQUE,
    user_identifier TEXT,
    start_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    query_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT sessions_session_id_check CHECK (length(session_id) > 0)
);

-- ============================================================================
-- INDEXES for Performance Optimization
-- ============================================================================

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_documents_upload_timestamp ON documents(upload_timestamp);
CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata);

-- Document chunks indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON document_chunks(chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_token_count ON document_chunks(token_count);
CREATE INDEX IF NOT EXISTS idx_chunks_created_timestamp ON document_chunks(created_timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin ON document_chunks USING GIN (metadata);

-- Embeddings indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_name ON embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_embeddings_timestamp ON embeddings(embedding_timestamp);

-- Vector similarity index (for pgvector)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_cosine 
    ON embeddings USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

-- Collections indexes
CREATE INDEX IF NOT EXISTS idx_collections_name ON collections(name);
CREATE INDEX IF NOT EXISTS idx_collections_created ON collections(created_timestamp);

-- Document-Collections indexes
CREATE INDEX IF NOT EXISTS idx_doc_collections_document ON document_collections(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_collections_collection ON document_collections(collection_id);

-- Query history indexes
CREATE INDEX IF NOT EXISTS idx_query_history_timestamp ON query_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_history_success ON query_history(success);
CREATE INDEX IF NOT EXISTS idx_query_history_execution_time ON query_history(execution_time_ms);

-- User sessions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON user_sessions(last_activity);

-- ============================================================================
-- VIEWS for Common Queries
-- ============================================================================

-- View: Document summary with chunk count
CREATE OR REPLACE VIEW document_summary AS
SELECT 
    d.id,
    d.filename,
    d.content_type,
    d.file_size,
    d.upload_timestamp,
    COUNT(dc.id) as chunk_count,
    COUNT(e.id) as embedding_count,
    d.metadata
FROM documents d
LEFT JOIN document_chunks dc ON d.id = dc.document_id
LEFT JOIN embeddings e ON dc.id = e.chunk_id
GROUP BY d.id, d.filename, d.content_type, d.file_size, d.upload_timestamp, d.metadata;

-- View: Recent query performance
CREATE OR REPLACE VIEW recent_query_performance AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as total_queries,
    AVG(execution_time_ms) as avg_execution_time_ms,
    COUNT(*) FILTER (WHERE success = true) as successful_queries,
    COUNT(*) FILTER (WHERE success = false) as failed_queries
FROM query_history 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- ============================================================================
-- FUNCTIONS for Common Operations
-- ============================================================================

-- Function: Clean up old query history (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_queries()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM query_history 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function: Get document statistics
CREATE OR REPLACE FUNCTION get_document_stats()
RETURNS TABLE (
    total_documents BIGINT,
    total_chunks BIGINT,
    total_embeddings BIGINT,
    avg_chunks_per_document NUMERIC,
    total_storage_bytes BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT d.id)::BIGINT,
        COUNT(DISTINCT dc.id)::BIGINT,
        COUNT(DISTINCT e.id)::BIGINT,
        ROUND(COUNT(DISTINCT dc.id)::NUMERIC / NULLIF(COUNT(DISTINCT d.id), 0), 2),
        SUM(d.file_size)::BIGINT
    FROM documents d
    LEFT JOIN document_chunks dc ON d.id = dc.document_id
    LEFT JOIN embeddings e ON dc.id = e.chunk_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================

-- Insert default collection
INSERT INTO collections (name, description) 
VALUES ('default', 'Default document collection') 
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- GRANTS and PERMISSIONS
-- ============================================================================

-- Grant permissions to postgres_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO postgres_user;

-- Grant select permissions for read-only access (optional)
-- CREATE ROLE readonly_user;
-- GRANT CONNECT ON DATABASE ta_v8 TO readonly_user;
-- GRANT USAGE ON SCHEMA public TO readonly_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ TA_V8 RAG MCP PostgreSQL schema initialized successfully!';
    RAISE NOTICE '📊 Tables created: documents, document_chunks, embeddings, collections, document_collections, query_history, user_sessions';
    RAISE NOTICE '🔍 Indexes created for optimal performance';
    RAISE NOTICE '📈 Views and functions available for monitoring and management';
    RAISE NOTICE '🔐 Permissions granted to postgres_user';
END $$;
