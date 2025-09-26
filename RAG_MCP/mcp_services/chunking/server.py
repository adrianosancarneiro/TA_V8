#!/usr/bin/env python3
"""
# =============================================================================
# MCP SERVICE: CHUNKING 
# =============================================================================
# Purpose: Document chunking service for multi-agent RAG system
# Port: 8001
# Protocol: MCP-compliant via /mcp/execute endpoint
# 
# Integration Points:
# - TAO: Will register this service in Tool Registry as 'chunker_v1'
# - TAE: Will call via ToolCaller for agent tool usage during document processing
# - TAB: Will use for knowledge base setup workflows during team building
# 
# Dependencies: PostgreSQL, Neo4j, MinIO
# Status: MIGRATED - Ready for platform integration
# Database Integration: Stores chunks in PostgreSQL with Neo4j relationships
# Storage Integration: Reads documents from MinIO, stores processing metadata
# =============================================================================
"""

import os
import json
import uuid
import hashlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from minio import Minio
from neo4j import AsyncGraphDatabase
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chunking MCP Server", version="1.0.0")

# ============ MCP Request/Response Models ============
# Following MCP protocol standards for request/response envelope structure
# These models ensure compatibility with TAO's MCP Gateway and TAE's ToolCaller

class ChunkingPolicy(BaseModel):
    """Chunking policy configuration - defines how documents are split"""
    method: str = Field(default="recursive", description="Chunking method: recursive, semantic, or hybrid")
    target_tokens: int = Field(default=512, description="Target chunk size in tokens")
    overlap: int = Field(default=64, description="Token overlap between chunks")

class DocumentSource(BaseModel):
    """Document source specification - supports multiple input types"""
    type: str = Field(..., description="Source type: text, url, or file")
    text: Optional[str] = Field(None, description="Raw text content")
    url: Optional[str] = Field(None, description="URL to fetch")
    file_id: Optional[str] = Field(None, description="MinIO file ID")

class MCPChunkRequest(BaseModel):
    """MCP-compliant chunking request - standardized envelope for TAO/TAE integration"""
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenancy")
    domain_id: Optional[str] = Field(None, description="Domain identifier for knowledge organization")
    source: DocumentSource = Field(..., description="Document source specification")
    policy: ChunkingPolicy = Field(default_factory=ChunkingPolicy, description="Chunking policy configuration")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class ChunkRecord(BaseModel):
    """Individual chunk record - output format"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class MCPChunkResponse(BaseModel):
    """MCP-compliant chunking response - standardized envelope for TAO/TAE integration"""
    chunks: List[ChunkRecord]
    persisted: bool
    document_id: Optional[str] = None
    error: Optional[str] = None

# ============ Service Implementation ============
# Core service logic with full database and storage integration

class ChunkingService:
    """
    Core chunking service with complete storage integration
    
    Features:
    - Multi-tenant document processing
    - MinIO document storage and retrieval  
    - PostgreSQL chunk persistence
    - Neo4j relationship tracking
    - Token-based intelligent chunking
    - Configurable overlap and sizing
    """
    
    def __init__(self):
        # Database and storage connections
        self.pg_pool = None          # PostgreSQL for chunk storage
        self.neo4j_driver = None     # Neo4j for relationship tracking
        self.minio_client = None     # MinIO for document storage
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Token counting
        
    async def startup(self):
        """Initialize all service connections - called at startup"""
        logger.info("Starting Chunking MCP Service initialization...")
        
        # PostgreSQL connection pool
        # Used for storing chunk text, metadata, and processing status
        self.pg_pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres_user"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres_pass"),
            database=os.getenv("POSTGRES_DATABASE", "ta_v8"),
            min_size=1,
            max_size=10,
            server_settings={"search_path": "rag_system, public"}
        )
        
        # Neo4j connection for relationship tracking
        # Used for domain knowledge graphs and document relationships
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j_password")
        self.neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # MinIO client for document storage
        # Used for persistent document storage and retrieval
        self.minio_client = Minio(
            os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False
        )
        
        logger.info("Chunking MCP Service initialized successfully")
    
    async def shutdown(self):
        """Clean up connections - called at shutdown"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()
    
    async def chunk_document(self, request: MCPChunkRequest) -> MCPChunkResponse:
        """
        Main chunking logic with full storage integration
        
        Process:
        1. Load document from specified source (MinIO, URL, or direct text)
        2. Apply chunking policy (recursive, semantic, or hybrid)
        3. Generate chunk IDs and metadata
        4. Store chunks in PostgreSQL 
        5. Update Neo4j relationships
        6. Return MCP-compliant response
        """
        try:
            # Step 1: Load document content
            if request.source.type == "file" and request.source.file_id:
                # Load from MinIO storage
                text = await self._load_from_minio(request.source.file_id, request.tenant_id)
            elif request.source.type == "text" and request.source.text:
                # Direct text input
                text = request.source.text
            elif request.source.type == "url" and request.source.url:
                # URL fetching (implement as needed)
                text = await self._load_from_url(request.source.url)
            else:
                raise ValueError("Invalid document source specification")
            
            # Step 2: Apply chunking strategy
            chunks = await self._apply_chunking_policy(text, request.policy)
            
            # Step 3: Generate document and chunk IDs
            document_id = f"doc_{request.tenant_id}_{int(datetime.now().timestamp())}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
            
            chunk_records = []
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i:04d}"
                chunk_records.append(ChunkRecord(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "tenant_id": request.tenant_id,
                        "domain_id": request.domain_id,
                        "chunking_method": request.policy.method,
                        "token_count": len(self.tokenizer.encode(chunk_text)),
                        "created_at": datetime.now().isoformat()
                    }
                ))
            
            # Step 4: Store in PostgreSQL
            await self._store_chunks_postgresql(chunk_records, document_id, request)
            
            # Step 5: Update Neo4j relationships
            await self._update_neo4j_relationships(document_id, chunk_records, request)
            
            return MCPChunkResponse(
                chunks=chunk_records,
                persisted=True,
                document_id=document_id,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            return MCPChunkResponse(
                chunks=[],
                persisted=False,
                document_id=None,
                error=str(e)
            )
    
    async def _load_from_minio(self, file_id: str, tenant_id: str) -> str:
        """Load document content from MinIO storage"""
        try:
            # MinIO bucket structure: tenant_id/documents/file_id
            bucket_name = f"tenant-{tenant_id}"
            object_name = f"documents/{file_id}"
            
            response = self.minio_client.get_object(bucket_name, object_name)
            return response.read().decode('utf-8')
            
        except Exception as e:
            raise Exception(f"Failed to load document from MinIO: {str(e)}")
    
    async def _load_from_url(self, url: str) -> str:
        """Load document content from URL (implement as needed)"""
        # TODO: Implement URL fetching with proper error handling
        raise NotImplementedError("URL loading not yet implemented")
    
    async def _apply_chunking_policy(self, text: str, policy: ChunkingPolicy) -> List[str]:
        """
        Apply chunking strategy based on policy configuration
        
        Supports:
        - recursive: Split by paragraphs, then sentences, then tokens
        - semantic: Use sentence boundaries and semantic coherence
        - hybrid: Combine recursive and semantic approaches
        """
        if policy.method == "recursive":
            return await self._recursive_chunking(text, policy.target_tokens, policy.overlap)
        elif policy.method == "semantic":
            return await self._semantic_chunking(text, policy.target_tokens, policy.overlap)
        elif policy.method == "hybrid":
            return await self._hybrid_chunking(text, policy.target_tokens, policy.overlap)
        else:
            # Default to recursive
            return await self._recursive_chunking(text, policy.target_tokens, policy.overlap)
    
    async def _recursive_chunking(self, text: str, target_tokens: int, overlap: int) -> List[str]:
        """Recursive chunking: paragraphs -> sentences -> tokens"""
        # Simple implementation for MVP
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            # Take target number of words (approximate token counting)
            chunk_words = words[i:i + target_tokens]
            chunks.append(" ".join(chunk_words))
            
            # Move forward with overlap
            i += max(1, target_tokens - overlap)
        
        return chunks
    
    async def _semantic_chunking(self, text: str, target_tokens: int, overlap: int) -> List[str]:
        """Semantic chunking using sentence boundaries"""
        # Simple sentence-based chunking for MVP
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                # Start new chunk
                chunks.append('. '.join(current_chunk) + '.')
                
                # Handle overlap
                overlap_sentences = max(1, overlap // 50)  # Approximate
                current_chunk = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []
                current_tokens = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    async def _hybrid_chunking(self, text: str, target_tokens: int, overlap: int) -> List[str]:
        """Hybrid chunking combining recursive and semantic approaches"""
        # For MVP, use semantic chunking as the hybrid approach
        return await self._semantic_chunking(text, target_tokens, overlap)
    
    async def _store_chunks_postgresql(self, chunk_records: List[ChunkRecord], document_id: str, request: MCPChunkRequest):
        """Store chunk records in PostgreSQL database"""
        async with self.pg_pool.acquire() as conn:
            # Store document record using comprehensive schema
            await conn.execute("""
                INSERT INTO rag_system.documents 
                (document_id, tenant_id, domain_id, title, content_type, full_text,
                 chunking_method, status, document_tags, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'completed', $8, $9, $9)
                ON CONFLICT (document_id) DO UPDATE SET 
                    status = 'completed',
                    full_text = EXCLUDED.full_text,
                    document_tags = EXCLUDED.document_tags,
                    updated_at = $9
            """, document_id, request.tenant_id, request.domain_id,
                f"Document {document_id[:8]}...", request.source.type,
                request.source.text, request.policy.method, 
                json.dumps(request.metadata), datetime.now())
            
            # Store chunk records using comprehensive schema
            for chunk in chunk_records:
                await conn.execute("""
                    INSERT INTO rag_system.chunks 
                    (chunk_id, document_id, tenant_id, domain_id, chunk_index,
                     chunk_text, chunk_text_with_overlap, token_count, char_count,
                     method, chunk_tags, embedding_status, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'pending', $12)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE SET 
                        chunk_text = EXCLUDED.chunk_text,
                        chunk_text_with_overlap = EXCLUDED.chunk_text_with_overlap,
                        token_count = EXCLUDED.token_count,
                        char_count = EXCLUDED.char_count,
                        chunk_tags = EXCLUDED.chunk_tags,
                        embedding_status = 'pending'
                """, chunk.chunk_id, document_id, request.tenant_id, request.domain_id,
                    chunk.metadata["chunk_index"], chunk.text, chunk.text,
                    chunk.metadata["token_count"], len(chunk.text),
                    request.policy.method, json.dumps(chunk.metadata), datetime.now())
    
    async def _update_neo4j_relationships(self, document_id: str, chunk_records: List[ChunkRecord], request: MCPChunkRequest):
        """Update Neo4j graph relationships for document and chunks"""
        async with self.neo4j_driver.session() as session:
            # Create document node
            await session.run("""
                MERGE (d:Document {id: $document_id})
                SET d.tenant_id = $tenant_id,
                    d.domain_id = $domain_id,
                    d.chunk_count = $chunk_count,
                    d.created_at = datetime($created_at)
                
                // Link to domain if specified
                WITH d
                MATCH (domain:Domain {id: $domain_id})
                MERGE (domain)-[:CONTAINS_DOCUMENT]->(d)
            """, document_id=document_id, tenant_id=request.tenant_id,
                domain_id=request.domain_id, chunk_count=len(chunk_records),
                created_at=datetime.now().isoformat())
            
            # Create chunk nodes and relationships
            for chunk in chunk_records:
                await session.run("""
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.document_id = $document_id,
                        c.chunk_index = $chunk_index,
                        c.token_count = $token_count,
                        c.created_at = datetime($created_at)
                    
                    // Link to document
                    WITH c
                    MATCH (d:Document {id: $document_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, chunk_id=chunk.chunk_id, document_id=document_id,
                    chunk_index=chunk.metadata["chunk_index"],
                    token_count=chunk.metadata["token_count"],
                    created_at=chunk.metadata["created_at"])

# ============ Service Setup and API Endpoints ============

# Global service instance
chunking_service = ChunkingService()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup - called by FastAPI"""
    await chunking_service.startup()

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean up service on shutdown - called by FastAPI"""
    await chunking_service.shutdown()

# ============ MCP-Compliant API Endpoints ============
# Following MCP protocol standards for TAO/TAE integration

@app.post("/mcp/execute")
async def mcp_execute_endpoint(request: MCPChunkRequest) -> MCPChunkResponse:
    """
    Main MCP-compliant execution endpoint
    
    This endpoint follows the Model Context Protocol standard and integrates with:
    - TAO: Registered as 'chunker_v1' tool in ToolRegistry
    - TAE: Called via ToolCaller when agents need document chunking
    - TAB: Used during domain knowledge setup workflows
    
    Request format follows MCP envelope structure for consistency
    Response format provides structured output for downstream processing
    """
    logger.info(f"MCP chunking request for tenant: {request.tenant_id}, domain: {request.domain_id}")
    
    try:
        response = await chunking_service.chunk_document(request)
        
        if response.error:
            logger.error(f"Chunking failed: {response.error}")
        else:
            logger.info(f"Chunking completed: {len(response.chunks)} chunks generated, document_id: {response.document_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"MCP execution error: {str(e)}")
        return MCPChunkResponse(
            chunks=[],
            persisted=False,
            document_id=None,
            error=f"Service error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and service discovery
    
    Used by:
    - Docker health checks
    - Load balancers
    - TAO service registry for availability monitoring
    - Deployment orchestration
    """
    try:
        # Test database connections
        if chunking_service.pg_pool:
            async with chunking_service.pg_pool.acquire() as conn:
                await conn.execute("SELECT 1")
        
        # Test Neo4j connection
        if chunking_service.neo4j_driver:
            async with chunking_service.neo4j_driver.session() as session:
                await session.run("RETURN 1")
        
        # Test MinIO connection
        if chunking_service.minio_client:
            chunking_service.minio_client.list_buckets()
        
        return {
            "status": "healthy",
            "service": "chunking-mcp",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "postgresql": "connected",
                "neo4j": "connected", 
                "minio": "connected"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "chunking-mcp",
            "version": "1.0.0", 
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# ============ Main Application Entry Point ============

if __name__ == "__main__":
    import uvicorn
    
    # Production-ready ASGI server configuration
    uvicorn.run(
        "server:app",  # Module:app reference
        host="0.0.0.0",  # Bind to all interfaces for container deployment
        port=8001,       # Standard port for chunking MCP service
        reload=False,    # Disable reload in production
        workers=1,       # Single worker for MVP, scale as needed
        log_level="info"
    )