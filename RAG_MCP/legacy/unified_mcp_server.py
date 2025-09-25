#!/usr/bin/env python3
"""
Unified MCP Server for TA_V8 RAG System with Complete Storage Integration

This server provides comprehensive document processing, embedding, and retrieval
capabilities with full MinIO, PostgreSQL, and Qdrant integration.

Architecture Flow:
1. Documents are uploaded to MinIO for persistent storage
2. Documents are chunked using intelligent strategies (LLM-guided)
3. Chunk text is stored in PostgreSQL for retrieval
4. Chunk embeddings are stored in Qdrant with chunk_id references
5. Retrieval searches Qdrant for similar vectors then fetches text from PostgreSQL

Key Features:
- MinIO object storage for document persistence
- PostgreSQL for chunk text and metadata storage
- Qdrant for high-performance vector similarity search
- Intelligent chunking with LLM-based strategy selection
- Multi-tenant data isolation
- Comprehensive health monitoring

API Endpoints:
- POST /upload: Upload document to MinIO and process
- POST /chunk: Process and chunk documents with auto-strategy selection
- POST /embed: Generate embeddings and store in Qdrant + PostgreSQL
- POST /retrieve: Semantic search with PostgreSQL text retrieval
- GET /health: Comprehensive health check

Author: TA_V8 Team
Version: 2.0
Created: 2025-09-24
"""

import asyncio
import json
import logging
import os
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from io import BytesIO

# Third-party imports
import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    CollectionStatus
)
from minio import Minio
from minio.error import S3Error

# Advanced processing imports
import tiktoken
try:
    import spacy
    from sentence_transformers import SentenceTransformer
    import ruptures as rpt
    import numpy as np
    import ollama
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Import our custom chunking module
from document_chunker import AdvancedChunker, ChunkingStrategy, DocumentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ Configuration ============
class Config:
    """Configuration for all services"""
    # PostgreSQL
    POSTGRES_HOST = "postgres"
    POSTGRES_PORT = 5432
    POSTGRES_USER = "postgres_user"
    POSTGRES_PASSWORD = "postgres_pass"
    POSTGRES_DATABASE = "ta_v8"
    
    # Qdrant
    QDRANT_HOST = "qdrant"
    QDRANT_PORT = 6333
    
    # MinIO
    MINIO_ENDPOINT = "minio:9000"
    MINIO_ACCESS_KEY = "minioadmin"
    MINIO_SECRET_KEY = "minioadmin"
    MINIO_BUCKET = "ta-v8-documents"
    
    # Embedding Service
    EMBEDDING_URL = "http://multilingual-e5-large:8080"
    
    # Ollama LLM
    OLLAMA_URL = "http://ta_v8_ollama:11434"
    OLLAMA_MODEL = "gpt-oss:20b"

config = Config()


# ============ Pydantic Models ============

class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    document_id: str
    minio_path: str
    file_size: int
    upload_timestamp: str
    message: str


class ChunkRequest(BaseModel):
    """Request for document chunking"""
    document_id: Optional[str] = Field(None, description="Document ID (will be generated if not provided)")
    text: str = Field(..., description="Document text to chunk")
    method: str = Field(
        default="auto",
        description="Chunking strategy: 'auto', 'semantic_coherence', 'hybrid', 'llm_assisted'"
    )
    target_chunk_tokens: int = Field(default=500, description="Target tokens per chunk")
    max_chunk_tokens: int = Field(default=1500, description="Maximum tokens per chunk")
    chunk_overlap_tokens: int = Field(default=50, description="Overlap between chunks")
    tenant_id: str = Field(..., description="Tenant identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbedRequest(BaseModel):
    """Request for embedding generation"""
    chunk_ids: List[str] = Field(..., description="List of chunk IDs to embed")
    tenant_id: str = Field(..., description="Tenant identifier")
    collection_name: str = Field(default="ta_v8_embeddings", description="Qdrant collection")


class RetrieveRequest(BaseModel):
    """Request for semantic retrieval"""
    query: str = Field(..., description="Search query")
    tenant_id: str = Field(..., description="Tenant identifier")
    collection_name: str = Field(default="ta_v8_embeddings", description="Qdrant collection")
    top_k: int = Field(default=5, description="Number of results to return")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")


# ============ Main Server Class ============

class UnifiedMCPServer:
    """Unified MCP Server with complete storage integration"""
    
    def __init__(self):
        """Initialize the server with all components"""
        self.app = FastAPI(
            title="TA_V8 Unified MCP Server",
            description="Enterprise RAG system with MinIO, PostgreSQL, and Qdrant",
            version="2.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Connection pools (initialized during startup)
        self.pg_pool = None
        self.qdrant_client = None
        self.minio_client = None
        self.embedding_client = None
        self.ollama_client = None
        
        # Processing components
        self.tokenizer = None
        self.sentence_model = None
        self.spacy_nlp = None
        self.chunker = None
        
        # Register event handlers
        self.app.add_event_handler("startup", self.startup)
        self.app.add_event_handler("shutdown", self.shutdown)
        
        # Register routes
        self._register_routes()
    
    async def startup(self):
        """Initialize all connections and resources"""
        logger.info("🚀 Starting TA_V8 Unified MCP Server...")
        
        try:
            # PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                host=config.POSTGRES_HOST,
                port=config.POSTGRES_PORT,
                user=config.POSTGRES_USER,
                password=config.POSTGRES_PASSWORD,
                database=config.POSTGRES_DATABASE,
                min_size=5,
                max_size=20
            )
            await self._init_postgres_schema()
            logger.info("✅ PostgreSQL initialized with schema")
            
            # Qdrant client
            self.qdrant_client = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
                timeout=30.0
            )
            await self._init_qdrant_collections()
            logger.info("✅ Qdrant initialized with collections")
            
            # MinIO client
            self.minio_client = Minio(
                config.MINIO_ENDPOINT,
                access_key=config.MINIO_ACCESS_KEY,
                secret_key=config.MINIO_SECRET_KEY,
                secure=False
            )
            self._ensure_minio_bucket()
            logger.info("✅ MinIO initialized with bucket")
            
            # Embedding service client
            self.embedding_client = httpx.AsyncClient(
                base_url=config.EMBEDDING_URL,
                timeout=60.0
            )
            logger.info("✅ Embedding service client initialized")
            
            # Initialize advanced components if available
            if ADVANCED_FEATURES_AVAILABLE:
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    try:
                        self.spacy_nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        self.spacy_nlp = spacy.blank("en")
                        self.spacy_nlp.add_pipe('sentencizer')
                    
                    # Initialize Ollama client
                    self.ollama_client = ollama.AsyncClient(host=config.OLLAMA_URL)
                    
                    # Initialize advanced chunker with storage integration
                    self.chunker = AdvancedChunker(
                        tokenizer=self.tokenizer,
                        sentence_model=self.sentence_model,
                        spacy_nlp=self.spacy_nlp,
                        ollama_client=self.ollama_client,
                        minio_client=self.minio_client,
                        postgres_pool=self.pg_pool
                    )
                    
                    logger.info("✅ Advanced processing components initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Some advanced features unavailable: {e}")
                    ADVANCED_FEATURES_AVAILABLE = False
            
            # Validate health
            await self._validate_startup_health()
            logger.info("✅ All systems operational - Server ready!")
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Graceful shutdown of all connections"""
        logger.info("🔄 Shutting down server...")
        
        if self.embedding_client:
            await self.embedding_client.aclose()
        
        if self.pg_pool:
            await self.pg_pool.close()
        
        logger.info("✅ Server shutdown complete")
    
    async def _init_postgres_schema(self):
        """Initialize PostgreSQL schema for chunk storage"""
        async with self.pg_pool.acquire() as conn:
            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id VARCHAR(255) PRIMARY KEY,
                    tenant_id VARCHAR(100) NOT NULL,
                    minio_path TEXT NOT NULL,
                    original_filename VARCHAR(500),
                    file_size BIGINT,
                    content_hash VARCHAR(64),
                    chunking_method VARCHAR(50),
                    total_chunks INT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create chunks table with text storage
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id VARCHAR(255) PRIMARY KEY,
                    document_id VARCHAR(255) REFERENCES documents(document_id) ON DELETE CASCADE,
                    tenant_id VARCHAR(100) NOT NULL,
                    chunk_index INT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_text_with_overlap TEXT,
                    token_count INT,
                    start_char INT,
                    end_char INT,
                    method VARCHAR(50),
                    overlap_previous TEXT,
                    overlap_next TEXT,
                    metadata JSONB,
                    embedding_status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_id, chunk_index)
                );
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_tenant 
                ON documents(tenant_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document 
                ON chunks(document_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_tenant 
                ON chunks(tenant_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_status 
                ON chunks(embedding_status);
            """)
    
    async def _init_qdrant_collections(self):
        """Initialize Qdrant collections for vector storage"""
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self.qdrant_client.get_collections)
            collection_names = [col.name for col in collections.collections]
            
            if "ta_v8_embeddings" not in collection_names:
                # Create collection with appropriate vector size
                await asyncio.to_thread(
                    self.qdrant_client.create_collection,
                    collection_name="ta_v8_embeddings",
                    vectors_config=VectorParams(
                        size=1024,  # E5-large embedding size
                        distance=Distance.COSINE
                    )
                )
                logger.info("Created Qdrant collection: ta_v8_embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collections: {e}")
            raise
    
    def _ensure_minio_bucket(self):
        """Ensure MinIO bucket exists"""
        try:
            if not self.minio_client.bucket_exists(config.MINIO_BUCKET):
                self.minio_client.make_bucket(config.MINIO_BUCKET)
                logger.info(f"Created MinIO bucket: {config.MINIO_BUCKET}")
        except S3Error as e:
            logger.error(f"Failed to ensure MinIO bucket: {e}")
            raise
    
    async def _validate_startup_health(self):
        """Validate all services are healthy"""
        # Test PostgreSQL
        async with self.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Test Qdrant
        await asyncio.to_thread(self.qdrant_client.get_collections)
        
        # Test MinIO
        self.minio_client.list_buckets()
        
        # Test embedding service
        response = await self.embedding_client.get("/health")
        if response.status_code != 200:
            raise Exception(f"Embedding service unhealthy: {response.status_code}")
    
    async def extract_text_from_file(self, content: bytes, content_type: str, filename: str) -> str:
        """Extract text content from uploaded file
        
        Args:
            content: File content bytes
            content_type: MIME type
            filename: Original filename
            
        Returns:
            Extracted text content
        """
        try:
            if content_type.startswith('text/'):
                return content.decode('utf-8')
            elif filename.endswith('.txt'):
                return content.decode('utf-8')
            else:
                # For other file types, we'd need additional libraries
                # For now, just try to decode as text
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    raise Exception(f"Unsupported file type: {content_type}")
        except Exception as e:
            logger.error(f"Failed to extract text from {filename}: {e}")
            raise
    
    async def store_chunks_in_postgres(self, document_id: str, tenant_id: str, 
                                      chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Store chunk text and metadata in PostgreSQL
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            chunks: List of chunk dictionaries
            metadata: Document metadata
        """
        async with self.pg_pool.acquire() as conn:
            # Store document record
            await conn.execute("""
                INSERT INTO documents (
                    document_id, tenant_id, minio_path, original_filename,
                    file_size, content_hash, chunking_method, total_chunks, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (document_id) DO UPDATE SET
                    total_chunks = EXCLUDED.total_chunks,
                    chunking_method = EXCLUDED.chunking_method,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, 
                document_id,
                tenant_id,
                metadata.get('minio_path', ''),
                metadata.get('original_filename', ''),
                metadata.get('file_size', 0),
                metadata.get('content_hash', ''),
                metadata.get('chunking_method', 'auto'),
                len(chunks),
                json.dumps(metadata)
            )
            
            # Store chunks
            for chunk in chunks:
                await conn.execute("""
                    INSERT INTO chunks (
                        chunk_id, document_id, tenant_id, chunk_index,
                        chunk_text, chunk_text_with_overlap, token_count,
                        start_char, end_char, method, overlap_previous,
                        overlap_next, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        chunk_text_with_overlap = EXCLUDED.chunk_text_with_overlap,
                        token_count = EXCLUDED.token_count,
                        metadata = EXCLUDED.metadata
                """,
                    chunk['chunk_id'],
                    document_id,
                    tenant_id,
                    chunk['chunk_index'],
                    chunk['text'],
                    chunk.get('text_with_overlap', chunk['text']),
                    chunk['token_count'],
                    chunk.get('start_char', 0),
                    chunk.get('end_char', 0),
                    chunk.get('method', 'unknown'),
                    chunk.get('overlap_previous', ''),
                    chunk.get('overlap_next', ''),
                    json.dumps(chunk.get('metadata', {}))
                )
            
            logger.info(f"💾 Stored {len(chunks)} chunks in PostgreSQL for document {document_id}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            response = await self.embedding_client.post(
                "/embed",
                json={"texts": texts}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('embeddings', [])
            else:
                raise Exception(f"Embedding service error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def store_embeddings_in_qdrant(self, chunk_ids: List[str], 
                                        embeddings: List[List[float]], 
                                        tenant_id: str,
                                        collection_name: str = "ta_v8_embeddings"):
        """Store embeddings in Qdrant with chunk_id references
        
        Args:
            chunk_ids: List of chunk identifiers
            embeddings: List of embedding vectors
            tenant_id: Tenant identifier
            collection_name: Qdrant collection name
        """
        points = []
        
        for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
            # Generate unique point ID
            point_id = hashlib.sha256(chunk_id.encode()).hexdigest()[:16]
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk_id,
                        "tenant_id": tenant_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            )
        
        # Store in Qdrant
        await asyncio.to_thread(
            self.qdrant_client.upsert,
            collection_name=collection_name,
            points=points
        )
        
        # Update embedding status in PostgreSQL
        async with self.pg_pool.acquire() as conn:
            for chunk_id in chunk_ids:
                await conn.execute("""
                    UPDATE chunks 
                    SET embedding_status = 'completed' 
                    WHERE chunk_id = $1
                """, chunk_id)
        
        logger.info(f"🔮 Stored {len(embeddings)} embeddings in Qdrant")
    
    async def semantic_search_with_postgres(self, query: str, tenant_id: str, 
                                           top_k: int = 5,
                                           collection_name: str = "ta_v8_embeddings") -> List[Dict[str, Any]]:
        """Perform semantic search in Qdrant then fetch text from PostgreSQL
        
        Args:
            query: Search query
            tenant_id: Tenant identifier
            top_k: Number of results
            collection_name: Qdrant collection
            
        Returns:
            List of chunks with text and metadata
        """
        # Generate query embedding
        query_embeddings = await self.generate_embeddings([query])
        if not query_embeddings:
            return []
        
        query_vector = query_embeddings[0]
        
        # Search in Qdrant
        search_results = await asyncio.to_thread(
            self.qdrant_client.search,
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            ),
            limit=top_k
        )
        
        # Extract chunk IDs from search results
        chunk_ids = [hit.payload['chunk_id'] for hit in search_results]
        scores = [hit.score for hit in search_results]
        
        if not chunk_ids:
            return []
        
        # Fetch chunk text from PostgreSQL
        async with self.pg_pool.acquire() as conn:
            chunks_data = await conn.fetch("""
                SELECT 
                    c.chunk_id,
                    c.chunk_text,
                    c.chunk_text_with_overlap,
                    c.token_count,
                    c.chunk_index,
                    c.method,
                    c.metadata as chunk_metadata,
                    d.document_id,
                    d.original_filename,
                    d.metadata as document_metadata
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE c.chunk_id = ANY($1)
                ORDER BY array_position($1::text[], c.chunk_id)
            """, chunk_ids)
        
        # Combine results with scores
        results = []
        for chunk_data, score in zip(chunks_data, scores):
            results.append({
                "chunk_id": chunk_data['chunk_id'],
                "document_id": chunk_data['document_id'],
                "chunk_text": chunk_data['chunk_text'],
                "chunk_text_with_overlap": chunk_data['chunk_text_with_overlap'],
                "score": float(score),
                "token_count": chunk_data['token_count'],
                "chunk_index": chunk_data['chunk_index'],
                "method": chunk_data['method'],
                "original_filename": chunk_data['original_filename'],
                "chunk_metadata": json.loads(chunk_data['chunk_metadata']) if chunk_data['chunk_metadata'] else {},
                "document_metadata": json.loads(chunk_data['document_metadata']) if chunk_data['document_metadata'] else {}
            })
        
        return results
    
    def _register_routes(self):
        """Register all API routes"""
        
        @self.app.post("/upload", response_model=DocumentUploadResponse)
        async def upload_document(
            file: UploadFile = File(...),
            tenant_id: str = Form(...)
        ):
            """Upload and process a document with integrated chunking
            
            This endpoint:
            1. Extracts text content from the uploaded file
            2. Uses enhanced chunker to save to MinIO and get MinIO document ID
            3. Automatically chunks and stores in PostgreSQL
            4. Returns complete processing information
            """
            try:
                if not self.chunker:
                    raise HTTPException(status_code=503, detail="Document processing service unavailable")
                
                # Read and extract text content
                content_bytes = await file.read()
                text_content = await self.extract_text_from_file(
                    content_bytes, 
                    file.content_type or "text/plain", 
                    file.filename or "document.txt"
                )
                
                # Process document with integrated storage (MinIO + PostgreSQL)
                result = await self.chunker.chunk_document(
                    text=text_content,
                    method="auto",  # Use intelligent strategy selection
                    tenant_id=tenant_id,
                    filename=file.filename or "document.txt",
                    metadata={
                        "original_filename": file.filename,
                        "content_type": file.content_type,
                        "file_size": len(content_bytes)
                    },
                    auto_store=True  # Enable automatic storage
                )
                
                return DocumentUploadResponse(
                    document_id=result['document_id'],
                    minio_path=result.get('storage', {}).get('minio', {}).get('minio_path', ''),
                    file_size=len(content_bytes),
                    upload_timestamp=result['timestamp'],
                    message=f"Document processed successfully using {result['method']} method with {result['statistics']['total_chunks']} chunks"
                )
                
            except Exception as e:
                logger.error(f"Document processing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/chunk")
        async def chunk_document(request: ChunkRequest):
            """Chunk a document using intelligent strategy selection with integrated storage
            
            This endpoint:
            1. Uses LLM to analyze document and select optimal chunking strategy
            2. Automatically saves document to MinIO if not already stored
            3. Chunks the document with overlap support
            4. Automatically stores chunks in PostgreSQL
            5. Returns comprehensive processing information
            """
            try:
                if not self.chunker:
                    raise HTTPException(status_code=503, detail="Chunking service unavailable")
                
                # Use enhanced chunker with integrated storage
                result = await self.chunker.chunk_document(
                    text=request.text,
                    method=request.method,
                    target_chunk_tokens=request.target_chunk_tokens,
                    max_chunk_tokens=request.max_chunk_tokens,
                    chunk_overlap_tokens=request.chunk_overlap_tokens,
                    metadata=request.metadata,
                    document_id=request.document_id,
                    tenant_id=request.tenant_id,
                    filename=request.metadata.get('filename', 'document.txt'),
                    auto_store=True  # Enable automatic MinIO and PostgreSQL storage
                )
                
                return {
                    "success": True,
                    "document_id": result['document_id'],
                    "method_used": result['method'],
                    "total_chunks": result['statistics']['total_chunks'],
                    "statistics": result['statistics'],
                    "storage": result.get('storage', {}),
                    "analysis": result.get('analysis', {}),
                    "message": f"Document chunked and stored successfully using {result['method']} method"
                }
                
            except Exception as e:
                logger.error(f"Chunking failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/embed")
        async def generate_and_store_embeddings(request: EmbedRequest):
            """Generate embeddings for chunks and store in Qdrant
            
            This endpoint:
            1. Fetches chunk text from PostgreSQL
            2. Generates embeddings using the embedding service
            3. Stores embeddings in Qdrant with chunk_id references
            """
            try:
                # Fetch chunk texts from PostgreSQL
                async with self.pg_pool.acquire() as conn:
                    chunks = await conn.fetch("""
                        SELECT chunk_id, chunk_text_with_overlap 
                        FROM chunks 
                        WHERE chunk_id = ANY($1)
                    """, request.chunk_ids)
                
                if not chunks:
                    raise HTTPException(status_code=404, detail="Chunks not found")
                
                # Extract texts for embedding
                texts = [chunk['chunk_text_with_overlap'] for chunk in chunks]
                chunk_ids = [chunk['chunk_id'] for chunk in chunks]
                
                # Generate embeddings
                embeddings = await self.generate_embeddings(texts)
                
                # Store in Qdrant
                await self.store_embeddings_in_qdrant(
                    chunk_ids=chunk_ids,
                    embeddings=embeddings,
                    tenant_id=request.tenant_id,
                    collection_name=request.collection_name
                )
                
                return {
                    "success": True,
                    "chunks_embedded": len(chunk_ids),
                    "message": "Embeddings generated and stored successfully"
                }
                
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/retrieve")
        async def semantic_retrieve(request: RetrieveRequest):
            """Perform semantic search and retrieve chunk text
            
            This endpoint:
            1. Generates embedding for the query
            2. Searches Qdrant for similar vectors
            3. Fetches corresponding chunk text from PostgreSQL
            4. Returns ranked results with text and metadata
            """
            try:
                results = await self.semantic_search_with_postgres(
                    query=request.query,
                    tenant_id=request.tenant_id,
                    top_k=request.top_k,
                    collection_name=request.collection_name
                )
                
                return {
                    "success": True,
                    "query": request.query,
                    "results_count": len(results),
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Comprehensive health check for all services"""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {}
            }
            
            # Check PostgreSQL
            try:
                async with self.pg_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                health_status["services"]["postgresql"] = "healthy"
            except Exception as e:
                health_status["services"]["postgresql"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check Qdrant
            try:
                await asyncio.to_thread(self.qdrant_client.get_collections)
                health_status["services"]["qdrant"] = "healthy"
            except Exception as e:
                health_status["services"]["qdrant"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check MinIO
            try:
                self.minio_client.list_buckets()
                health_status["services"]["minio"] = "healthy"
            except Exception as e:
                health_status["services"]["minio"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check Embedding Service
            try:
                response = await self.embedding_client.get("/health")
                if response.status_code == 200:
                    health_status["services"]["embedding"] = "healthy"
                else:
                    health_status["services"]["embedding"] = f"unhealthy: status {response.status_code}"
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["services"]["embedding"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            return health_status


# Create server instance
server = UnifiedMCPServer()
app = server.app

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting TA_V8 Unified MCP Server...")
    
    uvicorn.run(
        "unified_mcp_server_v2:app",
        host="0.0.0.0",
        port=8005,
        workers=1,
        log_level="info",
        access_log=True
    )
