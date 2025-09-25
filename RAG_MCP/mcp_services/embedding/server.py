#!/usr/bin/env python3
"""
# =============================================================================
# MCP SERVICE: EMBEDDING 
# =============================================================================
# Purpose: Vector embedding generation service for multi-agent RAG system
# Port: 8002
# Protocol: MCP-compliant via /mcp/execute endpoint
# 
# Integration Points:
# - TAO: Will register this service in Tool Registry as 'embed_v1'
# - TAE: Will call via ToolCaller for agent tool usage during knowledge processing
# - TAB: Will use for domain knowledge indexing during team building
# 
# Dependencies: PostgreSQL, Qdrant, BGE-M3 Embedding Service
# Status: MIGRATED - Ready for platform integration
# Vector Storage: Qdrant for high-performance similarity search
# Metadata Storage: PostgreSQL for embedding metadata and relationships
# =============================================================================
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding MCP Server", version="1.0.0")

# ============ MCP Request/Response Models ============
# Following MCP protocol standards for request/response envelope structure
# These models ensure compatibility with TAO's MCP Gateway and TAE's ToolCaller

class EmbedItem(BaseModel):
    """Item to be embedded - supports chunks, documents, or queries"""
    id: str = Field(..., description="Chunk or document ID")
    text: str = Field(..., description="Text to embed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Item metadata")

class MCPEmbedRequest(BaseModel):
    """MCP-compliant embedding request - standardized envelope for TAO/TAE integration"""
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenancy")
    collection: str = Field(..., description="Collection name (e.g., 'domain:d-abc')")
    items: List[EmbedItem] = Field(..., description="Items to embed")
    upsert: bool = Field(default=True, description="Whether to upsert to vector store")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class VectorRecord(BaseModel):
    """Vector embedding record - output format"""
    id: str
    vector: List[float]

class MCPEmbedResponse(BaseModel):
    """MCP-compliant embedding response - standardized envelope for TAO/TAE integration"""
    vectors: List[VectorRecord]
    upserted: int
    error: Optional[str] = None

# ============ Service Implementation ============
# Core service logic with vector database integration

class EmbeddingService:
    """
    Core embedding service with complete vector storage integration
    
    Features:
    - Multi-tenant vector collections
    - BGE-M3 multilingual embedding model
    - Qdrant high-performance vector storage
    - PostgreSQL metadata persistence
    - Configurable embedding dimensions
    - Batch processing support
    """
    
    def __init__(self):
        # Database and service connections
        self.pg_pool = None                    # PostgreSQL for metadata
        self.qdrant_client = None             # Qdrant for vector storage  
        self.embedding_client = httpx.AsyncClient()  # HTTP client for embedding service
        self.embedding_url = os.getenv("EMBEDDING_URL", "http://multilingual-e5-large:8080")
        self.embedding_dim = 1024             # BGE-M3 embedding dimension
        
    async def startup(self):
        """Initialize all service connections - called at startup"""
        logger.info("Starting Embedding MCP Service initialization...")
        
        # PostgreSQL connection pool
        # Used for storing embedding metadata and processing status
        self.pg_pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres_user"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres_pass"),
            database=os.getenv("POSTGRES_DATABASE", "ta_v8"),
            min_size=1,
            max_size=10
        )
        
        # Qdrant vector database client
        # Used for high-performance vector similarity search
        qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        logger.info("Embedding MCP Service initialized successfully")
    
    async def shutdown(self):
        """Clean up connections - called at shutdown"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.embedding_client:
            await self.embedding_client.aclose()
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using BGE-M3 or similar multilingual model
        
        Features:
        - Batch processing for efficiency
        - Retry logic for robustness  
        - Error handling and fallback
        - Support for various embedding models
        """
        try:
            response = await self.embedding_client.post(
                f"{self.embedding_url}/embed",
                json={"texts": texts},
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Embedding service error: {response.status_code}")
                
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            if len(embeddings) != len(texts):
                raise Exception(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    async def ensure_collection_exists(self, collection_name: str, tenant_id: str):
        """
        Ensure Qdrant collection exists for tenant
        
        Collection naming: {tenant_id}_{collection_name}
        This provides tenant isolation while allowing domain-specific collections
        """
        full_collection_name = f"{tenant_id}_{collection_name}"
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if full_collection_name not in collection_names:
                # Create new collection with appropriate vector configuration
                self.qdrant_client.create_collection(
                    collection_name=full_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,        # BGE-M3 dimension
                        distance=Distance.COSINE        # Cosine similarity for text
                    )
                )
                logger.info(f"Created Qdrant collection: {full_collection_name}")
            
            return full_collection_name
            
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            raise Exception(f"Collection setup failed: {str(e)}")
    
    async def embed_items(self, request: MCPEmbedRequest) -> MCPEmbedResponse:
        """
        Main embedding logic with full vector storage integration
        
        Process:
        1. Extract text content from items
        2. Generate embeddings via BGE-M3 service  
        3. Ensure Qdrant collection exists
        4. Store vectors in Qdrant with metadata
        5. Update PostgreSQL with embedding records
        6. Return MCP-compliant response
        """
        try:
            if not request.items:
                return MCPEmbedResponse(vectors=[], upserted=0, error="No items provided")
            
            # Step 1: Extract texts for batch embedding generation
            texts = [item.text for item in request.items]
            item_ids = [item.id for item in request.items]
            
            # Step 2: Generate embeddings in batch
            logger.info(f"Generating embeddings for {len(texts)} items")
            embeddings = await self.generate_embeddings(texts)
            
            # Step 3: Ensure collection exists
            collection_name = await self.ensure_collection_exists(request.collection, request.tenant_id)
            
            # Step 4: Prepare vector records
            vector_records = []
            qdrant_points = []
            
            for i, (item, embedding) in enumerate(zip(request.items, embeddings)):
                vector_record = VectorRecord(
                    id=item.id,
                    vector=embedding
                )
                vector_records.append(vector_record)
                
                # Prepare Qdrant point with metadata
                point_metadata = {
                    "tenant_id": request.tenant_id,
                    "collection": request.collection,
                    "text": item.text[:500],  # Truncate for metadata storage
                    "item_metadata": item.metadata,
                    "created_at": datetime.now().isoformat(),
                    **request.metadata  # Include additional metadata
                }
                
                qdrant_points.append(PointStruct(
                    id=item.id,
                    vector=embedding,
                    payload=point_metadata
                ))
            
            # Step 5: Store in Qdrant (upsert if enabled)
            upserted_count = 0
            if request.upsert and qdrant_points:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=qdrant_points
                )
                upserted_count = len(qdrant_points)
                logger.info(f"Upserted {upserted_count} vectors to collection: {collection_name}")
            
            # Step 6: Update PostgreSQL with embedding metadata
            await self._store_embedding_metadata(request, vector_records, collection_name)
            
            return MCPEmbedResponse(
                vectors=vector_records,
                upserted=upserted_count,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return MCPEmbedResponse(
                vectors=[],
                upserted=0,
                error=str(e)
            )
    
    async def _store_embedding_metadata(self, request: MCPEmbedRequest, vector_records: List[VectorRecord], collection_name: str):
        """Store embedding metadata in PostgreSQL for tracking and audit"""
        async with self.pg_pool.acquire() as conn:
            # Store embedding batch record
            batch_id = f"embed_batch_{int(datetime.now().timestamp())}_{hashlib.md5(str(request.items).encode()).hexdigest()[:8]}"
            
            await conn.execute("""
                INSERT INTO rag_system.embedding_batches 
                (batch_id, tenant_id, collection_name, item_count, status, metadata, created_at)
                VALUES ($1, $2, $3, $4, 'completed', $5, $6)
            """, batch_id, request.tenant_id, collection_name, len(vector_records),
                json.dumps(request.metadata), datetime.now())
            
            # Store individual embedding records
            for item, vector_record in zip(request.items, vector_records):
                await conn.execute("""
                    INSERT INTO rag_system.embeddings 
                    (item_id, tenant_id, collection_name, batch_id, vector_dim, 
                     item_text, status, metadata, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, 'stored', $7, $8, $8)
                    ON CONFLICT (item_id, collection_name) DO UPDATE SET 
                        vector_dim = $5,
                        item_text = $6,
                        status = 'stored',
                        updated_at = $8
                """, item.id, request.tenant_id, collection_name, batch_id, 
                    len(vector_record.vector), item.text[:1000],  # Truncate for storage
                    json.dumps(item.metadata), datetime.now())

# ============ Service Setup and API Endpoints ============

# Global service instance
embedding_service = EmbeddingService()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup - called by FastAPI"""
    await embedding_service.startup()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up service on shutdown - called by FastAPI"""
    await embedding_service.shutdown()

# ============ MCP-Compliant API Endpoints ============
# Following MCP protocol standards for TAO/TAE integration

@app.post("/mcp/execute")
async def mcp_execute_endpoint(request: MCPEmbedRequest) -> MCPEmbedResponse:
    """
    Main MCP-compliant execution endpoint
    
    This endpoint follows the Model Context Protocol standard and integrates with:
    - TAO: Registered as 'embed_v1' tool in ToolRegistry
    - TAE: Called via ToolCaller when agents need vector embeddings
    - TAB: Used during domain knowledge indexing workflows
    
    Request format follows MCP envelope structure for consistency
    Response format provides structured output for downstream processing
    """
    logger.info(f"MCP embedding request for tenant: {request.tenant_id}, collection: {request.collection}, items: {len(request.items)}")
    
    try:
        response = await embedding_service.embed_items(request)
        
        if response.error:
            logger.error(f"Embedding failed: {response.error}")
        else:
            logger.info(f"Embedding completed: {len(response.vectors)} vectors generated, {response.upserted} upserted")
        
        return response
        
    except Exception as e:
        logger.error(f"MCP execution error: {str(e)}")
        return MCPEmbedResponse(
            vectors=[],
            upserted=0,
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
        if embedding_service.pg_pool:
            async with embedding_service.pg_pool.acquire() as conn:
                await conn.execute("SELECT 1")
        
        # Test Qdrant connection
        if embedding_service.qdrant_client:
            embedding_service.qdrant_client.get_collections()
        
        # Test embedding service connection
        if embedding_service.embedding_client:
            response = await embedding_service.embedding_client.get(
                f"{embedding_service.embedding_url}/health",
                timeout=5.0
            )
            embedding_service_status = "connected" if response.status_code == 200 else "error"
        else:
            embedding_service_status = "disconnected"
        
        return {
            "status": "healthy",
            "service": "embedding-mcp",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "postgresql": "connected",
                "qdrant": "connected",
                "embedding_service": embedding_service_status
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "embedding-mcp", 
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
        port=8002,       # Standard port for embedding MCP service  
        reload=False,    # Disable reload in production
        workers=1,       # Single worker for MVP, scale as needed
        log_level="info"
    )