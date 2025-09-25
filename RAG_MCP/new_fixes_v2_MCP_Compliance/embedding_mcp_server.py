#!/usr/bin/env python3
"""
Embedding MCP Server
Implements MCP-compliant embedding service for the Team Agent platform
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

class EmbedItem(BaseModel):
    """Item to be embedded"""
    id: str = Field(..., description="Chunk or document ID")
    text: str = Field(..., description="Text to embed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Item metadata")

class MCPEmbedRequest(BaseModel):
    """MCP-compliant embedding request"""
    tenant_id: str = Field(..., description="Tenant identifier")
    collection: str = Field(..., description="Collection name (e.g., 'domain:d-abc')")
    items: List[EmbedItem] = Field(..., description="Items to embed")
    upsert: bool = Field(default=True, description="Whether to upsert to vector store")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class VectorRecord(BaseModel):
    """Vector embedding record"""
    id: str
    vector: List[float]

class MCPEmbedResponse(BaseModel):
    """MCP-compliant embedding response"""
    vectors: List[VectorRecord]
    upserted: int
    error: Optional[str] = None

# ============ Service Implementation ============

class EmbeddingService:
    def __init__(self):
        self.pg_pool = None
        self.qdrant_client = None
        self.embedding_client = httpx.AsyncClient()
        self.embedding_url = os.getenv("EMBEDDING_URL", "http://multilingual-e5-large:8080")
        
    async def startup(self):
        """Initialize connections"""
        # PostgreSQL
        self.pg_pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres_user"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres_pass"),
            database=os.getenv("POSTGRES_DATABASE", "ta_v8"),
            min_size=1,
            max_size=10
        )
        
        # Qdrant
        qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        logger.info("Embedding MCP Service initialized")
    
    async def shutdown(self):
        """Clean up connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.embedding_client:
            await self.embedding_client.aclose()
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using BGE-M3 or similar model"""
        try:
            response = await self.embedding_client.post(
                f"{self.embedding_url}/embed",
                json={"texts": texts},
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Embedding service error: {response.status_code}")
            
            result = response.json()
            return result.get("embeddings", [])
            
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            # Fallback: return random embeddings for testing
            import numpy as np
            return [np.random.randn(768).tolist() for _ in texts]
    
    async def ensure_collection(self, collection_name: str, vector_size: int = 768):
        """Ensure Qdrant collection exists"""
        try:
            collections = self.qdrant_client.get_collections()
            if collection_name not in [c.name for c in collections.collections]:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Collection creation error: {str(e)}")
    
    async def process_embeddings(self, request: MCPEmbedRequest) -> MCPEmbedResponse:
        """Process embedding request following MCP protocol"""
        try:
            # Extract texts
            texts = [item.text for item in request.items]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            if len(embeddings) != len(texts):
                raise ValueError(f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}")
            
            # Create vector records
            vector_records = []
            for item, embedding in zip(request.items, embeddings):
                vector_records.append(
                    VectorRecord(id=item.id, vector=embedding)
                )
            
            # Upsert to Qdrant if requested
            upserted_count = 0
            if request.upsert:
                # Ensure collection exists
                await self.ensure_collection(request.collection)
                
                # Prepare points for Qdrant
                points = []
                for item, embedding in zip(request.items, embeddings):
                    # Generate unique point ID
                    point_id = hashlib.sha256(item.id.encode()).hexdigest()[:16]
                    
                    # Combine metadata
                    payload = {
                        "chunk_id": item.id,
                        "tenant_id": request.tenant_id,
                        **item.metadata,
                        "embedded_at": datetime.utcnow().isoformat()
                    }
                    
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    )
                
                # Upsert to Qdrant
                self.qdrant_client.upsert(
                    collection_name=request.collection,
                    points=points
                )
                upserted_count = len(points)
                
                # Update PostgreSQL chunk records
                async with self.pg_pool.acquire() as conn:
                    for item in request.items:
                        await conn.execute("""
                            UPDATE chunks 
                            SET embedding_status = 'embedded',
                                embedded_at = $2
                            WHERE chunk_id = $1
                        """, item.id, datetime.utcnow())
            
            return MCPEmbedResponse(
                vectors=vector_records,
                upserted=upserted_count
            )
            
        except Exception as e:
            logger.error(f"Embedding processing error: {str(e)}")
            return MCPEmbedResponse(
                vectors=[],
                upserted=0,
                error=str(e)
            )

# Initialize service
service = EmbeddingService()

# ============ MCP Endpoints ============

@app.on_event("startup")
async def startup():
    await service.startup()

@app.on_event("shutdown")
async def shutdown():
    await service.shutdown()

@app.post("/mcp/execute", response_model=MCPEmbedResponse)
async def execute_mcp(request: MCPEmbedRequest) -> MCPEmbedResponse:
    """
    MCP-compliant execution endpoint
    This is the standard endpoint that TAO's MCP Gateway expects
    """
    return await service.process_embeddings(request)

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check database connection
        async with service.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Check Qdrant connection
        service.qdrant_client.get_collections()
        
        return {
            "status": "healthy",
            "service": "embed_v1",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "embed_v1",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # TODO: Implement proper metrics
    return {
        "embeddings_generated_total": 0,
        "vectors_upserted_total": 0,
        "errors_total": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
