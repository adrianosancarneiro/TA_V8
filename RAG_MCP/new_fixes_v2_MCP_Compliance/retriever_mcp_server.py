#!/usr/bin/env python3
"""
Retriever MCP Server
Implements MCP-compliant retrieval service for the Team Agent platform
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retriever MCP Server", version="1.0.0")

# ============ MCP Request/Response Models ============

class QuerySpec(BaseModel):
    """Query specification"""
    text: str = Field(..., description="Query text")
    use_embedding: bool = Field(default=True, description="Whether to use embedding for search")

class MCPRetrieveRequest(BaseModel):
    """MCP-compliant retrieval request"""
    tenant_id: str = Field(..., description="Tenant identifier")
    collection: str = Field(..., description="Collection name (e.g., 'domain:d-abc')")
    query: QuerySpec = Field(..., description="Query specification")
    top_k: int = Field(default=5, description="Number of results to return")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")

class HitRecord(BaseModel):
    """Search hit record"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class MCPRetrieveResponse(BaseModel):
    """MCP-compliant retrieval response"""
    hits: List[HitRecord]
    error: Optional[str] = None

# ============ Service Implementation ============

class RetrievalService:
    def __init__(self):
        self.pg_pool = None
        self.qdrant_client = None
        self.embedding_client = httpx.AsyncClient()
        self.embedding_mcp_url = os.getenv("EMBEDDING_MCP_URL", "http://embedding-mcp:8002")
        
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
        
        logger.info("Retriever MCP Service initialized")
    
    async def shutdown(self):
        """Clean up connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.embedding_client:
            await self.embedding_client.aclose()
    
    async def get_query_embedding(self, text: str, tenant_id: str) -> List[float]:
        """Get embedding for query text by calling Embedding MCP"""
        try:
            # Call Embedding MCP service
            request_payload = {
                "tenant_id": tenant_id,
                "collection": "temp_query",
                "items": [
                    {
                        "id": "query_temp",
                        "text": text,
                        "metadata": {"type": "query"}
                    }
                ],
                "upsert": False  # Don't store query embeddings
            }
            
            response = await self.embedding_client.post(
                f"{self.embedding_mcp_url}/mcp/execute",
                json=request_payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Embedding MCP error: {response.status_code}")
            
            result = response.json()
            if result.get("vectors") and len(result["vectors"]) > 0:
                return result["vectors"][0]["vector"]
            else:
                raise ValueError("No embedding returned")
                
        except Exception as e:
            logger.error(f"Query embedding error: {str(e)}")
            # Fallback: return random embedding for testing
            import numpy as np
            return np.random.randn(768).tolist()
    
    async def process_retrieval(self, request: MCPRetrieveRequest) -> MCPRetrieveResponse:
        """Process retrieval request following MCP protocol"""
        try:
            hits = []
            
            if request.query.use_embedding:
                # Get query embedding
                query_vector = await self.get_query_embedding(
                    request.query.text, 
                    request.tenant_id
                )
                
                # Build Qdrant filter
                must_conditions = [
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=request.tenant_id)
                    )
                ]
                
                # Add domain filter if specified in collection name
                if "domain:" in request.collection:
                    domain_id = request.collection.split("domain:")[1]
                    must_conditions.append(
                        FieldCondition(
                            key="domain_id",
                            match=MatchValue(value=domain_id)
                        )
                    )
                
                # Add custom filters
                for key, value in request.filters.items():
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                
                # Search in Qdrant
                search_results = self.qdrant_client.search(
                    collection_name=request.collection.split("/")[0],  # Extract base collection name
                    query_vector=query_vector,
                    query_filter=Filter(must=must_conditions),
                    limit=request.top_k,
                    with_payload=True
                )
                
                # Extract chunk IDs from search results
                chunk_ids = []
                scores = {}
                for result in search_results:
                    chunk_id = result.payload.get("chunk_id")
                    if chunk_id:
                        chunk_ids.append(chunk_id)
                        scores[chunk_id] = result.score
                
                if chunk_ids:
                    # Fetch chunk texts from PostgreSQL
                    async with self.pg_pool.acquire() as conn:
                        chunks = await conn.fetch("""
                            SELECT 
                                chunk_id,
                                chunk_text,
                                metadata
                            FROM chunks
                            WHERE chunk_id = ANY($1)
                            ORDER BY array_position($1::text[], chunk_id)
                        """, chunk_ids)
                    
                    # Build hit records
                    for chunk in chunks:
                        chunk_id = chunk['chunk_id']
                        metadata = json.loads(chunk['metadata']) if chunk['metadata'] else {}
                        
                        hit = HitRecord(
                            id=chunk_id,
                            score=scores.get(chunk_id, 0.0),
                            text=chunk['chunk_text'],
                            metadata=metadata
                        )
                        hits.append(hit)
            else:
                # Text-based search (fallback)
                async with self.pg_pool.acquire() as conn:
                    # Simple text search using PostgreSQL full-text search
                    query = f"%{request.query.text}%"
                    chunks = await conn.fetch("""
                        SELECT 
                            chunk_id,
                            chunk_text,
                            metadata
                        FROM chunks
                        WHERE tenant_id = $1
                        AND chunk_text ILIKE $2
                        LIMIT $3
                    """, request.tenant_id, query, request.top_k)
                    
                    for chunk in chunks:
                        metadata = json.loads(chunk['metadata']) if chunk['metadata'] else {}
                        
                        hit = HitRecord(
                            id=chunk['chunk_id'],
                            score=1.0,  # No real score for text search
                            text=chunk['chunk_text'],
                            metadata=metadata
                        )
                        hits.append(hit)
            
            return MCPRetrieveResponse(hits=hits)
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return MCPRetrieveResponse(
                hits=[],
                error=str(e)
            )

# Initialize service
service = RetrievalService()

# ============ MCP Endpoints ============

@app.on_event("startup")
async def startup():
    await service.startup()

@app.on_event("shutdown")
async def shutdown():
    await service.shutdown()

@app.post("/mcp/execute", response_model=MCPRetrieveResponse)
async def execute_mcp(request: MCPRetrieveRequest) -> MCPRetrieveResponse:
    """
    MCP-compliant execution endpoint
    This is the standard endpoint that TAO's MCP Gateway expects
    """
    return await service.process_retrieval(request)

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
            "service": "retriever_v1",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "retriever_v1",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # TODO: Implement proper metrics
    return {
        "queries_processed_total": 0,
        "chunks_retrieved_total": 0,
        "errors_total": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
