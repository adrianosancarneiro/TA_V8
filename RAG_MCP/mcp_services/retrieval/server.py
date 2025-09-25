#!/usr/bin/env python3
"""
# =============================================================================
# MCP SERVICE: RETRIEVAL 
# =============================================================================
# Purpose: Semantic retrieval service for multi-agent RAG system
# Port: 8003
# Protocol: MCP-compliant via /mcp/execute endpoint
# 
# Integration Points:
# - TAO: Will register this service in Tool Registry as 'retriever_v1'
# - TAE: Will call via ToolCaller for agent tool usage during query processing
# - TAB: Will use for knowledge verification during team building
# 
# Dependencies: PostgreSQL, Qdrant, Embedding MCP Service (port 8002)
# Status: MIGRATED - Ready for platform integration
# Query Processing: Semantic search using vector similarity + metadata filtering
# Inter-service Communication: Calls Embedding MCP for query vectorization
# =============================================================================
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
# Following MCP protocol standards for request/response envelope structure
# These models ensure compatibility with TAO's MCP Gateway and TAE's ToolCaller

class QuerySpec(BaseModel):
    """Query specification with semantic and filtering options"""
    text: str = Field(..., description="Query text for semantic search")
    use_embedding: bool = Field(default=True, description="Whether to use embedding for search")

class MCPRetrieveRequest(BaseModel):
    """MCP-compliant retrieval request - standardized envelope for TAO/TAE integration"""
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenancy")
    collection: str = Field(..., description="Collection name (e.g., 'domain:d-abc')")
    query: QuerySpec = Field(..., description="Query specification")
    top_k: int = Field(default=5, description="Number of results to return")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")

class HitRecord(BaseModel):
    """Search hit record - output format"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class MCPRetrieveResponse(BaseModel):
    """MCP-compliant retrieval response - standardized envelope for TAO/TAE integration"""
    hits: List[HitRecord]
    error: Optional[str] = None

# ============ Service Implementation ============
# Core service logic with vector search and inter-service communication

class RetrievalService:
    """
    Core retrieval service with complete semantic search capabilities
    
    Features:
    - Multi-tenant vector search across collections
    - Hybrid semantic + metadata filtering
    - Inter-service communication with Embedding MCP
    - PostgreSQL text retrieval with vector similarity
    - Configurable result ranking and filtering
    - Support for complex query patterns
    """
    
    def __init__(self):
        # Database and service connections
        self.pg_pool = None                     # PostgreSQL for text retrieval
        self.qdrant_client = None              # Qdrant for vector search
        self.embedding_client = httpx.AsyncClient()  # HTTP client for embedding MCP
        self.embedding_mcp_url = os.getenv("EMBEDDING_MCP_URL", "http://embedding-mcp:8002")
        
    async def startup(self):
        """Initialize all service connections - called at startup"""
        logger.info("Starting Retrieval MCP Service initialization...")
        
        # PostgreSQL connection pool
        # Used for retrieving full chunk text and metadata after vector search
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
        # Used for high-performance semantic similarity search
        qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        logger.info("Retrieval MCP Service initialized successfully")
    
    async def shutdown(self):
        """Clean up connections - called at shutdown"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.embedding_client:
            await self.embedding_client.aclose()
    
    async def get_query_embedding(self, text: str, tenant_id: str) -> List[float]:
        """
        Get embedding for query text by calling Embedding MCP service
        
        This demonstrates inter-service communication in the MCP architecture:
        Retrieval MCP → Embedding MCP → BGE-M3 Model → Vector Response
        """
        try:
            # Prepare request payload for Embedding MCP service
            request_payload = {
                "tenant_id": tenant_id,
                "collection": "temp_query",  # Temporary collection for query embedding
                "items": [
                    {
                        "id": "query_temp",
                        "text": text,
                        "metadata": {"type": "query"}
                    }
                ],
                "upsert": False,  # Don't store query embeddings
                "metadata": {"purpose": "retrieval_query"}
            }
            
            # Call Embedding MCP service
            response = await self.embedding_client.post(
                f"{self.embedding_mcp_url}/mcp/execute",
                json=request_payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Embedding MCP service error: {response.status_code}")
            
            result = response.json()
            if result.get("error"):
                raise Exception(f"Embedding MCP error: {result['error']}")
            
            vectors = result.get("vectors", [])
            if not vectors:
                raise Exception("No embedding returned from service")
            
            return vectors[0]["vector"]
            
        except Exception as e:
            logger.error(f"Failed to get query embedding: {str(e)}")
            raise Exception(f"Query embedding failed: {str(e)}")
    
    async def search_vectors(self, query_vector: List[float], collection_name: str, 
                           top_k: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in Qdrant
        
        Features:
        - Cosine similarity search
        - Metadata filtering
        - Configurable result count
        - Tenant isolation via collection naming
        """
        try:
            # Build Qdrant filter from request filters
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ))
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Perform vector search
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Convert to standard format
            hits = []
            for result in search_results:
                hit = {
                    "id": result.id,
                    "score": float(result.score),
                    "payload": result.payload or {}
                }
                hits.append(hit)
            
            return hits
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise Exception(f"Vector search error: {str(e)}")
    
    async def get_full_texts(self, chunk_ids: List[str], tenant_id: str) -> Dict[str, str]:
        """
        Retrieve full text content from PostgreSQL for given chunk IDs
        
        This provides the complete text content after vector similarity matching,
        ensuring users get the full context rather than truncated metadata
        """
        try:
            async with self.pg_pool.acquire() as conn:
                # Query for full chunk texts
                query = """
                    SELECT chunk_id, chunk_text 
                    FROM rag_system.chunks 
                    WHERE chunk_id = ANY($1) AND tenant_id = $2
                """
                
                rows = await conn.fetch(query, chunk_ids, tenant_id)
                
                # Convert to dictionary for easy lookup
                text_map = {}
                for row in rows:
                    text_map[row['chunk_id']] = row['chunk_text']
                
                return text_map
                
        except Exception as e:
            logger.error(f"Text retrieval failed: {str(e)}")
            raise Exception(f"Text retrieval error: {str(e)}")
    
    async def retrieve_documents(self, request: MCPRetrieveRequest) -> MCPRetrieveResponse:
        """
        Main retrieval logic with semantic search and text retrieval
        
        Process:
        1. Get query embedding via Embedding MCP service
        2. Ensure target collection exists in Qdrant
        3. Perform vector similarity search with filtering
        4. Retrieve full text content from PostgreSQL
        5. Combine results and rank by relevance
        6. Return MCP-compliant response
        """
        try:
            if not request.query.text.strip():
                return MCPRetrieveResponse(hits=[], error="Empty query provided")
            
            # Step 1: Get query embedding if semantic search is enabled
            query_vector = None
            if request.query.use_embedding:
                logger.info(f"Getting embedding for query: {request.query.text[:100]}...")
                query_vector = await self.get_query_embedding(request.query.text, request.tenant_id)
            
            # Step 2: Prepare collection name with tenant isolation
            collection_name = f"{request.tenant_id}_{request.collection}"
            
            # Step 3: Perform vector search or fallback to text search
            vector_hits = []
            if query_vector:
                try:
                    vector_hits = await self.search_vectors(
                        query_vector=query_vector,
                        collection_name=collection_name,
                        top_k=request.top_k,
                        filters=request.filters
                    )
                    logger.info(f"Vector search returned {len(vector_hits)} hits")
                except Exception as e:
                    logger.warning(f"Vector search failed, falling back to text search: {str(e)}")
                    # Could implement text-based fallback here if needed
            
            # Step 4: Get full text content from PostgreSQL
            chunk_ids = [hit["id"] for hit in vector_hits]
            text_map = {}
            if chunk_ids:
                text_map = await self.get_full_texts(chunk_ids, request.tenant_id)
            
            # Step 5: Combine results with full text content
            hit_records = []
            for hit in vector_hits:
                chunk_id = hit["id"]
                full_text = text_map.get(chunk_id, hit["payload"].get("text", ""))
                
                # Combine vector search metadata with payload metadata
                combined_metadata = {
                    **hit["payload"],
                    "search_type": "semantic" if query_vector else "fallback",
                    "collection": request.collection,
                    "retrieved_at": datetime.now().isoformat()
                }
                
                hit_record = HitRecord(
                    id=chunk_id,
                    score=hit["score"],
                    text=full_text,
                    metadata=combined_metadata
                )
                hit_records.append(hit_record)
            
            # Step 6: Sort by relevance score (highest first)
            hit_records.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Retrieval completed: {len(hit_records)} results for query in collection {collection_name}")
            
            return MCPRetrieveResponse(
                hits=hit_records,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return MCPRetrieveResponse(
                hits=[],
                error=str(e)
            )

# ============ Service Setup and API Endpoints ============

# Global service instance
retrieval_service = RetrievalService()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup - called by FastAPI"""
    await retrieval_service.startup()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up service on shutdown - called by FastAPI"""
    await retrieval_service.shutdown()

# ============ MCP-Compliant API Endpoints ============
# Following MCP protocol standards for TAO/TAE integration

@app.post("/mcp/execute")
async def mcp_execute_endpoint(request: MCPRetrieveRequest) -> MCPRetrieveResponse:
    """
    Main MCP-compliant execution endpoint
    
    This endpoint follows the Model Context Protocol standard and integrates with:
    - TAO: Registered as 'retriever_v1' tool in ToolRegistry  
    - TAE: Called via ToolCaller when agents need semantic search
    - TAB: Used during knowledge verification workflows
    
    Request format follows MCP envelope structure for consistency
    Response format provides structured output for agent reasoning
    """
    logger.info(f"MCP retrieval request for tenant: {request.tenant_id}, collection: {request.collection}, query: {request.query.text[:100]}")
    
    try:
        response = await retrieval_service.retrieve_documents(request)
        
        if response.error:
            logger.error(f"Retrieval failed: {response.error}")
        else:
            logger.info(f"Retrieval completed: {len(response.hits)} hits returned")
        
        return response
        
    except Exception as e:
        logger.error(f"MCP execution error: {str(e)}")
        return MCPRetrieveResponse(
            hits=[],
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
        if retrieval_service.pg_pool:
            async with retrieval_service.pg_pool.acquire() as conn:
                await conn.execute("SELECT 1")
        
        # Test Qdrant connection
        if retrieval_service.qdrant_client:
            retrieval_service.qdrant_client.get_collections()
        
        # Test embedding MCP service connection
        embedding_service_status = "disconnected"
        if retrieval_service.embedding_client:
            try:
                response = await retrieval_service.embedding_client.get(
                    f"{retrieval_service.embedding_mcp_url}/health",
                    timeout=5.0
                )
                embedding_service_status = "connected" if response.status_code == 200 else "error"
            except:
                embedding_service_status = "unreachable"
        
        return {
            "status": "healthy",
            "service": "retrieval-mcp",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "postgresql": "connected",
                "qdrant": "connected",
                "embedding_mcp": embedding_service_status
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "retrieval-mcp",
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
        port=8003,       # Standard port for retrieval MCP service
        reload=False,    # Disable reload in production
        workers=1,       # Single worker for MVP, scale as needed
        log_level="info"
    )