#!/usr/bin/env python3
"""
Chunking MCP Server
Implements MCP-compliant chunking service for the Team Agent platform
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

class ChunkingPolicy(BaseModel):
    """Chunking policy configuration"""
    method: str = Field(default="recursive", description="Chunking method: recursive, semantic, or hybrid")
    target_tokens: int = Field(default=512, description="Target chunk size in tokens")
    overlap: int = Field(default=64, description="Token overlap between chunks")

class DocumentSource(BaseModel):
    """Document source specification"""
    type: str = Field(..., description="Source type: text, url, or file")
    text: Optional[str] = Field(None, description="Raw text content")
    url: Optional[str] = Field(None, description="URL to fetch")
    file_id: Optional[str] = Field(None, description="MinIO file ID")

class MCPChunkRequest(BaseModel):
    """MCP-compliant chunking request"""
    tenant_id: str = Field(..., description="Tenant identifier")
    domain_id: Optional[str] = Field(None, description="Domain identifier")
    source: DocumentSource = Field(..., description="Document source")
    policy: ChunkingPolicy = Field(default_factory=ChunkingPolicy, description="Chunking policy")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class ChunkRecord(BaseModel):
    """Individual chunk record"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class MCPChunkResponse(BaseModel):
    """MCP-compliant chunking response"""
    chunks: List[ChunkRecord]
    persisted: bool
    document_id: Optional[str] = None
    error: Optional[str] = None

# ============ Service Implementation ============

class ChunkingService:
    def __init__(self):
        self.pg_pool = None
        self.neo4j_driver = None
        self.minio_client = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
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
        
        # Neo4j
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j_password")
        self.neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # MinIO
        self.minio_client = Minio(
            os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False
        )
        
        logger.info("Chunking MCP Service initialized")
    
    async def shutdown(self):
        """Clean up connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()
    
    def chunk_recursive(self, text: str, policy: ChunkingPolicy) -> List[Dict[str, Any]]:
        """Implement recursive chunking strategy"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        chunk_size = policy.target_tokens
        overlap = policy.overlap
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "start_token": i,
                "end_token": min(i + chunk_size, len(tokens)),
                "position": len(chunks)
            })
            
            if i + chunk_size >= len(tokens):
                break
        
        return chunks
    
    def chunk_semantic(self, text: str, policy: ChunkingPolicy) -> List[Dict[str, Any]]:
        """Implement semantic chunking strategy (simplified)"""
        # Split on paragraph boundaries first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = len(self.tokenizer.encode(paragraph))
            
            if current_tokens + paragraph_tokens > policy.target_tokens and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": current_chunk,
                    "position": len(chunks)
                })
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                # Add to current chunk
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_tokens += paragraph_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "position": len(chunks)
            })
        
        return chunks
    
    async def process_chunks(self, request: MCPChunkRequest) -> MCPChunkResponse:
        """Process chunking request following MCP protocol"""
        try:
            # Extract text based on source type
            if request.source.type == "text":
                text = request.source.text
            elif request.source.type == "url":
                # TODO: Implement URL fetching
                text = request.source.text  # Placeholder
            elif request.source.type == "file":
                # TODO: Fetch from MinIO
                text = request.source.text  # Placeholder
            else:
                raise ValueError(f"Unsupported source type: {request.source.type}")
            
            if not text:
                raise ValueError("No text content provided")
            
            # Generate document ID
            document_id = f"doc_{uuid.uuid4().hex[:12]}"
            
            # Choose chunking strategy
            if request.policy.method == "recursive":
                raw_chunks = self.chunk_recursive(text, request.policy)
            elif request.policy.method == "semantic":
                raw_chunks = self.chunk_semantic(text, request.policy)
            else:
                raw_chunks = self.chunk_recursive(text, request.policy)  # Default
            
            # Generate chunk records with IDs
            chunk_records = []
            for i, chunk_data in enumerate(raw_chunks):
                chunk_id = f"c-{document_id}-{i:04d}"
                
                chunk_record = ChunkRecord(
                    chunk_id=chunk_id,
                    text=chunk_data["text"],
                    metadata={
                        "tenant_id": request.tenant_id,
                        "domain_id": request.domain_id,
                        "document_id": document_id,
                        "position": chunk_data["position"],
                        "chunk_method": request.policy.method,
                        **request.metadata
                    }
                )
                chunk_records.append(chunk_record)
            
            # Persist to PostgreSQL
            persisted = await self.persist_chunks(
                document_id, 
                chunk_records, 
                request.tenant_id,
                request.domain_id
            )
            
            return MCPChunkResponse(
                chunks=chunk_records,
                persisted=persisted,
                document_id=document_id
            )
            
        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            return MCPChunkResponse(
                chunks=[],
                persisted=False,
                error=str(e)
            )
    
    async def persist_chunks(
        self, 
        document_id: str, 
        chunks: List[ChunkRecord], 
        tenant_id: str,
        domain_id: Optional[str]
    ) -> bool:
        """Persist chunks to PostgreSQL"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Insert document record
                await conn.execute("""
                    INSERT INTO documents (document_id, tenant_id, domain_id, created_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (document_id) DO NOTHING
                """, document_id, tenant_id, domain_id, datetime.utcnow())
                
                # Insert chunks
                for chunk in chunks:
                    await conn.execute("""
                        INSERT INTO chunks (
                            chunk_id, document_id, tenant_id, 
                            chunk_text, chunk_index, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """, 
                        chunk.chunk_id,
                        document_id,
                        tenant_id,
                        chunk.text,
                        chunk.metadata["position"],
                        json.dumps(chunk.metadata)
                    )
                    
                    # Insert chunk-domain relationship if domain specified
                    if domain_id:
                        await conn.execute("""
                            INSERT INTO chunk_domains (chunk_id, domain_id, relevance_score)
                            VALUES ($1, $2, $3)
                        """, chunk.chunk_id, domain_id, 1.0)
            
            return True
            
        except Exception as e:
            logger.error(f"Persistence error: {str(e)}")
            return False

# Initialize service
service = ChunkingService()

# ============ MCP Endpoints ============

@app.on_event("startup")
async def startup():
    await service.startup()

@app.on_event("shutdown")
async def shutdown():
    await service.shutdown()

@app.post("/mcp/execute", response_model=MCPChunkResponse)
async def execute_mcp(request: MCPChunkRequest) -> MCPChunkResponse:
    """
    MCP-compliant execution endpoint
    This is the standard endpoint that TAO's MCP Gateway expects
    """
    return await service.process_chunks(request)

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check database connection
        async with service.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {
            "status": "healthy",
            "service": "chunker_v1",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "chunker_v1",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # TODO: Implement proper metrics
    return {
        "chunks_processed_total": 0,
        "documents_processed_total": 0,
        "errors_total": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
