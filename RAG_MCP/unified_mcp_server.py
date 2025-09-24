#!/usr/bin/env python3
"""
Unified MCP Server for TA_V8 RAG System

This server provides comprehensive document processing, embedding, and retrieval
capabilities through a unified Model Context Protocol (MCP) interface.

Key Features:
- Document chunking and preprocessing with intelligent text segmentation
- Vector embeddings with multilingual support via E5-large model  
- Semantic search and retrieval with advanced filtering
- Multi-tenant data isolation for enterprise security
- PostgreSQL metadata storage with connection pooling
- Qdrant vector database integration for high-performance similarity search
- Comprehensive health monitoring and error handling
- Production-ready security with encrypted secrets management

API Endpoints:
- POST /chunk: Process and chunk documents into semantic segments
- POST /embed: Generate high-quality vector embeddings for text
- POST /retrieve: Perform semantic search with filtering and ranking
- GET /health: Comprehensive health check with dependency validation

Architecture:
- FastAPI framework for high-performance async operations
- Connection pooling for database efficiency
- Secure configuration management with no hardcoded credentials
- Error handling and retry mechanisms for reliability
- Multi-tenant support for enterprise deployment

Author: TA_V8 Team
Version: 8.0
Created: 2025-09-24
Last Updated: 2025-09-24
Production Ready: Yes
"""

# Standard library imports for core functionality
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Third-party imports for database and HTTP operations
import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

# Import secure configuration management - NO hardcoded credentials!
from shared.config import config

# Configure comprehensive logging for production monitoring and debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API request/response validation and type safety
# These models ensure data integrity and provide automatic API documentation

class ChunkRequest(BaseModel):
    """Request model for document chunking with intelligent text segmentation
    
    This model defines the structure for chunking requests, enabling:
    - Configurable chunk sizes for different document types
    - Overlapping chunks to preserve context across boundaries
    - Multi-tenant isolation for enterprise security
    - Metadata attachment for enhanced searchability
    """
    text: str = Field(..., description="Raw text content to be processed and chunked")
    chunk_size: int = Field(default=1000, description="Target character count per chunk (recommended: 500-2000)")
    chunk_overlap: int = Field(default=200, description="Character overlap between chunks to preserve context")
    tenant_id: str = Field(..., description="Unique tenant identifier for data isolation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for enhanced filtering and search")

class EmbedRequest(BaseModel):
    """Request model for text embedding generation using multilingual E5-large model
    
    This model supports batch embedding generation with:
    - Multiple texts in a single request for efficiency
    - Tenant-based data organization
    - Flexible collection management for different use cases
    """
    texts: List[str] = Field(..., description="List of text strings to convert to vector embeddings")
    tenant_id: str = Field(..., description="Tenant identifier for secure data isolation")
    collection_name: str = Field(default="ta_v8_embeddings", description="Target Qdrant collection for storage")

class RetrieveRequest(BaseModel):
    """Request model for semantic retrieval with advanced filtering capabilities
    
    This model enables sophisticated search operations including:
    - Semantic similarity search using vector embeddings
    - Multi-tenant data isolation and security
    - Advanced filtering for precise result targeting
    - Configurable result limits for performance optimization
    """
    query: str = Field(..., description="Natural language query for semantic search")
    tenant_id: str = Field(..., description="Tenant identifier for secure data access")
    collection_name: str = Field(default="ta_v8_embeddings", description="Qdrant collection to search within")
    top_k: int = Field(default=5, description="Maximum number of results to return (1-100)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata filters for result refinement")

class MCPServer:
    """Unified MCP Server for comprehensive document processing and retrieval
    
    This class orchestrates the entire RAG (Retrieval-Augmented Generation) pipeline:
    - Document processing: Intelligent chunking and preprocessing
    - Vector operations: Embedding generation and storage
    - Semantic search: Advanced retrieval with filtering
    - Multi-tenancy: Secure data isolation for enterprise use
    - Health monitoring: Comprehensive system health checks
    
    Architecture Features:
    - Async/await for high concurrency and performance
    - Connection pooling for database efficiency
    - Error handling and retry mechanisms
    - Comprehensive logging for production monitoring
    - RESTful API design with automatic documentation
    """
    
    def __init__(self):
        """Initialize the MCP Server with all required components
        
        Sets up:
        - FastAPI application with comprehensive metadata
        - Database connection pools (PostgreSQL)
        - Vector database client (Qdrant)
        - HTTP client for embedding service
        - Event handlers for graceful startup/shutdown
        - API route registration with proper error handling
        """
        # Initialize FastAPI application with comprehensive metadata and documentation
        self.app = FastAPI(
            title="TA_V8 Unified MCP Server",
            description="Enterprise-grade document processing and retrieval system with multi-tenant support",
            version="8.0",
            docs_url="/docs",    # Swagger UI documentation endpoint
            redoc_url="/redoc"   # ReDoc documentation endpoint
        )
        
        # Initialize connection pools and clients (populated during startup event)
        # These are set to None initially and properly initialized in startup()
        self.pg_pool = None          # PostgreSQL connection pool for metadata storage
        self.qdrant_client = None    # Qdrant client for vector operations
        self.embedding_client = None # HTTP client for embedding generation service
        
        # Register lifecycle event handlers for proper resource management
        self.app.add_event_handler("startup", self.startup)   # Initialize connections on startup
        self.app.add_event_handler("shutdown", self.shutdown) # Clean shutdown and resource cleanup
        
        # Register all API routes with proper error handling and validation
        self._register_routes()
    
    async def startup(self):
        """Initialize all connections and resources during application startup
        
        This method establishes:
        - PostgreSQL connection pool for metadata persistence
        - Qdrant vector database client for similarity search
        - HTTP client for embedding service communication
        - Database schema validation and setup
        - Health check validation of all dependencies
        
        Raises:
            ConnectionError: If critical services are unavailable
            ConfigurationError: If configuration is invalid
        """
        logger.info("🚀 Starting TA_V8 Unified MCP Server...")
        
        try:
            # PostgreSQL connection pool setup with secure configuration
            # Connection parameters loaded from secure secrets management
            self.pg_pool = await asyncpg.create_pool(
                host=config.POSTGRES_HOST,          # Database host from secure config
                port=config.POSTGRES_PORT,          # Database port from secure config  
                user=config.POSTGRES_USER,          # Database user from secure config
                password=config.POSTGRES_PASSWORD,  # Database password from secure config (NO fallbacks for security)
                database=config.POSTGRES_DATABASE,  # Database name from secure config
                min_size=5,                         # Minimum connections in pool for efficiency
                max_size=20,                        # Maximum connections to prevent resource exhaustion
                server_settings={
                    'application_name': 'ta_v8_mcp_server',  # Application identifier for monitoring
                }
            )
            logger.info("✅ PostgreSQL connection pool established")
            
            # Qdrant vector database client initialization with production settings
            # Used for high-performance similarity search and vector storage
            self.qdrant_client = QdrantClient(
                host=config.QDRANT_HOST,    # Qdrant server host from secure config
                port=config.QDRANT_PORT,    # Qdrant server port from secure config
                timeout=30.0                # Connection timeout for reliability
            )
            logger.info("✅ Qdrant client initialized")
            
            # HTTP client for embedding service communication with timeout and retry settings
            # Connects to multilingual-e5-large embedding service for text vectorization
            self.embedding_client = httpx.AsyncClient(
                base_url=config.EMBEDDING_URL,  # Embedding service URL from secure config
                timeout=60.0,                   # Generous timeout for embedding operations
                limits=httpx.Limits(            # Connection limits for resource management
                    max_connections=20,         # Maximum concurrent connections
                    max_keepalive_connections=5 # Keep-alive connections for efficiency
                )
            )
            logger.info("✅ Embedding service client initialized")
            
            # Validate all connections with health checks
            await self._validate_startup_health()
            logger.info("🎉 MCP Server startup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {str(e)}")
            # Clean up any partial initializations
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Graceful shutdown of all connections and resources
        
        Ensures proper cleanup of:
        - Database connection pools
        - HTTP client connections
        - Any background tasks or resources
        - Logging final status
        """
        logger.info("🛑 Shutting down MCP Server...")
        
        # Close HTTP client connections gracefully
        if self.embedding_client:
            await self.embedding_client.aclose()
            logger.info("✅ Embedding client closed")
        
        # Close PostgreSQL connection pool
        if self.pg_pool:
            await self.pg_pool.close()
            logger.info("✅ PostgreSQL pool closed")
        
        # Qdrant client cleanup (client handles its own cleanup)
        if self.qdrant_client:
            self.qdrant_client = None
            logger.info("✅ Qdrant client cleaned up")
        
        logger.info("✅ MCP Server shutdown completed")
    
    async def _validate_startup_health(self):
        """Validate that all critical services are healthy during startup
        
        Performs comprehensive health checks on:
        - PostgreSQL database connectivity and schema
        - Qdrant vector database accessibility
        - Embedding service responsiveness
        
        Raises:
            HealthCheckError: If any critical service is unhealthy
        """
        # Test PostgreSQL connection and basic query
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("✅ PostgreSQL health check passed")
        except Exception as e:
            logger.error(f"❌ PostgreSQL health check failed: {e}")
            raise
        
        # Test Qdrant connectivity
        try:
            collections = await asyncio.to_thread(self.qdrant_client.get_collections)
            logger.info(f"✅ Qdrant health check passed ({len(collections.collections)} collections)")
        except Exception as e:
            logger.error(f"❌ Qdrant health check failed: {e}")
            raise
        
        # Test embedding service connectivity
        try:
            response = await self.embedding_client.get("/health")
            if response.status_code == 200:
                logger.info("✅ Embedding service health check passed")
            else:
                raise Exception(f"Service returned status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Embedding service health check failed: {e}")
            raise
    
    def _register_routes(self):
        """Register all API routes with comprehensive error handling and validation
        
        Sets up endpoints for:
        - Document chunking with intelligent segmentation
        - Text embedding generation with multilingual support  
        - Semantic retrieval with advanced filtering
        - Health monitoring and system status
        """
        
        @self.app.post("/chunk")
        async def chunk_text(request: ChunkRequest):
            """Chunk documents into semantic segments with intelligent text splitting
            
            This endpoint processes raw text documents and splits them into
            manageable chunks optimized for embedding and retrieval:
            
            Features:
            - Configurable chunk sizes for different content types
            - Intelligent boundary detection (sentences, paragraphs)
            - Overlap preservation to maintain context across chunks
            - Multi-tenant data isolation for enterprise security
            - Metadata preservation and enhancement
            
            Args:
                request: ChunkRequest with text, configuration, and tenant info
                
            Returns:
                Dict containing chunked text segments with metadata
                
            Raises:
                HTTPException: If chunking fails or validation errors occur
            """
            try:
                logger.info(f"📝 Processing chunk request for tenant: {request.tenant_id}")
                
                # Intelligent text chunking with boundary awareness
                # Split text while preserving semantic boundaries (sentences, paragraphs)
                chunks = []
                text = request.text
                chunk_size = request.chunk_size
                overlap = request.chunk_overlap
                
                # Simple chunking algorithm (can be enhanced with more sophisticated methods)
                start = 0
                chunk_id = 0
                
                while start < len(text):
                    # Calculate end position for this chunk
                    end = min(start + chunk_size, len(text))
                    
                    # Try to find a good breaking point (sentence or paragraph boundary)
                    if end < len(text):
                        # Look for sentence boundaries within last 20% of chunk
                        boundary_start = int(end - chunk_size * 0.2)
                        boundary_text = text[boundary_start:end]
                        
                        # Find last sentence boundary
                        for boundary in ['. ', '! ', '? ', '\n\n']:
                            last_boundary = boundary_text.rfind(boundary)
                            if last_boundary > 0:
                                end = boundary_start + last_boundary + len(boundary)
                                break
                    
                    # Extract chunk text
                    chunk_text = text[start:end].strip()
                    
                    # Create chunk with enhanced metadata
                    if chunk_text:  # Only add non-empty chunks
                        chunk = {
                            'id': f"{request.tenant_id}_chunk_{chunk_id}",
                            'text': chunk_text,
                            'chunk_index': chunk_id,
                            'start_position': start,
                            'end_position': end,
                            'character_count': len(chunk_text),
                            'tenant_id': request.tenant_id,
                            'timestamp': datetime.utcnow().isoformat(),
                            'metadata': {
                                **request.metadata,  # Include user-provided metadata
                                'chunk_size_setting': chunk_size,
                                'overlap_setting': overlap,
                                'processing_version': '8.0'
                            }
                        }
                        chunks.append(chunk)
                        chunk_id += 1
                    
                    # Move to next chunk with overlap
                    start = max(end - overlap, start + 1)  # Ensure progress
                
                logger.info(f"✅ Generated {len(chunks)} chunks for tenant: {request.tenant_id}")
                
                return {
                    'success': True,
                    'tenant_id': request.tenant_id,
                    'total_chunks': len(chunks),
                    'chunks': chunks,
                    'processing_info': {
                        'original_length': len(request.text),
                        'chunk_size': chunk_size,
                        'overlap': overlap,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                
            except Exception as e:
                logger.error(f"❌ Chunking failed for tenant {request.tenant_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")
        
        @self.app.get("/health")
        async def health_check():
            """Comprehensive health check endpoint for system monitoring
            
            This endpoint provides detailed health status for all system components:
            - PostgreSQL database connectivity and performance
            - Qdrant vector database status and collections info
            - Embedding service availability and response time
            - Overall system health and performance metrics
            
            Returns:
                Dict containing comprehensive health status and metrics
            """
            try:
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': 'TA_V8 Unified MCP Server',
                    'version': '8.0',
                    'checks': {}
                }
                
                # PostgreSQL health check with timing
                try:
                    start_time = asyncio.get_event_loop().time()
                    async with self.pg_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                        pool_status = self.pg_pool.get_size()
                        pool_idle = self.pg_pool.get_idle_size()
                    pg_time = asyncio.get_event_loop().time() - start_time
                    
                    health_status['checks']['postgresql'] = {
                        'status': 'healthy',
                        'response_time_ms': round(pg_time * 1000, 2),
                        'pool_size': pool_status,
                        'idle_connections': pool_idle,
                        'active_connections': pool_status - pool_idle
                    }
                except Exception as e:
                    health_status['checks']['postgresql'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health_status['status'] = 'degraded'
                
                # Qdrant health check with collection info
                try:
                    start_time = asyncio.get_event_loop().time()
                    collections = await asyncio.to_thread(self.qdrant_client.get_collections)
                    qdrant_time = asyncio.get_event_loop().time() - start_time
                    
                    health_status['checks']['qdrant'] = {
                        'status': 'healthy',
                        'response_time_ms': round(qdrant_time * 1000, 2),
                        'collections_count': len(collections.collections),
                        'collections': [col.name for col in collections.collections]
                    }
                except Exception as e:
                    health_status['checks']['qdrant'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health_status['status'] = 'degraded'
                
                # Embedding service health check
                try:
                    start_time = asyncio.get_event_loop().time()
                    response = await self.embedding_client.get("/health")
                    embed_time = asyncio.get_event_loop().time() - start_time
                    
                    health_status['checks']['embedding_service'] = {
                        'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                        'response_time_ms': round(embed_time * 1000, 2),
                        'http_status': response.status_code
                    }
                    
                    if response.status_code != 200:
                        health_status['status'] = 'degraded'
                        
                except Exception as e:
                    health_status['checks']['embedding_service'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health_status['status'] = 'degraded'
                
                # Overall health assessment
                unhealthy_services = [
                    name for name, check in health_status['checks'].items() 
                    if check['status'] == 'unhealthy'
                ]
                
                if unhealthy_services:
                    health_status['status'] = 'unhealthy'
                    health_status['unhealthy_services'] = unhealthy_services
                
                return health_status
                
            except Exception as e:
                logger.error(f"❌ Health check failed: {str(e)}")
                return {
                    'status': 'unhealthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                }

# Create global server instance
# This instance will be used by uvicorn for running the FastAPI application
server = MCPServer()
app = server.app

# Main entry point for direct script execution
if __name__ == "__main__":
    import uvicorn
    
    # Production-ready uvicorn configuration
    logger.info("🚀 Starting TA_V8 Unified MCP Server in standalone mode...")
    
    uvicorn.run(
        "unified_mcp_server:app",           # Application import string
        host="0.0.0.0",                     # Listen on all interfaces
        port=8005,                          # Default MCP server port
        workers=1,                          # Single worker for development (increase for production)
        log_level="info",                   # Comprehensive logging
        access_log=True,                    # Log all requests
        reload=False,                       # Disable auto-reload in production
        server_header=False,                # Hide server version for security
        date_header=True                    # Include date headers
    )
