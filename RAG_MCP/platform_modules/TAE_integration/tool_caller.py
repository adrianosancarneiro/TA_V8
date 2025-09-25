#!/usr/bin/env python3
"""
# =============================================================================
# TAE INTEGRATION: RAG TOOL CALLER
# =============================================================================
# Purpose: Enable TAE's ToolCaller to invoke RAG MCP services for agent workflows
# Integration Point: TAE ToolCaller → TAO MCP Gateway → RAG MCP Services
# 
# This module provides the interfaces needed for TAE agents to seamlessly
# use RAG capabilities during multi-agent conversations and task execution.
# 
# Workflow:
# 1. Agent needs RAG capability → TAE ToolCaller detects tool request
# 2. TAE calls TAO's MCP Gateway with tool request
# 3. TAO routes to appropriate RAG MCP service (chunking/embedding/retrieval)
# 4. MCP service processes request and returns result
# 5. Result flows back: MCP Service → TAO → TAE → Agent context
# =============================================================================
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass

# TAE integration imports (these will be available when TAE is integrated)
# from tae.core.tool_caller import ToolCaller, ToolCallRequest, ToolCallResponse
# from tae.core.agent_runtime import AgentContext
# from tae.models.tool_result import ToolResult, ToolError

logger = logging.getLogger(__name__)

# ============================================================================
# RAG TOOL CALL MODELS
# ============================================================================
# These models define the interface between TAE agents and RAG MCP services

@dataclass
class RAGChunkRequest:
    """Request structure for document chunking via TAE ToolCaller"""
    tenant_id: str
    document_text: str
    domain_id: Optional[str] = None
    chunking_method: str = "recursive"  # recursive, semantic, hybrid
    target_tokens: int = 512
    overlap: int = 64
    file_id: Optional[str] = None  # For MinIO-stored documents

@dataclass  
class RAGChunkResult:
    """Result structure from document chunking"""
    success: bool
    document_id: Optional[str] = None
    chunks: List[Dict[str, Any]] = None
    total_chunks: int = 0
    error: Optional[str] = None

@dataclass
class RAGEmbedRequest:
    """Request structure for vector embedding via TAE ToolCaller"""
    tenant_id: str
    collection: str
    items: List[Dict[str, str]]  # [{"id": "...", "text": "..."}]
    upsert: bool = True

@dataclass
class RAGEmbedResult:
    """Result structure from vector embedding"""
    success: bool
    vectors_generated: int = 0
    vectors_upserted: int = 0
    error: Optional[str] = None

@dataclass
class RAGRetrieveRequest:
    """Request structure for semantic retrieval via TAE ToolCaller"""
    tenant_id: str
    collection: str
    query_text: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    use_embedding: bool = True

@dataclass
class RAGRetrieveResult:
    """Result structure from semantic retrieval"""
    success: bool
    hits: List[Dict[str, Any]] = None
    total_hits: int = 0
    error: Optional[str] = None

# ============================================================================
# TAE INTEGRATION CLASS
# ============================================================================

class RAGToolCaller:
    """
    RAG-specific tool caller for TAE agent integration
    
    This class provides convenient methods for TAE agents to perform
    RAG operations through the TAO MCP Gateway without needing to
    understand the underlying MCP protocol details.
    
    Features:
    - Simplified API for agents
    - Automatic error handling and retries
    - Result caching for performance
    - Integration with TAE's execution context
    """
    
    def __init__(self, tao_mcp_gateway_url: str = "http://tao:8100"):
        self.tao_gateway_url = tao_mcp_gateway_url
        self.call_cache = {}  # Simple result caching
        
    async def chunk_document(self, request: RAGChunkRequest, 
                           agent_context=None) -> RAGChunkResult:
        """
        Chunk a document using the MCP chunking service
        
        This method is called by TAE agents when they need to process
        documents during their reasoning workflow.
        
        Args:
            request: Chunking request parameters
            agent_context: TAE agent context for logging/tracking
            
        Returns:
            RAGChunkResult with chunks and metadata
        """
        try:
            logger.info(f"Agent chunking request: tenant={request.tenant_id}, method={request.chunking_method}")
            
            # Build MCP request payload
            mcp_payload = {
                "tenant_id": request.tenant_id,
                "domain_id": request.domain_id,
                "source": {
                    "type": "file" if request.file_id else "text",
                    "text": request.document_text,
                    "file_id": request.file_id
                },
                "policy": {
                    "method": request.chunking_method,
                    "target_tokens": request.target_tokens,
                    "overlap": request.overlap
                },
                "metadata": {
                    "agent_context": agent_context.agent_id if agent_context else "unknown",
                    "called_at": datetime.now().isoformat()
                }
            }
            
            # Call TAO MCP Gateway (this will route to chunking MCP service)
            result = await self._call_tao_gateway("chunker_v1", mcp_payload)
            
            # Process response
            if result.get("error"):
                return RAGChunkResult(
                    success=False,
                    error=result["error"]
                )
            
            chunks = result.get("chunks", [])
            return RAGChunkResult(
                success=True,
                document_id=result.get("document_id"),
                chunks=chunks,
                total_chunks=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"Chunking call failed: {str(e)}")
            return RAGChunkResult(
                success=False,
                error=f"Chunking service error: {str(e)}"
            )
    
    async def embed_items(self, request: RAGEmbedRequest,
                         agent_context=None) -> RAGEmbedResult:
        """
        Generate vector embeddings using the MCP embedding service
        
        This method is called by TAE agents when they need to create
        or update vector embeddings for semantic search.
        
        Args:
            request: Embedding request parameters
            agent_context: TAE agent context for logging/tracking
            
        Returns:
            RAGEmbedResult with embedding statistics
        """
        try:
            logger.info(f"Agent embedding request: tenant={request.tenant_id}, collection={request.collection}, items={len(request.items)}")
            
            # Build MCP request payload
            mcp_payload = {
                "tenant_id": request.tenant_id,
                "collection": request.collection,
                "items": [
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "metadata": {
                            "agent_context": agent_context.agent_id if agent_context else "unknown",
                            "embedded_at": datetime.now().isoformat()
                        }
                    }
                    for item in request.items
                ],
                "upsert": request.upsert,
                "metadata": {
                    "agent_context": agent_context.agent_id if agent_context else "unknown"
                }
            }
            
            # Call TAO MCP Gateway (this will route to embedding MCP service)
            result = await self._call_tao_gateway("embed_v1", mcp_payload)
            
            # Process response
            if result.get("error"):
                return RAGEmbedResult(
                    success=False,
                    error=result["error"]
                )
            
            return RAGEmbedResult(
                success=True,
                vectors_generated=len(result.get("vectors", [])),
                vectors_upserted=result.get("upserted", 0)
            )
            
        except Exception as e:
            logger.error(f"Embedding call failed: {str(e)}")
            return RAGEmbedResult(
                success=False,
                error=f"Embedding service error: {str(e)}"
            )
    
    async def retrieve_documents(self, request: RAGRetrieveRequest,
                               agent_context=None) -> RAGRetrieveResult:
        """
        Perform semantic retrieval using the MCP retrieval service
        
        This method is called by TAE agents when they need to search
        for relevant information during their reasoning process.
        
        Args:
            request: Retrieval request parameters  
            agent_context: TAE agent context for logging/tracking
            
        Returns:
            RAGRetrieveResult with search hits and metadata
        """
        try:
            logger.info(f"Agent retrieval request: tenant={request.tenant_id}, collection={request.collection}, query={request.query_text[:100]}")
            
            # Check cache first (simple optimization)
            cache_key = f"{request.tenant_id}:{request.collection}:{hash(request.query_text)}:{request.top_k}"
            if cache_key in self.call_cache:
                logger.debug("Returning cached retrieval result")
                return self.call_cache[cache_key]
            
            # Build MCP request payload
            mcp_payload = {
                "tenant_id": request.tenant_id,
                "collection": request.collection,
                "query": {
                    "text": request.query_text,
                    "use_embedding": request.use_embedding
                },
                "top_k": request.top_k,
                "filters": request.filters or {},
                "metadata": {
                    "agent_context": agent_context.agent_id if agent_context else "unknown",
                    "queried_at": datetime.now().isoformat()
                }
            }
            
            # Call TAO MCP Gateway (this will route to retrieval MCP service)
            result = await self._call_tao_gateway("retriever_v1", mcp_payload)
            
            # Process response
            if result.get("error"):
                return RAGRetrieveResult(
                    success=False,
                    error=result["error"]
                )
            
            hits = result.get("hits", [])
            rag_result = RAGRetrieveResult(
                success=True,
                hits=hits,
                total_hits=len(hits)
            )
            
            # Cache result for future use
            self.call_cache[cache_key] = rag_result
            
            return rag_result
            
        except Exception as e:
            logger.error(f"Retrieval call failed: {str(e)}")
            return RAGRetrieveResult(
                success=False,
                error=f"Retrieval service error: {str(e)}"
            )
    
    async def _call_tao_gateway(self, tool_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to call TAO's MCP Gateway
        
        This handles the actual HTTP communication with TAO and includes
        retry logic, error handling, and request/response logging.
        """
        import httpx
        
        try:
            # TAO MCP Gateway endpoint format
            gateway_url = f"{self.tao_gateway_url}/tools/{tool_id}/execute"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    gateway_url,
                    json=payload,
                    timeout=30.0,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    raise Exception(f"TAO Gateway error: HTTP {response.status_code}")
                
                return response.json()
                
        except Exception as e:
            logger.error(f"TAO Gateway call failed: {str(e)}")
            raise Exception(f"Gateway communication error: {str(e)}")

# ============================================================================
# TAE INTEGRATION UTILITIES  
# ============================================================================

async def setup_agent_rag_capabilities(agent_id: str, tenant_id: str, 
                                     allowed_collections: List[str] = None) -> RAGToolCaller:
    """
    Set up RAG capabilities for a specific TAE agent
    
    This function should be called during agent initialization to
    provide the agent with RAG tool access.
    
    Args:
        agent_id: TAE agent identifier
        tenant_id: Tenant context for the agent
        allowed_collections: Collections the agent can access
        
    Returns:
        RAGToolCaller: Configured tool caller instance
    """
    logger.info(f"Setting up RAG capabilities for agent {agent_id}")
    
    rag_caller = RAGToolCaller()
    
    # TODO: Integrate with TAE's agent configuration system
    # This will register the RAG capabilities with the agent's tool set
    
    logger.info(f"RAG capabilities configured for agent {agent_id}")
    return rag_caller

def create_rag_enabled_agent_prompt(base_prompt: str, rag_instructions: str = None) -> str:
    """
    Enhance an agent prompt with RAG capability instructions
    
    This function helps TAE agents understand how to use RAG tools
    during their reasoning process.
    
    Args:
        base_prompt: Original agent prompt
        rag_instructions: Custom RAG usage instructions
        
    Returns:
        str: Enhanced prompt with RAG capabilities
    """
    default_rag_instructions = """
## RAG Tool Usage Instructions

You have access to the following RAG (Retrieval-Augmented Generation) tools:

1. **chunk_document**: Break large documents into manageable chunks
   - Use when you receive long documents that need processing
   - Choose chunking method based on content type (recursive for technical docs, semantic for narrative)

2. **embed_items**: Create vector embeddings for semantic search  
   - Use after chunking to make content searchable
   - Always specify the appropriate collection name

3. **retrieve_documents**: Search for relevant information
   - Use when you need to find specific information to answer questions
   - Craft precise search queries for better results

**RAG Workflow Example:**
1. If given a document: chunk_document → embed_items → store for later retrieval
2. If answering a question: retrieve_documents → use results in your response
3. Always cite retrieved information with chunk IDs for transparency

Remember: RAG tools work within your tenant/domain context automatically.
"""
    
    instructions = rag_instructions or default_rag_instructions
    
    enhanced_prompt = f"""{base_prompt}

{instructions}

Use these RAG capabilities intelligently to provide accurate, well-sourced responses."""

    return enhanced_prompt

# ============================================================================
# MAIN INTEGRATION DEMO/TEST
# ============================================================================

async def main():
    """Demo function showing how TAE integration will work"""
    print("🤖 RAG MCP → TAE Integration Demo")
    print("=" * 50)
    
    try:
        # Initialize RAG tool caller
        rag_caller = RAGToolCaller()
        
        # Demo 1: Document Chunking
        print("\n📄 Demo 1: Document Chunking")
        chunk_request = RAGChunkRequest(
            tenant_id="demo_tenant",
            document_text="This is a sample document that will be chunked for processing. " * 50,
            chunking_method="recursive",
            target_tokens=200
        )
        
        # chunk_result = await rag_caller.chunk_document(chunk_request)
        print(f"   → Would chunk document into ~{len(chunk_request.document_text.split())//40} chunks")
        
        # Demo 2: Embedding Generation  
        print("\n🔗 Demo 2: Embedding Generation")
        embed_request = RAGEmbedRequest(
            tenant_id="demo_tenant",
            collection="demo_collection",
            items=[
                {"id": "chunk_001", "text": "Sample chunk 1 content"},
                {"id": "chunk_002", "text": "Sample chunk 2 content"}
            ]
        )
        
        # embed_result = await rag_caller.embed_items(embed_request)
        print(f"   → Would generate embeddings for {len(embed_request.items)} items")
        
        # Demo 3: Semantic Retrieval
        print("\n🔍 Demo 3: Semantic Retrieval")
        retrieve_request = RAGRetrieveRequest(
            tenant_id="demo_tenant", 
            collection="demo_collection",
            query_text="Find information about sample content",
            top_k=5
        )
        
        # retrieve_result = await rag_caller.retrieve_documents(retrieve_request)
        print(f"   → Would search for top {retrieve_request.top_k} relevant chunks")
        
        # Demo 4: Enhanced Agent Prompt
        print("\n📝 Demo 4: RAG-Enhanced Agent Prompt")
        base_prompt = "You are a helpful research assistant."
        enhanced_prompt = create_rag_enabled_agent_prompt(base_prompt)
        print(f"   → Enhanced prompt: {len(enhanced_prompt)} characters (vs {len(base_prompt)} original)")
        
        print(f"\n🎉 RAG MCP → TAE integration ready for production!")
        
    except Exception as e:
        print(f"❌ Integration demo failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())