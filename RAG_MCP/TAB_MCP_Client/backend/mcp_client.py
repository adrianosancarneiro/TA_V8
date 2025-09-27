"""
MCP stdio client for TAB_MCP_Client
===================================
Handles communication with MCP services using stdio transport
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import subprocess
import os
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class MCPStdioClient:
    """Client for communicating with MCP services via stdio transport"""
    
    def __init__(self):
        self.services = {
            "chunking": "chunking-mcp",
            "embedding": "embedding-mcp", 
            "retrieval": "retrieval-mcp",
            "rag_agent": "rag-agent-team"
        }
    
    async def call_service(self, service_name: str, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a service method via systemd service communication"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service_id = self.services[service_name]
        
        # For now, we'll use HTTP as fallback until stdio MCP is fully implemented
        # This allows the client to work while we transition to stdio
        return await self._call_http_fallback(service_name, method, params)
    
    async def _call_http_fallback(self, service_name: str, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback to HTTP communication with services"""
        import httpx
        
        # Map service names to ports
        port_map = {
            "chunking": 8001,
            "embedding": 8004,
            "retrieval": 8003,
            "rag_agent": 8001  # RAG agent team runs on 8001
        }
        
        if service_name not in port_map:
            raise ValueError(f"No port mapping for service: {service_name}")
        
        port = port_map[service_name]
        url = f"http://localhost:{port}"
        
        # Create method-specific endpoints
        if method == "chunk_document":
            endpoint = f"{url}/chunk"
        elif method == "embed_chunks":
            endpoint = f"{url}/embed"
        elif method == "search":
            endpoint = f"{url}/search"
        elif method == "query":
            endpoint = f"{url}/query"
        else:
            endpoint = f"{url}/{method}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(endpoint, json=params or {})
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP request failed for {service_name}.{method}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling {service_name}.{method}: {e}")
            raise
    
    async def check_service_status(self, service_name: str) -> bool:
        """Check if a service is running via systemd"""
        if service_name not in self.services:
            return False
        
        service_id = self.services[service_name]
        
        try:
            result = await asyncio.create_subprocess_exec(
                'systemctl', '--user', 'is-active', service_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            is_active = stdout.decode().strip() == 'active'
            if not is_active:
                logger.warning(f"Service {service_id} is not active")
            
            return is_active
        except Exception as e:
            logger.error(f"Failed to check status of {service_id}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all MCP services"""
        results = {}
        
        for service_name, service_id in self.services.items():
            results[service_name] = await self.check_service_status(service_name)
        
        return results


class RAGQueryClient:
    """High-level client for RAG queries using MCP services"""
    
    def __init__(self):
        self.mcp_client = MCPStdioClient()
    
    async def process_query(self, query: str, tenant_id: str = "default", domain_id: str = "default") -> Dict[str, Any]:
        """Process a RAG query through the agent team"""
        try:
            # Call the RAG agent team service
            response = await self.mcp_client.call_service(
                "rag_agent",
                "query",
                {
                    "query": query,
                    "tenant_id": tenant_id,
                    "domain_id": domain_id,
                    "session_id": f"tab_client_{asyncio.current_task().get_name() if asyncio.current_task() else 'default'}"
                }
            )
            
            return {
                "success": True,
                "response": response.get("response", "No response generated"),
                "retrieved_chunks": response.get("retrieved_chunks", []),
                "execution_time_ms": response.get("execution_time_ms", 0),
                "metadata": response.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "Sorry, I encountered an error processing your query.",
                "retrieved_chunks": [],
                "execution_time_ms": 0,
                "metadata": {}
            }
    
    async def upload_document(self, document_content: str, filename: str, tenant_id: str = "default", domain_id: str = "default") -> Dict[str, Any]:
        """Upload and process a document"""
        try:
            # First, chunk the document
            chunk_response = await self.mcp_client.call_service(
                "chunking",
                "chunk_document",
                {
                    "document_content": document_content,
                    "filename": filename,
                    "tenant_id": tenant_id,
                    "domain_id": domain_id,
                    "chunking_method": "auto",
                    "chunk_size": 500,
                    "chunk_overlap": 50
                }
            )
            
            if not chunk_response.get("success", False):
                raise Exception(f"Document chunking failed: {chunk_response.get('error', 'Unknown error')}")
            
            chunks = chunk_response.get("chunks", [])
            
            # Then embed the chunks
            embed_response = await self.mcp_client.call_service(
                "embedding",
                "embed_chunks",
                {
                    "chunks": chunks,
                    "tenant_id": tenant_id,
                    "domain_id": domain_id,
                    "model": "multilingual-e5-large"
                }
            )
            
            if not embed_response.get("success", False):
                raise Exception(f"Chunk embedding failed: {embed_response.get('error', 'Unknown error')}")
            
            return {
                "success": True,
                "document_id": chunk_response.get("document_id"),
                "chunks_processed": len(chunks),
                "embeddings_created": len(embed_response.get("embeddings", [])),
                "message": f"Successfully processed '{filename}' with {len(chunks)} chunks"
            }
            
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process document '{filename}': {str(e)}"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        service_health = await self.mcp_client.health_check()
        
        all_healthy = all(service_health.values())
        
        return {
            "overall_health": "healthy" if all_healthy else "unhealthy",
            "services": service_health,
            "timestamp": datetime.now().isoformat()
        }


# Global clients
mcp_client = MCPStdioClient()
rag_client = RAGQueryClient()
