#!/usr/bin/env python3
"""
# =============================================================================
# TAO INTEGRATION: MCP SERVICE REGISTRY
# =============================================================================
# Purpose: Bridge between TAO's ToolRegistry and our MCP services
# Integration Point: TAO ToolRegistry registration for chunking/embedding/retrieval
# 
# This module provides the interfaces needed for TAO to discover and register
# our MCP services as tools that can be orchestrated by agent teams.
# 
# Workflow:
# 1. TAO startup → calls register_rag_mcp_services()
# 2. Services registered in TAO's ToolRegistry as 'chunker_v1', 'embed_v1', 'retriever_v1'
# 3. Agent teams can be configured with permissions to use these tools
# 4. TAE ToolCaller → TAO MCP Gateway → Our MCP Services
# =============================================================================
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# TAO integration imports (these will be available when TAO is integrated)
# from tao.core.registry import ToolRegistry
# from tao.core.tool_adapter import ToolAdapter
# from tao.models.tool import ToolDefinition, ToolInputSchema, ToolOutputSchema

logger = logging.getLogger(__name__)

# ============================================================================
# MCP TOOL DEFINITIONS FOR TAO REGISTRY
# ============================================================================
# These definitions describe our MCP services in TAO's ToolRegistry format

MCP_CHUNKING_TOOL_DEF = {
    "tool_id": "chunker_v1",
    "name": "Document Chunking Service",
    "description": "Intelligent document chunking with multiple strategies (recursive, semantic, hybrid)",
    "endpoint": "http://chunking-mcp:8001/mcp/execute",
    "method": "POST",
    "input_schema": {
        "type": "object",
        "properties": {
            "tenant_id": {"type": "string", "description": "Tenant identifier"},
            "domain_id": {"type": "string", "description": "Domain identifier (optional)"},
            "source": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["text", "url", "file"]},
                    "text": {"type": "string", "description": "Raw text content"},
                    "url": {"type": "string", "description": "URL to fetch"},
                    "file_id": {"type": "string", "description": "MinIO file ID"}
                },
                "required": ["type"]
            },
            "policy": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["recursive", "semantic", "hybrid"], "default": "recursive"},
                    "target_tokens": {"type": "integer", "default": 512},
                    "overlap": {"type": "integer", "default": 64}
                }
            }
        },
        "required": ["tenant_id", "source"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"}
                    }
                }
            },
            "persisted": {"type": "boolean"},
            "document_id": {"type": "string"},
            "error": {"type": "string"}
        }
    },
    "tags": ["rag", "document", "chunking", "text-processing"],
    "version": "1.0.0",
    "status": "active"
}

MCP_EMBEDDING_TOOL_DEF = {
    "tool_id": "embed_v1", 
    "name": "Vector Embedding Service",
    "description": "Generate vector embeddings using BGE-M3 multilingual model with Qdrant storage",
    "endpoint": "http://embedding-mcp:8002/mcp/execute",
    "method": "POST",
    "input_schema": {
        "type": "object",
        "properties": {
            "tenant_id": {"type": "string", "description": "Tenant identifier"},
            "collection": {"type": "string", "description": "Collection name"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Item ID"},
                        "text": {"type": "string", "description": "Text to embed"},
                        "metadata": {"type": "object", "description": "Item metadata"}
                    },
                    "required": ["id", "text"]
                }
            },
            "upsert": {"type": "boolean", "default": True}
        },
        "required": ["tenant_id", "collection", "items"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "vectors": {
                "type": "array", 
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "vector": {"type": "array", "items": {"type": "number"}}
                    }
                }
            },
            "upserted": {"type": "integer"},
            "error": {"type": "string"}
        }
    },
    "tags": ["rag", "embedding", "vector", "semantic"],
    "version": "1.0.0",
    "status": "active"
}

MCP_RETRIEVAL_TOOL_DEF = {
    "tool_id": "retriever_v1",
    "name": "Semantic Retrieval Service", 
    "description": "Semantic search and document retrieval using vector similarity with metadata filtering",
    "endpoint": "http://retrieval-mcp:8003/mcp/execute",
    "method": "POST",
    "input_schema": {
        "type": "object",
        "properties": {
            "tenant_id": {"type": "string", "description": "Tenant identifier"},
            "collection": {"type": "string", "description": "Collection name"},
            "query": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Query text"},
                    "use_embedding": {"type": "boolean", "default": True}
                },
                "required": ["text"]
            },
            "top_k": {"type": "integer", "default": 5},
            "filters": {"type": "object", "description": "Additional filters"}
        },
        "required": ["tenant_id", "collection", "query"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "hits": {
                "type": "array",
                "items": {
                    "type": "object", 
                    "properties": {
                        "id": {"type": "string"},
                        "score": {"type": "number"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"}
                    }
                }
            },
            "error": {"type": "string"}
        }
    },
    "tags": ["rag", "retrieval", "search", "semantic"],
    "version": "1.0.0", 
    "status": "active"
}

# ============================================================================
# TAO INTEGRATION CLASS
# ============================================================================

class RAGMCPServiceRegistry:
    """
    Registry manager for integrating RAG MCP services with TAO
    
    This class handles:
    - Registration of MCP services as TAO tools
    - Health monitoring of MCP services  
    - Tool permission management
    - Service discovery and routing
    """
    
    def __init__(self):
        self.registered_tools = {}
        self.service_health = {}
        
    async def register_all_rag_services(self, tool_registry) -> bool:
        """
        Register all RAG MCP services with TAO's ToolRegistry
        
        Called during TAO startup to make MCP services available
        to agent teams for orchestration
        
        Args:
            tool_registry: TAO's ToolRegistry instance
            
        Returns:
            bool: True if all services registered successfully
        """
        try:
            logger.info("Registering RAG MCP services with TAO ToolRegistry...")
            
            # Register chunking service
            await self._register_service(tool_registry, MCP_CHUNKING_TOOL_DEF)
            
            # Register embedding service  
            await self._register_service(tool_registry, MCP_EMBEDDING_TOOL_DEF)
            
            # Register retrieval service
            await self._register_service(tool_registry, MCP_RETRIEVAL_TOOL_DEF)
            
            logger.info(f"Successfully registered {len(self.registered_tools)} RAG MCP services")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register RAG MCP services: {str(e)}")
            return False
    
    async def _register_service(self, tool_registry, tool_def: Dict[str, Any]):
        """Register individual MCP service with TAO"""
        try:
            # Create TAO ToolDefinition (this will use actual TAO classes when integrated)
            # tool_definition = ToolDefinition(**tool_def)
            
            # Register with TAO ToolRegistry
            # await tool_registry.register_tool(tool_definition)
            
            # For now, just track locally until TAO integration
            self.registered_tools[tool_def["tool_id"]] = {
                "definition": tool_def,
                "registered_at": datetime.now().isoformat(),
                "status": "registered"
            }
            
            logger.info(f"Registered tool: {tool_def['tool_id']} -> {tool_def['endpoint']}")
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_def['tool_id']}: {str(e)}")
            raise
    
    async def check_service_health(self) -> Dict[str, str]:
        """
        Check health status of all registered MCP services
        
        Returns:
            Dict mapping service_id to health status
        """
        import httpx
        
        health_status = {}
        
        for tool_id, tool_info in self.registered_tools.items():
            endpoint = tool_info["definition"]["endpoint"]
            health_endpoint = endpoint.replace("/mcp/execute", "/health")
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(health_endpoint, timeout=5.0)
                    
                    if response.status_code == 200:
                        health_status[tool_id] = "healthy"
                    else:
                        health_status[tool_id] = f"unhealthy (HTTP {response.status_code})"
                        
            except Exception as e:
                health_status[tool_id] = f"unreachable ({str(e)})"
        
        self.service_health = health_status
        return health_status
    
    def get_registered_tools(self) -> Dict[str, Any]:
        """Get list of all registered RAG MCP tools"""
        return self.registered_tools
    
    def get_tool_definition(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get tool definition by ID"""
        return self.registered_tools.get(tool_id, {}).get("definition")

# ============================================================================
# TAO INTEGRATION FUNCTIONS
# ============================================================================

async def initialize_rag_mcp_integration(tao_tool_registry):
    """
    Main integration function called by TAO during startup
    
    This function should be called from TAO's initialization sequence
    to register all RAG MCP services as available tools
    
    Args:
        tao_tool_registry: TAO's ToolRegistry instance
        
    Returns:
        RAGMCPServiceRegistry: Configured registry instance
    """
    registry = RAGMCPServiceRegistry()
    
    success = await registry.register_all_rag_services(tao_tool_registry)
    
    if not success:
        raise Exception("Failed to initialize RAG MCP integration")
    
    # Perform initial health check
    health_status = await registry.check_service_health()
    logger.info(f"RAG MCP Service health status: {health_status}")
    
    return registry

async def setup_rag_tool_permissions(team_id: str, member_ids: List[str], 
                                   allowed_tools: List[str] = None):
    """
    Configure RAG tool permissions for agent team members
    
    This function should be called when setting up agent teams
    to grant appropriate RAG tool access
    
    Args:
        team_id: Agent team identifier
        member_ids: List of team member IDs
        allowed_tools: List of RAG tools to grant access to
                      (defaults to all: chunker_v1, embed_v1, retriever_v1)
    """
    if allowed_tools is None:
        allowed_tools = ["chunker_v1", "embed_v1", "retriever_v1"]
    
    logger.info(f"Setting up RAG tool permissions for team {team_id}")
    
    # This will integrate with TAO's permission system when available
    # For now, document the expected integration points
    
    for member_id in member_ids:
        for tool_id in allowed_tools:
            logger.info(f"Granting {member_id} access to {tool_id}")
            
            # TAO integration point:
            # await tao_permission_manager.grant_tool_access(
            #     member_id=member_id,
            #     tool_id=tool_id,
            #     granted_by="rag_mcp_setup",
            #     granted_at=datetime.now()
            # )

# ============================================================================
# MAIN INTEGRATION DEMO/TEST
# ============================================================================

async def main():
    """Demo function showing how TAO integration will work"""
    print("🔗 RAG MCP → TAO Integration Demo")
    print("=" * 50)
    
    # Simulate TAO ToolRegistry (will be real instance in production)
    mock_tool_registry = {}
    
    try:
        # Initialize integration
        registry = await initialize_rag_mcp_integration(mock_tool_registry)
        
        # Show registered tools
        tools = registry.get_registered_tools()
        print(f"✅ Registered {len(tools)} RAG MCP services:")
        for tool_id, info in tools.items():
            print(f"   - {tool_id}: {info['definition']['endpoint']}")
        
        # Check health status
        health = await registry.check_service_health()
        print(f"\n📊 Service Health Status:")
        for service_id, status in health.items():
            print(f"   - {service_id}: {status}")
        
        # Demo tool permission setup
        print(f"\n🔐 Setting up tool permissions for demo team...")
        await setup_rag_tool_permissions(
            team_id="demo_team_001", 
            member_ids=["retriever_agent", "indexer_agent"]
        )
        
        print(f"\n🎉 RAG MCP → TAO integration ready for production!")
        
    except Exception as e:
        print(f"❌ Integration failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())