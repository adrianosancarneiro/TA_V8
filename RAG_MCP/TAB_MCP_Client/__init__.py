"""
TAB_MCP_Client - MCP Client for Team Agent Builder
==================================================
A simplified MVC chatbot client for the RAG_MCP system that enables:
1. Tenant configuration management (YAML upload → JSON → PostgreSQL versioning → Neo4j)
2. Domain knowledge configuration (YAML upload → JSON → PostgreSQL versioning → Neo4j)
3. Document upload and processing (MinIO → Chunking → Embedding → Qdrant)
4. Query processing using the RAG Agent Team

This client provides a web-based interface for interacting with the MCP services
and managing the complete tenant/domain/document lifecycle.
"""

__version__ = "1.0.0"
__author__ = "TA_V8 Team"

from .backend.app import app
from .backend.models import TenantConfig, DomainKnowledgeConfig, DocumentUpload
from .backend.services import TenantService, DomainService, DocumentService, QueryService

__all__ = [
    "app",
    "TenantConfig",
    "DomainKnowledgeConfig",
    "DocumentUpload",
    "TenantService",
    "DomainService", 
    "DocumentService",
    "QueryService"
]
