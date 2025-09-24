# TA_V8/RAG_MCP/shared/base.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class MCPRequest(BaseModel):
    """Base MCP request compatible with TAO"""
    action: str = Field(default="execute")
    tenant_id: str
    domain_id: Optional[str] = None
    params: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = {}

class MCPResponse(BaseModel):
    """Base MCP response compatible with TAO"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class ChunkRequest(BaseModel):
    tenant_id: str
    domain_id: str
    source: Dict[str, Any]
    policy: Dict[str, Any] = {
        "method": "recursive",
        "target_tokens": 512,
        "overlap": 64
    }

class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    persisted: bool

class EmbedRequest(BaseModel):
    tenant_id: str
    collection: str
    items: List[Dict[str, Any]]
    upsert: bool = True

class EmbedResponse(BaseModel):
    vectors: List[Dict[str, Any]]
    upserted: int

class RetrieveRequest(BaseModel):
    tenant_id: str
    collection: str
    query: Dict[str, Any]
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = {}

class RetrieveResponse(BaseModel):
    hits: List[Dict[str, Any]]
