"""
FastAPI Backend Application for TAB_MCP_Client
==============================================
Main API server for the TAB MCP Client chatbot interface
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .models import (
    TenantConfig, DomainKnowledgeConfig, DocumentUpload,
    QueryRequest, QueryResponse, AgentTeamConfig
)
from .services import (
    TenantService, DomainService, DocumentService,
    QueryService, AgentTeamService
)
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="TAB MCP Client API",
    description="API for managing tenant configurations, domain knowledge, documents, and queries",
    version="1.0.0",
    debug=config.API_DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SERVICE INSTANCES
# ============================================================================

tenant_service = TenantService()
domain_service = DomainService()
document_service = DocumentService()
query_service = QueryService()
agent_team_service = AgentTeamService()

# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting TAB MCP Client API...")
    
    # Initialize all services
    await tenant_service.initialize()
    await domain_service.initialize()
    await document_service.initialize()
    await query_service.initialize()
    await agent_team_service.initialize()
    
    logger.info("All services initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down TAB MCP Client API...")
    
    # Close all connections
    await tenant_service.close()
    await domain_service.close()
    await document_service.close()
    await query_service.close()
    await agent_team_service.close()
    
    logger.info("All services shut down successfully")

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # TODO: Check database connections
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# TENANT CONFIGURATION ENDPOINTS
# ============================================================================

@app.post("/api/tenant/upload")
async def upload_tenant_config(
    file: UploadFile = File(...),
    created_by: str = Form(...)
):
    """Upload and process tenant configuration YAML file"""
    try:
        # Read YAML content
        content = await file.read()
        yaml_content = content.decode('utf-8')
        
        # Process tenant configuration
        tenant_config, is_new = await tenant_service.create_or_update_tenant(yaml_content, created_by)
        
        return {
            "success": True,
            "message": f"Tenant configuration {'created' if is_new else 'updated'} successfully",
            "tenant_id": tenant_config.tenant_id,
            "tenant_name": tenant_config.name,
            "version": tenant_config.current_version.version_number if tenant_config.current_version else 1
        }
        
    except Exception as e:
        logger.error(f"Error uploading tenant config: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/tenant/{tenant_id}")
async def get_tenant_config(tenant_id: str):
    """Get tenant configuration by ID"""
    try:
        tenant_config = await tenant_service.get_tenant(tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")
        
        return tenant_config.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tenant config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DOMAIN KNOWLEDGE CONFIGURATION ENDPOINTS
# ============================================================================

@app.post("/api/domain/upload")
async def upload_domain_config(
    file: UploadFile = File(...),
    created_by: str = Form(...)
):
    """Upload and process domain knowledge configuration YAML file"""
    try:
        # Read YAML content
        content = await file.read()
        yaml_content = content.decode('utf-8')
        
        # Process domain configuration
        domain_config, is_new = await domain_service.create_or_update_domain(yaml_content, created_by)
        
        return {
            "success": True,
            "message": f"Domain knowledge configuration {'created' if is_new else 'updated'} successfully",
            "domain_id": domain_config.domain_id,
            "domain_name": domain_config.name,
            "tenant_id": domain_config.tenant_id,
            "version": domain_config.current_version.version_number if domain_config.current_version else 1
        }
        
    except Exception as e:
        logger.error(f"Error uploading domain config: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/domain/{domain_id}")
async def get_domain_config(domain_id: str):
    """Get domain configuration by ID"""
    try:
        domain_config = await domain_service.get_domain(domain_id)
        if not domain_config:
            raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")
        
        return domain_config.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting domain config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DOCUMENT UPLOAD ENDPOINTS
# ============================================================================

@app.post("/api/document/upload")
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    domain_id: str = Form(...),
    title: str = Form(...),
    uploaded_by: str = Form(...),
    knowledge_entities: str = Form(default="[]"),  # JSON array as string
    main_topics: str = Form(default="[]"),  # JSON array as string
    document_tags: str = Form(default="{}"),  # JSON object as string
    chunking_method: str = Form(default="auto"),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50)
):
    """Upload and process a document"""
    try:
        # Read file content
        content = await file.read()
        
        # Parse JSON fields
        knowledge_entities_list = json.loads(knowledge_entities) if knowledge_entities else []
        main_topics_list = json.loads(main_topics) if main_topics else []
        document_tags_dict = json.loads(document_tags) if document_tags else {}
        
        # Create document upload model
        document_upload = DocumentUpload(
            tenant_id=tenant_id,
            domain_id=domain_id,
            title=title,
            filename=file.filename,
            content_type=file.content_type,
            content=content,
            uploaded_by=uploaded_by,
            knowledge_entities=knowledge_entities_list,
            main_topics=main_topics_list,
            document_tags=document_tags_dict,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Process document
        result = await document_service.upload_and_process_document(document_upload)
        
        return {
            "success": True,
            "message": "Document uploaded and processed successfully",
            **result
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# QUERY ENDPOINTS
# ============================================================================

@app.post("/api/query", response_model=QueryResponse)
async def process_query(query_request: QueryRequest):
    """Process a query using the RAG Agent Team"""
    try:
        response = await query_service.process_query(query_request)
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AGENT TEAM CONFIGURATION ENDPOINTS
# ============================================================================

@app.post("/api/team/upload")
async def upload_team_config(
    file: UploadFile = File(...),
    created_by: str = Form(...)
):
    """Upload and process agent team configuration YAML file"""
    try:
        # Read YAML content
        content = await file.read()
        yaml_content = content.decode('utf-8')
        
        # Process team configuration
        team_config, is_new = await agent_team_service.create_or_update_team(yaml_content, created_by)
        
        return {
            "success": True,
            "message": f"Agent team configuration {'created' if is_new else 'updated'} successfully",
            "team_id": team_config.team_id,
            "team_name": team_config.name,
            "tenant_id": team_config.tenant_id
        }
        
    except Exception as e:
        logger.error(f"Error uploading team config: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# CHATBOT INTERFACE ENDPOINTS
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message model"""
    message: str
    tenant_id: str
    domain_id: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/api/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """
    Main chat endpoint for the TAB MCP Client chatbot
    Handles both configuration commands and queries
    """
    try:
        message = chat_message.message.lower().strip()
        
        # Parse commands
        if message.startswith("/upload tenant"):
            return {
                "response": "Please upload your tenant configuration YAML file using the upload button.",
                "action": "upload_tenant"
            }
        
        elif message.startswith("/upload domain"):
            return {
                "response": "Please upload your domain knowledge configuration YAML file using the upload button.",
                "action": "upload_domain"
            }
        
        elif message.startswith("/upload document"):
            return {
                "response": "Please upload your document using the upload button and specify the domain.",
                "action": "upload_document"
            }
        
        elif message.startswith("/status"):
            # Get system status
            return {
                "response": "System is operational. All services are running.",
                "status": {
                    "postgres": "connected",
                    "neo4j": "connected",
                    "minio": "connected",
                    "qdrant": "connected",
                    "mcp_services": "available"
                }
            }
        
        else:
            # Process as a regular query
            query_request = QueryRequest(
                query=chat_message.message,
                tenant_id=chat_message.tenant_id,
                domain_id=chat_message.domain_id,
                session_id=chat_message.session_id
            )
            
            response = await query_service.process_query(query_request)
            
            return {
                "response": response.response_text,
                "sources": response.sources if response.sources else None,
                "metadata": response.metadata
            }
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return {
            "response": f"I encountered an error processing your request: {str(e)}",
            "error": True
        }

# ============================================================================
# STATIC FILES (for UI)
# ============================================================================

# Mount static files directory for the frontend
from pathlib import Path
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD
    )
