"""
Configuration for TAB_MCP_Client
================================
Environment configuration and settings
"""

import os
from pathlib import Path
from typing import Optional

class Config:
    """Configuration settings for TAB_MCP_Client"""
    
    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.UPLOAD_DIR = self.DATA_DIR / "uploads"
        self.LOG_DIR = self.DATA_DIR / "logs"
        
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        
        # PostgreSQL configuration - Using local services (no Docker)
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "ta_v8")
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres_user")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres_pass")
        
        # Neo4j configuration - Using local services (no Docker)
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "pJnssz3khcLtn6T")
        
        # MinIO configuration - Using local services (no Docker)
        self.MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
        self.MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ta-v8-documents")
        
        # Qdrant configuration - Using local services (no Docker)
        self.QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
        self.QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
        self.QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ta_v8_embeddings")
        
        # MCP Transport Configuration - Using HTTP + SSE for better compatibility
        self.MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "http")
        
        # MCP Services - Updated for HTTP + SSE transport
        if self.MCP_TRANSPORT == "http":
            # HTTP + SSE transport URLs with updated ports
            self.MCP_CHUNKING_URL = os.getenv("CHUNKING_MCP_URL", "http://localhost:8001")
            self.MCP_EMBEDDING_URL = os.getenv("EMBEDDING_MCP_URL", "http://localhost:8004")
            self.MCP_RETRIEVAL_URL = os.getenv("RETRIEVAL_MCP_URL", "http://localhost:8003")
        else:
            # For stdio transport (fallback), we don't use URLs but service commands
            self.MCP_CHUNKING_CMD = os.getenv("CHUNKING_MCP_CMD", "systemctl --user status chunking-mcp")
            self.MCP_EMBEDDING_CMD = os.getenv("EMBEDDING_MCP_CMD", "systemctl --user status embedding-mcp")
            self.MCP_RETRIEVAL_CMD = os.getenv("RETRIEVAL_MCP_CMD", "systemctl --user status retrieval-mcp")
        
        # RAG Agent Team - Using HTTP transport
        self.RAG_AGENT_TEAM_URL = os.getenv("RAG_AGENT_TEAM_URL", "http://localhost:8006")
        
        # Default processing settings
        self.DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "multilingual-e5-large")
        self.DEFAULT_CHUNKING_METHOD = os.getenv("DEFAULT_CHUNKING_METHOD", "auto")
        self.DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
        self.DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
        
        # API settings
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8005"))
        self.API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
        self.API_DEBUG = os.getenv("API_DEBUG", "true").lower() == "true"
        
        # Security settings - Using production secrets
        self.SECRET_KEY = os.getenv("API_SECRET_KEY", "ta_v8_secure_api_key_prod_2024_xK9mP2vB8nQ4")
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
        
        # Logging settings - Using production configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Session settings
        self.SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
        
        # Rate limiting
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    
    def get_postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    def get_neo4j_url(self) -> str:
        """Get Neo4j connection URL without credentials"""
        return self.NEO4J_URI
    
    def get_minio_url(self) -> str:
        """Get MinIO URL"""
        protocol = "https" if self.MINIO_SECURE else "http"
        return f"{protocol}://{self.MINIO_ENDPOINT}"
    
    def get_qdrant_url(self) -> str:
        """Get Qdrant URL"""
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

# Global config instance
config = Config()