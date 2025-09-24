"""
Environment configuration loader for TA_V8 RAG MCP
Loads secrets from secure location and provides environment variables
"""
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Secure secrets file location
SECRETS_FILE = Path("/etc/TA_V8/RAG_MCP/secrets.env")

def load_secrets() -> dict[str, str]:
    """
    Load secrets from the secure secrets.env file
    Returns a dictionary of environment variables
    """
    secrets = {}
    
    if not SECRETS_FILE.exists():
        logger.warning(f"Secrets file not found at {SECRETS_FILE}")
        return secrets
    
    try:
        with open(SECRETS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        secrets[key.strip()] = value.strip()
        
        logger.info(f"Loaded {len(secrets)} secrets from {SECRETS_FILE}")
        return secrets
        
    except Exception as e:
        logger.error(f"Failed to load secrets from {SECRETS_FILE}: {e}")
        return {}

def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with fallback to secrets file
    Priority: 1. Environment variable, 2. Secrets file, 3. Default value
    """
    # First try environment variable
    value = os.getenv(key)
    if value is not None:
        return value
    
    # Then try secrets file
    secrets = load_secrets()
    value = secrets.get(key)
    if value is not None:
        return value
    
    # Finally use default
    return default

def setup_environment():
    """
    Load all secrets and set them as environment variables
    This should be called at application startup
    """
    secrets = load_secrets()
    
    for key, value in secrets.items():
        # Only set if not already in environment (env vars take priority)
        if key not in os.environ:
            os.environ[key] = value
    
    logger.info("Environment setup completed")

# Configuration with secure defaults
class Config:
    """Application configuration with secure defaults"""
    
    def __init__(self):
        # Load secrets on initialization
        setup_environment()
    
    # Database Configuration
    POSTGRES_HOST = get_env_var("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(get_env_var("POSTGRES_PORT", "5432"))
    POSTGRES_USER = get_env_var("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = get_env_var("POSTGRES_PASSWORD", "")
    POSTGRES_DB = get_env_var("POSTGRES_DB", "ta_v8")
    
    # Vector Database Configuration
    QDRANT_HOST = get_env_var("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(get_env_var("QDRANT_PORT", "6333"))
    QDRANT_API_KEY = get_env_var("QDRANT_API_KEY", "")
    
    # Embedding Service Configuration
    EMBEDDING_URL = get_env_var("EMBEDDING_URL", "http://localhost:8080")
    EMBEDDING_API_KEY = get_env_var("EMBEDDING_API_KEY", "")
    
    # LLM Configuration
    OLLAMA_URL = get_env_var("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_API_KEY = get_env_var("OLLAMA_API_KEY", "")
    
    # MCP Configuration
    MCP_SECRET_KEY = get_env_var("MCP_SECRET_KEY", "")
    MCP_JWT_SECRET = get_env_var("MCP_JWT_SECRET", "")
    
    # Security Configuration
    API_SECRET_KEY = get_env_var("API_SECRET_KEY", "")
    ENCRYPTION_KEY = get_env_var("ENCRYPTION_KEY", "")
    
    # Application Configuration
    ENVIRONMENT = get_env_var("ENVIRONMENT", "development")
    LOG_LEVEL = get_env_var("LOG_LEVEL", "INFO")
    DEBUG = get_env_var("DEBUG", "false").lower() == "true"
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct PostgreSQL URL from components"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property 
    def QDRANT_URL(self) -> str:
        """Construct Qdrant URL from components"""
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

# Global configuration instance
config = Config()