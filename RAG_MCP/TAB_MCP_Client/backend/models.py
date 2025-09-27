"""
Models for TAB_MCP_Client
=========================
Defines data models for tenant configs, domain knowledge, and document processing
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import yaml
import json
import hashlib

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ConfigStatus(str, Enum):
    """Status of configuration files"""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class DomainType(str, Enum):
    """Types of domains in the knowledge system"""
    APPLICATION = "Application"
    BUSINESS_AREA = "BusinessArea"
    DEPARTMENT = "Department"
    PRODUCT_LINE = "ProductLine"
    SERVICE = "Service"
    CUSTOM = "Custom"

class StalenesLevel(str, Enum):
    """Staleness levels for knowledge and documents"""
    DEPRECATED = "deprecated"
    LEGACY_CONTEXT = "legacy_context"
    CURRENT = "current"
    IN_DEVELOPMENT = "in_development"

# ============================================================================
# TENANT CONFIGURATION MODELS
# ============================================================================

class TenantConfigVersion(BaseModel):
    """Version tracking for tenant configurations"""
    version_id: str = Field(..., description="Unique version identifier")
    version_number: int = Field(..., description="Sequential version number")
    config_hash: str = Field(..., description="Hash of configuration content")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="User who created this version")
    change_summary: Optional[str] = Field(None, description="Summary of changes")
    is_active: bool = Field(True, description="Whether this version is active")

class TenantConfig(BaseModel):
    """Complete tenant configuration model"""
    tenant_id: str = Field(..., description="Unique tenant identifier")
    name: str = Field(..., description="Tenant name")
    display_name: str = Field(..., description="Display name for UI")
    industry: Optional[str] = Field(None, description="Industry sector")
    subscription_tier: str = Field(default="standard", description="Subscription level")
    
    # Organization details
    headquarters_location: Optional[str] = None
    founding_year: Optional[int] = None
    employee_count: Optional[int] = None
    annual_revenue: Optional[float] = None
    currency: str = Field(default="USD")
    
    # Compliance & Governance
    compliance_frameworks: List[str] = Field(default_factory=list)
    data_residency: str = Field(default="US")
    
    # Technical Configuration
    core_tech_stack: List[str] = Field(default_factory=list)
    api_rate_limit: int = Field(default=1000)
    storage_quota_gb: int = Field(default=100)
    max_users: int = Field(default=50)
    
    # Settings
    default_embedding_model: str = Field(default="multilingual-e5-large")
    default_chunking_method: str = Field(default="auto")
    default_chunk_size: int = Field(default=500)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: ConfigStatus = Field(default=ConfigStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Version tracking
    versions: List[TenantConfigVersion] = Field(default_factory=list)
    current_version: Optional[TenantConfigVersion] = None
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> "TenantConfig":
        """Create TenantConfig from YAML content"""
        data = yaml.safe_load(yaml_content)
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string for PostgreSQL storage"""
        return json.dumps(self.model_dump(), default=str)
    
    def calculate_hash(self) -> str:
        """Calculate hash of configuration for version tracking"""
        config_str = json.dumps(self.model_dump(exclude={"versions", "current_version"}), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def create_version(self, created_by: str, change_summary: Optional[str] = None) -> TenantConfigVersion:
        """Create a new version of the configuration"""
        version_number = len(self.versions) + 1
        version = TenantConfigVersion(
            version_id=f"{self.tenant_id}_v{version_number}",
            version_number=version_number,
            config_hash=self.calculate_hash(),
            created_by=created_by,
            change_summary=change_summary
        )
        self.versions.append(version)
        self.current_version = version
        return version

# ============================================================================
# DOMAIN KNOWLEDGE CONFIGURATION MODELS
# ============================================================================

class KnowledgeEntity(BaseModel):
    """Knowledge entity within a domain"""
    entity_id: str = Field(..., description="Unique entity identifier")
    entity_type: str = Field(..., description="Type of entity (Feature, Module, Process, etc.)")
    entity_subtype: Optional[str] = None
    name: str = Field(..., description="Entity name")
    display_name: str = Field(..., description="Display name")
    description: Optional[str] = None
    
    # Hierarchical structure
    path: Optional[str] = None
    hierarchy_level: int = Field(default=0)
    parent_entity_id: Optional[str] = None
    
    # Properties
    properties: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Staleness tracking
    staleness: StalenesLevel = Field(default=StalenesLevel.CURRENT)
    staleness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    last_validated: datetime = Field(default_factory=datetime.utcnow)

class MetadataTemplate(BaseModel):
    """Metadata template for domain knowledge"""
    features: List[str] = Field(default_factory=list)
    content_types: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    staleness_levels: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)

class DomainKnowledgeConfigVersion(BaseModel):
    """Version tracking for domain knowledge configurations"""
    version_id: str
    version_number: int
    config_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    change_summary: Optional[str] = None
    is_active: bool = Field(default=True)

class DomainKnowledgeConfig(BaseModel):
    """Complete domain knowledge configuration model"""
    domain_id: str = Field(..., description="Unique domain identifier")
    tenant_id: str = Field(..., description="Associated tenant ID")
    name: str = Field(..., description="Domain name")
    display_name: str = Field(..., description="Display name")
    
    # Domain classification
    domain_type: DomainType = Field(..., description="Type of domain")
    domain_category: Optional[str] = None
    
    # Hierarchical structure
    path: Optional[str] = None
    level: int = Field(default=0)
    parent_domain_id: Optional[str] = None
    
    # Domain-specific attributes
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    # Knowledge configuration
    knowledge_sources: List[str] = Field(default_factory=list)
    indexing_frequency: str = Field(default="daily")
    retention_days: int = Field(default=365)
    
    # Metadata template
    metadata_template: Optional[MetadataTemplate] = None
    
    # Knowledge entities
    knowledge_entities: List[KnowledgeEntity] = Field(default_factory=list)
    
    # Metadata
    status: ConfigStatus = Field(default=ConfigStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Version tracking
    versions: List[DomainKnowledgeConfigVersion] = Field(default_factory=list)
    current_version: Optional[DomainKnowledgeConfigVersion] = None
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> "DomainKnowledgeConfig":
        """Create DomainKnowledgeConfig from YAML content"""
        data = yaml.safe_load(yaml_content)
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string for PostgreSQL storage"""
        return json.dumps(self.model_dump(), default=str)
    
    def calculate_hash(self) -> str:
        """Calculate hash of configuration for version tracking"""
        config_str = json.dumps(self.model_dump(exclude={"versions", "current_version"}), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def create_version(self, created_by: str, change_summary: Optional[str] = None) -> DomainKnowledgeConfigVersion:
        """Create a new version of the configuration"""
        version_number = len(self.versions) + 1
        version = DomainKnowledgeConfigVersion(
            version_id=f"{self.domain_id}_v{version_number}",
            version_number=version_number,
            config_hash=self.calculate_hash(),
            created_by=created_by,
            change_summary=change_summary
        )
        self.versions.append(version)
        self.current_version = version
        return version

# ============================================================================
# DOCUMENT PROCESSING MODELS
# ============================================================================

class DocumentUpload(BaseModel):
    """Document upload request model"""
    document_id: Optional[str] = None
    tenant_id: str = Field(..., description="Tenant ID for document")
    domain_id: str = Field(..., description="Domain ID for document context")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of document")
    content: Optional[bytes] = Field(None, description="Document content (for small files)")
    file_path: Optional[str] = Field(None, description="Path to file (for large files)")
    
    # Processing options
    chunking_method: str = Field(default="auto")
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    
    # Metadata
    document_tags: Dict[str, Any] = Field(default_factory=dict)
    main_topics: List[str] = Field(default_factory=list)
    knowledge_entities: List[str] = Field(default_factory=list, description="Related knowledge entity IDs")
    
    # User tracking
    uploaded_by: str = Field(..., description="User who uploaded the document")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

class ChunkMetadata(BaseModel):
    """Metadata for document chunks"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    chunk_index: int = Field(..., description="Sequential index in document")
    
    # Content
    chunk_text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., description="Number of tokens")
    
    # Semantic tags (generated by LLM)
    primary_topic: Optional[str] = None
    subtopics: List[str] = Field(default_factory=list)
    technical_concepts: List[str] = Field(default_factory=list)
    business_concepts: List[str] = Field(default_factory=list)
    
    # Knowledge references
    related_features: List[str] = Field(default_factory=list)
    related_entities: List[str] = Field(default_factory=list)
    
    # Context metadata
    context_type: Optional[str] = None
    importance: Optional[str] = None
    complexity: Optional[str] = None
    
    # Quality metrics
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)

# ============================================================================
# QUERY AND RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Query request model for RAG Agent Team"""
    query: str = Field(..., description="User query")
    tenant_id: str = Field(..., description="Tenant ID for context")
    domain_id: Optional[str] = Field(None, description="Optional domain filter")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    max_results: int = Field(default=5, description="Maximum results to retrieve")
    include_sources: bool = Field(default=True, description="Include source citations")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")

class QueryResponse(BaseModel):
    """Response model for queries"""
    query_id: str = Field(..., description="Unique query identifier")
    response_text: str = Field(..., description="Generated response")
    chunks_retrieved: List[str] = Field(default_factory=list, description="Chunk IDs retrieved")
    relevance_scores: List[float] = Field(default_factory=list, description="Relevance scores for chunks")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time_ms: int = Field(..., description="Total processing time")

# ============================================================================
# AGENT TEAM CONFIGURATION MODELS
# ============================================================================

class AgentConfig(BaseModel):
    """Configuration for an individual agent"""
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role")
    persona: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    assigned_tools: List[str] = Field(default_factory=list)
    llm_config: Dict[str, Any] = Field(default_factory=dict)

class AgentTeamConfig(BaseModel):
    """Configuration for an agent team"""
    team_id: str = Field(..., description="Unique team identifier")
    tenant_id: str = Field(..., description="Associated tenant ID")
    name: str = Field(..., description="Team name")
    description: Optional[str] = None
    
    # Team composition
    team_type: str = Field(default="analytical")
    agents: List[AgentConfig] = Field(default_factory=list)
    
    # Domain associations
    domain_focus: List[str] = Field(default_factory=list, description="Domain IDs this team focuses on")
    
    # Configuration
    max_agents: int = Field(default=5)
    orchestration_mode: str = Field(default="sequential")
    timeout_seconds: int = Field(default=300)
    retry_policy: str = Field(default="exponential_backoff")
    
    # Metadata
    status: ConfigStatus = Field(default=ConfigStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> "AgentTeamConfig":
        """Create AgentTeamConfig from YAML content"""
        data = yaml.safe_load(yaml_content)
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string for storage"""
        return json.dumps(self.model_dump(), default=str)
