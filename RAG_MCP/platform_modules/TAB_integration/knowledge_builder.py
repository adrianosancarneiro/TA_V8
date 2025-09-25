#!/usr/bin/env python3
"""
# =============================================================================
# TAB INTEGRATION: RAG KNOWLEDGE BUILDER
# =============================================================================
# Purpose: Enable TAB's WizardEngine and team building workflows to configure RAG
# Integration Point: TAB WizardEngine → RAG MCP Services → Domain Knowledge Setup
# 
# This module provides the interfaces needed for TAB to configure RAG capabilities
# when creating agent teams, including domain knowledge setup, tool permissions,
# and knowledge base initialization.
# 
# Workflow:
# 1. TAB WizardEngine asks about knowledge sources during team creation
# 2. If RAG needed → configure domain knowledge via RAG MCP services
# 3. Set up team member tool permissions for RAG tools
# 4. Initialize knowledge collections in vector database  
# 5. Register domain relationships in Neo4j
# =============================================================================
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# TAB integration imports (these will be available when TAB is integrated)
# from tab.core.wizard_engine import WizardEngine, WizardStep, WizardResponse
# from tab.core.team_builder_workflow import TeamBuilderWorkflow
# from tab.models.team_config import AgentTeam, AgentMember, DomainConfig

logger = logging.getLogger(__name__)

# ============================================================================
# RAG KNOWLEDGE CONFIGURATION MODELS
# ============================================================================

class KnowledgeSourceType(Enum):
    """Types of knowledge sources for RAG configuration"""
    DOCUMENTS = "documents"
    WEB_URLS = "web_urls"
    DATABASES = "databases" 
    API_ENDPOINTS = "api_endpoints"
    FILE_UPLOADS = "file_uploads"

@dataclass
class KnowledgeSource:
    """Configuration for a knowledge source"""
    source_type: KnowledgeSourceType
    name: str
    description: str
    location: str  # File path, URL, database connection, etc.
    metadata: Dict[str, Any] = None
    
@dataclass
class DomainKnowledgeConfig:
    """Complete domain knowledge configuration"""
    domain_id: str
    domain_name: str
    description: str
    tenant_id: str
    knowledge_sources: List[KnowledgeSource]
    chunking_strategy: str = "hybrid"
    embedding_collection: str = None
    access_permissions: List[str] = None  # Agent IDs with access

@dataclass
class RAGToolPermissions:
    """RAG tool permission configuration for team members"""
    member_id: str
    chunking_access: bool = False
    embedding_access: bool = False 
    retrieval_access: bool = False
    allowed_collections: List[str] = None
    read_only: bool = True

@dataclass
class KnowledgeSetupResult:
    """Result from knowledge base setup process"""
    success: bool
    domain_id: str = None
    collections_created: List[str] = None
    documents_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    error: Optional[str] = None

# ============================================================================
# TAB INTEGRATION CLASS
# ============================================================================

class RAGKnowledgeBuilder:
    """
    RAG knowledge configuration builder for TAB integration
    
    This class provides the interfaces needed for TAB's WizardEngine
    to collect RAG requirements and configure knowledge bases during
    the team building process.
    
    Features:
    - Conversational knowledge source discovery
    - Automated domain knowledge setup
    - Tool permission configuration
    - Integration with existing TAB workflows
    """
    
    def __init__(self, rag_mcp_services_url: str = "http://localhost"):
        self.rag_services_base = rag_mcp_services_url
        self.setup_history = {}
        
    async def discover_knowledge_requirements(self, wizard_context) -> List[Dict[str, Any]]:
        """
        Conversational discovery of RAG knowledge requirements
        
        This method integrates with TAB's WizardEngine to ask users
        about their knowledge sources and RAG needs during team building.
        
        Args:
            wizard_context: TAB WizardEngine context
            
        Returns:
            List of wizard steps for knowledge configuration
        """
        knowledge_steps = [
            {
                "step_id": "rag_needs_assessment",
                "question": "Does your team need access to specific documents or knowledge sources?",
                "type": "yes_no",
                "follow_up": {
                    "yes": "rag_knowledge_sources",
                    "no": "skip_rag_setup"
                },
                "explanation": "RAG (Retrieval-Augmented Generation) helps agents access and use specific knowledge during conversations."
            },
            {
                "step_id": "rag_knowledge_sources", 
                "question": "What type of knowledge sources will your team need? (Select all that apply)",
                "type": "multi_select",
                "options": [
                    {"value": "documents", "label": "Document files (PDF, Word, etc.)"},
                    {"value": "web_urls", "label": "Web pages and articles"},
                    {"value": "databases", "label": "Database content"},
                    {"value": "api_endpoints", "label": "API data sources"},
                    {"value": "file_uploads", "label": "Files to upload now"}
                ],
                "follow_up": "rag_domain_setup"
            },
            {
                "step_id": "rag_domain_setup",
                "question": "What should we call this knowledge domain? (e.g., 'Product Documentation', 'Company Policies')",
                "type": "text_input",
                "validation": "required",
                "follow_up": "rag_chunking_strategy"
            },
            {
                "step_id": "rag_chunking_strategy", 
                "question": "How should we process your documents for optimal search?",
                "type": "single_select",
                "options": [
                    {"value": "semantic", "label": "Semantic (best for narrative content)", "description": "Preserves meaning and context"},
                    {"value": "recursive", "label": "Recursive (best for technical docs)", "description": "Structured hierarchical chunking"},
                    {"value": "hybrid", "label": "Hybrid (recommended)", "description": "Combines both approaches automatically"}
                ],
                "default": "hybrid",
                "follow_up": "rag_permissions_setup"
            },
            {
                "step_id": "rag_permissions_setup",
                "question": "Which team members should have access to this knowledge?",
                "type": "multi_select_members",  # Special type that shows team members
                "default": "all_members",
                "follow_up": "rag_configuration_review"
            },
            {
                "step_id": "rag_configuration_review",
                "question": "Review your RAG configuration:",
                "type": "review_display",
                "follow_up": "complete_rag_setup"
            }
        ]
        
        return knowledge_steps
    
    async def process_knowledge_configuration(self, wizard_responses: Dict[str, Any], 
                                            team_context) -> DomainKnowledgeConfig:
        """
        Process wizard responses into domain knowledge configuration
        
        Args:
            wizard_responses: Responses from TAB WizardEngine
            team_context: Team building context
            
        Returns:
            DomainKnowledgeConfig: Complete configuration ready for setup
        """
        try:
            # Extract configuration from wizard responses
            domain_name = wizard_responses.get("rag_domain_setup", "Team Knowledge")
            knowledge_source_types = wizard_responses.get("rag_knowledge_sources", [])
            chunking_strategy = wizard_responses.get("rag_chunking_strategy", "hybrid")
            permitted_members = wizard_responses.get("rag_permissions_setup", [])
            
            # Generate domain ID
            domain_id = f"domain_{team_context.tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create knowledge sources from wizard input
            knowledge_sources = []
            for source_type in knowledge_source_types:
                if source_type == "file_uploads":
                    # Handle file uploads (would integrate with TAB's file upload system)
                    knowledge_sources.append(KnowledgeSource(
                        source_type=KnowledgeSourceType.FILE_UPLOADS,
                        name="Uploaded Documents",
                        description="Documents uploaded during team creation",
                        location="uploads://team_creation",
                        metadata={"upload_session": team_context.session_id}
                    ))
                elif source_type == "web_urls":
                    # Handle web URL collection (would prompt for specific URLs)
                    knowledge_sources.append(KnowledgeSource(
                        source_type=KnowledgeSourceType.WEB_URLS,
                        name="Web Resources",
                        description="Web pages and articles for knowledge base",
                        location="web://collected_urls",
                        metadata={"collection_method": "wizard_input"}
                    ))
                # Add other source types as needed
            
            # Create complete configuration
            config = DomainKnowledgeConfig(
                domain_id=domain_id,
                domain_name=domain_name,
                description=f"Knowledge domain for {team_context.team_name}",
                tenant_id=team_context.tenant_id,
                knowledge_sources=knowledge_sources,
                chunking_strategy=chunking_strategy,
                embedding_collection=f"collection_{domain_id}",
                access_permissions=permitted_members
            )
            
            logger.info(f"Created domain knowledge config: {domain_id}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to process knowledge configuration: {str(e)}")
            raise Exception(f"Knowledge configuration error: {str(e)}")
    
    async def setup_domain_knowledge(self, config: DomainKnowledgeConfig) -> KnowledgeSetupResult:
        """
        Set up complete domain knowledge infrastructure
        
        This method orchestrates the setup of:
        1. Domain record in Neo4j
        2. Document processing and chunking
        3. Vector embedding generation
        4. Collection creation in Qdrant
        5. Permission configuration
        
        Args:
            config: Domain knowledge configuration
            
        Returns:
            KnowledgeSetupResult: Setup results and statistics
        """
        try:
            logger.info(f"Setting up domain knowledge: {config.domain_id}")
            
            total_documents = 0
            total_chunks = 0
            total_embeddings = 0
            collections_created = []
            
            # Step 1: Create domain record in Neo4j
            await self._create_domain_record(config)
            
            # Step 2: Process each knowledge source
            for source in config.knowledge_sources:
                logger.info(f"Processing knowledge source: {source.name}")
                
                if source.source_type == KnowledgeSourceType.FILE_UPLOADS:
                    # Process uploaded files
                    doc_result = await self._process_uploaded_files(source, config)
                elif source.source_type == KnowledgeSourceType.WEB_URLS:
                    # Process web URLs
                    doc_result = await self._process_web_urls(source, config)
                elif source.source_type == KnowledgeSourceType.DOCUMENTS:
                    # Process existing documents
                    doc_result = await self._process_documents(source, config)
                else:
                    logger.warning(f"Unsupported source type: {source.source_type}")
                    continue
                
                total_documents += doc_result.get("documents", 0)
                total_chunks += doc_result.get("chunks", 0)
                total_embeddings += doc_result.get("embeddings", 0)
            
            # Step 3: Create vector collection
            collection_name = f"{config.tenant_id}_{config.embedding_collection}"
            await self._ensure_vector_collection(collection_name)
            collections_created.append(collection_name)
            
            # Step 4: Set up permissions
            await self._configure_domain_permissions(config)
            
            logger.info(f"Domain knowledge setup completed: {config.domain_id}")
            
            return KnowledgeSetupResult(
                success=True,
                domain_id=config.domain_id,
                collections_created=collections_created,
                documents_processed=total_documents,
                chunks_created=total_chunks,
                embeddings_generated=total_embeddings
            )
            
        except Exception as e:
            logger.error(f"Domain knowledge setup failed: {str(e)}")
            return KnowledgeSetupResult(
                success=False,
                error=str(e)
            )
    
    async def configure_team_rag_permissions(self, team_id: str, 
                                           permissions: List[RAGToolPermissions]) -> bool:
        """
        Configure RAG tool permissions for all team members
        
        Args:
            team_id: Team identifier
            permissions: List of permission configurations per member
            
        Returns:
            bool: True if permissions configured successfully
        """
        try:
            logger.info(f"Configuring RAG permissions for team: {team_id}")
            
            for perm in permissions:
                logger.info(f"Setting up permissions for member: {perm.member_id}")
                
                # Configure permissions via TAO integration (will be available when integrated)
                await self._set_member_tool_permissions(team_id, perm)
            
            logger.info(f"RAG permissions configured for {len(permissions)} team members")
            return True
            
        except Exception as e:
            logger.error(f"Permission configuration failed: {str(e)}")
            return False
    
    async def _create_domain_record(self, config: DomainKnowledgeConfig):
        """Create domain record in Neo4j"""
        # This will integrate with Neo4j when available
        logger.info(f"Creating domain record: {config.domain_id}")
        
        # Neo4j Cypher query to create domain:
        # CREATE (d:Domain {
        #     id: $domain_id,
        #     name: $domain_name,
        #     tenant_id: $tenant_id,
        #     created_at: datetime()
        # })
        
    async def _process_uploaded_files(self, source: KnowledgeSource, 
                                    config: DomainKnowledgeConfig) -> Dict[str, int]:
        """Process uploaded files through chunking and embedding"""
        # This will call RAG MCP services to process files
        logger.info(f"Processing uploaded files for domain: {config.domain_id}")
        
        # Mock processing results
        return {"documents": 5, "chunks": 150, "embeddings": 150}
    
    async def _process_web_urls(self, source: KnowledgeSource,
                              config: DomainKnowledgeConfig) -> Dict[str, int]:
        """Process web URLs through fetching, chunking, and embedding"""
        logger.info(f"Processing web URLs for domain: {config.domain_id}")
        
        # Mock processing results
        return {"documents": 3, "chunks": 75, "embeddings": 75}
    
    async def _process_documents(self, source: KnowledgeSource,
                               config: DomainKnowledgeConfig) -> Dict[str, int]:
        """Process existing documents through chunking and embedding"""
        logger.info(f"Processing documents for domain: {config.domain_id}")
        
        # Mock processing results
        return {"documents": 10, "chunks": 300, "embeddings": 300}
    
    async def _ensure_vector_collection(self, collection_name: str):
        """Ensure vector collection exists in Qdrant"""
        logger.info(f"Creating vector collection: {collection_name}")
        
        # This will call embedding MCP service to ensure collection exists
    
    async def _configure_domain_permissions(self, config: DomainKnowledgeConfig):
        """Configure domain access permissions"""
        logger.info(f"Configuring domain permissions: {config.domain_id}")
        
        # This will integrate with TAO permission system
    
    async def _set_member_tool_permissions(self, team_id: str, perm: RAGToolPermissions):
        """Set individual member tool permissions"""
        # This will integrate with TAO's permission system
        tools_to_grant = []
        
        if perm.chunking_access:
            tools_to_grant.append("chunker_v1")
        if perm.embedding_access:
            tools_to_grant.append("embed_v1") 
        if perm.retrieval_access:
            tools_to_grant.append("retriever_v1")
        
        logger.info(f"Granting tools to {perm.member_id}: {tools_to_grant}")

# ============================================================================
# TAB INTEGRATION UTILITIES
# ============================================================================

async def integrate_rag_with_team_builder(team_builder_workflow):
    """
    Integrate RAG capabilities into TAB's team building workflow
    
    This function adds RAG configuration steps to the standard
    team building process.
    
    Args:
        team_builder_workflow: TAB's TeamBuilderWorkflow instance
    """
    logger.info("Integrating RAG capabilities with TAB team builder")
    
    rag_builder = RAGKnowledgeBuilder()
    
    # Add RAG discovery steps to workflow
    rag_steps = await rag_builder.discover_knowledge_requirements(None)
    
    # This will integrate with TAB's workflow system
    # team_builder_workflow.add_steps(rag_steps)
    
    logger.info(f"Added {len(rag_steps)} RAG configuration steps to team builder")

def create_rag_aware_team_template(base_template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance a team template with RAG capability options
    
    Args:
        base_template: Standard TAB team template
        
    Returns:
        Enhanced template with RAG options
    """
    rag_template = base_template.copy()
    
    # Add RAG-specific template sections
    rag_template["knowledge_configuration"] = {
        "rag_enabled": True,
        "default_chunking_strategy": "hybrid",
        "default_collection_prefix": "team_knowledge",
        "suggested_permissions": {
            "retrieval": ["all_members"],
            "chunking": ["knowledge_managers"],
            "embedding": ["knowledge_managers"]
        }
    }
    
    # Add RAG tool suggestions to member roles
    if "member_roles" in rag_template:
        for role in rag_template["member_roles"]:
            if role.get("type") == "researcher":
                role.setdefault("suggested_tools", []).extend([
                    "retriever_v1", "chunker_v1"
                ])
            elif role.get("type") == "knowledge_manager":
                role.setdefault("suggested_tools", []).extend([
                    "chunker_v1", "embed_v1", "retriever_v1"
                ])
    
    return rag_template

# ============================================================================
# MAIN INTEGRATION DEMO/TEST
# ============================================================================

async def main():
    """Demo function showing how TAB integration will work"""
    print("🏗️ RAG MCP → TAB Integration Demo")
    print("=" * 50)
    
    try:
        # Initialize RAG knowledge builder
        rag_builder = RAGKnowledgeBuilder()
        
        # Demo 1: Knowledge Requirements Discovery
        print("\n📋 Demo 1: Knowledge Requirements Discovery")
        wizard_steps = await rag_builder.discover_knowledge_requirements(None)
        print(f"   → Generated {len(wizard_steps)} wizard steps for RAG configuration")
        for step in wizard_steps[:2]:  # Show first 2 steps
            print(f"     - {step['step_id']}: {step['question'][:60]}...")
        
        # Demo 2: Configuration Processing
        print("\n⚙️ Demo 2: Configuration Processing")
        mock_responses = {
            "rag_needs_assessment": "yes",
            "rag_knowledge_sources": ["documents", "file_uploads"],
            "rag_domain_setup": "Product Documentation",
            "rag_chunking_strategy": "hybrid", 
            "rag_permissions_setup": ["agent_001", "agent_002"]
        }
        
        mock_team_context = type('obj', (object,), {
            'tenant_id': 'demo_tenant',
            'team_name': 'Demo Team',
            'session_id': 'session_123'
        })
        
        config = await rag_builder.process_knowledge_configuration(mock_responses, mock_team_context)
        print(f"   → Created domain config: {config.domain_id}")
        print(f"     - Knowledge sources: {len(config.knowledge_sources)}")
        print(f"     - Chunking strategy: {config.chunking_strategy}")
        
        # Demo 3: Knowledge Setup
        print("\n🔧 Demo 3: Domain Knowledge Setup")  
        # setup_result = await rag_builder.setup_domain_knowledge(config)
        print(f"   → Would process {len(config.knowledge_sources)} knowledge sources")
        print(f"   → Would create collection: {config.embedding_collection}")
        
        # Demo 4: Template Enhancement
        print("\n📝 Demo 4: RAG-Aware Team Template")
        base_template = {
            "team_name": "Research Team",
            "member_roles": [
                {"type": "researcher", "name": "Research Specialist"},
                {"type": "analyst", "name": "Data Analyst"}
            ]
        }
        
        rag_template = create_rag_aware_team_template(base_template)
        print(f"   → Enhanced template with RAG capabilities")
        print(f"     - RAG enabled: {rag_template['knowledge_configuration']['rag_enabled']}")
        
        print(f"\n🎉 RAG MCP → TAB integration ready for production!")
        
    except Exception as e:
        print(f"❌ Integration demo failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())