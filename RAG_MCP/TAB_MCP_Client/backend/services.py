"""
Services for TAB_MCP_Client
===========================
Business logic for tenant, domain, document, and query processing
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import yaml
import httpx
import asyncio
from pathlib import Path

from neo4j import GraphDatabase, AsyncGraphDatabase
import asyncpg
from minio import Minio
from minio.error import S3Error

from .models import (
    TenantConfig, DomainKnowledgeConfig, DocumentUpload,
    QueryRequest, QueryResponse, ChunkMetadata,
    ConfigStatus, AgentTeamConfig
)
from config import Config

logger = logging.getLogger(__name__)

# ============================================================================
# BASE SERVICE CLASS
# ============================================================================

class BaseService:
    """Base service class with common database connections"""
    
    def __init__(self):
        self.config = Config()
        self.neo4j_driver = None
        self.pg_pool = None
        self.minio_client = None
        self.mcp_client = httpx.AsyncClient(timeout=30.0)
    
    async def initialize(self):
        """Initialize database connections"""
        # Neo4j connection
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.config.NEO4J_URI,
            auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
        )
        
        # PostgreSQL connection pool
        self.pg_pool = await asyncpg.create_pool(
            host=self.config.POSTGRES_HOST,
            port=self.config.POSTGRES_PORT,
            database=self.config.POSTGRES_DB,
            user=self.config.POSTGRES_USER,
            password=self.config.POSTGRES_PASSWORD,
            min_size=5,
            max_size=20
        )
        
        # MinIO client
        self.minio_client = Minio(
            self.config.MINIO_ENDPOINT,
            access_key=self.config.MINIO_ACCESS_KEY,
            secret_key=self.config.MINIO_SECRET_KEY,
            secure=self.config.MINIO_SECURE
        )
        
        # Ensure MinIO bucket exists
        bucket_name = self.config.MINIO_BUCKET
        if not self.minio_client.bucket_exists(bucket_name):
            self.minio_client.make_bucket(bucket_name)
            logger.info(f"Created MinIO bucket: {bucket_name}")
    
    async def close(self):
        """Close all connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.pg_pool:
            await self.pg_pool.close()
        await self.mcp_client.aclose()

# ============================================================================
# TENANT SERVICE
# ============================================================================

class TenantService(BaseService):
    """Service for managing tenant configurations"""
    
    async def create_or_update_tenant(self, yaml_content: str, created_by: str) -> Tuple[TenantConfig, bool]:
        """
        Create or update tenant configuration from YAML
        Returns: (TenantConfig, is_new)
        """
        try:
            # Parse YAML to TenantConfig
            tenant_config = TenantConfig.from_yaml(yaml_content)
            
            # Check if tenant exists
            existing_tenant = await self.get_tenant(tenant_config.tenant_id)
            is_new = existing_tenant is None
            
            if not is_new:
                # Create new version if configuration changed
                if existing_tenant.calculate_hash() != tenant_config.calculate_hash():
                    tenant_config.versions = existing_tenant.versions
                    tenant_config.create_version(created_by, "Configuration update via YAML upload")
            else:
                # Create initial version for new tenant
                tenant_config.create_version(created_by, "Initial configuration")
            
            # Save to PostgreSQL with versioning
            await self.save_tenant_to_postgres(tenant_config)
            
            # Save to Neo4j graph
            await self.save_tenant_to_neo4j(tenant_config)
            
            logger.info(f"{'Created' if is_new else 'Updated'} tenant: {tenant_config.tenant_id}")
            return tenant_config, is_new
            
        except Exception as e:
            logger.error(f"Error processing tenant configuration: {e}")
            raise
    
    async def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration by ID"""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT config_data, versions 
                FROM rag_system.tenant_configs 
                WHERE tenant_id = $1 AND status = 'active'
                """,
                tenant_id
            )
            
            if row:
                config_data = json.loads(row['config_data'])
                tenant_config = TenantConfig(**config_data)
                if row['versions']:
                    tenant_config.versions = json.loads(row['versions'])
                return tenant_config
            return None
    
    async def save_tenant_to_postgres(self, tenant_config: TenantConfig):
        """Save tenant configuration to PostgreSQL with versioning"""
        async with self.pg_pool.acquire() as conn:
            # First, ensure the table exists (extend the existing schema)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_system.tenant_configs (
                    tenant_id VARCHAR(255) PRIMARY KEY,
                    config_data JSONB NOT NULL,
                    versions JSONB DEFAULT '[]',
                    current_version_id VARCHAR(255),
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Upsert tenant configuration
            await conn.execute("""
                INSERT INTO rag_system.tenant_configs (tenant_id, config_data, versions, current_version_id, status, updated_at)
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
                ON CONFLICT (tenant_id)
                DO UPDATE SET 
                    config_data = EXCLUDED.config_data,
                    versions = EXCLUDED.versions,
                    current_version_id = EXCLUDED.current_version_id,
                    updated_at = CURRENT_TIMESTAMP
            """,
                tenant_config.tenant_id,
                tenant_config.to_json(),
                json.dumps([v.model_dump() for v in tenant_config.versions], default=str),
                tenant_config.current_version.version_id if tenant_config.current_version else None,
                tenant_config.status.value
            )
            
            # Also update the main tenants table
            await conn.execute("""
                INSERT INTO rag_system.tenants (
                    tenant_id, name, display_name, industry, subscription_tier,
                    api_rate_limit, storage_quota_gb, max_users,
                    default_embedding_model, default_chunking_method, default_chunk_size,
                    metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (tenant_id)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    display_name = EXCLUDED.display_name,
                    industry = EXCLUDED.industry,
                    subscription_tier = EXCLUDED.subscription_tier,
                    api_rate_limit = EXCLUDED.api_rate_limit,
                    storage_quota_gb = EXCLUDED.storage_quota_gb,
                    max_users = EXCLUDED.max_users,
                    default_embedding_model = EXCLUDED.default_embedding_model,
                    default_chunking_method = EXCLUDED.default_chunking_method,
                    default_chunk_size = EXCLUDED.default_chunk_size,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """,
                tenant_config.tenant_id,
                tenant_config.name,
                tenant_config.display_name,
                tenant_config.industry,
                tenant_config.subscription_tier,
                tenant_config.api_rate_limit,
                tenant_config.storage_quota_gb,
                tenant_config.max_users,
                tenant_config.default_embedding_model,
                tenant_config.default_chunking_method,
                tenant_config.default_chunk_size,
                json.dumps(tenant_config.metadata)
            )
    
    async def save_tenant_to_neo4j(self, tenant_config: TenantConfig):
        """Save tenant configuration to Neo4j graph"""
        async with self.neo4j_driver.session() as session:
            await session.execute_write(self._create_tenant_node, tenant_config)
    
    @staticmethod
    async def _create_tenant_node(tx, tenant_config: TenantConfig):
        """Create or update tenant node in Neo4j"""
        query = """
        MERGE (t:Tenant {id: $tenant_id})
        SET t.name = $name,
            t.display_name = $display_name,
            t.industry = $industry,
            t.subscription_tier = $subscription_tier,
            t.headquarters_location = $headquarters_location,
            t.founding_year = $founding_year,
            t.employee_count = $employee_count,
            t.annual_revenue = $annual_revenue,
            t.currency = $currency,
            t.compliance_frameworks = $compliance_frameworks,
            t.data_residency = $data_residency,
            t.core_tech_stack = $core_tech_stack,
            t.api_rate_limit = $api_rate_limit,
            t.storage_quota_gb = $storage_quota_gb,
            t.metadata = $metadata,
            t.updated_at = datetime()
        RETURN t
        """
        
        result = await tx.run(
            query,
            tenant_id=tenant_config.tenant_id,
            name=tenant_config.name,
            display_name=tenant_config.display_name,
            industry=tenant_config.industry,
            subscription_tier=tenant_config.subscription_tier,
            headquarters_location=tenant_config.headquarters_location,
            founding_year=tenant_config.founding_year,
            employee_count=tenant_config.employee_count,
            annual_revenue=tenant_config.annual_revenue,
            currency=tenant_config.currency,
            compliance_frameworks=tenant_config.compliance_frameworks,
            data_residency=tenant_config.data_residency,
            core_tech_stack=tenant_config.core_tech_stack,
            api_rate_limit=tenant_config.api_rate_limit,
            storage_quota_gb=tenant_config.storage_quota_gb,
            metadata=json.dumps(tenant_config.metadata)
        )
        return await result.single()

# ============================================================================
# DOMAIN SERVICE
# ============================================================================

class DomainService(BaseService):
    """Service for managing domain knowledge configurations"""
    
    async def create_or_update_domain(self, yaml_content: str, created_by: str) -> Tuple[DomainKnowledgeConfig, bool]:
        """
        Create or update domain knowledge configuration from YAML
        Returns: (DomainKnowledgeConfig, is_new)
        """
        try:
            # Parse YAML to DomainKnowledgeConfig
            domain_config = DomainKnowledgeConfig.from_yaml(yaml_content)
            
            # Validate tenant exists
            tenant_service = TenantService()
            await tenant_service.initialize()
            tenant = await tenant_service.get_tenant(domain_config.tenant_id)
            if not tenant:
                raise ValueError(f"Tenant {domain_config.tenant_id} not found")
            
            # Check if domain exists
            existing_domain = await self.get_domain(domain_config.domain_id)
            is_new = existing_domain is None
            
            if not is_new:
                # Create new version if configuration changed
                if existing_domain.calculate_hash() != domain_config.calculate_hash():
                    domain_config.versions = existing_domain.versions
                    domain_config.create_version(created_by, "Configuration update via YAML upload")
            else:
                # Create initial version for new domain
                domain_config.create_version(created_by, "Initial configuration")
            
            # Save to PostgreSQL with versioning
            await self.save_domain_to_postgres(domain_config)
            
            # Save to Neo4j graph with relationships
            await self.save_domain_to_neo4j(domain_config)
            
            logger.info(f"{'Created' if is_new else 'Updated'} domain: {domain_config.domain_id}")
            return domain_config, is_new
            
        except Exception as e:
            logger.error(f"Error processing domain configuration: {e}")
            raise
    
    async def get_domain(self, domain_id: str) -> Optional[DomainKnowledgeConfig]:
        """Get domain configuration by ID"""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT config_data, versions 
                FROM rag_system.domain_configs 
                WHERE domain_id = $1 AND status = 'active'
                """,
                domain_id
            )
            
            if row:
                config_data = json.loads(row['config_data'])
                domain_config = DomainKnowledgeConfig(**config_data)
                if row['versions']:
                    domain_config.versions = json.loads(row['versions'])
                return domain_config
            return None
    
    async def save_domain_to_postgres(self, domain_config: DomainKnowledgeConfig):
        """Save domain configuration to PostgreSQL with versioning"""
        async with self.pg_pool.acquire() as conn:
            # First, ensure the table exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_system.domain_configs (
                    domain_id VARCHAR(255) PRIMARY KEY,
                    tenant_id VARCHAR(255) REFERENCES rag_system.tenants(tenant_id),
                    config_data JSONB NOT NULL,
                    versions JSONB DEFAULT '[]',
                    current_version_id VARCHAR(255),
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Upsert domain configuration
            await conn.execute("""
                INSERT INTO rag_system.domain_configs (domain_id, tenant_id, config_data, versions, current_version_id, status, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                ON CONFLICT (domain_id)
                DO UPDATE SET 
                    config_data = EXCLUDED.config_data,
                    versions = EXCLUDED.versions,
                    current_version_id = EXCLUDED.current_version_id,
                    updated_at = CURRENT_TIMESTAMP
            """,
                domain_config.domain_id,
                domain_config.tenant_id,
                domain_config.to_json(),
                json.dumps([v.model_dump() for v in domain_config.versions], default=str),
                domain_config.current_version.version_id if domain_config.current_version else None,
                domain_config.status.value
            )
            
            # Also update the main domains table
            await conn.execute("""
                INSERT INTO rag_system.domains (
                    domain_id, tenant_id, name, display_name, domain_type, domain_category,
                    path, level, parent_domain_id,
                    metadata_template, knowledge_sources, indexing_frequency, retention_days,
                    attributes, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (domain_id)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    display_name = EXCLUDED.display_name,
                    domain_type = EXCLUDED.domain_type,
                    domain_category = EXCLUDED.domain_category,
                    path = EXCLUDED.path,
                    level = EXCLUDED.level,
                    parent_domain_id = EXCLUDED.parent_domain_id,
                    metadata_template = EXCLUDED.metadata_template,
                    knowledge_sources = EXCLUDED.knowledge_sources,
                    indexing_frequency = EXCLUDED.indexing_frequency,
                    retention_days = EXCLUDED.retention_days,
                    attributes = EXCLUDED.attributes,
                    status = EXCLUDED.status,
                    updated_at = CURRENT_TIMESTAMP
            """,
                domain_config.domain_id,
                domain_config.tenant_id,
                domain_config.name,
                domain_config.display_name,
                domain_config.domain_type.value,
                domain_config.domain_category,
                domain_config.path,
                domain_config.level,
                domain_config.parent_domain_id,
                json.dumps(domain_config.metadata_template.model_dump()) if domain_config.metadata_template else None,
                domain_config.knowledge_sources,
                domain_config.indexing_frequency,
                domain_config.retention_days,
                json.dumps(domain_config.attributes),
                domain_config.status.value
            )
    
    async def save_domain_to_neo4j(self, domain_config: DomainKnowledgeConfig):
        """Save domain configuration and knowledge entities to Neo4j graph"""
        async with self.neo4j_driver.session() as session:
            await session.execute_write(self._create_domain_node, domain_config)
            
            # Create knowledge entity nodes
            for entity in domain_config.knowledge_entities:
                await session.execute_write(self._create_knowledge_entity_node, domain_config.domain_id, entity)
    
    @staticmethod
    async def _create_domain_node(tx, domain_config: DomainKnowledgeConfig):
        """Create or update domain node in Neo4j"""
        query = """
        MERGE (d:Domain {id: $domain_id})
        SET d.tenant_id = $tenant_id,
            d.name = $name,
            d.display_name = $display_name,
            d.domain_type = $domain_type,
            d.domain_category = $domain_category,
            d.path = $path,
            d.level = $level,
            d.parent_id = $parent_domain_id,
            d.attributes = $attributes,
            d.knowledge_sources = $knowledge_sources,
            d.indexing_frequency = $indexing_frequency,
            d.retention_days = $retention_days,
            d.metadata_template = $metadata_template,
            d.status = $status,
            d.updated_at = datetime()
        
        WITH d
        MATCH (t:Tenant {id: $tenant_id})
        MERGE (t)-[r:HAS_DOMAIN]->(d)
        SET r.assigned_at = datetime(),
            r.access_level = 'full',
            r.data_isolation = true
        
        RETURN d
        """
        
        result = await tx.run(
            query,
            domain_id=domain_config.domain_id,
            tenant_id=domain_config.tenant_id,
            name=domain_config.name,
            display_name=domain_config.display_name,
            domain_type=domain_config.domain_type.value,
            domain_category=domain_config.domain_category,
            path=domain_config.path,
            level=domain_config.level,
            parent_domain_id=domain_config.parent_domain_id,
            attributes=json.dumps(domain_config.attributes),
            knowledge_sources=domain_config.knowledge_sources,
            indexing_frequency=domain_config.indexing_frequency,
            retention_days=domain_config.retention_days,
            metadata_template=json.dumps(domain_config.metadata_template.model_dump()) if domain_config.metadata_template else None,
            status=domain_config.status.value
        )
        return await result.single()
    
    @staticmethod
    async def _create_knowledge_entity_node(tx, domain_id: str, entity):
        """Create or update knowledge entity node in Neo4j"""
        query = """
        MERGE (k:KnowledgeEntity {id: $entity_id})
        SET k.domain_id = $domain_id,
            k.entity_type = $entity_type,
            k.entity_subtype = $entity_subtype,
            k.name = $name,
            k.display_name = $display_name,
            k.description = $description,
            k.path = $path,
            k.hierarchy_level = $hierarchy_level,
            k.parent_entity_id = $parent_entity_id,
            k.properties = $properties,
            k.tags = $tags,
            k.keywords = $keywords,
            k.staleness = $staleness,
            k.staleness_score = $staleness_score,
            k.last_validated = $last_validated,
            k.updated_at = datetime()
        
        WITH k
        MATCH (d:Domain {id: $domain_id})
        MERGE (d)-[r:CONTAINS_KNOWLEDGE]->(k)
        SET r.category = $entity_type,
            r.importance = 'high'
        
        RETURN k
        """
        
        result = await tx.run(
            query,
            entity_id=entity.entity_id,
            domain_id=domain_id,
            entity_type=entity.entity_type,
            entity_subtype=entity.entity_subtype,
            name=entity.name,
            display_name=entity.display_name,
            description=entity.description,
            path=entity.path,
            hierarchy_level=entity.hierarchy_level,
            parent_entity_id=entity.parent_entity_id,
            properties=json.dumps(entity.properties),
            tags=entity.tags,
            keywords=entity.keywords,
            staleness=entity.staleness.value,
            staleness_score=entity.staleness_score,
            last_validated=entity.last_validated.isoformat()
        )
        return await result.single()

# ============================================================================
# DOCUMENT SERVICE
# ============================================================================

class DocumentService(BaseService):
    """Service for document upload and processing"""
    
    async def upload_and_process_document(self, document_upload: DocumentUpload) -> Dict[str, Any]:
        """
        Upload document to MinIO and process through chunking and embedding pipeline
        """
        try:
            # Generate document ID if not provided
            if not document_upload.document_id:
                document_upload.document_id = f"doc_{uuid.uuid4().hex[:12]}"
            
            # Upload to MinIO
            minio_path = await self.upload_to_minio(document_upload)
            
            # Save document metadata to PostgreSQL
            await self.save_document_metadata(document_upload, minio_path)
            
            # Save document node to Neo4j
            await self.save_document_to_neo4j(document_upload, minio_path)
            
            # Process through MCP chunking service
            chunks = await self.chunk_document(document_upload, minio_path)
            
            # Process through MCP embedding service with metadata enrichment
            embeddings = await self.embed_chunks_with_metadata(chunks, document_upload)
            
            # Store in Qdrant with enriched metadata
            await self.store_in_qdrant(embeddings, document_upload)
            
            logger.info(f"Successfully processed document: {document_upload.document_id}")
            
            return {
                "document_id": document_upload.document_id,
                "minio_path": minio_path,
                "total_chunks": len(chunks),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    async def upload_to_minio(self, document_upload: DocumentUpload) -> str:
        """Upload document to MinIO"""
        try:
            bucket_name = self.config.MINIO_BUCKET
            object_name = f"{document_upload.tenant_id}/{document_upload.domain_id}/documents/{document_upload.document_id}/{document_upload.filename}"
            
            if document_upload.content:
                # Upload from bytes
                from io import BytesIO
                data = BytesIO(document_upload.content)
                self.minio_client.put_object(
                    bucket_name,
                    object_name,
                    data,
                    length=len(document_upload.content),
                    content_type=document_upload.content_type
                )
            elif document_upload.file_path:
                # Upload from file path
                self.minio_client.fput_object(
                    bucket_name,
                    object_name,
                    document_upload.file_path,
                    content_type=document_upload.content_type
                )
            else:
                raise ValueError("No content or file path provided")
            
            minio_path = f"s3://{bucket_name}/{object_name}"
            logger.info(f"Uploaded document to MinIO: {minio_path}")
            return minio_path
            
        except S3Error as e:
            logger.error(f"MinIO upload error: {e}")
            raise
    
    async def save_document_metadata(self, document_upload: DocumentUpload, minio_path: str):
        """Save document metadata to PostgreSQL"""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO rag_system.documents (
                    document_id, tenant_id, domain_id,
                    title, original_filename, content_hash,
                    minio_path, minio_bucket,
                    content_type, document_tags, main_topics, detected_entities,
                    status, created_by
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
                document_upload.document_id,
                document_upload.tenant_id,
                document_upload.domain_id,
                document_upload.title,
                document_upload.filename,
                "", # content_hash - calculate if needed
                minio_path,
                self.config.MINIO_BUCKET,
                document_upload.content_type,
                json.dumps(document_upload.document_tags),
                document_upload.main_topics,
                document_upload.knowledge_entities,
                'processing',
                document_upload.uploaded_by
            )
    
    async def save_document_to_neo4j(self, document_upload: DocumentUpload, minio_path: str):
        """Save document node to Neo4j"""
        async with self.neo4j_driver.session() as session:
            query = """
            CREATE (doc:Document {
                id: $document_id,
                tenant_id: $tenant_id,
                domain_id: $domain_id,
                title: $title,
                filename: $filename,
                minio_path: $minio_path,
                document_type: $content_type,
                document_tags: $document_tags,
                main_topics: $main_topics,
                detected_entities: $knowledge_entities,
                created_at: datetime(),
                created_by: $uploaded_by
            })
            
            WITH doc
            MATCH (d:Domain {id: $domain_id})
            CREATE (doc)-[:BELONGS_TO_DOMAIN {primary: true, relevance: 1.0}]->(d)
            
            WITH doc
            UNWIND $knowledge_entities AS entity_id
            MATCH (k:KnowledgeEntity {id: entity_id})
            CREATE (doc)-[:DESCRIBES {coverage: 'partial', detail_level: 'technical'}]->(k)
            
            RETURN doc
            """
            
            await session.run(
                query,
                document_id=document_upload.document_id,
                tenant_id=document_upload.tenant_id,
                domain_id=document_upload.domain_id,
                title=document_upload.title,
                filename=document_upload.filename,
                minio_path=minio_path,
                content_type=document_upload.content_type,
                document_tags=json.dumps(document_upload.document_tags),
                main_topics=document_upload.main_topics,
                knowledge_entities=document_upload.knowledge_entities,
                uploaded_by=document_upload.uploaded_by
            )
    
    async def chunk_document(self, document_upload: DocumentUpload, minio_path: str) -> List[Dict[str, Any]]:
        """Call MCP chunking service to chunk the document"""
        try:
            response = await self.mcp_client.post(
                f"{self.config.MCP_CHUNKING_URL}/chunk",
                json={
                    "document_id": document_upload.document_id,
                    "tenant_id": document_upload.tenant_id,
                    "minio_path": minio_path,
                    "method": document_upload.chunking_method,
                    "chunk_size": document_upload.chunk_size,
                    "chunk_overlap": document_upload.chunk_overlap
                }
            )
            response.raise_for_status()
            chunks = response.json()["chunks"]
            
            # Save chunks to PostgreSQL
            await self.save_chunks_to_postgres(chunks, document_upload)
            
            return chunks
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Chunking service error: {e}")
            raise
    
    async def save_chunks_to_postgres(self, chunks: List[Dict[str, Any]], document_upload: DocumentUpload):
        """Save chunks to PostgreSQL"""
        async with self.pg_pool.acquire() as conn:
            for chunk in chunks:
                await conn.execute("""
                    INSERT INTO rag_system.chunks (
                        chunk_id, document_id, tenant_id, domain_id,
                        chunk_index, chunk_hash,
                        chunk_text, chunk_text_with_overlap,
                        token_count, char_count, start_char, end_char,
                        method, overlap_tokens,
                        related_entities
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                    chunk["chunk_id"],
                    document_upload.document_id,
                    document_upload.tenant_id,
                    document_upload.domain_id,
                    chunk["chunk_index"],
                    chunk.get("chunk_hash", ""),
                    chunk["text"],
                    chunk.get("text_with_overlap", chunk["text"]),
                    chunk["token_count"],
                    chunk.get("char_count", len(chunk["text"])),
                    chunk.get("start_char", 0),
                    chunk.get("end_char", len(chunk["text"])),
                    document_upload.chunking_method,
                    document_upload.chunk_overlap,
                    document_upload.knowledge_entities
                )
    
    async def embed_chunks_with_metadata(self, chunks: List[Dict[str, Any]], document_upload: DocumentUpload) -> List[Dict[str, Any]]:
        """
        Call MCP embedding service to embed chunks with metadata enrichment
        """
        try:
            # Get domain and tenant metadata for enrichment
            domain_service = DomainService()
            await domain_service.initialize()
            domain = await domain_service.get_domain(document_upload.domain_id)
            
            # Prepare chunks with enriched metadata
            enriched_chunks = []
            for chunk in chunks:
                chunk_metadata = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": document_upload.document_id,
                    "tenant_id": document_upload.tenant_id,
                    "domain_id": document_upload.domain_id,
                    "domain_type": domain.domain_type.value if domain else None,
                    "domain_name": domain.name if domain else None,
                    "knowledge_entities": document_upload.knowledge_entities,
                    "document_tags": document_upload.document_tags,
                    "main_topics": document_upload.main_topics,
                    **chunk.get("metadata", {})
                }
                
                enriched_chunks.append({
                    "id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk_metadata
                })
            
            # Call embedding service
            response = await self.mcp_client.post(
                f"{self.config.MCP_EMBEDDING_URL}/embed",
                json={
                    "tenant_id": document_upload.tenant_id,
                    "collection_name": self.config.QDRANT_COLLECTION,
                    "items": enriched_chunks
                }
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            
            return embeddings
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding service error: {e}")
            raise
    
    async def store_in_qdrant(self, embeddings: List[Dict[str, Any]], document_upload: DocumentUpload):
        """Store embeddings in Qdrant with enriched metadata"""
        try:
            # The embedding service already stores in Qdrant, but we need to update PostgreSQL
            async with self.pg_pool.acquire() as conn:
                for embedding in embeddings:
                    await conn.execute("""
                        UPDATE rag_system.chunks
                        SET qdrant_point_id = $1,
                            embedding_status = 'embedded',
                            embedding_model = $2,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE chunk_id = $3
                    """,
                        embedding["point_id"],
                        self.config.DEFAULT_EMBEDDING_MODEL,
                        embedding["chunk_id"]
                    )
            
            # Update document status
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE rag_system.documents
                    SET status = 'completed',
                        indexed_at = CURRENT_TIMESTAMP,
                        total_chunks = $1
                    WHERE document_id = $2
                """,
                    len(embeddings),
                    document_upload.document_id
                )
                
        except Exception as e:
            logger.error(f"Error updating Qdrant references: {e}")
            raise

# ============================================================================
# QUERY SERVICE
# ============================================================================

class QueryService(BaseService):
    """Service for processing queries using RAG Agent Team"""
    
    async def process_query(self, query_request: QueryRequest) -> QueryResponse:
        """
        Process query using the RAG Agent Team with MCP retrieval
        """
        try:
            # Generate query ID
            query_id = f"query_{uuid.uuid4().hex[:12]}"
            
            # Call RAG Agent Team endpoint
            response = await self.mcp_client.post(
                f"{self.config.RAG_AGENT_TEAM_URL}/query",
                json={
                    "query": query_request.query,
                    "tenant_id": query_request.tenant_id,
                    "session_id": query_request.session_id,
                    "max_results": query_request.max_results,
                    "include_sources": query_request.include_sources,
                    "context": {
                        "domain_id": query_request.domain_id,
                        **query_request.filters
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Save query to PostgreSQL
            await self.save_query(query_id, query_request, result)
            
            # Create response
            query_response = QueryResponse(
                query_id=query_id,
                response_text=result["answer"],
                chunks_retrieved=result.get("chunks_retrieved", []),
                relevance_scores=result.get("relevance_scores", []),
                sources=result.get("sources", []),
                metadata=result.get("metadata", {}),
                processing_time_ms=result.get("processing_time_ms", 0)
            )
            
            return query_response
            
        except httpx.HTTPStatusError as e:
            logger.error(f"RAG Agent Team error: {e}")
            raise
    
    async def save_query(self, query_id: str, query_request: QueryRequest, result: Dict[str, Any]):
        """Save query and results to PostgreSQL"""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO rag_system.queries (
                    query_id, tenant_id, domain_id, user_id, session_id,
                    query_text, query_embedding_id,
                    chunks_retrieved, relevance_scores,
                    response_text,
                    latency_ms, tokens_used
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                query_id,
                query_request.tenant_id,
                query_request.domain_id,
                "user_" + query_request.session_id if query_request.session_id else "anonymous",
                query_request.session_id,
                query_request.query,
                result.get("query_embedding_id"),
                result.get("chunks_retrieved", []),
                result.get("relevance_scores", []),
                result.get("answer"),
                result.get("processing_time_ms", 0),
                result.get("tokens_used", 0)
            )
            
            # Track chunk retrievals
            for i, chunk_id in enumerate(result.get("chunks_retrieved", [])):
                relevance_score = result.get("relevance_scores", [])[i] if i < len(result.get("relevance_scores", [])) else 0.0
                await conn.execute("""
                    INSERT INTO rag_system.chunk_retrievals (chunk_id, query_id, relevance_score, rank)
                    VALUES ($1, $2, $3, $4)
                """,
                    chunk_id,
                    query_id,
                    relevance_score,
                    i + 1
                )

# ============================================================================
# AGENT TEAM SERVICE
# ============================================================================

class AgentTeamService(BaseService):
    """Service for managing agent team configurations"""
    
    async def create_or_update_team(self, yaml_content: str, created_by: str) -> Tuple[AgentTeamConfig, bool]:
        """
        Create or update agent team configuration from YAML
        Returns: (AgentTeamConfig, is_new)
        """
        try:
            # Parse YAML to AgentTeamConfig
            team_config = AgentTeamConfig.from_yaml(yaml_content)
            
            # Save to PostgreSQL
            is_new = await self.save_team_to_postgres(team_config)
            
            # Save to Neo4j graph
            await self.save_team_to_neo4j(team_config)
            
            logger.info(f"{'Created' if is_new else 'Updated'} agent team: {team_config.team_id}")
            return team_config, is_new
            
        except Exception as e:
            logger.error(f"Error processing agent team configuration: {e}")
            raise
    
    async def save_team_to_postgres(self, team_config: AgentTeamConfig) -> bool:
        """Save agent team configuration to PostgreSQL"""
        async with self.pg_pool.acquire() as conn:
            # Create table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_system.agent_teams (
                    team_id VARCHAR(255) PRIMARY KEY,
                    tenant_id VARCHAR(255) REFERENCES rag_system.tenants(tenant_id),
                    config_data JSONB NOT NULL,
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if exists
            existing = await conn.fetchrow(
                "SELECT team_id FROM rag_system.agent_teams WHERE team_id = $1",
                team_config.team_id
            )
            is_new = existing is None
            
            # Upsert team configuration
            await conn.execute("""
                INSERT INTO rag_system.agent_teams (team_id, tenant_id, config_data, status)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (team_id)
                DO UPDATE SET 
                    config_data = EXCLUDED.config_data,
                    updated_at = CURRENT_TIMESTAMP
            """,
                team_config.team_id,
                team_config.tenant_id,
                team_config.to_json(),
                team_config.status.value
            )
            
            return is_new
    
    async def save_team_to_neo4j(self, team_config: AgentTeamConfig):
        """Save agent team to Neo4j graph"""
        async with self.neo4j_driver.session() as session:
            query = """
            MERGE (team:Team {id: $team_id})
            SET team.tenant_id = $tenant_id,
                team.name = $name,
                team.description = $description,
                team.team_type = $team_type,
                team.domain_focus = $domain_focus,
                team.configuration = $configuration,
                team.status = $status,
                team.updated_at = datetime()
            
            WITH team
            MATCH (t:Tenant {id: $tenant_id})
            MERGE (t)-[r:OWNS_TEAM]->(team)
            SET r.created_at = datetime()
            
            RETURN team
            """
            
            await session.run(
                query,
                team_id=team_config.team_id,
                tenant_id=team_config.tenant_id,
                name=team_config.name,
                description=team_config.description,
                team_type=team_config.team_type,
                domain_focus=team_config.domain_focus,
                configuration=json.dumps({
                    "max_agents": team_config.max_agents,
                    "orchestration_mode": team_config.orchestration_mode,
                    "timeout_seconds": team_config.timeout_seconds,
                    "retry_policy": team_config.retry_policy
                }),
                status=team_config.status.value
            )
