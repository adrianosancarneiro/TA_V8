#!/usr/bin/env python3
"""
# =============================================================================
# TA_V8 RAG SYSTEM - DEMO PREPARATION SCRIPT
# =============================================================================
# Purpose: Clean existing demo/test data and prepare system for first tenant
# 
# What this script does:
# 1. Clean all existing demo/test data from databases
# 2. Reset vector collections in Qdrant
# 3. Prepare clean database schemas
# 4. Set up first tenant configuration
# 5. Validate system readiness for new documents
# 
# Usage: python prepare_demo_system.py
# =============================================================================
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import asyncpg
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoSystemPreparation:
    """
    Comprehensive demo system preparation and cleanup
    
    Features:
    - Clean all existing tenant data
    - Reset database schemas
    - Prepare vector collections
    - Configure first tenant
    - Validate system readiness
    """
    
    def __init__(self):
        # Database connections
        self.pg_pool = None
        self.qdrant_client = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Configuration
        self.postgres_url = "postgresql://postgres_user:postgres_pass@localhost:5432/ta_v8"
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        
        # First tenant configuration (you can modify this)
        self.first_tenant_config = {
            "tenant_id": "demo_org",
            "name": "Demo Organization",
            "display_name": "Demo Organization - AI Knowledge System",
            "industry": "Technology Demonstration",
            "subscription_tier": "enterprise",
            "api_rate_limit": 10000,
            "storage_quota_gb": 1000,
            "max_users": 100,
            "default_embedding_model": "multilingual-e5-large",
            "default_chunking_method": "auto",
            "default_chunk_size": 512,
            "metadata": {
                "setup_date": datetime.now().isoformat(),
                "demo_purpose": "stakeholder_demo",
                "clean_install": True
            }
        }
        
        self.demo_domain_config = {
            "domain_id": "knowledge_base",
            "name": "Knowledge Base",
            "display_name": "Organizational Knowledge Base",
            "domain_type": "KnowledgeRepository",
            "domain_category": "Information",
            "path": "/knowledge_base",
            "level": 0,
            "attributes": {
                "description": "Central repository for organizational knowledge",
                "content_types": ["documents", "procedures", "policies", "guides"],
                "access_level": "all_users"
            },
            "knowledge_sources": ["documents", "uploads", "manual_entry"],
            "indexing_frequency": "immediate",
            "retention_days": 365
        }

    async def startup(self):
        """Initialize database connections"""
        logger.info("🔧 Initializing database connections...")
        
        try:
            # PostgreSQL connection
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=1,
                max_size=5,
                server_settings={"search_path": "rag_system, public"}
            )
            
            # Qdrant connection
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            
            logger.info("✅ Database connections established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize connections: {str(e)}")
            raise

    async def cleanup_existing_data(self):
        """Clean all existing demo and test data"""
        logger.info("🧹 Cleaning existing demo/test data...")
        
        try:
            async with self.pg_pool.acquire() as conn:
                # Clean in dependency order
                
                logger.info("  Cleaning chunk_retrievals...")
                await conn.execute("DELETE FROM rag_system.chunk_retrievals")
                
                logger.info("  Cleaning queries...")
                await conn.execute("DELETE FROM rag_system.queries")
                
                logger.info("  Cleaning embedding_queue...")
                await conn.execute("DELETE FROM rag_system.embedding_queue")
                
                logger.info("  Cleaning embeddings...")
                await conn.execute("DELETE FROM rag_system.embeddings")
                
                logger.info("  Cleaning embedding_batches...")
                await conn.execute("DELETE FROM rag_system.embedding_batches")
                
                logger.info("  Cleaning chunks...")
                await conn.execute("DELETE FROM rag_system.chunks")
                
                logger.info("  Cleaning documents...")
                await conn.execute("DELETE FROM rag_system.documents")
                
                logger.info("  Cleaning knowledge_entities...")
                await conn.execute("DELETE FROM rag_system.knowledge_entities")
                
                logger.info("  Cleaning domains...")
                await conn.execute("DELETE FROM rag_system.domains")
                
                logger.info("  Cleaning tenants...")
                await conn.execute("DELETE FROM rag_system.tenants")
                
                logger.info("✅ PostgreSQL data cleaned")
                
        except Exception as e:
            logger.error(f"❌ PostgreSQL cleanup failed: {str(e)}")
            raise

    async def cleanup_vector_collections(self):
        """Clean all Qdrant vector collections"""
        logger.info("🧹 Cleaning Qdrant vector collections...")
        
        try:
            # Get all collections
            collections = self.qdrant_client.get_collections()
            
            for collection in collections.collections:
                collection_name = collection.name
                logger.info(f"  Deleting collection: {collection_name}")
                
                try:
                    self.qdrant_client.delete_collection(collection_name)
                except Exception as e:
                    logger.warning(f"  Failed to delete {collection_name}: {str(e)}")
            
            logger.info("✅ Qdrant collections cleaned")
            
        except Exception as e:
            logger.error(f"❌ Qdrant cleanup failed: {str(e)}")
            raise

    async def setup_first_tenant(self):
        """Set up the first tenant with clean configuration"""
        logger.info("👥 Setting up first tenant...")
        
        try:
            async with self.pg_pool.acquire() as conn:
                # Insert tenant
                await conn.execute("""
                    INSERT INTO rag_system.tenants (
                        tenant_id, name, display_name, industry, subscription_tier,
                        api_rate_limit, storage_quota_gb, max_users,
                        default_embedding_model, default_chunking_method, default_chunk_size,
                        metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $13)
                """, 
                    self.first_tenant_config["tenant_id"],
                    self.first_tenant_config["name"],
                    self.first_tenant_config["display_name"],
                    self.first_tenant_config["industry"],
                    self.first_tenant_config["subscription_tier"],
                    self.first_tenant_config["api_rate_limit"],
                    self.first_tenant_config["storage_quota_gb"],
                    self.first_tenant_config["max_users"],
                    self.first_tenant_config["default_embedding_model"],
                    self.first_tenant_config["default_chunking_method"],
                    self.first_tenant_config["default_chunk_size"],
                    json.dumps(self.first_tenant_config["metadata"]),
                    datetime.now()
                )
                
                # Insert default domain
                await conn.execute("""
                    INSERT INTO rag_system.domains (
                        domain_id, tenant_id, name, display_name, domain_type, domain_category,
                        path, level, parent_domain_id, metadata_template, knowledge_sources,
                        indexing_frequency, retention_days, attributes, created_at, updated_at, status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $15, $16)
                """,
                    self.demo_domain_config["domain_id"],
                    self.first_tenant_config["tenant_id"],
                    self.demo_domain_config["name"],
                    self.demo_domain_config["display_name"],
                    self.demo_domain_config["domain_type"],
                    self.demo_domain_config["domain_category"],
                    self.demo_domain_config["path"],
                    self.demo_domain_config["level"],
                    None,  # parent_domain_id
                    json.dumps({}),  # metadata_template
                    self.demo_domain_config["knowledge_sources"],
                    self.demo_domain_config["indexing_frequency"],
                    self.demo_domain_config["retention_days"],
                    json.dumps(self.demo_domain_config["attributes"]),
                    datetime.now(),
                    "active"
                )
                
                logger.info(f"✅ First tenant created: {self.first_tenant_config['tenant_id']}")
                logger.info(f"✅ Default domain created: {self.demo_domain_config['domain_id']}")
                
        except Exception as e:
            logger.error(f"❌ Tenant setup failed: {str(e)}")
            raise

    async def prepare_vector_collection(self):
        """Prepare vector collection for the first tenant"""
        logger.info("🗄️ Preparing vector collections...")
        
        try:
            collection_name = f"{self.first_tenant_config['tenant_id']}_{self.demo_domain_config['domain_id']}"
            
            # Create collection with proper configuration
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1024,  # multilingual-e5-large embedding dimension
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"✅ Vector collection created: {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Vector collection setup failed: {str(e)}")
            raise

    async def validate_services(self):
        """Validate that all required services are running and accessible"""
        logger.info("🔍 Validating service availability...")
        
        services = [
            ("PostgreSQL", "http://localhost:5432"),
            ("Qdrant", "http://localhost:6333"),
            ("Ollama", "http://localhost:11434"),
            ("Chunking MCP", "http://localhost:8001/health"),
            ("Embedding MCP", "http://localhost:8012/health"),
            ("Retrieval MCP", "http://localhost:8003/health"),
            ("RAG Agent Team", "http://localhost:8006/health")
        ]
        
        all_healthy = True
        
        for service_name, url in services:
            try:
                if service_name in ["PostgreSQL", "Qdrant", "Ollama"]:
                    # For core services, check if connection is established
                    if service_name == "PostgreSQL" and self.pg_pool:
                        logger.info(f"✅ {service_name:20} Connected")
                    elif service_name == "Qdrant" and self.qdrant_client:
                        # Test Qdrant connection
                        collections = self.qdrant_client.get_collections()
                        logger.info(f"✅ {service_name:20} Connected")
                    elif service_name == "Ollama":
                        response = await self.http_client.get("http://localhost:11434/api/tags", timeout=5)
                        if response.status_code == 200:
                            logger.info(f"✅ {service_name:20} Available")
                        else:
                            logger.warning(f"⚠️ {service_name:20} HTTP {response.status_code}")
                            all_healthy = False
                else:
                    # For MCP services, check health endpoints
                    response = await self.http_client.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ {service_name:20} Healthy")
                    else:
                        logger.warning(f"⚠️ {service_name:20} HTTP {response.status_code}")
                        all_healthy = False
                        
            except Exception as e:
                logger.error(f"❌ {service_name:20} {str(e)[:50]}")
                all_healthy = False
        
        return all_healthy

    async def create_system_info_file(self):
        """Create a system information file for demo reference"""
        system_info = {
            "system_prepared_at": datetime.now().isoformat(),
            "first_tenant": self.first_tenant_config,
            "default_domain": self.demo_domain_config,
            "system_status": "ready_for_demo",
            "next_steps": [
                "Add your documents using the chunking service",
                "Documents will be automatically embedded and indexed",
                "Use the RAG Agent Team for intelligent question answering",
                "Monitor progress via health endpoints"
            ],
            "api_endpoints": {
                "chunk_documents": "POST http://localhost:8001/mcp/execute",
                "embed_content": "POST http://localhost:8012/mcp/execute", 
                "retrieve_info": "POST http://localhost:8003/mcp/execute",
                "agent_query": "POST http://localhost:8006/query",
                "system_health": "GET http://localhost:8006/health"
            },
            "example_document_upload": {
                "endpoint": "POST http://localhost:8001/mcp/execute",
                "payload": {
                    "tenant_id": self.first_tenant_config["tenant_id"],
                    "domain_id": self.demo_domain_config["domain_id"],
                    "source": {
                        "type": "text",
                        "text": "Your document content here..."
                    },
                    "policy": {
                        "method": "auto",
                        "target_tokens": 512,
                        "overlap": 64
                    }
                }
            }
        }
        
        info_file = "/home/mentorius/AI_Services/TA_V8/RAG_MCP/SYSTEM_READY.json"
        with open(info_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        
        logger.info(f"✅ System info saved to: {info_file}")
        return system_info

    async def shutdown(self):
        """Clean up connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.http_client:
            await self.http_client.aclose()

async def main():
    """Main preparation workflow"""
    print("🎯 TA_V8 RAG SYSTEM - DEMO PREPARATION")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    prep = DemoSystemPreparation()
    
    try:
        # Initialize connections
        await prep.startup()
        
        # Validate services first
        print("🔍 VALIDATING SERVICES")
        print("-" * 30)
        services_ok = await prep.validate_services()
        
        if not services_ok:
            print("\n⚠️  Some services are not available!")
            print("Please ensure all required services are running:")
            print("1. Run: ./deploy_rag_mcp.sh start")
            print("2. Wait for all services to be healthy")
            print("3. Re-run this preparation script")
            return False
        
        print(f"\n✅ All services are available")
        
        # Clean existing data
        print("\n🧹 CLEANING EXISTING DATA")
        print("-" * 30)
        await prep.cleanup_existing_data()
        await prep.cleanup_vector_collections()
        
        # Set up first tenant
        print("\n👥 SETTING UP FIRST TENANT")
        print("-" * 30)
        await prep.setup_first_tenant()
        await prep.prepare_vector_collection()
        
        # Create system info
        print("\n📋 CREATING SYSTEM INFO")
        print("-" * 30)
        system_info = await prep.create_system_info_file()
        
        # Final summary
        print("\n🎉 DEMO SYSTEM READY!")
        print("=" * 30)
        print("✅ All demo/test data cleaned")
        print("✅ Database schemas prepared")
        print("✅ First tenant configured")
        print("✅ Vector collections ready")
        print("✅ Services validated")
        print()
        print("📋 FIRST TENANT DETAILS:")
        print(f"   Tenant ID: {prep.first_tenant_config['tenant_id']}")
        print(f"   Domain ID: {prep.demo_domain_config['domain_id']}")
        print(f"   Collection: {prep.first_tenant_config['tenant_id']}_{prep.demo_domain_config['domain_id']}")
        print()
        print("🚀 READY FOR YOUR DOCUMENTS!")
        print("   You can now upload your documents and they will be:")
        print("   • Chunked into semantic sections")
        print("   • Embedded using multilingual-e5-large")
        print("   • Indexed in Qdrant for retrieval")
        print("   • Available for RAG Agent Team queries")
        print()
        print("📁 System configuration saved to: SYSTEM_READY.json")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo preparation failed: {str(e)}")
        return False
    
    finally:
        await prep.shutdown()

if __name__ == "__main__":
    import uvloop
    try:
        uvloop.install()
    except:
        pass  # Use standard event loop if uvloop not available
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)