# TA_V8 RAG_MCP Migration Plan - MCP Compliance & Platform Integration

## 🎯 **MIGRATION OBJECTIVE**
Migrate from monolithic `unified_mcp_server.py` to MCP-compliant microservices architecture while organizing folder structure for future TAB, TAE, TAO integration.

## ⏱️ **TIMELINE: 2-3 Hours**

---

## 📋 **PHASE 1: FOLDER STRUCTURE ORGANIZATION (30 minutes)**

### **1.1 Create TA_V8 Platform-Ready Structure**
```bash
# Main platform modules (future homes for TAB, TAE, TAO)
RAG_MCP/
├── platform_modules/          # Future TA_V8 platform components
│   ├── TAB_integration/       # Team Agent Builder integration point
│   ├── TAE_integration/       # Team Agent Executor integration point  
│   ├── TAO_integration/       # Team Agent Orchestrator integration point
│   └── shared_platform/       # Shared utilities across TAB/TAE/TAO
│
├── mcp_services/              # MCP-compliant microservices
│   ├── chunking/              # Document chunking MCP service
│   ├── embedding/             # Embedding generation MCP service
│   ├── retrieval/             # Semantic retrieval MCP service
│   └── shared/                # Shared MCP utilities
│
├── legacy/                    # Current unified server (backup)
│   ├── unified_mcp_server.py  # MOVED: Original monolithic server
│   └── backup_configs/        # Original configs for rollback
│
├── infrastructure/            # Database schemas and configs
│   ├── databases/             # Database schemas (PostgreSQL, Neo4j)
│   ├── docker/                # Docker configurations
│   └── monitoring/            # Health checks and monitoring
│
├── testing/                   # Comprehensive test suites
│   ├── integration/           # Integration tests
│   ├── mcp_compliance/        # MCP protocol compliance tests
│   └── platform_tests/        # TAB/TAE/TAO integration tests
│
└── deployment/                # Production deployment configs
    ├── docker-compose/        # Service orchestration
    ├── kubernetes/            # K8s manifests (future)
    └── scripts/               # Deployment automation
```

### **1.2 TODOs for Folder Structure:**
- [ ] **Create platform_modules/ with TAB/TAE/TAO placeholders**
- [ ] **Move current files to legacy/ for backup**  
- [ ] **Copy MCP services from new_fixes_v2_MCP_Compliance/**
- [ ] **Organize databases/ under infrastructure/**
- [ ] **Create testing/ with proper test organization**

---

## 📋 **PHASE 2: MCP SERVICES MIGRATION (45 minutes)**

### **2.1 Copy & Organize MCP Services**
```bash
# Copy services from new_fixes_v2_MCP_Compliance to proper locations
# DIRECT COPY - NO MODIFICATIONS YET

Source: new_fixes_v2_MCP_Compliance/
Target: mcp_services/

chunking_mcp_server.py → mcp_services/chunking/server.py
embedding_mcp_server.py → mcp_services/embedding/server.py  
retriever_mcp_server.py → mcp_services/retrieval/server.py
test_mcp_servers.py → testing/mcp_compliance/test_all_services.py
MCP_Integration_Guide.md → mcp_services/README.md
```

### **2.2 TODOs for MCP Services Migration:**
- [ ] **Copy chunking_mcp_server.py → mcp_services/chunking/server.py**
- [ ] **Copy embedding_mcp_server.py → mcp_services/embedding/server.py**
- [ ] **Copy retriever_mcp_server.py → mcp_services/retrieval/server.py**
- [ ] **Copy Dockerfile.* → mcp_services/*/Dockerfile** 
- [ ] **Copy docker-compose-mcp.yml → deployment/docker-compose/mcp-services.yml**
- [ ] **Create __init__.py files in each service directory**
- [ ] **Copy test_mcp_servers.py → testing/mcp_compliance/**

### **2.3 Service Configuration Updates**
```python
# Each service needs these inline comments added:

# =============================================================================
# MCP SERVICE: [CHUNKING/EMBEDDING/RETRIEVAL] 
# =============================================================================
# Purpose: [Specific service purpose]
# Port: [8001/8002/8003]
# Protocol: MCP-compliant via /mcp/execute endpoint
# 
# Integration Points:
# - TAO: Will register this service in Tool Registry
# - TAE: Will call via ToolCaller for agent tool usage  
# - TAB: Will use for knowledge base setup workflows
# 
# Dependencies: PostgreSQL, Neo4j, Qdrant, MinIO
# Status: MIGRATED - Ready for platform integration
# =============================================================================
```

### **2.4 TODOs for Service Comments:**
- [ ] **Add detailed inline comments to chunking/server.py**
- [ ] **Add detailed inline comments to embedding/server.py**  
- [ ] **Add detailed inline comments to retrieval/server.py**
- [ ] **Document MCP protocol compliance in each service**
- [ ] **Add platform integration notes (TAB/TAE/TAO)**

---

## 📋 **PHASE 3: DATABASE & INFRASTRUCTURE MIGRATION (30 minutes)**

### **3.1 Database Schema Organization**
```bash
# Move database schemas to infrastructure/
Databases/graph/neo4j_schema.cypher → infrastructure/databases/neo4j/schema.cypher
Databases/relational/postgresql_schema.sql → infrastructure/databases/postgresql/schema.sql

# Add migration scripts
infrastructure/databases/migrations/
├── 001_initial_postgresql_schema.sql
├── 002_initial_neo4j_schema.cypher  
├── 003_mcp_services_tables.sql        # New tables for MCP services
└── rollback/                          # Rollback scripts
```

### **3.2 TODOs for Database Migration:**
- [ ] **Move neo4j_schema.cypher to infrastructure/databases/neo4j/**
- [ ] **Move postgresql_schema.sql to infrastructure/databases/postgresql/**
- [ ] **Create migration scripts with version numbers**
- [ ] **Add rollback scripts for safe deployment**
- [ ] **Document schema changes for MCP compliance**

### **3.3 Docker Configuration**
```bash
# Organize docker configs
infrastructure/docker/
├── base/                      # Base images
├── services/                  # Individual service Dockerfiles  
└── compose/                   # docker-compose files

deployment/docker-compose/
├── development.yml            # Dev environment
├── production.yml             # Production environment  
├── mcp-services.yml          # MCP services only
└── legacy.yml                # Fallback to unified server
```

### **3.4 TODOs for Docker Migration:**
- [ ] **Move Dockerfile.* to infrastructure/docker/services/**
- [ ] **Update docker-compose-mcp.yml → deployment/docker-compose/mcp-services.yml**
- [ ] **Create development.yml for local testing**
- [ ] **Create production.yml with proper networking**
- [ ] **Add health checks and monitoring to all services**

---

## 📋 **PHASE 4: TESTING & VALIDATION (45 minutes)**

### **4.1 Test Suite Organization**
```bash
testing/
├── integration/
│   ├── test_mcp_services_integration.py    # Services work together
│   ├── test_database_integration.py        # DB connections work
│   └── test_end_to_end_workflow.py         # Full RAG workflow
│
├── mcp_compliance/
│   ├── test_mcp_protocol.py               # MCP standard compliance
│   ├── test_chunking_mcp.py               # Chunking service tests
│   ├── test_embedding_mcp.py              # Embedding service tests
│   └── test_retrieval_mcp.py              # Retrieval service tests
│
├── platform_tests/                        # Future TAB/TAE/TAO tests
│   ├── test_tao_integration.py            # TAO MCP Gateway integration
│   ├── test_tae_tool_caller.py            # TAE ToolCaller compatibility  
│   └── test_tab_workflow.py               # TAB team building workflows
│
└── performance/
    ├── test_load_performance.py           # Load testing
    └── test_service_scaling.py            # Horizontal scaling tests
```

### **4.2 TODOs for Testing Migration:**
- [ ] **Copy test_mcp_servers.py → testing/mcp_compliance/test_all_services.py**
- [ ] **Create comprehensive integration tests**
- [ ] **Add MCP protocol compliance verification**
- [ ] **Create performance benchmarks vs unified server**
- [ ] **Add platform integration test stubs**

### **4.3 Test Execution Plan**
```bash
# Test execution sequence (inline with migration)
1. MCP Protocol Compliance: Verify /mcp/execute endpoints
2. Service Integration: Test service-to-service communication  
3. Database Integration: Test all CRUD operations
4. End-to-End RAG: Full document → chunk → embed → retrieve flow
5. Performance Comparison: MCP services vs unified server
```

### **4.4 TODOs for Test Execution:**
- [ ] **Run MCP compliance tests on all 3 services**
- [ ] **Verify inter-service HTTP communication**
- [ ] **Test database schema migrations**
- [ ] **Execute full RAG workflow test**
- [ ] **Benchmark performance vs legacy system**

---

## 📋 **PHASE 5: PLATFORM INTEGRATION PREPARATION (30 minutes)**

### **5.1 TAO Integration Points**
```python
# platform_modules/TAO_integration/mcp_registry.py
"""
TAO MCP Tool Registry Integration

This module provides the bridge between TAO's ToolRegistry and our MCP services.
Each MCP service will be registered as a tool that TAO can orchestrate.

Integration Steps:
1. Register chunking service as 'chunker_v1' tool in TAO
2. Register embedding service as 'embed_v1' tool in TAO  
3. Register retrieval service as 'retriever_v1' tool in TAO
4. Configure tool permissions per agent team member
5. Enable TAO's MCP Gateway to route requests to our services
"""

class MCPServiceRegistry:
    """Register our MCP services with TAO's ToolRegistry"""
    
    def register_chunking_service(self):
        # TAO ToolRegistry integration for chunking service
        pass
    
    def register_embedding_service(self):  
        # TAO ToolRegistry integration for embedding service
        pass
        
    def register_retrieval_service(self):
        # TAO ToolRegistry integration for retrieval service  
        pass
```

### **5.2 TAE Integration Points**
```python  
# platform_modules/TAE_integration/tool_caller.py
"""
TAE ToolCaller Integration

This module enables TAE's ToolCaller to invoke our MCP services
when agents need RAG capabilities during execution.

Integration Steps:
1. TAE ToolCaller calls TAO's MCP Gateway
2. TAO routes to appropriate MCP service (chunking/embedding/retrieval)
3. MCP service processes request and returns MCP-compliant response
4. Response flows back: MCP Service → TAO → TAE → Agent
"""

class RAGToolCaller:
    """TAE integration for RAG tool calling"""
    
    def call_chunking_service(self, document, tenant_id):
        # TAE → TAO → Chunking MCP integration
        pass
    
    def call_embedding_service(self, chunks, tenant_id):
        # TAE → TAO → Embedding MCP integration  
        pass
        
    def call_retrieval_service(self, query, tenant_id):
        # TAE → TAO → Retrieval MCP integration
        pass
```

### **5.3 TAB Integration Points**
```python
# platform_modules/TAB_integration/knowledge_builder.py  
"""
TAB Team Building Integration

This module enables TAB's WizardEngine and team building workflows
to configure RAG capabilities when creating agent teams.

Integration Steps:
1. TAB WizardEngine asks about knowledge sources during team creation
2. If RAG capabilities needed, configure domain knowledge via our services
3. Set up team member tool permissions for RAG tools
4. Register domain knowledge in Neo4j via our database schemas
"""

class RAGKnowledgeBuilder:
    """TAB integration for RAG-enabled team building"""
    
    def configure_domain_knowledge(self, domain_id, knowledge_sources):
        # TAB domain knowledge setup via our MCP services
        pass
        
    def setup_rag_permissions(self, team_id, member_id):
        # Configure agent permissions for RAG tools
        pass
```

### **5.4 TODOs for Platform Integration:**
- [ ] **Create TAO integration stubs and documentation**
- [ ] **Create TAE integration stubs and documentation**  
- [ ] **Create TAB integration stubs and documentation**
- [ ] **Document MCP service registration process**
- [ ] **Plan tool permission configuration**

---

## 📋 **PHASE 6: DEPLOYMENT & VALIDATION (20 minutes)**

### **6.1 Deployment Sequence**
```bash
# Step-by-step deployment with rollback capability

1. Deploy infrastructure (databases, networking)
   docker-compose -f deployment/docker-compose/production.yml up -d postgres neo4j qdrant minio

2. Run database migrations  
   ./infrastructure/databases/migrations/run_migrations.sh

3. Deploy MCP services
   docker-compose -f deployment/docker-compose/mcp-services.yml up -d

4. Run validation tests
   python testing/mcp_compliance/test_all_services.py
   python testing/integration/test_end_to_end_workflow.py

5. Performance validation
   python testing/performance/test_load_performance.py
```

### **6.2 Rollback Plan**
```bash
# If migration fails, immediate rollback to unified server

1. Stop MCP services
   docker-compose -f deployment/docker-compose/mcp-services.yml down

2. Start legacy unified server
   docker-compose -f deployment/docker-compose/legacy.yml up -d

3. Verify legacy system health
   curl http://localhost:8005/health

4. Investigation mode: Keep both systems running for comparison
   # MCP services on ports 8001-8003
   # Legacy unified on port 8005
```

### **6.3 TODOs for Deployment:**
- [ ] **Create automated deployment script**
- [ ] **Test rollback procedure thoroughly**
- [ ] **Set up monitoring and alerting**  
- [ ] **Document production deployment process**
- [ ] **Create operational runbook**

---

## 📋 **SUCCESS CRITERIA & VALIDATION CHECKLIST**

### **✅ Technical Validation**
- [ ] **All 3 MCP services respond on ports 8001-8003**
- [ ] **MCP protocol compliance verified (/mcp/execute endpoints)**
- [ ] **Inter-service communication working (retrieval → embedding)**
- [ ] **Database connectivity confirmed (PostgreSQL, Neo4j, Qdrant, MinIO)**
- [ ] **Full RAG workflow: document → chunks → embeddings → retrieval**

### **✅ Performance Validation**  
- [ ] **Response times ≤ legacy unified server**
- [ ] **Memory usage reasonable across 3 services** 
- [ ] **No data loss during migration**
- [ ] **Error rates < 0.1%**
- [ ] **Health checks passing on all services**

### **✅ Platform Readiness**
- [ ] **Folder structure supports TAB/TAE/TAO integration**
- [ ] **MCP services ready for TAO ToolRegistry**
- [ ] **Documentation complete for platform integration**
- [ ] **Integration stubs created for all platform modules**
- [ ] **Migration reversible with rollback plan**

### **✅ Operational Readiness**
- [ ] **Docker compose files production-ready**
- [ ] **Database schemas versioned and migration-ready**  
- [ ] **Comprehensive test suites in place**
- [ ] **Monitoring and health checks configured**
- [ ] **Documentation updated and complete**

---

## 🚀 **NEXT STEPS AFTER MIGRATION**

### **Immediate (Next Few Days)**
1. **TAO Integration**: Register MCP services in TAO's ToolRegistry
2. **TAE Integration**: Test agent tool calling via TAO → MCP services  
3. **Performance Tuning**: Optimize service communication and caching
4. **Production Hardening**: Add comprehensive error handling and resilience

### **Near Term (Next Few Weeks)**  
1. **TAB Integration**: Enable RAG configuration during team building
2. **Advanced Features**: Implement semantic search improvements
3. **Scaling**: Add horizontal scaling capabilities
4. **Security**: Implement proper authentication and authorization

### **Long Term (Next Few Months)**
1. **Full Platform Integration**: Complete TAB/TAE/TAO ecosystem
2. **Advanced RAG**: Multi-modal embeddings, advanced retrieval strategies  
3. **Enterprise Features**: Multi-tenancy, advanced analytics, governance
4. **Kubernetes Deployment**: Production-grade orchestration

---

## 💡 **MIGRATION EXECUTION COMMANDS**

```bash
# Execute migration in this exact order:

# Phase 1: Structure  
./scripts/create_folder_structure.sh

# Phase 2: Copy services
./scripts/copy_mcp_services.sh  

# Phase 3: Databases
./scripts/migrate_databases.sh

# Phase 4: Testing
./scripts/run_test_suite.sh

# Phase 5: Platform prep
./scripts/prepare_platform_integration.sh

# Phase 6: Deploy
./scripts/deploy_mcp_services.sh
```

**Estimated Total Time: 2-3 Hours**
**Risk Level: Low (rollback plan in place)**
**Success Probability: High (incremental migration)**