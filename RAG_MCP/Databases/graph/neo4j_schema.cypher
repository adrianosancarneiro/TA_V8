// ============================================
// NEO4J SCHEMA FOR MULTI-TENANT RAG SYSTEM
// ============================================
// This schema is designed to be flexible and accommodate various domain knowledge types
// (software, business areas, product lines, departments, etc.)

// ============================================
// CONSTRAINTS AND INDEXES
// ============================================

// Unique constraints for primary identifiers
CREATE CONSTRAINT tenant_unique IF NOT EXISTS
FOR (t:Tenant) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT domain_unique IF NOT EXISTS
FOR (d:Domain) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT team_unique IF NOT EXISTS
FOR (t:Team) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT member_unique IF NOT EXISTS
FOR (m:Member) REQUIRE m.id IS UNIQUE;

CREATE CONSTRAINT tool_unique IF NOT EXISTS
FOR (t:Tool) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT chunk_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT document_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT knowledge_entity_unique IF NOT EXISTS
FOR (k:KnowledgeEntity) REQUIRE k.id IS UNIQUE;

// Composite indexes for efficient querying
CREATE INDEX tenant_name IF NOT EXISTS
FOR (t:Tenant) ON (t.name);

CREATE INDEX domain_tenant_composite IF NOT EXISTS
FOR (d:Domain) ON (d.tenant_id, d.name);

CREATE INDEX knowledge_domain_type IF NOT EXISTS
FOR (k:KnowledgeEntity) ON (k.domain_id, k.entity_type);

// Full-text indexes for search
CREATE FULLTEXT INDEX domain_search IF NOT EXISTS
FOR (d:Domain) ON EACH [d.name, d.description, d.keywords];

CREATE FULLTEXT INDEX knowledge_search IF NOT EXISTS
FOR (k:KnowledgeEntity) ON EACH [k.name, k.description, k.tags];

// ============================================
// NODE DEFINITIONS
// ============================================

// TENANT NODE - Represents an organization/client
// Example: Questel
CREATE (t:Tenant {
    id: 'tenant_questel_001',
    name: 'Questel',
    display_name: 'Questel Corporation',
    industry: 'Intellectual Property Software',
    subscription_tier: 'enterprise',
    created_at: datetime(),
    updated_at: datetime(),
    
    // Tenant Configuration (from tenant_config_file_example_V2.json)
    headquarters_location: 'Paris, France',
    founding_year: 1978,
    employee_count: 2500,
    annual_revenue: 150000000,
    currency: 'EUR',
    
    // Compliance & Governance
    compliance_frameworks: ['ISO 27001', 'SOC 2', 'GDPR'],
    data_residency: 'EU',
    
    // Technical Configuration
    core_tech_stack: ['AWS', 'Kubernetes', 'PostgreSQL', 'Neo4j'],
    api_rate_limit: 10000,
    storage_quota_gb: 5000,
    
    // Metadata
    metadata: {
        schema_version: '2.0',
        last_sync: datetime(),
        confidence_score: 0.95,
        data_source: 'tenant_config_v2'
    }
});

// DOMAIN NODE - Represents a knowledge domain (flexible type)
// Can be: Application, BusinessArea, Department, ProductLine, etc.
CREATE (d:Domain {
    id: 'domain_savanta_001',
    tenant_id: 'tenant_questel_001',
    name: 'Savanta',
    display_name: 'Savanta Translation Management',
    
    // Flexible domain type system
    domain_type: 'Application',  // Can be: Application, BusinessArea, Department, ProductLine, Service, etc.
    domain_category: 'Software',  // Higher level categorization
    
    // Hierarchical path for nested domains
    path: '/applications/translation/savanta',
    level: 3,
    parent_id: 'domain_translation_001',
    
    // Domain-specific attributes (flexible based on domain_type)
    attributes: {
        // For Application type
        version: '5.2.1',
        platform: 'web',
        programming_languages: ['Python', 'JavaScript', 'Java'],
        
        // These would be different for other domain types
        // For Department: head_count, budget, cost_center
        // For ProductLine: sku_count, revenue_contribution
        // For BusinessArea: market_segment, competitors
    },
    
    // Knowledge configuration
    knowledge_sources: ['confluence', 'jira', 'github', 'meetings'],
    indexing_frequency: 'daily',
    retention_days: 365,
    
    // Metadata template (from metadata_template_savanta.json)
    metadata_template: {
        features: ['EZLEDES', 'ModelFront', 'NetSuite', 'PreBooking', 'ResourceTracking'],
        content_types: ['FeatureSpecificTechnical', 'FeatureSpecificBusiness', 'GeneralTechnical', 'GeneralBusiness'],
        sources: ['ConfluencePage', 'MeetingTranscript', 'DemoTranscript', 'DevOpsTicket'],
        staleness_levels: ['Deprecated', 'LegacyContext', 'Current', 'InDevelopment']
    },
    
    created_at: datetime(),
    updated_at: datetime(),
    status: 'active'
});

// KNOWLEDGE_ENTITY NODE - Flexible knowledge representation
// Replaces rigid Feature/SubFeature with flexible entity system
CREATE (k:KnowledgeEntity {
    id: 'ke_savanta_pricing_001',
    domain_id: 'domain_savanta_001',
    tenant_id: 'tenant_questel_001',
    
    // Flexible entity system
    entity_type: 'Feature',  // Can be: Feature, Module, Process, Policy, Guideline, etc.
    entity_subtype: 'PricingModule',
    
    name: 'TranslationPricing',
    display_name: 'Translation Pricing Module',
    description: 'Handles pricing calculations for translation services',
    
    // Hierarchical structure
    path: 'TranslationPricing > PricingQualityModifier',
    hierarchy_level: 2,
    parent_entity_id: 'ke_savanta_translation_001',
    
    // Flexible attributes based on entity_type
    properties: {
        complexity: 'high',
        business_impact: 'critical',
        technical_debt: 'medium',
        last_refactor: '2024-06-15',
        owner_team: 'pricing-team',
        dependencies: ['NetSuite', 'ModelFront']
    },
    
    // Tags for flexible categorization
    tags: ['pricing', 'financial', 'core-feature', 'customer-facing'],
    keywords: ['price', 'cost', 'quote', 'estimate', 'billing'],
    
    // Staleness and freshness tracking
    staleness: 'Current',
    staleness_score: 0.85,
    last_validated: datetime(),
    
    created_at: datetime(),
    updated_at: datetime(),
    created_by: 'system',
    updated_by: 'john.doe@questel.com'
});

// DOCUMENT NODE - Represents a document in the system
CREATE (doc:Document {
    id: 'doc_001',
    tenant_id: 'tenant_questel_001',
    domain_id: 'domain_savanta_001',
    
    title: 'Savanta Pricing Module Technical Specification',
    filename: 'savanta_pricing_spec_v5.pdf',
    
    // Storage references
    minio_path: 's3://ta-v8-documents/questel/savanta/docs/pricing_spec_v5.pdf',
    postgres_document_id: 'pg_doc_001',
    
    // Document metadata
    document_type: 'technical_specification',
    format: 'pdf',
    size_bytes: 2548000,
    page_count: 45,
    
    // Content analysis (generated by LLM)
    summary: 'Technical specification for the Savanta translation pricing module...',
    main_topics: ['pricing algorithms', 'quality modifiers', 'integration points'],
    
    // Extracted entities
    detected_features: ['TranslationPricing', 'PricingQualityModifier'],
    detected_entities: ['ke_savanta_pricing_001', 'ke_savanta_quality_001'],
    
    // Document-level tags (generated during initial processing)
    document_tags: {
        technical_level: 'advanced',
        audience: 'developers',
        confidentiality: 'internal',
        review_status: 'approved'
    },
    
    // Versioning
    version: '5.0',
    previous_version_id: 'doc_000',
    
    // Timestamps
    created_at: datetime(),
    updated_at: datetime(),
    indexed_at: datetime(),
    last_accessed: datetime()
});

// CHUNK NODE - Represents a chunk of a document
CREATE (c:Chunk {
    id: 'chunk_001',
    document_id: 'doc_001',
    tenant_id: 'tenant_questel_001',
    domain_id: 'domain_savanta_001',
    
    // Chunk identification
    chunk_index: 1,
    chunk_hash: 'sha256_hash_here',
    
    // Text reference (actual text in PostgreSQL)
    postgres_chunk_id: 'pg_chunk_001',
    text_snippet: 'First 100 characters of chunk for preview...',
    
    // Chunk metadata
    token_count: 450,
    char_count: 2100,
    start_char: 0,
    end_char: 2100,
    
    // Chunking method
    method: 'semantic_coherence',
    overlap_tokens: 50,
    
    // Semantic tags (generated by LLM during embedding)
    chunk_tags: {
        primary_topic: 'quality_modifiers',
        subtopics: ['machine_translation', 'human_translation', 'pricing'],
        technical_concepts: ['algorithm', 'calculation', 'formula'],
        business_concepts: ['cost', 'margin', 'discount'],
        
        // Knowledge entity references
        related_features: ['TranslationPricing', 'PricingQualityModifier'],
        related_entities: ['ke_savanta_pricing_001'],
        
        // Context tags
        context_type: 'implementation_detail',
        importance: 'high',
        complexity: 'medium'
    },
    
    // Embedding reference
    qdrant_point_id: 'qdrant_point_001',
    embedding_model: 'multilingual-e5-large',
    embedding_dimension: 1024,
    
    // Quality and freshness
    quality_score: 0.92,
    relevance_score: 0.88,
    staleness: 'current',
    
    created_at: datetime(),
    embedded_at: datetime()
});

// TEAM NODE - Agent team configuration
CREATE (team:Team {
    id: 'team_001',
    tenant_id: 'tenant_questel_001',
    name: 'PricingAnalysisTeam',
    description: 'Team for analyzing and optimizing pricing strategies',
    
    team_type: 'analytical',
    domain_focus: ['domain_savanta_001'],
    
    configuration: {
        max_agents: 5,
        orchestration_mode: 'sequential',
        timeout_seconds: 300,
        retry_policy: 'exponential_backoff'
    },
    
    status: 'active',
    created_at: datetime()
});

// MEMBER NODE - Agent/team member
CREATE (m:Member {
    id: 'member_001',
    team_id: 'team_001',
    name: 'PricingAnalyst',
    role: 'analyzer',
    
    persona: 'Expert in financial analysis and pricing strategies',
    
    capabilities: ['data_analysis', 'report_generation', 'trend_detection'],
    assigned_tools: ['tool_001', 'tool_002'],
    
    llm_config: {
        model: 'gpt-4',
        temperature: 0.7,
        max_tokens: 2000
    },
    
    status: 'active',
    created_at: datetime()
});

// TOOL NODE - Available tools
CREATE (tool:Tool {
    id: 'tool_001',
    name: 'VectorSearch',
    type: 'search',
    category: 'retrieval',
    
    description: 'Semantic search using vector embeddings',
    
    endpoint: 'http://vector-search-service:8080',
    api_version: 'v2',
    
    input_schema: {
        query: 'string',
        top_k: 'integer',
        filters: 'object'
    },
    
    output_schema: {
        results: 'array',
        scores: 'array',
        metadata: 'object'
    },
    
    configuration: {
        timeout: 30,
        retry_count: 3,
        cache_ttl: 3600
    },
    
    status: 'active',
    created_at: datetime()
});

// QUERY NODE - Track user queries
CREATE (q:Query {
    id: 'query_001',
    tenant_id: 'tenant_questel_001',
    user_id: 'user_001',
    
    query_text: 'How does the pricing quality modifier work?',
    query_embedding_id: 'emb_query_001',
    
    context: {
        domain_id: 'domain_savanta_001',
        team_id: 'team_001',
        session_id: 'session_001'
    },
    
    results: {
        chunks_retrieved: ['chunk_001', 'chunk_002'],
        relevance_scores: [0.92, 0.87],
        response_generated: true
    },
    
    performance: {
        latency_ms: 245,
        tokens_used: 1250
    },
    
    timestamp: datetime()
});

// ============================================
// RELATIONSHIPS
// ============================================

// Tenant relationships
MATCH (t:Tenant {id: 'tenant_questel_001'}), (d:Domain {id: 'domain_savanta_001'})
CREATE (t)-[:HAS_DOMAIN {
    assigned_at: datetime(),
    access_level: 'full',
    data_isolation: true
}]->(d);

MATCH (t:Tenant {id: 'tenant_questel_001'}), (team:Team {id: 'team_001'})
CREATE (t)-[:OWNS_TEAM {
    created_at: datetime()
}]->(team);

// Domain hierarchy
MATCH (parent:Domain {id: 'domain_translation_001'}), (child:Domain {id: 'domain_savanta_001'})
CREATE (parent)-[:HAS_SUBDOMAIN {
    hierarchy_level: 1,
    inheritance: true
}]->(child);

// Knowledge entity relationships
MATCH (d:Domain {id: 'domain_savanta_001'}), (k:KnowledgeEntity {id: 'ke_savanta_pricing_001'})
CREATE (d)-[:CONTAINS_KNOWLEDGE {
    category: 'feature',
    importance: 'high'
}]->(k);

MATCH (k1:KnowledgeEntity {id: 'ke_savanta_translation_001'}), 
      (k2:KnowledgeEntity {id: 'ke_savanta_pricing_001'})
CREATE (k1)-[:HAS_SUBENTITY {
    relationship_type: 'parent_child',
    inheritance: true
}]->(k2);

// Related features (many-to-many)
MATCH (k1:KnowledgeEntity {id: 'ke_savanta_pricing_001'}), 
      (k2:KnowledgeEntity {id: 'ke_savanta_netsuite_001'})
CREATE (k1)-[:RELATES_TO {
    relationship_type: 'dependency',
    strength: 'strong',
    bidirectional: true
}]->(k2);

// Document relationships
MATCH (doc:Document {id: 'doc_001'}), (d:Domain {id: 'domain_savanta_001'})
CREATE (doc)-[:BELONGS_TO_DOMAIN {
    primary: true,
    relevance: 1.0
}]->(d);

MATCH (doc:Document {id: 'doc_001'}), (k:KnowledgeEntity {id: 'ke_savanta_pricing_001'})
CREATE (doc)-[:DESCRIBES {
    coverage: 'comprehensive',
    detail_level: 'technical'
}]->(k);

// Chunk relationships
MATCH (doc:Document {id: 'doc_001'}), (c:Chunk {id: 'chunk_001'})
CREATE (doc)-[:HAS_CHUNK {
    sequence: 1,
    method: 'semantic_coherence'
}]->(c);

MATCH (c:Chunk {id: 'chunk_001'}), (k:KnowledgeEntity {id: 'ke_savanta_pricing_001'})
CREATE (c)-[:REFERENCES {
    relevance_score: 0.95,
    confidence: 0.90
}]->(k);

// Sequential chunk relationships
MATCH (c1:Chunk {id: 'chunk_001'}), (c2:Chunk {id: 'chunk_002'})
WHERE c1.document_id = c2.document_id AND c2.chunk_index = c1.chunk_index + 1
CREATE (c1)-[:NEXT_CHUNK {
    overlap_tokens: 50
}]->(c2);

// Team and Member relationships
MATCH (team:Team {id: 'team_001'}), (m:Member {id: 'member_001'})
CREATE (team)-[:HAS_MEMBER {
    role: 'analyzer',
    joined_at: datetime()
}]->(m);

MATCH (m:Member {id: 'member_001'}), (tool:Tool {id: 'tool_001'})
CREATE (m)-[:CAN_USE {
    permission_level: 'full',
    usage_limit: 1000
}]->(tool);

// Query tracking relationships
MATCH (q:Query {id: 'query_001'}), (c:Chunk {id: 'chunk_001'})
CREATE (q)-[:RETRIEVED {
    relevance_score: 0.92,
    rank: 1
}]->(c);

MATCH (q:Query {id: 'query_001'}), (team:Team {id: 'team_001'})
CREATE (q)-[:PROCESSED_BY {
    latency_ms: 245
}]->(team);

// ============================================
// INDEXES FOR PERFORMANCE
// ============================================

// Vector similarity search optimization
CREATE INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.qdrant_point_id);

// Query performance
CREATE INDEX query_timestamp IF NOT EXISTS
FOR (q:Query) ON (q.timestamp);

// Document retrieval
CREATE INDEX document_domain IF NOT EXISTS
FOR (d:Document) ON (d.domain_id);

// ============================================
// EXAMPLE QUERIES
// ============================================

// 1. Find all knowledge entities for a tenant's domain
MATCH (t:Tenant {id: $tenant_id})-[:HAS_DOMAIN]->(d:Domain)
MATCH (d)-[:CONTAINS_KNOWLEDGE]->(k:KnowledgeEntity)
RETURN k.name, k.entity_type, k.properties;

// 2. Get document chunks with their knowledge context
MATCH (c:Chunk {id: $chunk_id})-[:REFERENCES]->(k:KnowledgeEntity)
MATCH (c)<-[:HAS_CHUNK]-(d:Document)
RETURN c, d.title, collect(k.name) as referenced_entities;

// 3. Find related features across domains
MATCH (k:KnowledgeEntity {id: $entity_id})-[:RELATES_TO*1..2]-(related:KnowledgeEntity)
RETURN related.name, related.entity_type, related.domain_id;

// 4. Track query patterns for a domain
MATCH (q:Query)-[:RETRIEVED]->(c:Chunk)-[:REFERENCES]->(k:KnowledgeEntity)
WHERE k.domain_id = $domain_id
RETURN k.name, count(q) as query_count
ORDER BY query_count DESC;
