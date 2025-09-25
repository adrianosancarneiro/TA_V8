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
FOR (k:KnowledgeEntity) ON (d.domain_id, k.entity_type);

// Full-text indexes for search
CREATE FULLTEXT INDEX domain_search IF NOT EXISTS
FOR (d:Domain) ON EACH [d.name, d.description, d.keywords];

CREATE FULLTEXT INDEX knowledge_search IF NOT EXISTS
FOR (k:KnowledgeEntity) ON EACH [k.name, k.description, k.content];

CREATE FULLTEXT INDEX document_search IF NOT EXISTS
FOR (d:Document) ON EACH [d.title, d.summary, d.keywords];

// ============================================
// NODE CREATION PATTERNS AND EXAMPLES
// ============================================

// Sample tenant creation
// CREATE (t:Tenant {
//     id: 'demo_tenant',
//     name: 'Demo Organization',
//     industry: 'Technology',
//     created_at: datetime(),
//     metadata: {}
// });

// ============================================
// RELATIONSHIP TYPES
// ============================================

// Tenant relationships
// (:Tenant)-[:HAS_DOMAIN]->(:Domain)
// (:Tenant)-[:HAS_TEAM]->(:Team)

// Domain relationships  
// (:Domain)-[:CONTAINS]->(:KnowledgeEntity)
// (:Domain)-[:HAS_DOCUMENT]->(:Document)
// (:Domain)-[:RELATED_TO]->(:Domain)

// Team relationships
// (:Team)-[:HAS_MEMBER]->(:Member)
// (:Team)-[:USES_TOOL]->(:Tool)
// (:Team)-[:ACCESSES_DOMAIN]->(:Domain)

// Knowledge relationships
// (:KnowledgeEntity)-[:RELATED_TO]->(:KnowledgeEntity)
// (:KnowledgeEntity)-[:DERIVED_FROM]->(:Document)
// (:KnowledgeEntity)-[:HAS_CHUNK]->(:Chunk)

// Document relationships
// (:Document)-[:HAS_CHUNK]->(:Chunk)
// (:Document)-[:VERSION_OF]->(:Document)
// (:Document)-[:REFERENCES]->(:Document)

// Chunk relationships
// (:Chunk)-[:FOLLOWS]->(:Chunk)
// (:Chunk)-[:PART_OF]->(:Document)
// (:Chunk)-[:SEMANTICALLY_SIMILAR]->(:Chunk)

// ============================================
// SAMPLE QUERIES FOR VALIDATION
// ============================================

// Find all domains for a tenant
// MATCH (t:Tenant {id: $tenant_id})-[:HAS_DOMAIN]->(d:Domain)
// RETURN d.name, d.description, d.created_at;

// Find knowledge entities by type within a domain
// MATCH (d:Domain {id: $domain_id})-[:CONTAINS]->(k:KnowledgeEntity {entity_type: $type})
// RETURN k.name, k.description, k.confidence_score;

// Find related knowledge entities
// MATCH (k1:KnowledgeEntity {id: $entity_id})-[:RELATED_TO]-(k2:KnowledgeEntity)
// RETURN k2.name, k2.entity_type, k2.confidence_score
// ORDER BY k2.confidence_score DESC;

// Find chunks related to a document
// MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
// RETURN c.id, c.content_preview, c.chunk_index
// ORDER BY c.chunk_index;