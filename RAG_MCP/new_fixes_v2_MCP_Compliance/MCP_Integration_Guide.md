# MCP Server Integration Guide for Team Agent Platform

## Summary of Changes

### ✅ What Has Been Modified

The original monolithic implementation has been refactored into **three separate MCP-compliant servers** that can be called by TAO's MCP Gateway and used by agents through TAE's ToolCaller.

### Original Structure (Not MCP-Compliant):
```
unified_mcp_server_enhanced.py
├── /chunk endpoint
├── /embed endpoint  
└── /retrieve endpoint
```

### New Structure (MCP-Compliant):
```
chunking_mcp_server.py    → Port 8001 → /mcp/execute
embedding_mcp_server.py   → Port 8002 → /mcp/execute
retriever_mcp_server.py   → Port 8003 → /mcp/execute
```

## Key Changes Made

### 1. **MCP Protocol Compliance**
- Each server now has a standard `/mcp/execute` endpoint
- Request/response models follow MCP envelope structure
- Proper input/output schemas matching the Multi_Agent_RAG_MVC_Doc_V1.md specifications

### 2. **Service Separation**
- **Chunking MCP** (port 8001): Handles document chunking with persistence
- **Embedding MCP** (port 8002): Generates embeddings and stores in Qdrant
- **Retriever MCP** (port 8003): Performs semantic search using embeddings

### 3. **Inter-Service Communication**
- Retriever MCP calls Embedding MCP for query vectorization
- Services communicate via HTTP using MCP protocol
- Each service is independently deployable and scalable

## Integration with TAO

### Step 1: Register Tools in TAO's ToolRegistry

```sql
-- Add to TAO's tools table
INSERT INTO tools (tool_id, name, input_schema, output_schema, endpoint, status) VALUES
('chunker_v1', 'Chunking Service', '{"type":"object","properties":{"tenant_id":{"type":"string"},"domain_id":{"type":"string"},"source":{"type":"object"},"policy":{"type":"object"}}}', '{"type":"object","properties":{"chunks":{"type":"array"},"persisted":{"type":"boolean"}}}', 'http://chunking-mcp:8001/mcp/execute', 'active'),
('embed_v1', 'Embedding Service', '{"type":"object","properties":{"tenant_id":{"type":"string"},"collection":{"type":"string"},"items":{"type":"array"},"upsert":{"type":"boolean"}}}', '{"type":"object","properties":{"vectors":{"type":"array"},"upserted":{"type":"integer"}}}', 'http://embedding-mcp:8002/mcp/execute', 'active'),
('retriever_v1', 'Retriever Service', '{"type":"object","properties":{"tenant_id":{"type":"string"},"collection":{"type":"string"},"query":{"type":"object"},"top_k":{"type":"integer"},"filters":{"type":"object"}}}', '{"type":"object","properties":{"hits":{"type":"array"}}}', 'http://retriever-mcp:8003/mcp/execute', 'active');
```

### Step 2: Configure Tool Adapters in TAO

```python
# In TAO's tool_adapters.py
class ChunkFetchAdapter(ToolAdapter):
    def __init__(self):
        self.mcp_endpoint = "http://chunking-mcp:8001/mcp/execute"
    
    async def execute(self, params: Dict) -> Dict:
        # Forward to MCP server
        return await self.call_mcp(params)

class EmbeddingAdapter(ToolAdapter):
    def __init__(self):
        self.mcp_endpoint = "http://embedding-mcp:8002/mcp/execute"
    
    async def execute(self, params: Dict) -> Dict:
        # Forward to MCP server
        return await self.call_mcp(params)

class VectorSearchAdapter(ToolAdapter):
    def __init__(self):
        self.mcp_endpoint = "http://retriever-mcp:8003/mcp/execute"
    
    async def execute(self, params: Dict) -> Dict:
        # Forward to MCP server
        return await self.call_mcp(params)
```

### Step 3: Grant Tool Permissions to Agents

```sql
-- Grant retriever agent access to retrieval tool
INSERT INTO member_tools (member_id, tool_id, granted_at) VALUES
('retriever_agent_001', 'retriever_v1', NOW());

-- Grant indexer agent access to chunking and embedding
INSERT INTO member_tools (member_id, tool_id, granted_at) VALUES
('indexer_agent_001', 'chunker_v1', NOW()),
('indexer_agent_001', 'embed_v1', NOW());
```

## Integration with TAE

TAE's ToolCaller will automatically work with these MCP servers through TAO:

```python
# In TAE's agent execution loop
async def agent_action(self, agent_id: str, action: str, params: Dict):
    if action.startswith("TOOL:"):
        tool_name = action.replace("TOOL:", "").strip()
        
        # TAE calls TAO's MCP Gateway
        result = await self.tao_client.execute_tool(
            tool_name=tool_name,
            member_id=agent_id,
            params=params
        )
        
        # TAO forwards to appropriate MCP server
        # Returns result to TAE
        return result
```

## Integration with TAB (Optional)

### Domain Creation Hook

```python
# In TAB's domain creation workflow
async def post_domain_creation(domain_id: str, files: List[str]):
    """After domain is created, ingest and index documents"""
    
    for file_path in files:
        # Read file content
        with open(file_path, 'r') as f:
            text = f.read()
        
        # Call Chunking MCP
        chunk_response = await call_mcp_service(
            "http://chunking-mcp:8001/mcp/execute",
            {
                "tenant_id": current_tenant,
                "domain_id": domain_id,
                "source": {"type": "text", "text": text},
                "policy": {"method": "recursive", "target_tokens": 512}
            }
        )
        
        # Call Embedding MCP for each chunk
        if chunk_response["persisted"]:
            embed_response = await call_mcp_service(
                "http://embedding-mcp:8002/mcp/execute",
                {
                    "tenant_id": current_tenant,
                    "collection": f"domain:{domain_id}",
                    "items": [
                        {"id": c["chunk_id"], "text": c["text"], "metadata": c["metadata"]}
                        for c in chunk_response["chunks"]
                    ],
                    "upsert": True
                }
            )
```

### Default Team Template

```yaml
# Default RAG team template in TAB
name: "RAG Team"
members:
  - id: "retriever_001"
    role: "Retriever"
    persona: "You retrieve relevant information from the knowledge base"
    allowed_tools:
      - "retriever_v1"
  
  - id: "solver_001"
    role: "Solver"  
    persona: "You synthesize information to answer questions"
    allowed_tools: []  # LLM only

execution_plan:
  type: "sequential"
  steps:
    - member: "retriever_001"
      action: "retrieve"
    - member: "solver_001"
      action: "synthesize"
```

## Deployment Instructions

### 1. Deploy MCP Servers

```bash
# Start all MCP servers
docker-compose -f docker-compose-mcp.yml up -d

# Verify health
curl http://localhost:8001/health  # Chunking
curl http://localhost:8002/health  # Embedding
curl http://localhost:8003/health  # Retriever
```

### 2. Test MCP Endpoints

```bash
# Test chunking
curl -X POST http://localhost:8001/mcp/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "t-123",
    "domain_id": "d-abc",
    "source": {"type": "text", "text": "Sample document text..."},
    "policy": {"method": "recursive", "target_tokens": 512}
  }'

# Test retrieval
curl -X POST http://localhost:8003/mcp/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "t-123",
    "collection": "domain:d-abc",
    "query": {"text": "search query", "use_embedding": true},
    "top_k": 5
  }'
```

### 3. Register in TAO

Execute the SQL scripts above to register tools and configure permissions.

### 4. Test End-to-End

Use TAE to execute a team with a retriever agent:

```python
# Via TAE API
response = await tae_client.execute_team(
    team_id="rag_team_001",
    query="What are the key features of our product?"
)
```

## Benefits of This Architecture

### 1. **Platform Integration**
- Fully compatible with TAO's MCP Gateway
- Works seamlessly with TAE's ToolCaller
- Can be configured through TAB

### 2. **Modularity**
- Each service is independently deployable
- Can scale services based on load
- Easy to update individual components

### 3. **Reusability**
- Services can be reused across different teams
- Standard MCP protocol ensures compatibility
- Can be called by any MCP client

### 4. **Multi-Tenancy**
- Built-in tenant isolation
- Domain-scoped collections
- Secure data separation

## Monitoring and Observability

Each MCP server provides:
- `/health` - Health check endpoint
- `/metrics` - Prometheus metrics endpoint

Configure Prometheus to scrape:
```yaml
scrape_configs:
  - job_name: 'mcp-services'
    static_configs:
      - targets:
        - 'chunking-mcp:8001'
        - 'embedding-mcp:8002'  
        - 'retriever-mcp:8003'
```

## Future Enhancements

1. **Add caching layer** - Redis for frequently accessed chunks
2. **Implement batch processing** - For bulk document ingestion
3. **Add more chunking strategies** - LLM-assisted, entity-aware
4. **Enhanced filtering** - Complex query capabilities
5. **Performance optimization** - Connection pooling, async batch operations

## Conclusion

The refactored MCP-compliant services are now ready to be integrated into the Team Agent platform. They follow the exact specifications from the Multi_Agent_RAG_MVC_Doc_V1.md and can be:

1. **Called by TAO** through its MCP Gateway
2. **Used by agents** via TAE's ToolCaller
3. **Configured in TAB** for team building
4. **Reused across** the entire TA platform

This architecture ensures that your RAG capabilities are fully integrated into the Team Agent ecosystem while maintaining the modularity and reusability required for the platform's long-term success.
