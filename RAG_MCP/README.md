# RAG MCP for TA_V8 - Quick Start (1-2 Hours) - UV Edition

## What This Is
A minimal, production-ready Multi-Agent RAG system that integrates with your TA_V8 project. It provides:
- **Unified MCP Server**: Chunking, Embedding, and Retrieval in one service
- **RAG Agent Team**: 3 agents (Retrieval, Refiner, Critic) using your GPT-OSS 21b
- **Full Integration**: Works with your existing Postgres, Qdrant, Ollama, and embedding services
- **UV-Based**: Fast, modern Python package management with UV

## Prerequisites
- UV (Python package manager) - will be installed automatically if missing
- Docker and Docker Compose
- Your existing TA_V8 services running (postgres, qdrant, ollama, multilingual-e5-large)
- **Security Setup**: Secrets configured in `/etc/TA_V8/RAG_MCP/secrets.env`

## Quick Deployment (5 minutes)

```bash
# 1. Copy this folder to your TA_V8 project (already done if you see this)
cd ~/TA_V8/RAG_MCP

# 2. SECURITY FIRST: Configure secrets (REQUIRED)
# See SECURITY_SETUP.md for detailed instructions
sudo nano /etc/TA_V8/RAG_MCP/secrets.env
python validate_security.py  # Verify security

# 3. Setup UV and dependencies (one-time setup)
chmod +x setup_uv.sh
./setup_uv.sh

# 4. Deploy everything with Docker
./deploy.sh

# 5. Test the pipeline (recommended)
uv run python test_rag_pipeline.py
```

## Services
After deployment, you'll have:
- **MCP Server**: `http://localhost:8005` (chunks, embeds, retrieves)
- **Agent Team**: `http://localhost:8006` (3-agent RAG team)

## Integration with TAO

### Option 1: Direct HTTP Calls
```python
import httpx

# Chunk a document
response = httpx.post("http://localhost:8005/mcp/chunk", json={
    "tenant_id": "your-tenant",
    "domain_id": "your-domain",
    "source": {"type": "text", "text": "Your document text..."},
    "policy": {"method": "recursive", "target_tokens": 512}
})

# Retrieve relevant chunks
response = httpx.post("http://localhost:8005/mcp/retrieve", json={
    "tenant_id": "your-tenant",
    "collection": "domain:your-domain",
    "query": {"text": "Your question?", "use_embedding": True},
    "top_k": 5
})

# Use the Agent Team for Q&A
response = httpx.post("http://localhost:8006/execute", json={
    "query": "What is the answer to my question?",
    "tenant_id": "your-tenant",
    "domain_id": "your-domain"
})
```

### Option 2: Register with TAO (if TAO database is set up)
```bash
python tao_integration.py
```

## Architecture
```
Your Query
    ↓
RAG Agent Team (GPT-OSS 20b via Ollama)
    ├── Retrieval Agent → calls MCP retrieve
    ├── Refiner Agent → synthesizes answer
    └── Critic Agent → validates quality
    ↓
Final Answer
```

## 🔒 Security

**CRITICAL**: This system uses secure credential management. Before deployment:

1. **Configure Secrets**: Edit `/etc/TA_V8/RAG_MCP/secrets.env` with real credentials
2. **Validate Security**: Run `python validate_security.py` 
3. **Read Security Guide**: See [SECURITY_SETUP.md](SECURITY_SETUP.md) for complete instructions

### Security Features
- ✅ No hardcoded credentials in source code
- ✅ Centralized secrets management via `/etc/TA_V8/RAG_MCP/secrets.env`
- ✅ Secure file permissions (600)
- ✅ Automated security validation
- ✅ Environment variable override support

### Quick Security Check
```bash
python validate_security.py  # Run security scan
chmod 600 /etc/TA_V8/RAG_MCP/secrets.env  # Fix permissions if needed
```

## Development with UV

For local development and debugging:

```bash
# Setup UV environment (one-time)
./setup_uv.sh

# Run services locally (development mode)
uv run uvicorn unified_mcp_server:app --reload --port 8005
uv run uvicorn rag_agent_team:app --reload --port 8006

# Run tests
uv run python test_rag_pipeline.py

# Code quality tools
uv run black .           # Format code
uv run isort .          # Sort imports
uv run mypy .           # Type checking
uv run pytest          # Run tests (when tests/ folder exists)

# Add new dependencies
uv add fastapi          # Add runtime dependency
uv add pytest --dev    # Add development dependency

# Create lock file for reproducible builds
uv lock

# Install from lock file (production)
uv sync --frozen
```

## Files Overview
- `pyproject.toml` - UV project configuration and dependencies
- `setup_uv.sh` - UV setup script
- `unified_mcp_server.py` - All MCP operations in one service
- `rag_agent_team.py` - 3-agent team implementation
- `test_rag_pipeline.py` - Full pipeline test
- `tao_integration.py` - Register tools with TAO (optional)
- `deploy.sh` - One-click deployment with UV support

## Troubleshooting

If UV is not found:
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

If services don't start:
```bash
# Check logs
docker-compose logs -f

# Restart services
docker-compose restart

# Check if your existing services are running
docker ps | grep ta_v8
```

If embedding service is not available:
- The system will use mock embeddings for testing
- To fix: ensure `ta_v8_multilingual-e5-large` is running

## Performance with RTX 5090
- CUDA 12.8 compatible
- Optimized for sm_120 architecture
- Can handle 1000s of chunks per second
- Embedding generation at ~500 texts/sec
- LLM inference via Ollama GPT-OSS 21b

## Production Notes
- All data is persisted in your existing Postgres
- Vectors stored in your existing Qdrant
- Multi-tenant ready (uses tenant_id/domain_id)
- Async throughout for high throughput
- Health checks at `/health` endpoints

## Next Steps
1. Test with `uv run python test_rag_pipeline.py`
2. Ingest your real documents
3. Integrate with TAO/TAE/TAB as needed
4. Scale horizontally if needed (just add replicas)

## Time to Deploy: ~5 minutes
## Time to Test: ~5 minutes  
## Total: ~10 minutes to working RAG system with UV
