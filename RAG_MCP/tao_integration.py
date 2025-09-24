# TA_V8/RAG_MCP/tao_integration.py
"""
TAO Integration for RAG MCP
Registers RAG MCP tools with the TAO system for use by TAE agents
"""

import asyncio
import asyncpg
import json
from datetime import datetime

import sys
sys.path.append('.')
from shared.config import config

async def register_mcp_tools():
    """Register MCP tools in TAO's database"""
    
    # Connect to TAO's PostgreSQL
    conn = await asyncpg.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        database=config.POSTGRES_DB
    )
    
    try:
        # Tool definitions
        tools = [
            {
                "tool_id": "chunker_v1",
                "name": "Document Chunker",
                "description": "Chunks documents for RAG pipeline",
                "category": "processing",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string"},
                        "domain_id": {"type": "string"},
                        "source": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "text": {"type": "string"}
                            }
                        },
                        "policy": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string"},
                                "target_tokens": {"type": "integer"},
                                "overlap": {"type": "integer"}
                            }
                        }
                    },
                    "required": ["tenant_id", "domain_id", "source"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "chunks": {"type": "array"},
                        "persisted": {"type": "boolean"}
                    }
                },
                "endpoint": "http://ta_v8_rag_mcp_unified:8000/mcp/execute",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "timeout": 30,
                "enabled": True
            },
            {
                "tool_id": "embed_v1",
                "name": "Embedding Service",
                "description": "Creates and stores embeddings",
                "category": "embedding",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string"},
                        "collection": {"type": "string"},
                        "items": {"type": "array"},
                        "upsert": {"type": "boolean"}
                    },
                    "required": ["tenant_id", "collection", "items"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "vectors": {"type": "array"},
                        "upserted": {"type": "integer"}
                    }
                },
                "endpoint": "http://ta_v8_rag_mcp_unified:8000/mcp/execute",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "timeout": 30,
                "enabled": True
            },
            {
                "tool_id": "retriever_v1",
                "name": "RAG Retriever",
                "description": "Retrieves relevant chunks for queries",
                "category": "retrieval",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string"},
                        "collection": {"type": "string"},
                        "query": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "use_embedding": {"type": "boolean"}
                            }
                        },
                        "top_k": {"type": "integer"},
                        "filters": {"type": "object"}
                    },
                    "required": ["tenant_id", "collection", "query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "hits": {"type": "array"}
                    }
                },
                "endpoint": "http://ta_v8_rag_mcp_unified:8000/mcp/execute",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "timeout": 30,
                "enabled": True
            }
        ]
        
        # Register tools in TAO
        for tool in tools:
            # Check if tool already exists
            existing = await conn.fetchval(
                "SELECT tool_id FROM tools WHERE tool_id = $1",
                tool["tool_id"]
            )
            
            if existing:
                # Update existing tool
                await conn.execute("""
                    UPDATE tools 
                    SET name = $2, description = $3, category = $4,
                        input_schema = $5, output_schema = $6,
                        endpoint = $7, method = $8, headers = $9,
                        timeout = $10, enabled = $11, updated_at = NOW()
                    WHERE tool_id = $1
                """, tool["tool_id"], tool["name"], tool["description"],
                     tool["category"], json.dumps(tool["input_schema"]),
                     json.dumps(tool["output_schema"]), tool["endpoint"],
                     tool["method"], json.dumps(tool["headers"]),
                     tool["timeout"], tool["enabled"])
                print(f"✓ Updated tool: {tool['tool_id']}")
            else:
                # Insert new tool
                await conn.execute("""
                    INSERT INTO tools (tool_id, name, description, category,
                                     input_schema, output_schema, endpoint,
                                     method, headers, timeout, enabled,
                                     created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW())
                """, tool["tool_id"], tool["name"], tool["description"],
                     tool["category"], json.dumps(tool["input_schema"]),
                     json.dumps(tool["output_schema"]), tool["endpoint"],
                     tool["method"], json.dumps(tool["headers"]),
                     tool["timeout"], tool["enabled"])
                print(f"✓ Registered tool: {tool['tool_id']}")
        
        # Create a default RAG team configuration
        team_config = {
            "team_id": "rag_team_v1",
            "name": "RAG Agent Team",
            "description": "Multi-agent team for RAG operations",
            "members": [
                {
                    "agent_id": "rag_retriever",
                    "name": "Retrieval Agent",
                    "role": "Finds and retrieves relevant information",
                    "tools": ["retriever_v1"]
                },
                {
                    "agent_id": "rag_refiner",
                    "name": "Refiner Agent",
                    "role": "Refines and synthesizes information",
                    "tools": []
                },
                {
                    "agent_id": "rag_critic",
                    "name": "Critic Agent",
                    "role": "Evaluates and improves answers",
                    "tools": []
                }
            ]
        }
        
        # Check if team exists
        existing_team = await conn.fetchval(
            "SELECT team_id FROM teams WHERE team_id = $1",
            team_config["team_id"]
        )
        
        if not existing_team:
            # Insert team
            await conn.execute("""
                INSERT INTO teams (team_id, tenant_id, name, description, 
                                 config, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
            """, team_config["team_id"], "default", team_config["name"],
                 team_config["description"], json.dumps(team_config))
            print(f"✓ Created team: {team_config['team_id']}")
            
            # Insert team members
            for member in team_config["members"]:
                await conn.execute("""
                    INSERT INTO team_members (team_id, agent_id, name, role,
                                            tools, created_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                """, team_config["team_id"], member["agent_id"],
                     member["name"], member["role"],
                     json.dumps(member["tools"]))
                print(f"  ✓ Added member: {member['agent_id']}")
        
        print("\n✓ TAO integration complete!")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    print("Registering RAG MCP tools with TAO...")
    asyncio.run(register_mcp_tools())
