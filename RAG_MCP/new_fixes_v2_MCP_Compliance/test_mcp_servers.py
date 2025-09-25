#!/usr/bin/env python3
"""
Test script for MCP servers
Verifies that all three MCP servers are working correctly
"""

import asyncio
import httpx
import json
from typing import Dict, Any

class MCPTester:
    def __init__(self):
        self.chunking_url = "http://localhost:8001/mcp/execute"
        self.embedding_url = "http://localhost:8002/mcp/execute"
        self.retriever_url = "http://localhost:8003/mcp/execute"
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def test_health_checks(self):
        """Test health endpoints"""
        print("\n=== Testing Health Checks ===")
        
        services = [
            ("Chunking", "http://localhost:8001/health"),
            ("Embedding", "http://localhost:8002/health"),
            ("Retriever", "http://localhost:8003/health")
        ]
        
        for name, url in services:
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {name}: {data.get('status', 'unknown')}")
                else:
                    print(f"❌ {name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {name}: {str(e)}")
    
    async def test_chunking(self, tenant_id: str, domain_id: str) -> Dict[str, Any]:
        """Test chunking MCP server"""
        print("\n=== Testing Chunking MCP ===")
        
        sample_text = """
        The Team Agent Platform revolutionizes multi-agent AI systems by providing 
        a comprehensive framework for building, orchestrating, and executing 
        collaborative AI agent teams. 
        
        Key components include the Team Agent Orchestrator (TAO) which manages 
        agent coordination and tool access, the Team Agent Builder (TAB) for 
        creating agent configurations, and the Team Agent Executor (TAE) for 
        runtime execution.
        
        The platform supports multi-tenant deployments with domain-specific 
        knowledge bases and fine-grained permission control. Agents can access 
        various tools through the Model Context Protocol (MCP), enabling 
        seamless integration with external services.
        """
        
        request = {
            "tenant_id": tenant_id,
            "domain_id": domain_id,
            "source": {
                "type": "text",
                "text": sample_text
            },
            "policy": {
                "method": "recursive",
                "target_tokens": 100,
                "overlap": 20
            }
        }
        
        try:
            response = await self.client.post(self.chunking_url, json=request)
            data = response.json()
            
            if response.status_code == 200:
                print(f"✅ Chunking successful:")
                print(f"   - Document ID: {data.get('document_id', 'N/A')}")
                print(f"   - Chunks created: {len(data.get('chunks', []))}")
                print(f"   - Persisted: {data.get('persisted', False)}")
                
                # Display first chunk as example
                if data.get('chunks'):
                    first_chunk = data['chunks'][0]
                    print(f"   - Sample chunk ID: {first_chunk['chunk_id']}")
                    print(f"   - Sample text: {first_chunk['text'][:100]}...")
                
                return data
            else:
                print(f"❌ Chunking failed: HTTP {response.status_code}")
                print(f"   Error: {data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            print(f"❌ Chunking error: {str(e)}")
            return {}
    
    async def test_embedding(self, chunk_data: Dict[str, Any], tenant_id: str, domain_id: str):
        """Test embedding MCP server"""
        print("\n=== Testing Embedding MCP ===")
        
        if not chunk_data.get('chunks'):
            print("⚠️  No chunks available for embedding")
            return
        
        # Prepare embedding request
        items = []
        for chunk in chunk_data['chunks'][:3]:  # Test with first 3 chunks
            items.append({
                "id": chunk['chunk_id'],
                "text": chunk['text'],
                "metadata": chunk['metadata']
            })
        
        request = {
            "tenant_id": tenant_id,
            "collection": f"domain:{domain_id}",
            "items": items,
            "upsert": True
        }
        
        try:
            response = await self.client.post(self.embedding_url, json=request)
            data = response.json()
            
            if response.status_code == 200:
                print(f"✅ Embedding successful:")
                print(f"   - Vectors generated: {len(data.get('vectors', []))}")
                print(f"   - Upserted to Qdrant: {data.get('upserted', 0)}")
                
                # Display first vector info
                if data.get('vectors'):
                    first_vector = data['vectors'][0]
                    print(f"   - Sample vector ID: {first_vector['id']}")
                    print(f"   - Vector dimensions: {len(first_vector.get('vector', []))}")
            else:
                print(f"❌ Embedding failed: HTTP {response.status_code}")
                print(f"   Error: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Embedding error: {str(e)}")
    
    async def test_retrieval(self, tenant_id: str, domain_id: str):
        """Test retrieval MCP server"""
        print("\n=== Testing Retrieval MCP ===")
        
        queries = [
            "What is the Team Agent Platform?",
            "How does TAO work?",
            "multi-agent coordination"
        ]
        
        for query_text in queries:
            request = {
                "tenant_id": tenant_id,
                "collection": f"domain:{domain_id}",
                "query": {
                    "text": query_text,
                    "use_embedding": True
                },
                "top_k": 3,
                "filters": {}
            }
            
            try:
                print(f"\n   Query: '{query_text}'")
                response = await self.client.post(self.retriever_url, json=request)
                data = response.json()
                
                if response.status_code == 200:
                    hits = data.get('hits', [])
                    print(f"   ✅ Found {len(hits)} results")
                    
                    for i, hit in enumerate(hits[:2], 1):  # Show first 2 hits
                        print(f"      Hit {i}:")
                        print(f"        - ID: {hit['id']}")
                        print(f"        - Score: {hit['score']:.3f}")
                        print(f"        - Text: {hit['text'][:100]}...")
                else:
                    print(f"   ❌ Retrieval failed: HTTP {response.status_code}")
                    print(f"      Error: {data.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ Retrieval error: {str(e)}")
    
    async def test_end_to_end(self):
        """Run complete end-to-end test"""
        print("\n" + "="*50)
        print("MCP SERVERS END-TO-END TEST")
        print("="*50)
        
        tenant_id = "test_tenant_001"
        domain_id = "test_domain_001"
        
        # 1. Test health checks
        await self.test_health_checks()
        
        # 2. Test chunking
        chunk_data = await self.test_chunking(tenant_id, domain_id)
        
        if chunk_data:
            # 3. Test embedding
            await self.test_embedding(chunk_data, tenant_id, domain_id)
            
            # Wait a bit for indexing
            await asyncio.sleep(2)
            
            # 4. Test retrieval
            await self.test_retrieval(tenant_id, domain_id)
        
        print("\n" + "="*50)
        print("TEST COMPLETE")
        print("="*50)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.client.aclose()

async def main():
    """Main test function"""
    tester = MCPTester()
    
    try:
        await tester.test_end_to_end()
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    print("\n🚀 Starting MCP Server Tests...")
    print("   Make sure all MCP servers are running on ports 8001-8003")
    print("   Run: docker-compose -f docker-compose-mcp.yml up -d")
    
    asyncio.run(main())
