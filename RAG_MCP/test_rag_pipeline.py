#!/usr/bin/env python3
# TA_V8/RAG_MCP/test_rag_pipeline.py
"""
Quick test script for RAG MCP pipeline
Run after services are up: uv run python test_rag_pipeline.py
"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_pipeline():
    """Test the complete RAG pipeline"""
    
    # Configuration
    MCP_URL = "http://localhost:8005"  # Unified MCP server
    AGENT_URL = "http://localhost:8006"  # RAG Agent Team
    
    # Test data
    tenant_id = "test-tenant"
    domain_id = "test-domain"
    collection = f"domain:{domain_id}"
    
    # Sample documents about your RTX 5090
    test_documents = [
        {
            "text": """The NVIDIA RTX 5090 is powered by the Blackwell architecture and features 32GB of GDDR7 memory. 
            It has 21,760 CUDA cores and supports CUDA 12.8 with sm_120 compute capability. 
            The card delivers 20-50% better performance than RTX 4090 in AI workloads."""
        },
        {
            "text": """For RTX 5090 compatibility, use CUDA 12.8 or later. PyTorch 2.7.0 offers native support 
            through CUDA 12.8 wheels. Install with: pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128"""
        },
        {
            "text": """The RTX 5090 excels at multi-agent RAG systems due to its large memory and high compute. 
            It can handle embedding models like BGE-M3 and LLMs like GPT-OSS 21b simultaneously with room to spare."""
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("="*60)
        print("Testing RAG MCP Pipeline")
        print("="*60)
        
        # Step 1: Chunk documents
        print("\n1. CHUNKING DOCUMENTS...")
        for i, doc in enumerate(test_documents):
            try:
                response = await client.post(
                    f"{MCP_URL}/mcp/chunk",
                    json={
                        "tenant_id": tenant_id,
                        "domain_id": domain_id,
                        "source": {"type": "text", "text": doc["text"]},
                        "policy": {"method": "recursive", "target_tokens": 100, "overlap": 20}
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✓ Document {i+1}: Created {len(result['chunks'])} chunks")
                    
                    # Step 2: Embed chunks
                    print(f"   → Embedding chunks...")
                    embed_response = await client.post(
                        f"{MCP_URL}/mcp/embed",
                        json={
                            "tenant_id": tenant_id,
                            "collection": collection,
                            "items": [
                                {
                                    "id": chunk["chunk_id"],
                                    "text": chunk["text"],
                                    "metadata": chunk["metadata"]
                                }
                                for chunk in result["chunks"]
                            ],
                            "upsert": True
                        }
                    )
                    if embed_response.status_code == 200:
                        embed_result = embed_response.json()
                        print(f"   ✓ Embedded and stored {embed_result['upserted']} vectors")
                else:
                    print(f"   ✗ Error chunking document {i+1}: {response.status_code}")
            except Exception as e:
                print(f"   ✗ Exception: {e}")
        
        # Step 3: Test retrieval
        print("\n2. TESTING RETRIEVAL...")
        test_queries = [
            "What are the RTX 5090 specifications?",
            "How do I install PyTorch for RTX 5090?",
            "Why is RTX 5090 good for multi-agent systems?"
        ]
        
        for query in test_queries:
            try:
                response = await client.post(
                    f"{MCP_URL}/mcp/retrieve",
                    json={
                        "tenant_id": tenant_id,
                        "collection": collection,
                        "query": {"text": query, "use_embedding": True},
                        "top_k": 3
                    }
                )
                if response.status_code == 200:
                    hits = response.json()["hits"]
                    print(f"\n   Query: '{query}'")
                    print(f"   Found {len(hits)} relevant chunks:")
                    for hit in hits[:2]:
                        print(f"   - Score {hit['score']:.3f}: {hit['text'][:100]}...")
                else:
                    print(f"   ✗ Retrieval error: {response.status_code}")
            except Exception as e:
                print(f"   ✗ Exception: {e}")
        
        # Step 4: Test RAG Agent Team
        print("\n3. TESTING RAG AGENT TEAM...")
        print("   (This may take 30-60 seconds as agents collaborate)\n")
        
        test_question = "Based on the documentation, what CUDA version and PyTorch version should I use for RTX 5090, and why is it good for RAG?"
        
        try:
            response = await client.post(
                f"{AGENT_URL}/execute",
                json={
                    "query": test_question,
                    "tenant_id": tenant_id,
                    "domain_id": domain_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Query: {test_question}\n")
                print(f"   Final Answer:")
                print("   " + "-"*50)
                print(f"   {result['final_answer']}")
                print("   " + "-"*50)
                print(f"\n   ✓ Answer approved: {result['approved']}")
                print(f"   ✓ Iterations: {result['iterations']}")
            else:
                print(f"   ✗ Agent team error: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   ✗ Exception calling agent team: {e}")
        
        print("\n" + "="*60)
        print("Pipeline test complete!")
        print("="*60)

if __name__ == "__main__":
    print("Starting RAG MCP Pipeline Test...")
    print("Make sure services are running: docker-compose up -d")
    asyncio.run(test_pipeline())
