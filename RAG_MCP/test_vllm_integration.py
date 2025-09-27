#!/usr/bin/env python3
"""
Test vLLM Integration with RAG MCP System

This test validates that the RAG system works with vLLM instead of Ollama.
"""

import asyncio
import httpx
import json
from datetime import datetime

# Test basic vLLM connectivity
async def test_vllm_basic_connectivity():
    """Test basic vLLM connectivity"""
    print("🔗 Testing vLLM Basic Connectivity")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient() as client:
            # Test models endpoint
            response = await client.get("http://localhost:8000/v1/models")
            if response.status_code == 200:
                models = response.json()
                print(f"✅ vLLM service is running")
                print(f"📋 Available models: {[m['id'] for m in models['data']]}")
                return True
            else:
                print(f"❌ vLLM service returned status: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Failed to connect to vLLM: {e}")
        return False

async def test_vllm_chat_completion():
    """Test vLLM chat completion"""
    print("\n💬 Testing vLLM Chat Completion")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            request_data = {
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
                ],
                "temperature": 0.7,
                "max_tokens": 50
            }
            
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"✅ vLLM chat completion successful")
                print(f"📝 Response: {content.strip()}")
                return True
            else:
                print(f"❌ Chat completion failed: {response.status_code}")
                print(f"📄 Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return False

async def test_rag_agent_initialization():
    """Test RAG agent team initialization with vLLM"""
    print("\n🤖 Testing RAG Agent Team Initialization")
    print("=" * 50)
    
    try:
        # Change to RAG_MCP directory for imports
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from rag_agent_team import RAGAgentTeam
        
        # Initialize RAG agent team
        rag_team = RAGAgentTeam()
        print(f"✅ RAG Agent Team initialized")
        print(f"🔗 vLLM URL: {rag_team.vllm_url}")
        print(f"🤖 LLM Model: {rag_team.model}")
        
        # Test health check
        await rag_team.initialize()
        health_status = await rag_team.health_check()
        
        print(f"🏥 Health Status: {health_status.get('status', 'unknown')}")
        
        # Check vLLM service health specifically
        vllm_status = health_status.get('services', {}).get('vllm', {})
        if vllm_status.get('status') == 'healthy':
            print("✅ vLLM service health check passed")
            return True
        else:
            print(f"❌ vLLM service health check failed: {vllm_status}")
            return False
            
    except Exception as e:
        print(f"❌ RAG Agent initialization error: {e}")
        return False

async def test_document_analyzer_with_vllm():
    """Test document analyzer with vLLM"""
    print("\n📄 Testing Document Analyzer with vLLM")
    print("=" * 50)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from document_chunker import DocumentAnalyzer
        import httpx
        
        # Initialize document analyzer with vLLM
        vllm_client = httpx.AsyncClient()
        analyzer = DocumentAnalyzer(
            vllm_client=vllm_client,
            llm_model="openai/gpt-oss-20b",
            vllm_url="http://localhost:8000"
        )
        
        print("✅ DocumentAnalyzer initialized with vLLM")
        
        # Test document analysis (simple test)
        test_doc = """
        # Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that focuses on algorithms
        that can learn and make decisions from data without being explicitly programmed.
        
        ## Types of Machine Learning
        
        1. Supervised Learning: Uses labeled data to train models
        2. Unsupervised Learning: Finds patterns in unlabeled data
        3. Reinforcement Learning: Learns through interaction with environment
        """
        
        # Run analysis
        strategy, details = await analyzer.analyze_and_recommend_strategy(test_doc)
        
        print(f"✅ Document analysis completed")
        print(f"📋 Recommended strategy: {strategy}")
        print(f"📊 Analysis details: {details.get('reasoning', 'No reasoning provided')[:100]}...")
        
        await vllm_client.aclose()
        return True
        
    except Exception as e:
        print(f"❌ Document analyzer test error: {e}")
        return False

async def main():
    """Run all vLLM integration tests"""
    print("🚀 Starting vLLM Integration Tests")
    print("=" * 70)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Basic Connectivity", test_vllm_basic_connectivity),
        ("Chat Completion", test_vllm_chat_completion),
        ("RAG Agent Initialization", test_rag_agent_initialization),
        ("Document Analyzer", test_document_analyzer_with_vllm),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
        print()  # Add spacing between tests
    
    # Summary
    print("📊 Test Results Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print()
    print(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! vLLM integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
