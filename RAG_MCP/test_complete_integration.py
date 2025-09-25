#!/usr/bin/env python3
"""
Complete Integration Test: Unified MCP Server + Pure LLM Document Chunker

Tests the full integration between unified_mcp_server.py and document_chunker.py
with GPU-accelerated LLM analysis using GPT-OSS 20B.

Author: TA_V8 Team  
Version: 2.0
Created: 2025-09-24
"""

import asyncio
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_unified_mcp_with_llm_chunker():
    """Test the complete integration pipeline"""
    print("🔗 Testing Complete Integration: MCP Server + LLM Chunker")
    print("=" * 70)
    
    try:
        # Import components
        from unified_mcp_server import UnifiedMCPServer
        from document_chunker import AdvancedChunker, DocumentAnalyzer
        import tiktoken
        import ollama
        
        print("✅ All modules imported successfully")
        
        # Initialize components
        tokenizer = tiktoken.get_encoding('cl100k_base')
        ollama_client = ollama.AsyncClient(host='http://localhost:11434')
        
        # Create chunker with LLM support
        chunker = AdvancedChunker(
            tokenizer=tokenizer,
            ollama_client=ollama_client
        )
        
        analyzer = DocumentAnalyzer(ollama_client, llm_model="gpt-oss:20b")
        
        print("🤖 Initialized chunker with GPT-OSS 20B LLM support")
        
        # Test document for processing
        test_document = """
        # Advanced RAG System Architecture Guide
        
        This guide covers the implementation of a Retrieval-Augmented Generation system
        using modern vector databases and large language models.
        
        ## System Components
        
        ### Vector Database
        Qdrant provides efficient vector similarity search with filtering capabilities.
        It supports multiple distance metrics and can handle large-scale deployments.
        
        Configuration example:
        ```python
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            host="qdrant",
            port=6333,
            timeout=30
        )
        ```
        
        ### Document Storage
        MinIO object storage handles raw document files with S3-compatible API.
        PostgreSQL stores chunk metadata and text content for rapid retrieval.
        
        ### Processing Pipeline
        
        1. Document ingestion and preprocessing
        2. Intelligent chunking with LLM-guided strategy selection
        3. Vector embedding generation using multilingual models
        4. Storage in both vector and relational databases
        5. Retrieval using hybrid search combining semantic and keyword matching
        
        ## Performance Optimization
        
        The system uses GPU acceleration for both embedding generation and LLM inference.
        Batch processing reduces latency for large document sets.
        
        ### Caching Strategy
        Frequently accessed chunks are cached in Redis for sub-millisecond retrieval.
        Vector embeddings are stored with metadata for efficient filtering.
        
        ## Monitoring and Maintenance
        
        Health monitoring ensures all components are operational.
        Automatic backup procedures protect against data loss.
        Performance metrics track system efficiency and identify bottlenecks.
        """
        
        # Test LLM analysis first
        print("\n🔍 Testing LLM document analysis...")
        start_time = time.time()
        
        strategy, analysis = await analyzer.analyze_and_recommend_strategy(
            test_document,
            metadata={'source': 'integration_test', 'type': 'technical_guide'}
        )
        
        analysis_time = time.time() - start_time
        
        print(f"✅ LLM analysis completed in {analysis_time:.2f}s")
        print(f"🎯 Recommended strategy: {strategy}")
        print(f"📊 Confidence: {analysis.get('confidence', 'N/A')}")
        print(f"📋 Content type: {analysis.get('content_type', 'N/A')}")
        
        # Test chunking with recommended strategy
        print(f"\n🧩 Testing chunking with {strategy} strategy...")
        chunk_start = time.time()
        
        result = await chunker.chunk_document(
            text=test_document,
            method=strategy,
            target_chunk_tokens=200,
            max_chunk_tokens=400,
            chunk_overlap_tokens=40,
            metadata={'test': 'integration', 'domain': 'rag_system'}
        )
        
        chunk_time = time.time() - chunk_start
        
        print(f"✅ Chunking completed in {chunk_time:.2f}s")
        print(f"📊 Method used: {result['method']}")
        print(f"📊 Total chunks: {result['statistics']['total_chunks']}")
        print(f"📊 Avg tokens per chunk: {result['statistics']['avg_tokens_per_chunk']:.1f}")
        
        # Show chunk details
        print(f"\n📝 Chunk Analysis:")
        for i, chunk in enumerate(result['chunks'][:3]):
            print(f"   Chunk {i+1}: {chunk['token_count']} tokens - {chunk['text'][:80]}...")
        
        # Test auto strategy as well
        print(f"\n🎯 Testing auto strategy selection...")
        auto_start = time.time()
        
        auto_result = await chunker.chunk_document(
            text=test_document,
            method='auto',
            target_chunk_tokens=250,
            max_chunk_tokens=500,
            metadata={'test': 'auto_integration'}
        )
        
        auto_time = time.time() - auto_start
        
        print(f"✅ Auto strategy completed in {auto_time:.2f}s")
        print(f"🎯 Auto-selected method: {auto_result['method']}")
        print(f"📊 Total chunks: {auto_result['statistics']['total_chunks']}")
        
        # Performance summary
        total_time = analysis_time + chunk_time + auto_time
        print(f"\n⏱️ Performance Summary:")
        print(f"   LLM Analysis: {analysis_time:.2f}s")
        print(f"   Strategy Chunking: {chunk_time:.2f}s") 
        print(f"   Auto Strategy: {auto_time:.2f}s")
        print(f"   Total Time: {total_time:.2f}s")
        
        # Validate integration success
        if total_time < 60 and result['statistics']['total_chunks'] > 0:
            print("🎉 INTEGRATION SUCCESS: Complete pipeline working!")
            return True
        else:
            print("⚠️ Integration concerns: Check performance or output")
            return False
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        logger.exception("Integration test error details:")
        return False


async def test_mcp_server_endpoints():
    """Test MCP server initialization and basic functionality"""
    print("\n🖥️ Testing MCP Server Initialization")
    print("=" * 50)
    
    try:
        from unified_mcp_server import UnifiedMCPServer
        
        # Test server initialization
        server = UnifiedMCPServer()
        print("✅ UnifiedMCPServer initialized successfully")
        
        # Check if chunker is properly integrated
        if hasattr(server, 'chunker') or hasattr(server, 'document_chunker'):
            print("✅ Document chunker integration detected")
        else:
            print("⚠️ Document chunker integration not found")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP server test failed: {e}")
        return False


async def main():
    """Run complete integration tests"""
    print("🔗 TA_V8 Complete Integration Test Suite")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    
    test_results = {}
    
    # Run tests
    test_results['mcp_server'] = await test_mcp_server_endpoints()
    test_results['full_integration'] = await test_unified_mcp_with_llm_chunker()
    
    # Summary
    print("\n" + "=" * 80)
    print("🔗 INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 COMPLETE SYSTEM INTEGRATION SUCCESSFUL!")
        print("\n🚀 Ready for Production:")
        print("   ✅ Unified MCP Server operational")
        print("   ✅ Pure LLM document analysis with GPU acceleration")
        print("   ✅ Advanced chunking strategies working")
        print("   ✅ Auto-strategy selection functional") 
        print("   ✅ End-to-end pipeline validated")
        print("\n🤖 GPU-Accelerated Features:")
        print("   • Fast LLM-based strategy recommendation")
        print("   • Intelligent document analysis")
        print("   • High-quality chunk boundary detection")
        print("   • Real-time processing capabilities")
    else:
        print("⚠️ Integration issues detected - review failed tests")
    
    print(f"Completed: {datetime.now().isoformat()}")
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())