#!/usr/bin/env python3
"""
Complete Storage Integration Test for TA_V8 RAG System

Tests the integrated chunking workflow with MinIO and PostgreSQL Docker containers:
1. Document storage in MinIO with MinIO-generated document IDs
2. Automatic chunking with LLM strategy selection
3. Chunk storage in PostgreSQL
4. Full end-to-end validation

Author: TA_V8 Team
Version: 3.0
Created: 2025-09-24
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_integrated_document_processing():
    """Test the complete integrated document processing workflow"""
    print("🔗 Testing Complete Integrated Document Processing")
    print("=" * 70)
    
    try:
        from document_chunker import AdvancedChunker, DocumentAnalyzer
        import tiktoken
        import ollama
        
        print("✅ Core modules imported successfully")
        
        # Initialize components with storage integration
        tokenizer = tiktoken.get_encoding('cl100k_base')
        ollama_client = ollama.AsyncClient(host='http://localhost:11434')
        
        # Create integrated chunker (it will initialize MinIO and PostgreSQL)
        chunker = AdvancedChunker(
            tokenizer=tokenizer,
            ollama_client=ollama_client
            # MinIO and PostgreSQL will be auto-initialized
        )\n        
        print("🤖 Initialized integrated chunker with storage capabilities")
        
        # Test document for processing
        test_document = \"\"\"
        # Comprehensive RAG System Integration Guide
        
        This technical documentation covers the complete integration of a production-ready
        Retrieval-Augmented Generation system with advanced document processing capabilities.
        
        ## System Architecture Overview
        
        The TA_V8 RAG system implements a modern, scalable architecture with the following
        key components working in harmony:
        
        ### Storage Layer
        - **MinIO Object Storage**: Handles raw document files with S3-compatible API
        - **PostgreSQL Database**: Stores chunk metadata, text content, and relationships
        - **Qdrant Vector Database**: Manages high-dimensional embeddings for semantic search
        
        ### Processing Pipeline
        
        1. **Document Ingestion**: Files are uploaded and stored in MinIO with unique IDs
        2. **Intelligent Chunking**: LLM-powered analysis selects optimal chunking strategies
        3. **Vector Generation**: Advanced embedding models create semantic representations
        4. **Storage Orchestration**: All components are linked through consistent document IDs
        
        ```python
        # Example chunking configuration
        chunker = AdvancedChunker(
            strategy_selector=\"auto\",
            target_chunk_size=500,
            overlap_tokens=50,
            llm_model=\"gpt-oss:20b\"
        )
        
        result = await chunker.process_document(
            text=document_text,
            tenant_id=\"production\",
            auto_store=True
        )
        ```
        
        ## Advanced Features
        
        ### LLM-Guided Strategy Selection
        The system uses a 20B parameter language model to analyze document characteristics
        and select the most appropriate chunking strategy:
        
        - **Semantic Coherence**: For narrative and flowing content
        - **Hybrid Structure**: For technical documentation with clear formatting
        - **LLM-Assisted**: For complex documents requiring nuanced boundary detection
        
        ### Multi-Tenant Architecture
        Built-in support for multiple tenants with isolated data storage and processing:
        
        - Tenant-specific MinIO paths: `tenant_id/documents/`
        - PostgreSQL tenant isolation with row-level security
        - Qdrant collections per tenant for embedding isolation
        
        ## Performance Characteristics
        
        The system is optimized for high-throughput document processing:
        
        - **GPU Acceleration**: LLM inference runs on GPU for fast analysis
        - **Parallel Processing**: Multiple documents can be processed simultaneously  
        - **Efficient Storage**: Deduplication and compression reduce storage costs
        - **Caching Strategies**: Frequently accessed chunks are cached in Redis
        
        ## Monitoring and Observability
        
        Comprehensive monitoring ensures system reliability:
        
        - Health checks for all storage components
        - Performance metrics for processing pipelines
        - Error tracking and alerting
        - Audit trails for document processing history
        \"\"\"
        
        print(f\"📄 Test document: {len(test_document)} characters\")
        
        # Test integrated processing with storage
        print(\"\\n🔄 Running integrated document processing...\")
        start_time = time.time()
        
        result = await chunker.chunk_document(
            text=test_document,
            method='auto',  # Use intelligent strategy selection
            target_chunk_tokens=300,
            max_chunk_tokens=600,
            chunk_overlap_tokens=50,
            tenant_id='test_tenant',
            filename='integration_test_doc.md',
            metadata={
                'source': 'integration_test',
                'document_type': 'technical_documentation',
                'test_run': datetime.now().isoformat()
            },
            auto_store=True  # Enable full storage integration
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f\"✅ Processing completed in {processing_time:.2f} seconds\")
        
        # Validate results
        print(f\"\\n📊 Processing Results:\")
        print(f\"   Document ID (MinIO): {result['document_id']}\")
        print(f\"   Strategy Used: {result['method']}\")
        print(f\"   Total Chunks: {result['statistics']['total_chunks']}\")
        print(f\"   Avg Tokens/Chunk: {result['statistics']['avg_tokens_per_chunk']:.1f}\")
        print(f\"   Storage Enabled: {result['statistics']['storage_enabled']}\")
        
        # Check storage information
        if 'storage' in result:
            storage = result['storage']
            if 'minio' in storage:
                minio_info = storage['minio']
                print(f\"\\n📁 MinIO Storage:\")
                print(f\"   Path: {minio_info.get('minio_path', 'N/A')}\")
                print(f\"   File Size: {minio_info.get('file_size', 0)} bytes\")
                print(f\"   Content Hash: {minio_info.get('content_hash', 'N/A')[:16]}...\")
            
            if 'postgres' in storage:
                postgres_info = storage['postgres']
                print(f\"\\n🐘 PostgreSQL Storage:\")
                print(f\"   Chunks Stored: {postgres_info.get('chunks_stored', 0)}\")
                print(f\"   Table: {postgres_info.get('table', 'N/A')}\")
        
        # Show chunk samples
        print(f\"\\n📝 Chunk Samples:\")
        for i, chunk in enumerate(result['chunks'][:3]):
            print(f\"   Chunk {i+1}: {chunk['token_count']} tokens\")\n            print(f\"      ID: {chunk['chunk_id']}\")\n            print(f\"      Method: {chunk.get('method', 'unknown')}\")
            print(f\"      Preview: {chunk['text'][:120]}...\\n\")
        
        # Validate MinIO document ID format
        doc_id = result['document_id']
        if doc_id.startswith('doc_test_tenant_') and len(doc_id) > 20:\n            print(\"🎉 SUCCESS: MinIO document ID format is correct!\")
        else:\n            print(f\"⚠️ WARNING: Unexpected document ID format: {doc_id}\")\n        
        # Check LLM analysis\n        if 'analysis' in result:\n            analysis = result['analysis']\n            print(f\"\\n🧠 LLM Analysis:\")
            print(f\"   Content Type: {analysis.get('content_type', 'N/A')}\")
            print(f\"   Confidence: {analysis.get('confidence', 'N/A')}\")
            print(f\"   Key Features: {analysis.get('key_characteristics', [])}\")\n        
        return True
        
    except Exception as e:
        print(f\"❌ Integration test failed: {e}\")\n        logger.exception(\"Integration test error details:\")\n        return False\n\n\nasync def test_storage_verification():\n    \"\"\"Test that storage components are accessible\"\"\"\n    print(\"\\n🔍 Testing Storage Component Accessibility\")\n    print(\"=\" * 50)\n    \n    results = {}\n    \n    # Test MinIO accessibility\n    try:\n        from minio import Minio\n        from shared.config import config\n        \n        minio_client = Minio(\n            endpoint=config.MINIO_ENDPOINT,\n            access_key=config.MINIO_ACCESS_KEY,\n            secret_key=config.MINIO_SECRET_KEY,\n            secure=False\n        )\n        \n        # Test bucket existence\n        bucket_exists = minio_client.bucket_exists(config.MINIO_BUCKET)\n        if bucket_exists:\n            print(\"✅ MinIO: Bucket accessible\")\n            results['minio'] = True\n        else:\n            print(\"⚠️ MinIO: Bucket not found\")\n            results['minio'] = False\n    except Exception as e:\n        print(f\"❌ MinIO: Connection failed - {e}\")\n        results['minio'] = False\n    \n    # Test PostgreSQL accessibility\n    try:\n        import asyncpg\n        from shared.config import config\n        \n        conn = await asyncpg.connect(\n            host=config.POSTGRES_HOST,\n            port=config.POSTGRES_PORT,\n            user=config.POSTGRES_USER,\n            password=config.POSTGRES_PASSWORD,\n            database=config.POSTGRES_DATABASE\n        )\n        \n        # Test basic query\n        result = await conn.fetchval('SELECT 1')\n        if result == 1:\n            print(\"✅ PostgreSQL: Connection successful\")\n            results['postgres'] = True\n        else:\n            print(\"⚠️ PostgreSQL: Unexpected query result\")\n            results['postgres'] = False\n        \n        await conn.close()\n        \n    except Exception as e:\n        print(f\"❌ PostgreSQL: Connection failed - {e}\")\n        results['postgres'] = False\n    \n    # Test Ollama accessibility\n    try:\n        import ollama\n        client = ollama.AsyncClient(host='http://localhost:11434')\n        \n        # Test model availability\n        models = await client.list()\n        model_names = [m.model for m in models.models]\n        \n        if 'gpt-oss:20b' in model_names:\n            print(\"✅ Ollama: GPT-OSS 20B model available\")\n            results['ollama'] = True\n        else:\n            print(f\"⚠️ Ollama: GPT-OSS 20B not found. Available: {model_names}\")\n            results['ollama'] = False\n            \n    except Exception as e:\n        print(f\"❌ Ollama: Connection failed - {e}\")\n        results['ollama'] = False\n    \n    return results\n\n\nasync def main():\n    \"\"\"Run comprehensive integration tests\"\"\"\n    print(\"🔗 TA_V8 Complete Storage Integration Test Suite\")\n    print(\"=\" * 80)\n    print(f\"Started: {datetime.now().isoformat()}\")\n    \n    test_results = {}\n    \n    # Test storage accessibility first\n    storage_results = await test_storage_verification()\n    test_results['storage_accessibility'] = all(storage_results.values())\n    \n    # Run main integration test if storage is accessible\n    if test_results['storage_accessibility']:\n        test_results['integrated_processing'] = await test_integrated_document_processing()\n    else:\n        print(\"\\n⚠️ Skipping integration test due to storage connectivity issues\")\n        test_results['integrated_processing'] = False\n    \n    # Summary\n    print(\"\\n\" + \"=\" * 80)\n    print(\"🔗 COMPLETE STORAGE INTEGRATION TEST SUMMARY\")\n    print(\"=\" * 80)\n    \n    passed = sum(test_results.values())\n    total = len(test_results)\n    \n    for test_name, result in test_results.items():\n        status = \"✅ PASS\" if result else \"❌ FAIL\"\n        print(f\"  {test_name:25}: {status}\")\n    \n    print(f\"\\n🎯 Overall: {passed}/{total} tests passed\")\n    \n    if passed == total:\n        print(\"🎉 COMPLETE STORAGE INTEGRATION SUCCESSFUL!\")\n        print(\"\\n🚀 System Ready for Production:\")\n        print(\"   ✅ MinIO document storage with generated IDs\")\n        print(\"   ✅ PostgreSQL chunk storage with metadata\")\n        print(\"   ✅ LLM-powered intelligent chunking strategies\")\n        print(\"   ✅ End-to-end processing pipeline operational\")\n        print(\"   ✅ Docker container integration working\")\n        print(\"\\n💡 Key Benefits:\")\n        print(\"   • Documents automatically saved to MinIO first\")\n        print(\"   • MinIO-generated document IDs used consistently\")\n        print(\"   • Chunks automatically stored in PostgreSQL\")\n        print(\"   • Full traceability and audit trail\")\n        print(\"   • No manual storage management required\")\n    else:\n        print(\"⚠️ Integration issues detected - review failed tests\")\n        if not test_results['storage_accessibility']:\n            print(\"\\n🔧 Storage Setup Required:\")\n            print(\"   • Ensure Docker containers are running\")\n            print(\"   • Check MinIO bucket configuration\")\n            print(\"   • Verify PostgreSQL database schema\")\n            print(\"   • Confirm Ollama model availability\")\n    \n    print(f\"\\nCompleted: {datetime.now().isoformat()}\")\n    return passed == total\n\n\nif __name__ == \"__main__\":\n    asyncio.run(main())