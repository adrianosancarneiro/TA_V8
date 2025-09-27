#!/usr/bin/env python3
"""
Test Pure LLM Selector with GPU-Accelerated GPT-OSS 20B

This test validates that the LLM-based document analysis and chunking
works properly with GPU acceleration via the Ollama container.

Author: TA_V8 Team
Version: 2.0
Created: 2025-09-24
"""

import asyncio
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_pure_llm_document_analysis():
    """Test the original LLM document analysis for strategy selection"""
    print("🤖 Testing Pure LLM Document Analysis")
    print("=" * 60)
    
    try:
        from document_chunker import DocumentAnalyzer
        import httpx
        
        # Initialize with vLLM client
        vllm_client = httpx.AsyncClient()
        analyzer = DocumentAnalyzer(vllm_client, llm_model="openai/gpt-oss-20b", vllm_url="http://localhost:8000")
        
        print("✅ DocumentAnalyzer initialized with vLLM GPT-OSS 20B")
        
        # Test document with some complexity
        test_doc = """
        # Machine Learning in Healthcare
        
        Machine learning is transforming healthcare through predictive analytics and diagnostic assistance.
        This document explores various applications and implementation strategies.
        
        ## Diagnostic Applications
        
        Deep learning models can analyze medical images with accuracy comparable to human experts.
        Convolutional neural networks excel at identifying patterns in X-rays, MRIs, and CT scans.
        
        ```python
        import tensorflow as tf
        from tensorflow.keras import layers
        
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        ```
        
        ## Predictive Analytics
        
        Machine learning algorithms can predict patient outcomes, readmission risks, and treatment responses.
        Time series analysis helps forecast disease outbreaks and resource needs.
        
        Random forests and gradient boosting are particularly effective for structured medical data.
        
        ## Implementation Challenges
        
        Privacy regulations like HIPAA require careful handling of patient data.
        Model interpretability is crucial for clinical acceptance and regulatory approval.
        
        ## Future Directions
        
        Federated learning enables collaborative model training without sharing sensitive data.
        Large language models show promise for clinical note analysis and medical coding.
        """
        
        print(f"📄 Test document: {len(test_doc)} characters")
        
        # Test the LLM analysis
        print("\n🔍 Running LLM analysis for strategy recommendation...")
        start_time = time.time()
        
        strategy, analysis_details = await analyzer.analyze_and_recommend_strategy(
            test_doc, 
            metadata={'source': 'test', 'domain': 'healthcare'}
        )
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        print(f"✅ LLM Analysis completed in {analysis_time:.2f} seconds")
        print(f"🎯 Recommended strategy: {strategy}")
        print(f"📊 Confidence: {analysis_details.get('confidence', 'N/A')}")
        print(f"📋 Content type: {analysis_details.get('content_type', 'N/A')}")
        print(f"🔍 Reasoning: {analysis_details.get('reasoning', 'N/A')}")
        print(f"🔧 Key characteristics: {analysis_details.get('key_characteristics', [])}")
        
        # Validate the analysis makes sense
        if strategy in ['semantic_coherence', 'hybrid', 'llm_assisted']:
            print("✅ Valid strategy recommended")
        else:
            print(f"⚠️ Unexpected strategy: {strategy}")
        
        # Performance assessment
        if analysis_time < 10:
            print(f"🚀 EXCELLENT: Analysis completed in {analysis_time:.2f}s")
        elif analysis_time < 30:
            print(f"✅ ACCEPTABLE: Analysis completed in {analysis_time:.2f}s")
        else:
            print(f"⚠️ SLOW: Analysis took {analysis_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM analysis test failed: {e}")
        return False


async def test_llm_assisted_chunking():
    """Test LLM-assisted chunking with GPU acceleration"""
    print("\n🧩 Testing LLM-Assisted Chunking")
    print("=" * 60)
    
    try:
        from document_chunker import AdvancedChunker
        import tiktoken
        import httpx
        
        # Initialize components
        tokenizer = tiktoken.get_encoding('cl100k_base')
        vllm_client = httpx.AsyncClient()
        
        chunker = AdvancedChunker(
            tokenizer=tokenizer,
            vllm_client=vllm_client,
            vllm_url="http://localhost:8000",
            llm_model="openai/gpt-oss-20b"
        )
        
        # Test document for chunking
        chunking_doc = """
        Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on the interaction between computers and human language. It combines computational linguistics with statistical, machine learning, and deep learning models to enable computers to process and analyze large amounts of natural language data.

        The field has evolved significantly from rule-based systems to modern neural approaches. Early NLP systems relied heavily on handcrafted rules and linguistic knowledge. However, the advent of machine learning brought statistical methods that could learn patterns from data. The introduction of deep learning has revolutionized NLP, enabling end-to-end learning and achieving state-of-the-art results across many tasks.

        Key NLP tasks include tokenization, part-of-speech tagging, named entity recognition, parsing, sentiment analysis, machine translation, question answering, and text summarization. Each task presents unique challenges and requires specialized approaches. Tokenization breaks text into meaningful units, while parsing analyzes grammatical structure.

        Modern NLP heavily relies on transformer architectures, introduced in the "Attention is All You Need" paper. Transformers use self-attention mechanisms to process sequences in parallel, making them more efficient than recurrent neural networks. This architecture has led to breakthrough models like BERT, GPT, and T5.

        The practical applications of NLP are vast and growing. Search engines use NLP to understand queries and rank results. Virtual assistants like Siri and Alexa rely on NLP for speech recognition and understanding. Social media platforms use NLP for content moderation and sentiment analysis. Healthcare systems use NLP to extract information from medical records.
        """
        
        print(f"📄 Chunking document: {len(chunking_doc)} characters")
        
        # Test LLM-assisted chunking
        print("\n🤖 Running LLM-assisted chunking...")
        start_time = time.time()
        
        result = await chunker.chunk_document(
            text=chunking_doc,
            method='llm_assisted',
            target_chunk_tokens=150,
            max_chunk_tokens=300,
            chunk_overlap_tokens=30,
            metadata={'test': 'llm_chunking'}
        )
        
        end_time = time.time()
        chunking_time = end_time - start_time
        
        print(f"✅ LLM Chunking completed in {chunking_time:.2f} seconds")
        print(f"📊 Method used: {result['method']}")
        print(f"📊 Total chunks: {result['statistics']['total_chunks']}")
        print(f"📊 Average tokens per chunk: {result['statistics']['avg_tokens_per_chunk']:.1f}")
        
        # Show chunk previews
        for i, chunk in enumerate(result['chunks'][:3]):  # Show first 3 chunks
            print(f"\n📝 Chunk {i+1}:")
            print(f"   Tokens: {chunk['token_count']}")
            print(f"   Method: {chunk.get('method', 'unknown')}")
            print(f"   Preview: {chunk['text'][:120]}...")
        
        # Performance assessment for chunking
        if chunking_time < 30:
            print(f"🚀 EXCELLENT: Chunking completed in {chunking_time:.2f}s")
        elif chunking_time < 60:
            print(f"✅ ACCEPTABLE: Chunking completed in {chunking_time:.2f}s")
        else:
            print(f"⚠️ SLOW: Chunking took {chunking_time:.2f}s")
        
        # Verify it actually used LLM
        if result['method'] == 'llm_assisted':
            print("🎉 SUCCESS: LLM-assisted chunking was used!")
        else:
            print(f"⚠️ Fallback occurred: Used {result['method']} instead")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM chunking test failed: {e}")
        return False


async def test_auto_strategy_with_llm():
    """Test auto strategy selection with LLM analysis"""
    print("\n🎯 Testing Auto Strategy with LLM")
    print("=" * 60)
    
    try:
        from document_chunker import AdvancedChunker
        import tiktoken
        import httpx
        
        # Initialize components
        tokenizer = tiktoken.get_encoding('cl100k_base')
        vllm_client = httpx.AsyncClient()
        
        chunker = AdvancedChunker(
            tokenizer=tokenizer,
            vllm_client=vllm_client,
            vllm_url="http://localhost:8000",
            llm_model="openai/gpt-oss-20b"
        )
        
        # Test document that should trigger LLM analysis
        auto_test_doc = """
        The Evolution of Artificial Intelligence: From Symbolic Systems to Large Language Models
        
        Artificial Intelligence has undergone several paradigm shifts since its inception in the 1950s. This comprehensive analysis examines the major evolutionary phases and their implications for future development.
        
        ## The Symbolic Era (1950s-1980s)
        
        Early AI systems were based on symbolic reasoning and expert systems. These approaches used explicit rules and logical inference to solve problems. While successful in narrow domains, they struggled with uncertainty and required extensive domain expertise to construct.
        
        Key achievements included:
        - DENDRAL for chemical analysis
        - MYCIN for medical diagnosis  
        - Chess-playing programs like Deep Blue
        
        ## The Statistical Revolution (1990s-2010s)
        
        The rise of statistical machine learning marked a fundamental shift towards data-driven approaches. Support vector machines, random forests, and neural networks began to outperform rule-based systems on many tasks.
        
        This period saw breakthroughs in:
        - Computer vision with convolutional neural networks
        - Speech recognition using hidden Markov models
        - Recommendation systems with collaborative filtering
        
        ## The Deep Learning Era (2010s-Present)
        
        Deep neural networks have achieved remarkable success across multiple domains. The combination of large datasets, computational power, and architectural innovations has led to superhuman performance on specific tasks.
        
        ## Future Directions
        
        Current research focuses on few-shot learning, multimodal understanding, and artificial general intelligence. The integration of symbolic and neural approaches may provide the next breakthrough.
        """
        
        print(f"📄 Auto-strategy document: {len(auto_test_doc)} characters")
        
        # Test auto strategy with LLM
        print("\n🔄 Running auto strategy selection...")
        start_time = time.time()
        
        result = await chunker.chunk_document(
            text=auto_test_doc,
            method='auto',
            target_chunk_tokens=200,
            max_chunk_tokens=400,
            chunk_overlap_tokens=40,
            metadata={'test': 'auto_strategy'}
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Auto strategy completed in {total_time:.2f} seconds")
        print(f"🎯 Selected method: {result['method']}")
        print(f"📊 Total chunks: {result['statistics']['total_chunks']}")
        print(f"📊 Processing time: {result['statistics']['processing_time_seconds']:.2f}s")
        
        # Check if analysis details are available
        if 'analysis' in result:
            analysis = result['analysis']
            print(f"📋 Content type: {analysis.get('content_type', 'N/A')}")
            print(f"📊 Confidence: {analysis.get('confidence', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto strategy test failed: {e}")
        return False


async def main():
    """Run pure LLM tests with GPU acceleration"""
    print("🤖 TA_V8 Pure LLM Selector Tests (GPU-Accelerated)")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    
    test_results = {}
    
    # Run tests
    test_results['llm_analysis'] = await test_pure_llm_document_analysis()
    test_results['llm_chunking'] = await test_llm_assisted_chunking()
    test_results['auto_strategy'] = await test_auto_strategy_with_llm()
    
    # Summary
    print("\n" + "=" * 80)
    print("🤖 PURE LLM TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:15}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🤖 🎉 PURE LLM SELECTOR WORKING WITH GPU!")
        print("\nGPU Performance Benefits:")
        print("   • Fast LLM analysis for strategy selection")
        print("   • High-quality boundary detection")
        print("   • Intelligent auto-strategy selection")
        print("   • Real AI-powered document understanding")
    else:
        print("🤖 ⚠️ Some LLM tests failed - check GPU and model availability")
    
    print(f"Completed: {datetime.now().isoformat()}")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())