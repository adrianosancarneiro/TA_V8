#!/usr/bin/env python3
"""
Advanced Document Chunking Module for TA_V8 RAG System

This module provides sophisticated document chunking strategies with intelligent
strategy selection using LLM analysis. All methods support overlap for context preservation.

Key Features:
- Automatic strategy selection using LLM document analysis
- Semantic coherence chunking with embedding-based boundary detection
- Hybrid structure-aware chunking for formatted documents
- LLM-assisted chunking for optimal semantic boundaries
- Universal overlap support across all methods
- Comprehensive metadata and debug information

Chunking Strategies:
1. Semantic Coherence: Uses sentence embeddings and change point detection
2. Hybrid: Structure-aware chunking respecting document formatting
3. LLM-Assisted: Uses AI to identify optimal chunk boundaries

Author: TA_V8 Team
Version: 2.0
Created: 2025-09-24
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import hashlib
import uuid

import numpy as np
import tiktoken
from io import BytesIO

try:
    import spacy
    from sentence_transformers import SentenceTransformer
    import ruptures as rpt
    import ollama
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

try:
    import asyncpg
    from minio import Minio
    from minio.error import S3Error
    STORAGE_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Storage features not available: {e}")
    STORAGE_FEATURES_AVAILABLE = False

# Import configuration
try:
    from shared.config import config
except ImportError:
    # Fallback configuration
    class Config:
        MINIO_ENDPOINT = "minio:9000"
        MINIO_ACCESS_KEY = "minio_user"
        MINIO_SECRET_KEY = "minio_password"
        MINIO_BUCKET = "ta-v8-documents"
        POSTGRES_HOST = "postgres"
        POSTGRES_PORT = 5432
        POSTGRES_USER = "ta_v8_user"
        POSTGRES_PASSWORD = "ta_v8_secure_password"
        POSTGRES_DATABASE = "ta_v8_rag"
    config = Config()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Enumeration of available chunking strategies"""
    SEMANTIC_COHERENCE = "semantic_coherence"
    HYBRID = "hybrid"
    LLM_ASSISTED = "llm_assisted"
    AUTO = "auto"  # Let the system decide


class DocumentAnalyzer:
    """Sophisticated document analyzer for optimal chunking strategy selection"""
    
    def __init__(self, vllm_client=None, llm_model: str = "openai/gpt-oss-20b", vllm_url: str = "http://localhost:8000"):
        """Initialize the document analyzer
        
        Args:
            vllm_client: HTTP client for vLLM analysis (httpx.AsyncClient)
            llm_model: Model to use for analysis
            vllm_url: URL of the vLLM service
        """
        self.vllm_client = vllm_client
        self.llm_model = llm_model
        self.vllm_url = vllm_url
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.last_analyzed_text = None
    
    async def analyze_and_recommend_strategy(self, text: str, 
                                           metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Analyze document and recommend optimal chunking strategy using LLM
        
        This method performs deep document analysis to understand:
        - Content type and domain
        - Structural complexity
        - Semantic coherence patterns
        - Optimal preservation requirements
        
        Args:
            text: Document text to analyze
            metadata: Optional metadata about the document
            
        Returns:
            Tuple of (recommended_strategy, analysis_details)
        """
        if not self.vllm_client:
            # Fallback to heuristic analysis if LLM not available
            return self._heuristic_analysis(text, metadata)
        
        try:
            # Prepare comprehensive analysis prompt
            analysis_prompt = self._create_analysis_prompt(text, metadata)
            
            logger.info("🔍 Analyzing document with vLLM for optimal chunking strategy...")
            
            # Call vLLM for sophisticated analysis
            vllm_request = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are an expert document analyzer specializing in optimal text segmentation."},
                    {"role": "user", "content": analysis_prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistent analysis
                "max_tokens": 1000,
                "top_p": 0.9
            }
            
            response = await self.vllm_client.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=vllm_request,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            llm_response = result["choices"][0]["message"]["content"]
            
            # Parse LLM recommendation
            strategy, details = self._parse_llm_recommendation(llm_response)
            
            logger.info(f"✅ LLM recommends: {strategy} strategy")
            logger.info(f"📊 Analysis confidence: {details.get('confidence', 'N/A')}")
            
            return strategy, details
            
        except Exception as e:
            logger.error(f"❌ LLM analysis failed: {e}, falling back to heuristics")
            return self._heuristic_analysis(text, metadata)
    
    def _create_analysis_prompt(self, text: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Create comprehensive prompt for document analysis"""
        # Truncate text for analysis if too long
        max_analysis_length = 10000
        text_sample = text[:max_analysis_length]
        if len(text) > max_analysis_length:
            text_sample += f"\n\n[Document continues... Total length: {len(text)} characters]"
        
        # Extract document statistics
        stats = self._calculate_document_stats(text)
        
        prompt = f"""You are an expert document analyst specializing in optimal text segmentation for AI processing.

**TASK**: Analyze this document and recommend the BEST chunking strategy from the available options.

**AVAILABLE CHUNKING STRATEGIES**:

1. **SEMANTIC_COHERENCE**: 
   - Best for: Narrative text, articles, essays, reports with flowing topics
   - How it works: Uses sentence embeddings to detect topic shifts and semantic boundaries
   - Strengths: Maintains topical coherence, smooth transitions, preserves context
   - Weaknesses: May not respect structural boundaries, computationally intensive
   - Use when: Document has natural topic flow without rigid structure

2. **HYBRID**:
   - Best for: Technical documentation, structured reports, markdown files, code documentation
   - How it works: Identifies headers, lists, code blocks, and uses them as natural boundaries
   - Strengths: Respects document structure, preserves formatting, handles mixed content
   - Weaknesses: May create uneven chunks, depends on document formatting
   - Use when: Document has clear structural elements (headings, sections, lists)

3. **LLM_ASSISTED**:
   - Best for: Complex documents, academic papers, legal texts, nuanced content
   - How it works: Uses AI to understand content deeply and identify optimal boundaries
   - Strengths: Highest quality boundaries, understands context and meaning
   - Weaknesses: Slowest method, requires more compute resources
   - Use when: Quality is paramount and document has complex semantic relationships

**DOCUMENT TO ANALYZE**:
{text_sample}

**DOCUMENT STATISTICS**:
- Total length: {stats['total_chars']} characters
- Estimated tokens: {stats['estimated_tokens']}
- Line count: {stats['line_count']}
- Average sentence length: {stats['avg_sentence_length']:.1f} characters
- Paragraph count: {stats['paragraph_count']}
- Has headers: {stats['has_headers']}
- Has code blocks: {stats['has_code']}
- Has lists: {stats['has_lists']}
- Has tables: {stats['has_tables']}
- Vocabulary diversity: {stats['vocab_diversity']:.2f}
- Structural complexity: {stats['structural_complexity']}

**METADATA**:
{json.dumps(metadata or {}, indent=2)}

**ANALYSIS REQUIREMENTS**:
1. Consider the CONTENT TYPE (technical, narrative, mixed, etc.)
2. Evaluate SEMANTIC COMPLEXITY (simple facts vs complex relationships)
3. Assess STRUCTURAL INTEGRITY (formatting importance)
4. Determine OPTIMAL CHUNK SIZE based on content density
5. Consider DOWNSTREAM USE CASE if known
6. Note that LLM_ASSISTED can be recommended when highest quality is needed

**YOUR RESPONSE MUST BE IN JSON FORMAT**:
{{
    "recommended_strategy": "[semantic_coherence|hybrid|llm_assisted]",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of why this strategy is optimal",
    "content_type": "technical|narrative|mixed|structured|academic|legal",
    "complexity_level": "low|medium|high",
    "key_characteristics": ["list", "of", "notable", "features"],
    "expected_benefits": ["list", "of", "benefits", "from", "chosen", "strategy"],
    "alternative_strategy": "second best option if primary fails",
    "special_considerations": "any special handling needed"
}}"""
        
        return prompt
    
    def _parse_llm_recommendation(self, llm_response: str) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM recommendation from JSON response"""
        try:
            # Try to extract JSON from response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                recommendation = json.loads(json_str)
                
                strategy = recommendation.get('recommended_strategy', 'semantic_coherence')
                
                # Validate strategy
                if strategy not in ['semantic_coherence', 'hybrid', 'llm_assisted']:
                    logger.warning(f"Invalid strategy '{strategy}', defaulting to semantic_coherence")
                    strategy = 'semantic_coherence'
                
                return strategy, recommendation
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse LLM recommendation: {e}")
            # Fallback to heuristic analysis
            return self._heuristic_analysis(self.last_analyzed_text or "", {})
    
    def _heuristic_analysis(self, text: str, metadata: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Fallback heuristic analysis when LLM is unavailable"""
        stats = self._calculate_document_stats(text)
        
        # Sophisticated heuristic logic
        strategy = 'semantic_coherence'  # Default
        confidence = 0.6
        
        # Check for strong structural indicators
        if stats['has_headers'] and stats['structural_complexity'] == 'high':
            strategy = 'hybrid'
            confidence = 0.8
        elif stats['has_code'] and stats['paragraph_count'] > 5:
            strategy = 'hybrid'
            confidence = 0.75
        # Check for complexity requiring LLM
        elif stats['total_chars'] > 15000 and stats['vocab_diversity'] > 0.7:
            strategy = 'llm_assisted'
            confidence = 0.7
        elif stats['avg_sentence_length'] > 150 and stats['estimated_tokens'] > 2000:
            strategy = 'llm_assisted'
            confidence = 0.65
        
        # Store for potential reuse
        self.last_analyzed_text = text
        
        return strategy, {
            'recommended_strategy': strategy,
            'confidence': confidence,
            'reasoning': 'Heuristic analysis based on document structure and complexity',
            'content_type': 'mixed',
            'complexity_level': 'medium' if stats['vocab_diversity'] < 0.6 else 'high',
            'key_characteristics': [k for k, v in stats.items() if v and k.startswith('has_')],
            'method': 'heuristic'
        }
    
    def _calculate_document_stats(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive document statistics"""
        lines = text.split('\n')
        sentences = re.split(r'[.!?]+\s+', text)
        words = text.split()
        
        # Calculate vocabulary diversity
        unique_words = set(word.lower() for word in words)
        vocab_diversity = len(unique_words) / max(len(words), 1)
        
        # Detect structural elements
        has_headers = bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))
        has_code = bool(re.search(r'```', text))
        has_lists = bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE))
        has_tables = bool(re.search(r'\|.*\|.*\|', text))
        
        # Determine structural complexity
        structural_elements = sum([has_headers, has_code, has_lists, has_tables])
        if structural_elements >= 3:
            structural_complexity = 'high'
        elif structural_elements >= 1:
            structural_complexity = 'medium'
        else:
            structural_complexity = 'low'
        
        return {
            'total_chars': len(text),
            'estimated_tokens': len(self.tokenizer.encode(text)) if self.tokenizer else len(words) // 0.75,
            'line_count': len(lines),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_sentence_length': len(text) / max(len(sentences), 1),
            'vocab_diversity': vocab_diversity,
            'has_headers': has_headers,
            'has_code': has_code,
            'has_lists': has_lists,
            'has_tables': has_tables,
            'structural_complexity': structural_complexity
        }


class AdvancedChunker:
    """Advanced document chunker with overlap support and integrated storage"""
    
    def __init__(self, 
                 tokenizer=None,
                 sentence_model=None,
                 spacy_nlp=None,
                 vllm_client=None,
                 vllm_url="http://localhost:8000",
                 llm_model="openai/gpt-oss-20b",
                 minio_client=None,
                 postgres_pool=None):
        """Initialize the advanced chunker
        
        Args:
            tokenizer: Tiktoken tokenizer for token counting
            sentence_model: SentenceTransformer for embeddings
            spacy_nlp: spaCy model for sentence segmentation
            vllm_client: HTTP client for vLLM operations (httpx.AsyncClient)
            vllm_url: URL of the vLLM service
            llm_model: Model name for vLLM
            minio_client: MinIO client for document storage
            postgres_pool: PostgreSQL connection pool for chunk storage
        """
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self.sentence_model = sentence_model
        self.spacy_nlp = spacy_nlp
        self.vllm_client = vllm_client
        self.analyzer = DocumentAnalyzer(vllm_client, llm_model, vllm_url)
        
        # Storage clients
        self.minio_client = minio_client or self._init_minio_client()
        self.postgres_pool = postgres_pool
        
        # Initialize storage if available
        if STORAGE_FEATURES_AVAILABLE:
            self._storage_ready = True
        else:
            self._storage_ready = False
            logger.warning("Storage features not available - operating in memory-only mode")
    
    def generate_document_id(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate unique document ID based on content hash and metadata
        
        Args:
            content: Document content
            metadata: Optional metadata
            
        Returns:
            Unique document ID
        """
        # Create hash from content
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Add timestamp component
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Add random component for uniqueness
        random_component = str(uuid.uuid4())[:8]
        
        # Combine components
        document_id = f"doc_{timestamp}_{content_hash}_{random_component}"
        
        return document_id
    
    def generate_chunk_id(self, document_id: str, chunk_index: int, method: str) -> str:
        """Generate unique chunk ID
        
        Args:
            document_id: Parent document ID
            chunk_index: Index of chunk in document
            method: Chunking method used
            
        Returns:
            Unique chunk ID
        """
        return f"{document_id}_chunk_{chunk_index}_{method[:3]}"
    
    async def chunk_document(self, 
                            text: str,
                            method: str = "auto",
                            target_chunk_tokens: int = 500,
                            max_chunk_tokens: int = 1500,
                            chunk_overlap_tokens: int = 50,
                            metadata: Optional[Dict[str, Any]] = None,
                            document_id: Optional[str] = None,
                            tenant_id: str = "default",
                            filename: str = "document.txt",
                            auto_store: bool = True) -> Dict[str, Any]:
        """Main entry point for document chunking with integrated storage
        
        Args:
            text: Document text to chunk
            method: Chunking method or "auto" for automatic selection
            target_chunk_tokens: Target size per chunk in tokens
            max_chunk_tokens: Maximum tokens per chunk
            chunk_overlap_tokens: Overlap between chunks in tokens
            metadata: Optional document metadata
            document_id: Optional document ID (will be generated if not provided)
            tenant_id: Tenant identifier for multi-tenancy
            filename: Original filename for MinIO storage
            auto_store: Whether to automatically store document and chunks
            
        Returns:
            Dictionary with chunks, storage info, and processing information
        """
        start_time = datetime.now()
        storage_info = {}
        
        # Handle document storage and ID generation
        if auto_store and self._storage_ready:
            # Save document to MinIO first and get MinIO-generated document ID
            try:
                storage_result = await self.save_document_to_minio(
                    content=text,
                    filename=filename,
                    tenant_id=tenant_id,
                    content_type="text/plain"
                )
                document_id = storage_result["document_id"]
                storage_info["minio"] = storage_result
                logger.info(f"📄 Document saved to MinIO with ID: {document_id}")
            except Exception as e:
                logger.error(f"Failed to save to MinIO: {e}")
                # Fallback to local ID generation
                if not document_id:
                    document_id = self.generate_document_id(text, metadata)
        else:
            # Generate document ID if not provided
            if not document_id:
                document_id = self.generate_document_id(text, metadata)
        
        # Determine chunking strategy
        if method == "auto" or method == ChunkingStrategy.AUTO.value:
            recommended_strategy, analysis = await self.analyzer.analyze_and_recommend_strategy(text, metadata)
            method = recommended_strategy
            logger.info(f"🎯 Auto-selected strategy: {method}")
        else:
            analysis = {'method': 'manual_selection'}
        
        # Apply the selected chunking method
        if method == ChunkingStrategy.SEMANTIC_COHERENCE.value:
            chunks = await self._chunk_semantic_coherence(
                text, target_chunk_tokens, max_chunk_tokens, chunk_overlap_tokens
            )
        elif method == ChunkingStrategy.HYBRID.value:
            chunks = await self._chunk_hybrid(
                text, target_chunk_tokens, max_chunk_tokens, chunk_overlap_tokens
            )
        elif method == ChunkingStrategy.LLM_ASSISTED.value:
            chunks = await self._chunk_llm_assisted(
                text, target_chunk_tokens, max_chunk_tokens, chunk_overlap_tokens
            )
        else:
            raise ValueError(f"Unknown chunking method: {method}")
        
        # Add overlap information and IDs to chunks
        chunks = self._add_overlap_to_chunks(chunks, text, chunk_overlap_tokens)
        chunks = self._add_chunk_ids(chunks, document_id, method)
        
        # Store chunks in PostgreSQL if auto_store is enabled
        if auto_store and self._storage_ready:
            try:
                await self.store_chunks_in_postgres(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    chunks=chunks,
                    metadata={
                        **metadata,
                        'chunking_method': method,
                        'analysis': analysis
                    }
                )
                storage_info["postgres"] = {
                    "chunks_stored": len(chunks),
                    "table": "chunks"
                }
                logger.info(f"💾 Stored {len(chunks)} chunks in PostgreSQL")
            except Exception as e:
                logger.error(f"Failed to store chunks in PostgreSQL: {e}")
                storage_info["postgres"] = {"error": str(e)}
        
        # Calculate statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            'success': True,
            'document_id': document_id,
            'method': method,
            'chunks': chunks,
            'storage': storage_info,
            'statistics': {
                'total_chunks': len(chunks),
                'original_length': len(text),
                'total_tokens': sum(c['token_count'] for c in chunks),
                'avg_tokens_per_chunk': sum(c['token_count'] for c in chunks) / max(len(chunks), 1),
                'processing_time_seconds': processing_time,
                'overlap_tokens': chunk_overlap_tokens,
                'storage_enabled': auto_store and self._storage_ready
            },
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def _add_chunk_ids(self, chunks: List[Dict[str, Any]], document_id: str, method: str) -> List[Dict[str, Any]]:
        """Add unique IDs to chunks
        
        Args:
            chunks: List of chunks
            document_id: Parent document ID
            method: Chunking method used
            
        Returns:
            Chunks with IDs
        """
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = self.generate_chunk_id(document_id, i, method)
            chunk['document_id'] = document_id
        
        return chunks
    
    def _add_overlap_to_chunks(self, chunks: List[Dict[str, Any]], 
                               original_text: str,
                               overlap_tokens: int) -> List[Dict[str, Any]]:
        """Add overlap context to chunks
        
        Args:
            chunks: List of chunks
            original_text: Original document text
            overlap_tokens: Number of overlap tokens
            
        Returns:
            Chunks with overlap information
        """
        for i, chunk in enumerate(chunks):
            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_start = self._get_overlap_text(
                    prev_chunk['text'], overlap_tokens, from_end=True
                )
                chunk['overlap_previous'] = overlap_start
                # Prepend overlap to chunk text for context
                chunk['text_with_overlap'] = overlap_start + " " + chunk['text']
            else:
                chunk['overlap_previous'] = None
                chunk['text_with_overlap'] = chunk['text']
            
            # Add overlap to next chunk
            if i < len(chunks) - 1:
                overlap_end = self._get_overlap_text(
                    chunk['text'], overlap_tokens, from_end=False
                )
                chunk['overlap_next'] = overlap_end
            else:
                chunk['overlap_next'] = None
            
            # Store overlap metadata
            chunk['overlap_tokens'] = overlap_tokens
            chunk['has_overlap'] = True
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int, from_end: bool = False) -> str:
        """Extract overlap text from chunk
        
        Args:
            text: Chunk text
            overlap_tokens: Number of tokens for overlap
            from_end: Whether to extract from end (True) or start (False)
            
        Returns:
            Overlap text
        """
        if not self.tokenizer:
            # Fallback to character-based overlap
            overlap_chars = overlap_tokens * 4  # Approximate
            if from_end:
                return text[-overlap_chars:] if len(text) > overlap_chars else text
            else:
                return text[:overlap_chars] if len(text) > overlap_chars else text
        
        # Token-based overlap
        tokens = self.tokenizer.encode(text)
        if from_end:
            overlap_token_ids = tokens[-overlap_tokens:] if len(tokens) > overlap_tokens else tokens
        else:
            overlap_token_ids = tokens[:overlap_tokens] if len(tokens) > overlap_tokens else tokens
        
        return self.tokenizer.decode(overlap_token_ids)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback approximation
            return int(len(text.split()) / 0.75)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Simple regex-based splitting
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    async def _chunk_semantic_coherence(self, text: str,
                                       target_tokens: int,
                                       max_tokens: int,
                                       overlap_tokens: int) -> List[Dict[str, Any]]:
        """Semantic coherence chunking with overlap
        
        Args:
            text: Document text
            target_tokens: Target chunk size
            max_tokens: Maximum chunk size
            overlap_tokens: Overlap size
            
        Returns:
            List of chunks with semantic boundaries
        """
        if not self.sentence_model:
            logger.warning("Sentence model not available, using sentence-based chunking")
            return self._fallback_sentence_chunking(text, target_tokens, max_tokens)
        
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return [{
                'text': text,
                'chunk_index': 0,
                'token_count': self._count_tokens(text),
                'method': 'semantic_coherence',
                'sentence_count': len(sentences),
                'start_char': 0,
                'end_char': len(text)
            }]
        
        # Compute sentence embeddings
        embeddings = self.sentence_model.encode(sentences)
        
        # Detect change points
        try:
            model = rpt.Binseg(model="rbf").fit(embeddings)
            n_bkps = max(1, len(sentences) // (target_tokens // 25))
            change_points = model.predict(n_bkps=n_bkps)
            change_points = [0] + change_points
        except Exception as e:
            logger.warning(f"Change point detection failed: {e}")
            # Fallback to uniform splits
            sentences_per_chunk = max(2, target_tokens // 25)
            change_points = list(range(0, len(sentences), sentences_per_chunk)) + [len(sentences)]
        
        # Create chunks
        chunks = []
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            # Calculate character positions
            start_char = text.find(chunk_sentences[0]) if chunk_sentences else 0
            end_char = start_char + len(chunk_text)
            
            # Check token count
            token_count = self._count_tokens(chunk_text)
            if token_count > max_tokens:
                # Split large chunk
                sub_chunks = self._split_large_chunk(chunk_sentences, target_tokens, max_tokens)
                for sub_chunk in sub_chunks:
                    sub_chunk['start_char'] = start_char
                    sub_chunk['end_char'] = start_char + len(sub_chunk['text'])
                    start_char = sub_chunk['end_char']
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': len(chunks),
                    'token_count': token_count,
                    'method': 'semantic_coherence',
                    'sentence_range': [start_idx, end_idx],
                    'sentence_count': len(chunk_sentences),
                    'start_char': start_char,
                    'end_char': end_char
                })
        
        return chunks
    
    async def _chunk_hybrid(self, text: str,
                          target_tokens: int,
                          max_tokens: int,
                          overlap_tokens: int) -> List[Dict[str, Any]]:
        """Hybrid structure-aware chunking with overlap"""
        # Identify structural boundaries
        lines = text.split('\n')
        structure_points = self._identify_structure_points(lines)
        
        chunks = []
        for i in range(len(structure_points) - 1):
            start_line = structure_points[i]
            end_line = structure_points[i + 1]
            
            chunk_lines = lines[start_line:end_line]
            chunk_text = '\n'.join(chunk_lines).strip()
            
            if not chunk_text:
                continue
            
            # Calculate character positions
            start_char = sum(len(line) + 1 for line in lines[:start_line])
            end_char = start_char + len(chunk_text)
            
            token_count = self._count_tokens(chunk_text)
            
            if token_count > max_tokens:
                # Sub-chunk using semantic coherence
                sub_chunks = await self._chunk_semantic_coherence(
                    chunk_text, target_tokens, max_tokens, overlap_tokens
                )
                for sub_chunk in sub_chunks:
                    sub_chunk['parent_structure'] = i
                    sub_chunk['method'] = 'hybrid_semantic'
                    sub_chunk['start_char'] += start_char
                    sub_chunk['end_char'] += start_char
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': len(chunks),
                    'token_count': token_count,
                    'method': 'hybrid',
                    'line_range': [start_line, end_line],
                    'structure_section': i,
                    'start_char': start_char,
                    'end_char': end_char
                })
        
        return chunks
    
    def _identify_structure_points(self, lines: List[str]) -> List[int]:
        """Identify structural boundaries in document"""
        structure_points = [0]
        
        heading_patterns = [
            r'^#{1,6}\s+.+$',           # Markdown headings
            r'^[A-Z][A-Za-z\s]+:$',     # Section headers
            r'^\d+\.\s+[A-Z].+$',       # Numbered sections
            r'^[A-Z\s]+\n[-=]{3,}$'    # Underlined headers
        ]
        
        in_code_block = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track code blocks
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                if not in_code_block:
                    structure_points.append(i + 1)
                continue
            
            if in_code_block:
                continue
            
            # Check for headings
            for pattern in heading_patterns:
                if re.match(pattern, stripped):
                    structure_points.append(i)
                    break
        
        structure_points.append(len(lines))
        return sorted(set(structure_points))
    
    async def _chunk_llm_assisted(self, text: str,
                                 target_tokens: int,
                                 max_tokens: int,
                                 overlap_tokens: int) -> List[Dict[str, Any]]:
        """LLM-assisted chunking with overlap"""
        if not self.ollama_client:
            logger.warning("Ollama client not available, falling back to semantic coherence")
            return await self._chunk_semantic_coherence(text, target_tokens, max_tokens, overlap_tokens)
        
        try:
            # Get LLM analysis for chunk boundaries
            boundaries = await self._get_llm_boundaries(text, target_tokens, max_tokens)
            
            if not boundaries:
                logger.warning("No boundaries from LLM, using semantic coherence")
                return await self._chunk_semantic_coherence(text, target_tokens, max_tokens, overlap_tokens)
            
            # Create chunks from boundaries
            chunks = []
            boundaries = [0] + sorted(boundaries) + [len(text)]
            
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                
                chunk_text = text[start:end].strip()
                if not chunk_text:
                    continue
                
                token_count = self._count_tokens(chunk_text)
                
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': len(chunks),
                    'token_count': token_count,
                    'method': 'llm_assisted',
                    'boundary_positions': [start, end],
                    'llm_optimized': True,
                    'start_char': start,
                    'end_char': end
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"LLM-assisted chunking failed: {e}")
            return await self._chunk_semantic_coherence(text, target_tokens, max_tokens, overlap_tokens)
    
    async def _get_llm_boundaries(self, text: str, target_tokens: int, max_tokens: int) -> List[int]:
        """Get optimal chunk boundaries from LLM"""
        # Truncate text if necessary
        max_context = 10000
        truncated = text[:max_context] if len(text) > max_context else text
        
        prompt = f"""Analyze this text and identify optimal chunk boundaries.

TARGET: ~{target_tokens} tokens per chunk
MAX: {max_tokens} tokens per chunk

TEXT:
{truncated}

Return ONLY character positions as a JSON array of integers where chunks should split.
Example: [450, 920, 1380, 1890]"""

        try:
            response = await self.ollama_client.chat(
                model="gpt-oss:20b",
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )
            
            # Parse response
            response_text = response['message']['content']
            
            # Extract JSON array
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                boundaries = json.loads(response_text[json_start:json_end])
                return [b for b in boundaries if isinstance(b, int) and 0 < b < len(text)]
            
        except Exception as e:
            logger.error(f"Failed to get LLM boundaries: {e}")
        
        return []
    
    def _fallback_sentence_chunking(self, text: str, target_tokens: int, max_tokens: int) -> List[Dict[str, Any]]:
        """Fallback chunking based on sentences"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_start = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': len(chunks),
                    'token_count': current_tokens,
                    'method': 'sentence_fallback',
                    'start_char': current_start,
                    'end_char': current_start + len(chunk_text)
                })
                current_start += len(chunk_text) + 1
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_index': len(chunks),
                'token_count': current_tokens,
                'method': 'sentence_fallback',
                'start_char': current_start,
                'end_char': current_start + len(chunk_text)
            })
        
        return chunks
    
    def _split_large_chunk(self, sentences: List[str], target_tokens: int, max_tokens: int) -> List[Dict[str, Any]]:
        """Split sentences that form too large a chunk"""
        chunks = []
        current_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens and current_sentences:
                # Create chunk
                chunk_text = ' '.join(current_sentences)
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': len(chunks),
                    'token_count': current_tokens,
                    'method': 'split_large'
                })
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append({
                'text': chunk_text,
                'chunk_index': len(chunks),
                'token_count': current_tokens,
                'method': 'split_large'
            })
        
        return chunks
