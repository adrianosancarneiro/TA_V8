#!/usr/bin/env python3
"""
# =============================================================================
# TA_V8 RAG SYSTEM - DOCUMENT UPLOAD UTILITY
# =============================================================================
# Purpose: Easy document upload and processing for demo system
# 
# Usage:
#   python upload_documents.py --text "Your text content here"
#   python upload_documents.py --file /path/to/document.txt
#   python upload_documents.py --url https://example.com/document
# 
# This script will:
# 1. Chunk your documents
# 2. Generate embeddings
# 3. Store in vector database
# 4. Make them available for RAG queries
# =============================================================================
"""

import asyncio
import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
import httpx

class DocumentUploader:
    """
    Simple document uploader for TA_V8 RAG system
    """
    
    def __init__(self, tenant_id="demo_org", domain_id="knowledge_base"):
        self.tenant_id = tenant_id
        self.domain_id = domain_id
        self.chunking_url = "http://localhost:8001/mcp/execute"
        self.embedding_url = "http://localhost:8012/mcp/execute"
        self.http_client = httpx.AsyncClient(timeout=60.0)
    
    async def upload_text(self, text: str, title: str = None) -> dict:
        """Upload text content directly"""
        print(f"📄 Processing text content ({len(text)} chars)...")
        
        chunk_request = {
            "tenant_id": self.tenant_id,
            "domain_id": self.domain_id,
            "source": {
                "type": "text",
                "text": text
            },
            "policy": {
                "method": "auto",
                "target_tokens": 512,
                "overlap": 64
            },
            "metadata": {
                "title": title or f"Document_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "upload_time": datetime.now().isoformat(),
                "source_type": "direct_text"
            }
        }
        
        return await self._process_document(chunk_request, text[:100])
    
    async def upload_file(self, file_path: str) -> dict:
        """Upload content from a file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📁 Processing file: {file_path.name}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        
        chunk_request = {
            "tenant_id": self.tenant_id,
            "domain_id": self.domain_id,
            "source": {
                "type": "text",
                "text": text
            },
            "policy": {
                "method": "auto",
                "target_tokens": 512,
                "overlap": 64
            },
            "metadata": {
                "title": file_path.stem,
                "filename": file_path.name,
                "file_extension": file_path.suffix,
                "upload_time": datetime.now().isoformat(),
                "source_type": "file_upload",
                "file_size": len(text)
            }
        }
        
        return await self._process_document(chunk_request, f"file: {file_path.name}")
    
    async def upload_url(self, url: str) -> dict:
        """Upload content from a URL"""
        print(f"🌐 Processing URL: {url}")
        
        try:
            response = await self.http_client.get(url, timeout=30.0)
            response.raise_for_status()
            text = response.text
        except Exception as e:
            raise Exception(f"Failed to fetch URL content: {str(e)}")
        
        chunk_request = {
            "tenant_id": self.tenant_id,
            "domain_id": self.domain_id,
            "source": {
                "type": "text",
                "text": text
            },
            "policy": {
                "method": "auto",
                "target_tokens": 512,
                "overlap": 64
            },
            "metadata": {
                "title": url.split('/')[-1] or f"URL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "source_url": url,
                "upload_time": datetime.now().isoformat(),
                "source_type": "url_fetch",
                "content_length": len(text)
            }
        }
        
        return await self._process_document(chunk_request, f"URL: {url}")
    
    async def _process_document(self, chunk_request: dict, description: str) -> dict:
        """Process document through the complete pipeline"""
        try:
            # Step 1: Chunk the document
            print("🔧 Step 1: Chunking document...")
            
            chunk_response = await self.http_client.post(
                self.chunking_url,
                json=chunk_request
            )
            
            if chunk_response.status_code != 200:
                raise Exception(f"Chunking failed: HTTP {chunk_response.status_code}")
            
            chunk_result = chunk_response.json()
            chunks = chunk_result.get("chunks", [])
            
            if not chunks:
                raise Exception("No chunks were created from the document")
            
            print(f"✅ Created {len(chunks)} chunks")
            
            # Step 2: Embed the chunks
            print("🔧 Step 2: Generating embeddings...")
            
            embed_items = []
            for chunk in chunks:
                embed_items.append({
                    "id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                })
            
            embed_request = {
                "tenant_id": self.tenant_id,
                "collection": self.domain_id,
                "items": embed_items,
                "upsert": True,
                "metadata": {
                    "batch_type": "document_upload",
                    "document_description": description
                }
            }
            
            embed_response = await self.http_client.post(
                self.embedding_url,
                json=embed_request
            )
            
            if embed_response.status_code != 200:
                print(f"⚠️ Warning: Embedding failed: HTTP {embed_response.status_code}")
                print("   Document was chunked but may not be searchable yet")
            else:
                embed_result = embed_response.json()
                upserted_count = embed_result.get("upserted", 0)
                print(f"✅ Embedded {upserted_count} chunks in vector database")
            
            # Return processing summary
            result = {
                "success": True,
                "document_id": chunk_result.get("document_id"),
                "chunks_created": len(chunks),
                "chunks_embedded": embed_result.get("upserted", 0) if embed_response.status_code == 200 else 0,
                "tenant_id": self.tenant_id,
                "domain_id": self.domain_id,
                "processing_time": datetime.now().isoformat(),
                "description": description
            }
            
            print("🎉 Document processing completed!")
            print(f"   Document ID: {result['document_id']}")
            print(f"   Chunks: {result['chunks_created']}")
            print(f"   Embedded: {result['chunks_embedded']}")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "description": description
            }
            print(f"❌ Document processing failed: {str(e)}")
            return error_result
    
    async def test_retrieval(self, query: str = "What is this document about?") -> dict:
        """Test retrieval with a sample query"""
        print(f"🔍 Testing retrieval with query: '{query}'")
        
        try:
            retrieval_request = {
                "tenant_id": self.tenant_id,
                "collection": self.domain_id,
                "query": {
                    "text": query,
                    "use_embedding": True
                },
                "top_k": 3
            }
            
            retrieval_response = await self.http_client.post(
                "http://localhost:8003/mcp/execute",
                json=retrieval_request
            )
            
            if retrieval_response.status_code != 200:
                raise Exception(f"Retrieval failed: HTTP {retrieval_response.status_code}")
            
            result = retrieval_response.json()
            hits = result.get("hits", [])
            
            print(f"✅ Found {len(hits)} relevant chunks")
            for i, hit in enumerate(hits[:3]):
                print(f"   {i+1}. Score: {hit['score']:.3f} - {hit['text'][:100]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ Retrieval test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def shutdown(self):
        """Clean up HTTP client"""
        await self.http_client.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Upload documents to TA_V8 RAG system")
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Text content to upload")
    input_group.add_argument("--file", help="File path to upload")
    input_group.add_argument("--url", help="URL to fetch and upload")
    
    # Optional parameters
    parser.add_argument("--tenant", default="demo_org", help="Tenant ID (default: demo_org)")
    parser.add_argument("--domain", default="knowledge_base", help="Domain ID (default: knowledge_base)")
    parser.add_argument("--title", help="Document title (auto-generated if not provided)")
    parser.add_argument("--test-query", default="What is this document about?", 
                       help="Test query after upload (default: 'What is this document about?')")
    parser.add_argument("--no-test", action="store_true", help="Skip retrieval test after upload")
    
    args = parser.parse_args()
    
    print("🚀 TA_V8 RAG SYSTEM - DOCUMENT UPLOADER")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tenant: {args.tenant}")
    print(f"Domain: {args.domain}")
    print()
    
    uploader = DocumentUploader(tenant_id=args.tenant, domain_id=args.domain)
    
    try:
        # Upload document
        if args.text:
            result = await uploader.upload_text(args.text, args.title)
        elif args.file:
            result = await uploader.upload_file(args.file)
        elif args.url:
            result = await uploader.upload_url(args.url)
        
        if not result["success"]:
            print(f"\n❌ Upload failed: {result['error']}")
            return False
        
        # Test retrieval if requested
        if not args.no_test:
            print("\n🔍 TESTING RETRIEVAL")
            print("-" * 20)
            await uploader.test_retrieval(args.test_query)
        
        print("\n✅ DOCUMENT READY FOR QUERIES!")
        print("   You can now use the RAG Agent Team to ask questions about your document:")
        print(f"   POST http://localhost:8006/query")
        print(f"   {{")
        print(f'     "query": "Your question here",')
        print(f'     "tenant_id": "{args.tenant}",')
        print(f'     "domain_id": "{args.domain}"')
        print(f"   }}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload process failed: {str(e)}")
        return False
    
    finally:
        await uploader.shutdown()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)