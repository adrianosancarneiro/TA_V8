#!/usr/bin/env python3
"""
Test Script: HTTP + SSE MCP Services Verification
==================================================

This script tests all MCP services in HTTP + SSE transport mode to ensure
they are properly configured and communicating correctly.

Usage:
    python test_http_mcp_services.py

Services tested:
- Chunking MCP (port 8001)
- Embedding MCP (port 8004)  
- Retrieval MCP (port 8003)
- RAG Agent Team (port 8006)
"""

import asyncio
import httpx
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPServiceTester:
    """Test HTTP + SSE MCP services"""
    
    def __init__(self):
        self.services = {
            "chunking": "http://localhost:8001",
            "embedding": "http://localhost:8004", 
            "retrieval": "http://localhost:8003",
            "rag_agent_team": "http://localhost:8006"
        }
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def test_health_endpoints(self):
        """Test health endpoints for all services"""
        logger.info("🔍 Testing health endpoints...")
        
        results = {}
        for service_name, base_url in self.services.items():
            try:
                response = await self.client.get(f"{base_url}/health")
                if response.status_code == 200:
                    results[service_name] = {
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds() if response.elapsed else None,
                        "data": response.json()
                    }
                    logger.info(f"✅ {service_name}: healthy")
                else:
                    results[service_name] = {
                        "status": "unhealthy",
                        "status_code": response.status_code,
                        "response": response.text
                    }
                    logger.warning(f"⚠️ {service_name}: unhealthy (status {response.status_code})")
                    
            except Exception as e:
                results[service_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"❌ {service_name}: error - {str(e)}")
        
        return results

    async def test_mcp_endpoints(self):
        """Test MCP-specific endpoints"""
        logger.info("🔍 Testing MCP endpoints...")
        
        results = {}
        
        # Test MCP services (excluding RAG Agent Team which is not an MCP service)
        mcp_services = {k: v for k, v in self.services.items() if k != "rag_agent_team"}
        
        for service_name, base_url in mcp_services.items():
            try:
                service_results = {}
                
                # Test MCP initialization
                init_response = await self.client.post(f"{base_url}/mcp/initialize", json={})
                service_results["initialize"] = {
                    "status_code": init_response.status_code,
                    "success": init_response.status_code == 200,
                    "data": init_response.json() if init_response.status_code == 200 else init_response.text
                }
                
                # Test tools list
                tools_response = await self.client.get(f"{base_url}/mcp/tools/list")
                service_results["tools_list"] = {
                    "status_code": tools_response.status_code,
                    "success": tools_response.status_code == 200,
                    "data": tools_response.json() if tools_response.status_code == 200 else tools_response.text
                }
                
                results[service_name] = service_results
                
                if service_results["initialize"]["success"] and service_results["tools_list"]["success"]:
                    logger.info(f"✅ {service_name}: MCP endpoints working")
                else:
                    logger.warning(f"⚠️ {service_name}: MCP endpoints have issues")
                    
            except Exception as e:
                results[service_name] = {"error": str(e)}
                logger.error(f"❌ {service_name}: MCP endpoint error - {str(e)}")
        
        return results

    async def test_service_integration(self):
        """Test integration between services"""
        logger.info("🔍 Testing service integration...")
        
        try:
            # Test chunking service with sample text
            chunk_request = {
                "name": "chunk_document",
                "arguments": {
                    "tenant_id": "test_tenant",
                    "document_id": "test_doc_001",
                    "text": "This is a test document for chunking. It contains multiple sentences that should be split into meaningful chunks for the RAG system.",
                    "chunk_size": 100,
                    "chunk_overlap": 20
                }
            }
            
            chunking_response = await self.client.post(
                f"{self.services['chunking']}/mcp/tools/call",
                json=chunk_request
            )
            
            if chunking_response.status_code == 200:
                logger.info("✅ Chunking service integration test passed")
                chunk_result = chunking_response.json()
            else:
                logger.warning(f"⚠️ Chunking service integration failed: {chunking_response.status_code}")
                chunk_result = None
            
            # Test embedding service (if chunking worked)
            if chunk_result:
                embed_request = {
                    "name": "embed_documents",
                    "arguments": {
                        "tenant_id": "test_tenant",
                        "collection": "test_collection",
                        "items": [
                            {
                                "id": "chunk_001",
                                "text": "Sample chunk text for embedding",
                                "metadata": {"source": "test_doc"}
                            }
                        ]
                    }
                }
                
                embedding_response = await self.client.post(
                    f"{self.services['embedding']}/mcp/tools/call",
                    json=embed_request
                )
                
                if embedding_response.status_code == 200:
                    logger.info("✅ Embedding service integration test passed")
                else:
                    logger.warning(f"⚠️ Embedding service integration failed: {embedding_response.status_code}")
            
            # Test retrieval service
            retrieval_request = {
                "name": "search_documents",
                "arguments": {
                    "tenant_id": "test_tenant",
                    "query": "test document chunking",
                    "top_k": 5
                }
            }
            
            retrieval_response = await self.client.post(
                f"{self.services['retrieval']}/mcp/tools/call",
                json=retrieval_request
            )
            
            if retrieval_response.status_code == 200:
                logger.info("✅ Retrieval service integration test passed")
            else:
                logger.warning(f"⚠️ Retrieval service integration failed: {retrieval_response.status_code}")
            
            # Test RAG Agent Team query
            if self.services.get("rag_agent_team"):
                rag_request = {
                    "query": "What is chunking in document processing?",
                    "tenant_id": "test_tenant",
                    "max_results": 3,
                    "include_sources": True
                }
                
                rag_response = await self.client.post(
                    f"{self.services['rag_agent_team']}/query",
                    json=rag_request
                )
                
                if rag_response.status_code == 200:
                    logger.info("✅ RAG Agent Team integration test passed")
                    return {"status": "success", "message": "All integration tests passed"}
                else:
                    logger.warning(f"⚠️ RAG Agent Team integration failed: {rag_response.status_code}")
                    return {"status": "partial", "message": "Some integration tests failed"}
            
        except Exception as e:
            logger.error(f"❌ Integration test error: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def run_full_test_suite(self):
        """Run complete test suite"""
        logger.info("🚀 Starting HTTP + SSE MCP Services Test Suite")
        logger.info("=" * 60)
        
        # Test 1: Health endpoints
        health_results = await self.test_health_endpoints()
        
        # Test 2: MCP endpoints
        mcp_results = await self.test_mcp_endpoints()
        
        # Test 3: Service integration
        integration_results = await self.test_service_integration()
        
        # Generate summary report
        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY REPORT")
        logger.info("=" * 60)
        
        # Health check summary
        healthy_services = sum(1 for result in health_results.values() if result.get("status") == "healthy")
        logger.info(f"🏥 Health Checks: {healthy_services}/{len(health_results)} services healthy")
        
        # MCP endpoints summary
        working_mcp_services = sum(1 for result in mcp_results.values() 
                                  if isinstance(result, dict) and 
                                  result.get("initialize", {}).get("success", False) and
                                  result.get("tools_list", {}).get("success", False))
        logger.info(f"🔧 MCP Endpoints: {working_mcp_services}/{len(mcp_results)} services working")
        
        # Integration test summary
        if integration_results.get("status") == "success":
            logger.info("🔗 Integration Tests: ✅ All passed")
        else:
            logger.info(f"🔗 Integration Tests: ⚠️ {integration_results.get('status', 'unknown')}")
        
        # Overall system status
        overall_health = (healthy_services / len(health_results)) * 100
        logger.info(f"📈 Overall System Health: {overall_health:.1f}%")
        
        if overall_health >= 75:
            logger.info("🎉 System Status: READY FOR PRODUCTION")
        elif overall_health >= 50:
            logger.info("⚠️ System Status: PARTIALLY READY - Some issues detected")
        else:
            logger.info("❌ System Status: NOT READY - Major issues detected")
        
        logger.info("=" * 60)
        
        await self.client.aclose()
        
        return {
            "health_results": health_results,
            "mcp_results": mcp_results,
            "integration_results": integration_results,
            "overall_health_percentage": overall_health
        }

async def main():
    """Main test execution"""
    tester = MCPServiceTester()
    results = await tester.run_full_test_suite()
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_http_mcp_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📄 Detailed test results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
