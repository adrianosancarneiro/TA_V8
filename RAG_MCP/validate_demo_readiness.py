#!/usr/bin/env python3
"""
Pre-Demo Validation Script
Quickly validates that all RAG MCP services are ready for stakeholder demo
"""

import asyncio
import json
import time
import socket
from datetime import datetime
try:
    import requests
except ImportError:
    import httpx
    # Use httpx as fallback if requests not available
    class RequestsCompat:
        @staticmethod
        def get(url, **kwargs):
            with httpx.Client() as client:
                return client.get(url, **kwargs)
        
        @staticmethod  
        def post(url, **kwargs):
            with httpx.Client() as client:
                return client.post(url, **kwargs)
    
    requests = RequestsCompat()

def print_status(service, status, details=""):
    emoji = "✅" if status else "❌"
    print(f"{emoji} {service:20} {details}")

async def validate_demo_readiness():
    print("🎯 RAG MCP Demo Readiness Validation")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Service endpoints to check
    services = {
        "PostgreSQL": "http://localhost:5432",
        "Neo4j": "http://localhost:7474",
        "Qdrant": "http://localhost:6333",
        "MinIO": "http://localhost:9001",
        "Ollama": "http://localhost:11434",
        "Embedding Service": "http://localhost:8080",
        "Chunking MCP": "http://localhost:8001/health",
        "Embedding MCP": "http://localhost:8012/health", 
        "Retrieval MCP": "http://localhost:8003/health",
        "RAG Agent Team": "http://localhost:8006/health"
    }
    
    print("🔍 SERVICE HEALTH CHECKS")
    print("-" * 30)
    
    all_healthy = True
    
    for service_name, url in services.items():
        try:
            if service_name in ["PostgreSQL", "Neo4j", "Qdrant", "MinIO", "Ollama", "Embedding Service"]:
                # For core services, just check if port is open
                import socket
                host, port = url.replace("http://", "").split(":")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, int(port)))
                sock.close()
                
                if result == 0:
                    print_status(service_name, True, "Port accessible")
                else:
                    print_status(service_name, False, "Port not accessible")
                    all_healthy = False
            else:
                # For MCP services, check health endpoints
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print_status(service_name, True, f"HTTP {response.status_code}")
                else:
                    print_status(service_name, False, f"HTTP {response.status_code}")
                    all_healthy = False
        except Exception as e:
            print_status(service_name, False, f"Error: {str(e)[:30]}")
            all_healthy = False
    
    print()
    print("🧪 FUNCTIONAL TESTS")
    print("-" * 20)
    
    # Test document processing
    try:
        test_doc = """
        AI-Powered Business Intelligence Dashboard
        
        Our new BI dashboard provides real-time insights into:
        - Customer behavior patterns
        - Sales performance metrics  
        - Operational efficiency indicators
        - Predictive analytics forecasts
        
        Key benefits include 40% faster decision-making and 25% cost reduction.
        """
        
        chunk_payload = {
            "tenant_id": "demo",
            "domain_id": "business_intelligence",
            "source": {"type": "text", "text": test_doc},
            "policy": {"method": "auto", "target_tokens": 200}
        }
        
        response = requests.post(
            "http://localhost:8001/mcp/execute",
            json=chunk_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            chunk_count = len(result.get("chunks", []))
            print_status("Document Chunking", True, f"{chunk_count} chunks created")
        else:
            print_status("Document Chunking", False, f"HTTP {response.status_code}")
            all_healthy = False
            
    except Exception as e:
        print_status("Document Chunking", False, f"Error: {str(e)[:30]}")
        all_healthy = False
    
    # Test retrieval
    try:
        retrieval_payload = {
            "tenant_id": "demo",
            "collection": "domain:business_intelligence",
            "query": {"text": "What are the key benefits?", "use_embedding": True},
            "top_k": 3
        }
        
        response = requests.post(
            "http://localhost:8003/mcp/execute",
            json=retrieval_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            results_count = len(result.get("results", []))
            print_status("Semantic Retrieval", True, f"{results_count} results found")
        else:
            print_status("Semantic Retrieval", False, f"HTTP {response.status_code}")
            all_healthy = False
            
    except Exception as e:
        print_status("Semantic Retrieval", False, f"Error: {str(e)[:30]}")
        all_healthy = False
    
    # Test RAG Agent Team
    try:
        agent_payload = {
            "query": "What are the main benefits of the BI dashboard?",
            "tenant_id": "demo",
            "domain_id": "business_intelligence"
        }
        
        response = requests.post(
            "http://localhost:8006/execute",
            json=agent_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "")
            print_status("RAG Agent Team", True, f"Generated {len(answer)} char answer")
        else:
            print_status("RAG Agent Team", False, f"HTTP {response.status_code}")
            all_healthy = False
            
    except Exception as e:
        print_status("RAG Agent Team", False, f"Error: {str(e)[:30]}")
        all_healthy = False
    
    print()
    print("📊 DEMO READINESS SUMMARY")
    print("=" * 30)
    
    if all_healthy:
        print("🎉 DEMO READY!")
        print("✅ All services are operational")
        print("✅ Document processing working")
        print("✅ Semantic retrieval working")  
        print("✅ RAG agent team working")
        print()
        print("🚀 You can start your stakeholder demo!")
        print("📋 Follow the DEMO_GUIDE_1_HOUR.md for the presentation")
        return True
    else:
        print("⚠️ ISSUES DETECTED")
        print("❌ Some services need attention before demo")
        print("🔧 Run: ./deploy_rag_mcp.sh restart")
        print("🧪 Then re-run this validation script")
        return False

if __name__ == "__main__":
    asyncio.run(validate_demo_readiness())