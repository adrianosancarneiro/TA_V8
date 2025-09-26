#!/usr/bin/env python3
"""
# =============================================================================
# TA_V8 RAG SYSTEM - QUERY UTILITY
# =============================================================================
# Purpose: Easy query interface for testing the RAG Agent Team
# 
# Usage:
#   python query_system.py "What are the main benefits?"
#   python query_system.py --query "How does this work?" --max-rounds 5
# 
# This script provides a simple interface to query the RAG Agent Team
# and see the complete critic-driven multi-agent response process.
# =============================================================================
"""

import asyncio
import json
import argparse
import sys
from datetime import datetime
import httpx

class RAGQueryInterface:
    """
    Simple query interface for TA_V8 RAG Agent Team
    """
    
    def __init__(self, tenant_id="demo_org", domain_id="knowledge_base"):
        self.tenant_id = tenant_id
        self.domain_id = domain_id
        self.agent_team_url = "http://localhost:8006/query"
        self.health_url = "http://localhost:8006/health"
        self.http_client = httpx.AsyncClient(timeout=120.0)
    
    async def query(self, question: str, max_rounds: int = 3, session_id: str = None) -> dict:
        """Send a query to the RAG Agent Team"""
        print(f"🤖 Querying RAG Agent Team...")
        print(f"   Question: {question}")
        print(f"   Max rounds: {max_rounds}")
        print()
        
        query_request = {
            "query": question,
            "tenant_id": self.tenant_id,
            "session_id": session_id,
            "max_results": 10,
            "max_rounds": max_rounds,
            "include_sources": True,
            "context": {
                "domain_id": self.domain_id,
                "query_time": datetime.now().isoformat(),
                "interface": "cli_query_tool"
            }
        }
        
        try:
            print("🔄 Processing query through critic-driven multi-agent system...")
            
            response = await self.http_client.post(
                self.agent_team_url,
                json=query_request
            )
            
            if response.status_code != 200:
                error_detail = response.text if response.text else f"HTTP {response.status_code}"
                raise Exception(f"Query failed: {error_detail}")
            
            result = response.json()
            
            # Display results
            self._display_results(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _display_results(self, result: dict):
        """Display query results in a readable format"""
        print("📋 QUERY RESULTS")
        print("=" * 50)
        
        if result.get("success"):
            # Main response
            response_text = result.get("response", "No response generated")
            print("🤖 AGENT RESPONSE:")
            print("-" * 20)
            print(response_text)
            print()
            
            # Metadata
            metadata = result.get("metadata", {})
            print("📊 PROCESSING DETAILS:")
            print("-" * 20)
            print(f"   Processing time: {metadata.get('total_processing_time', 0):.2f}s")
            print(f"   Rounds completed: {metadata.get('rounds_completed', 0)}/{metadata.get('max_rounds', 3)}")
            print(f"   Information sufficient: {metadata.get('information_sufficient', False)}")
            print(f"   Critic feedback count: {metadata.get('critic_feedback_count', 0)}")
            print(f"   Model used: {metadata.get('model_used', 'N/A')}")
            print()
            
            # Sources
            sources = result.get("sources", [])
            if sources:
                print(f"📚 SOURCES USED ({len(sources)} total):")
                print("-" * 20)
                for i, source in enumerate(sources[:5]):  # Show top 5 sources
                    score = source.get("similarity_score", source.get("score", 0))
                    text_preview = source.get("text_snippet", source.get("text", ""))[:150]
                    print(f"   {i+1}. Score: {score:.3f}")
                    print(f"      {text_preview}...")
                    print()
                
                if len(sources) > 5:
                    print(f"   ... and {len(sources) - 5} more sources")
                    print()
            else:
                print("📚 No sources found")
                print()
                
        else:
            print("❌ Query failed")
            error = result.get("error", "Unknown error")
            print(f"   Error: {error}")
            print()
    
    async def health_check(self):
        """Check RAG Agent Team health"""
        print("🔍 Checking RAG Agent Team health...")
        
        try:
            response = await self.http_client.get(self.health_url)
            
            if response.status_code != 200:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                return False
            
            health_data = response.json()
            status = health_data.get("status", "unknown")
            
            if status == "healthy":
                print("✅ RAG Agent Team is healthy")
                
                # Show service details
                services = health_data.get("services", {})
                if services:
                    print("   Connected services:")
                    for service_name, service_info in services.items():
                        service_status = service_info.get("status", "unknown")
                        emoji = "✅" if service_status == "healthy" else "⚠️"
                        print(f"   {emoji} {service_name}")
                
                agents = health_data.get("agents", {})
                if agents:
                    print(f"   Available agents: {len(agents)}")
                
                return True
            else:
                print(f"⚠️ RAG Agent Team status: {status}")
                return False
                
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return False
    
    async def interactive_mode(self):
        """Interactive query mode"""
        print("🎯 INTERACTIVE QUERY MODE")
        print("=" * 30)
        print("Type your questions and press Enter.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'health' to check system status.")
        print()
        
        session_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        query_count = 0
        
        while True:
            try:
                question = input("🔍 Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'health':
                    await self.health_check()
                    print()
                    continue
                
                query_count += 1
                print(f"\n--- Query #{query_count} ---")
                
                await self.query(
                    question=question,
                    max_rounds=3,
                    session_id=session_id
                )
                
                print("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in interactive mode: {str(e)}")
                print()
    
    async def shutdown(self):
        """Clean up HTTP client"""
        await self.http_client.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Query TA_V8 RAG Agent Team")
    
    parser.add_argument("query", nargs="?", help="Query to ask (or use --interactive)")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Start interactive query mode")
    parser.add_argument("--tenant", default="demo_org", 
                       help="Tenant ID (default: demo_org)")
    parser.add_argument("--domain", default="knowledge_base", 
                       help="Domain ID (default: knowledge_base)")
    parser.add_argument("--max-rounds", type=int, default=3,
                       help="Maximum rounds for critic-driven refinement (default: 3)")
    parser.add_argument("--session-id", help="Session ID for conversational context")
    parser.add_argument("--health", action="store_true", 
                       help="Check system health and exit")
    
    args = parser.parse_args()
    
    print("🚀 TA_V8 RAG SYSTEM - QUERY INTERFACE")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tenant: {args.tenant}")
    print(f"Domain: {args.domain}")
    print()
    
    interface = RAGQueryInterface(tenant_id=args.tenant, domain_id=args.domain)
    
    try:
        # Health check mode
        if args.health:
            await interface.health_check()
            return True
        
        # Interactive mode
        if args.interactive:
            # Check health first
            healthy = await interface.health_check()
            if not healthy:
                print("⚠️ System may not be fully operational")
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return False
            
            print()
            await interface.interactive_mode()
            return True
        
        # Single query mode
        if args.query:
            # Check health first
            healthy = await interface.health_check()
            if not healthy:
                print("❌ System is not healthy - query may fail")
                return False
            
            print()
            result = await interface.query(
                question=args.query,
                max_rounds=args.max_rounds,
                session_id=args.session_id
            )
            
            return result.get("success", False)
        else:
            parser.print_help()
            return False
        
    except Exception as e:
        print(f"❌ Query interface failed: {str(e)}")
        return False
    
    finally:
        await interface.shutdown()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)