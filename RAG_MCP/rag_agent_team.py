#!/usr/bin/env python3
"""
# =============================================================================
# TA_V8 RAG AGENT TEAM - MULTI-AGENT SYSTEM
# =============================================================================
# Purpose: Orchestrate multiple specialized agents for intelligent document retrieval
#          and response generation using RAG (Retrieval-Augmented Generation)
#
# Architecture: Multi-Agent System with specialized roles:
# - Retrieval Agent: Information discovery and search optimization
# - Refiner Agent: Information synthesis and quality enhancement  
# - Response Agent: Final response generation with proper citations
#
# Integration: Works with the new MCP microservices architecture
# - Uses MCP services for chunking, embedding, and retrieval
# - Provides high-level orchestration layer for complex queries
# - Maintains conversational context and session management
# =============================================================================
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

# Configure logging for comprehensive monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Comprehensive query request model for multi-agent processing"""
    query: str = Field(..., description="Natural language query from user")
    tenant_id: str = Field(..., description="Tenant identifier for secure data isolation")
    session_id: Optional[str] = Field(None, description="Session ID for conversational context")
    max_results: int = Field(5, description="Maximum number of results to retrieve")
    include_sources: bool = Field(True, description="Whether to include source citations")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context and metadata")

@dataclass
class Agent:
    """Agent configuration with role, capabilities, and tools"""
    name: str
    role: str
    system_prompt: str
    tools: List[str]
    capabilities: Optional[Dict[str, Any]] = None

@dataclass
class AgentResponse:
    """Comprehensive response from individual agent execution"""
    success: bool
    agent_name: str
    response_data: Dict[str, Any]
    metadata: Dict[str, Any]
    sources: List[Dict[str, Any]]
    processing_time: float

# ============================================================================
# MULTI-AGENT ORCHESTRATION CLASS
# ============================================================================

class RAGAgentTeam:
    """
    Multi-Agent RAG System Orchestrator
    
    Coordinates specialized agents to provide comprehensive, accurate responses
    to user queries through sophisticated retrieval-augmented generation.
    
    Features:
    - Multi-agent pipeline with specialized roles
    - Integration with MCP microservices
    - Conversational session management  
    - Comprehensive error handling and monitoring
    - Production-ready FastAPI endpoints
    """
    
    def __init__(self):
        """Initialize the multi-agent RAG system with comprehensive configuration
        
        Sets up:
        - Secure connections to MCP microservices
        - Specialized agent configurations with role-specific prompts
        - HTTP clients with proper timeout and retry settings
        - FastAPI application for team coordination endpoints
        - Comprehensive logging and monitoring infrastructure
        """
        logger.info("🤖 Initializing TA_V8 RAG Agent Team...")
        
        # MCP Microservices endpoints (connects to our new architecture)
        self.chunking_mcp_url = "http://localhost:8001"    # Chunking MCP service
        self.embedding_mcp_url = "http://localhost:8002"   # Embedding MCP service  
        self.retrieval_mcp_url = "http://localhost:8003"   # Retrieval MCP service
        self.ollama_url = "http://localhost:11434"         # Ollama LLM service
        
        # Production LLM model configuration
        self.model = "llama3.2:latest"  # Production-ready model with 8B+ parameters
        
        # HTTP client configuration with production-ready settings
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),     # Generous timeout for complex operations
            limits=httpx.Limits(             # Connection limits for resource management
                max_connections=20,          # Maximum concurrent connections
                max_keepalive_connections=5  # Keep-alive connections for efficiency
            ),
            retries=3                        # Automatic retry for transient failures
        )
        
        # Initialize FastAPI application for team coordination
        self.app = FastAPI(
            title="TA_V8 RAG Agent Team",
            description="Multi-agent system for intelligent document retrieval and response generation",
            version="8.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize specialized agents with comprehensive role definitions
        self._initialize_agents()
        
        # Register API endpoints for team coordination
        self._register_endpoints()
        
        # Initialize session management for conversational interactions
        self.active_sessions = {}  # Session ID -> conversation context mapping
        
        logger.info("✅ RAG Agent Team initialization completed successfully")
    
    def _initialize_agents(self):
        """Initialize all specialized agents with comprehensive role definitions"""
        logger.info("🔧 Initializing specialized RAG agents...")
        
        self.agents = {
            "retrieval": Agent(
                name="Retrieval Agent",
                role="Information Discovery Specialist",
                system_prompt="""You are an expert information retrieval specialist responsible for finding
the most relevant information to answer user queries. Your expertise includes:

CORE RESPONSIBILITIES:
1. Query Analysis: Break down complex queries into searchable components
2. Search Strategy: Generate multiple search queries to ensure comprehensive coverage
3. Relevance Ranking: Evaluate and rank retrieved information by relevance and quality
4. Source Diversity: Ensure information comes from diverse, reliable sources

SEARCH METHODOLOGY:
- Generate 2-3 related search queries for comprehensive coverage
- Use synonyms and related terms to capture semantic variations
- Consider different perspectives and aspects of the query
- Prioritize recent and authoritative sources when available

QUALITY STANDARDS:
- Only return information that directly addresses the user's query
- Clearly indicate confidence levels for retrieved information
- Flag any potentially outdated or conflicting information
- Provide context about source reliability when relevant

OUTPUT FORMAT:
Organize retrieved information by relevance and provide clear source attribution.""",
                tools=["retrieve"],
                capabilities={
                    "max_searches": 3,              # Maximum parallel searches
                    "relevance_threshold": 0.7,     # Minimum relevance score
                    "source_diversity": True        # Ensure diverse sources
                }
            ),
            
            "refiner": Agent(
                name="Refiner Agent", 
                role="Information Synthesis Specialist",
                system_prompt="""You are an expert information synthesis specialist who transforms
raw retrieved information into coherent, comprehensive responses. Your expertise includes:

CORE RESPONSIBILITIES:
1. Information Integration: Combine information from multiple sources coherently
2. Fact Verification: Cross-reference facts and identify potential conflicts
3. Content Organization: Structure information logically and comprehensively
4. Quality Enhancement: Improve clarity, accuracy, and completeness

SYNTHESIS METHODOLOGY:
- Identify key themes and concepts across retrieved information
- Resolve conflicts between sources using authoritative references
- Fill information gaps by identifying what additional context is needed
- Organize content in logical, easy-to-understand structures

QUALITY STANDARDS:
- Ensure factual accuracy through cross-referencing multiple sources
- Maintain appropriate technical depth for the intended audience
- Clearly distinguish between established facts and interpretations
- Highlight any limitations or uncertainties in the information

OUTPUT FORMAT:
Provide well-structured, comprehensive information with clear organization and source attribution.""",
                tools=["synthesize", "cross_reference"],
                capabilities={
                    "max_sources": 10,              # Maximum sources to synthesize
                    "conflict_resolution": True,    # Handle conflicting information
                    "context_enhancement": True     # Add relevant context
                }
            ),
            
            "response": Agent(
                name="Response Agent",
                role="Expert Response Generator", 
                system_prompt="""You are an expert response generation specialist who creates the final,
polished responses for users. Your expertise includes:

CORE RESPONSIBILITIES:
1. Response Crafting: Generate clear, comprehensive, and engaging responses
2. Audience Adaptation: Tailor language and detail level to user needs
3. Citation Management: Provide proper source attribution and references
4. Quality Assurance: Ensure responses meet high standards of accuracy and clarity

RESPONSE METHODOLOGY:
- Begin with direct answers to the user's specific question
- Provide comprehensive explanations with appropriate detail
- Use clear, professional language appropriate for the context
- Structure responses logically with smooth transitions

QUALITY STANDARDS:
- Ensure complete accuracy based on available information
- Provide balanced perspectives when multiple viewpoints exist
- Clearly indicate any limitations or uncertainties
- Maintain professional yet accessible tone throughout

CITATION PRACTICES:
- Include proper source citations for all factual claims
- Use consistent citation format throughout responses
- Provide enough source information for verification
- Distinguish between direct quotes and paraphrased information

OUTPUT FORMAT:
Generate well-structured, professional responses with comprehensive source citations.""",
                tools=["generate_response", "format_citations"],
                capabilities={
                    "response_length": "adaptive",   # Adjust length based on query complexity
                    "citation_style": "academic",   # Use academic citation standards
                    "tone_adaptation": True         # Adapt tone based on context
                }
            )
        }
        
        logger.info(f"✅ Initialized {len(self.agents)} specialized agents")
    
    def _register_endpoints(self):
        """Register API endpoints for multi-agent team coordination"""
        
        @self.app.post("/query", response_model=Dict[str, Any])
        async def process_query(request: QueryRequest):
            """Process queries through the complete multi-agent RAG pipeline"""
            start_time = asyncio.get_event_loop().time()
            
            try:
                logger.info(f"🔄 Processing query for tenant: {request.tenant_id}")
                logger.info(f"📝 Query: {request.query}")
                
                # Step 1: Query Analysis and Preprocessing
                query_analysis = await self._analyze_query(request.query, request.context)
                logger.info(f"📊 Query analysis: {query_analysis['query_type']}")
                
                # Step 2: Information Retrieval using MCP services
                retrieval_response = await self._execute_agent(
                    "retrieval", 
                    request.query, 
                    request.tenant_id,
                    {
                        "max_results": request.max_results,
                        "session_id": request.session_id,
                        **request.context
                    }
                )
                
                if not retrieval_response.success:
                    logger.error(f"❌ Retrieval failed: {retrieval_response.metadata}")
                    raise HTTPException(status_code=500, detail="Information retrieval failed")
                
                logger.info(f"✅ Retrieved {len(retrieval_response.sources)} relevant sources")
                
                # Step 3: Information Refinement and Synthesis
                refinement_response = await self._execute_agent(
                    "refiner",
                    request.query,
                    request.tenant_id,
                    {
                        "retrieved_info": retrieval_response.response_data,
                        "sources": retrieval_response.sources,
                        "query_analysis": query_analysis,
                        **request.context
                    }
                )
                
                if not refinement_response.success:
                    logger.warning("⚠️ Refinement failed, using raw retrieval results")
                    refinement_response = retrieval_response
                
                # Step 4: Final Response Generation
                final_response = await self._execute_agent(
                    "response",
                    request.query,
                    request.tenant_id,
                    {
                        "refined_info": refinement_response.response_data,
                        "sources": refinement_response.sources,
                        "query_analysis": query_analysis,
                        "include_sources": request.include_sources,
                        **request.context
                    }
                )
                
                if not final_response.success:
                    logger.error(f"❌ Response generation failed: {final_response.metadata}")
                    raise HTTPException(status_code=500, detail="Response generation failed")
                
                # Calculate total processing time
                total_time = asyncio.get_event_loop().time() - start_time
                
                # Compile comprehensive response
                complete_response = {
                    "success": True,
                    "tenant_id": request.tenant_id,
                    "query": request.query,
                    "response": final_response.response_data.get("generated_response", ""),
                    "sources": final_response.sources if request.include_sources else [],
                    "metadata": {
                        "total_processing_time": round(total_time, 3),
                        "query_analysis": query_analysis,
                        "retrieval_time": retrieval_response.processing_time,
                        "refinement_time": refinement_response.processing_time,
                        "response_time": final_response.processing_time,
                        "sources_used": len(final_response.sources),
                        "session_id": request.session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_used": self.model,
                        "pipeline_version": "8.0"
                    }
                }
                
                # Update session context if session ID provided
                if request.session_id:
                    await self._update_session_context(request.session_id, request.query, complete_response)
                
                logger.info(f"🎉 Query processed successfully in {total_time:.2f}s")
                return complete_response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Query processing failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
        
        @self.app.get("/health")
        async def health_check():
            """Comprehensive health check for the multi-agent system"""
            try:
                health_status = {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": "TA_V8 RAG Agent Team",
                    "version": "8.0",
                    "agents": {},
                    "services": {},
                    "sessions": {
                        "active_count": len(self.active_sessions),
                        "max_sessions": 1000
                    }
                }
                
                # Check agent readiness
                for agent_name, agent in self.agents.items():
                    health_status["agents"][agent_name] = {
                        "status": "ready",
                        "role": agent.role,
                        "tools": agent.tools,
                        "capabilities": agent.capabilities or {}
                    }
                
                # Check MCP service connectivity
                services_to_check = [
                    ("chunking_mcp", self.chunking_mcp_url),
                    ("embedding_mcp", self.embedding_mcp_url),
                    ("retrieval_mcp", self.retrieval_mcp_url),
                    ("ollama", self.ollama_url)
                ]
                
                for service_name, service_url in services_to_check:
                    try:
                        response = await self.http_client.get(
                            f"{service_url}/health", 
                            timeout=5.0
                        )
                        health_status["services"][service_name] = {
                            "status": "healthy" if response.status_code == 200 else "unhealthy",
                            "response_time_ms": response.elapsed.total_seconds() * 1000,
                            "url": service_url
                        }
                    except Exception as e:
                        health_status["services"][service_name] = {
                            "status": "unhealthy",
                            "error": str(e),
                            "url": service_url
                        }
                        health_status["status"] = "degraded"
                
                return health_status
                
            except Exception as e:
                logger.error(f"❌ Health check failed: {str(e)}")
                return {
                    "status": "unhealthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
    
    async def _analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query to determine optimal processing strategy"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Determine query type based on patterns
        if any(word in query_lower for word in ["what", "who", "when", "where", "which"]):
            query_type = "factual"
        elif any(word in query_lower for word in ["how", "why", "explain", "describe"]):
            query_type = "analytical" 
        elif any(word in query_lower for word in ["compare", "contrast", "difference", "versus"]):
            query_type = "comparative"
        elif "?" not in query and word_count > 10:
            query_type = "open_ended"
        else:
            query_type = "general"
        
        # Assess query complexity
        complexity = "simple"
        if word_count > 20 or len(query) > 200:
            complexity = "complex"
        elif word_count > 10 or len(query) > 100:
            complexity = "medium"
        
        return {
            "query_type": query_type,
            "complexity": complexity,
            "word_count": word_count,
            "character_count": len(query),
            "has_question_mark": "?" in query,
            "context_provided": bool(context),
            "processing_strategy": f"{query_type}_{complexity}",
            "estimated_sources_needed": min(10, max(3, word_count // 5))
        }
    
    async def _execute_agent(self, agent_name: str, query: str, tenant_id: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute a specific agent with comprehensive error handling and monitoring"""
        start_time = asyncio.get_event_loop().time()
        
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        logger.info(f"🤖 Executing {agent.name} for tenant: {tenant_id}")
        
        try:
            agent_context = {
                "query": query,
                "tenant_id": tenant_id,
                "agent_capabilities": agent.capabilities or {},
                **context
            }
            
            # Execute agent based on its specialization
            if agent_name == "retrieval":
                result = await self._execute_retrieval_agent(agent, agent_context)
            elif agent_name == "refiner":
                result = await self._execute_refiner_agent(agent, agent_context)
            elif agent_name == "response":
                result = await self._execute_response_agent(agent, agent_context)
            else:
                raise ValueError(f"No execution handler for agent: {agent_name}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"✅ {agent.name} completed in {processing_time:.2f}s")
            
            return AgentResponse(
                success=True,
                agent_name=agent.name,
                response_data=result,
                metadata={
                    "agent_type": agent_name,
                    "processing_time": processing_time,
                    "context_size": len(str(context)),
                    "tenant_id": tenant_id
                },
                sources=result.get("sources", []),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ {agent.name} failed after {processing_time:.2f}s: {str(e)}")
            
            return AgentResponse(
                success=False,
                agent_name=agent.name,
                response_data={"error": str(e)},
                metadata={
                    "agent_type": agent_name,
                    "processing_time": processing_time,
                    "error": str(e),
                    "tenant_id": tenant_id
                },
                sources=[],
                processing_time=processing_time
            )
    
    async def _execute_retrieval_agent(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieval agent using MCP retrieval service"""
        query = context["query"]
        tenant_id = context["tenant_id"]
        max_results = context.get("max_results", 5)
        
        try:
            # Call the retrieval MCP service
            response = await self.http_client.post(
                f"{self.retrieval_mcp_url}/mcp/execute",
                json={
                    "method": "retrieve",
                    "params": {
                        "query": query,
                        "tenant_id": tenant_id,
                        "top_k": max_results
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            
            retrieval_data = response.json()
            if retrieval_data.get("success") and retrieval_data.get("result"):
                results = retrieval_data["result"].get("results", [])
                sources = []
                
                for result in results:
                    source = {
                        "id": result.get("id", ""),
                        "text": result.get("text", ""),
                        "score": result.get("score", 0),
                        "metadata": result.get("metadata", {}),
                        "retrieval_timestamp": datetime.utcnow().isoformat()
                    }
                    sources.append(source)
                
                return {
                    "retrieved_info": [source["text"] for source in sources],
                    "sources": sources,
                    "total_results_found": len(results),
                    "retrieval_method": "mcp_service"
                }
            else:
                logger.warning("No results from retrieval MCP service")
                return {
                    "retrieved_info": [],
                    "sources": [],
                    "total_results_found": 0,
                    "retrieval_method": "mcp_service",
                    "warning": "No results found"
                }
                
        except Exception as e:
            logger.error(f"❌ Retrieval MCP service failed: {str(e)}")
            return {
                "retrieved_info": [],
                "sources": [],
                "total_results_found": 0,
                "error": str(e)
            }
    
    async def _execute_refiner_agent(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute refiner agent for information synthesis"""
        retrieved_info = context.get("retrieved_info", [])
        sources = context.get("sources", [])
        query = context["query"]
        
        if not retrieved_info:
            return {
                "refined_info": "",
                "sources": sources,
                "refinement_applied": False
            }
        
        # Combine information from multiple sources
        combined_text = "\n\n".join(retrieved_info)
        
        # Use LLM to synthesize and refine information
        refinement_prompt = f"""
{agent.system_prompt}

Original Query: {query}

Retrieved Information:
{combined_text}

Your task is to synthesize this information into a coherent, comprehensive response that:
1. Directly addresses the user's query
2. Integrates information from multiple sources
3. Resolves any conflicts between sources
4. Adds relevant context where helpful
5. Maintains accuracy and proper attribution

Provide a well-structured synthesis that enhances the retrieved information.
"""
        
        try:
            refined_response = await self._generate_llm_response(refinement_prompt)
            
            return {
                "refined_info": refined_response,
                "sources": sources,
                "original_sources_count": len(sources),
                "refinement_applied": True,
                "synthesis_method": "llm_enhanced"
            }
            
        except Exception as e:
            logger.warning(f"⚠️ LLM refinement failed, using simple combination: {str(e)}")
            
            return {
                "refined_info": combined_text,
                "sources": sources,
                "refinement_applied": False,
                "fallback_used": True,
                "error": str(e)
            }
    
    async def _execute_response_agent(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute response agent for final response generation"""
        refined_info = context.get("refined_info", "")
        sources = context.get("sources", [])
        query = context["query"]
        include_sources = context.get("include_sources", True)
        
        response_prompt = f"""
{agent.system_prompt}

User Query: {query}

Information to base your response on:
{refined_info}

Sources Available: {len(sources)} sources

Your task is to generate a comprehensive, professional response that:
1. Directly and completely answers the user's question
2. Uses clear, accessible language appropriate for the context
3. Provides sufficient detail without being overwhelming
4. Includes proper citations if sources should be included: {include_sources}
5. Maintains accuracy based on the provided information

Generate a well-structured, informative response.
"""
        
        try:
            final_response = await self._generate_llm_response(response_prompt)
            
            # Format sources for citation if requested
            formatted_sources = []
            if include_sources and sources:
                for i, source in enumerate(sources, 1):
                    formatted_source = {
                        "citation_id": i,
                        "text_snippet": source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"],
                        "similarity_score": source.get("score", 0),
                        "source_id": source.get("id", ""),
                        "metadata": source.get("metadata", {})
                    }
                    formatted_sources.append(formatted_source)
            
            return {
                "generated_response": final_response,
                "sources": formatted_sources,
                "response_length": len(final_response),
                "sources_cited": len(formatted_sources),
                "generation_method": "llm_generated"
            }
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {str(e)}")
            
            fallback_response = f"""Based on the available information, here's what I found regarding your query: {query}

{refined_info[:1000]}...

Note: This is a simplified response due to processing limitations. Please try your query again for a more comprehensive answer."""
            
            return {
                "generated_response": fallback_response,
                "sources": sources if include_sources else [],
                "fallback_used": True,
                "error": str(e)
            }
    
    async def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using Ollama LLM service"""
        try:
            ollama_request = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 0.9,
                    "stop": ["\\n\\nUser:", "\\n\\nHuman:"]
                }
            }
            
            response = await self.http_client.post(
                f"{self.ollama_url}/api/generate",
                json=ollama_request,
                timeout=60.0
            )
            response.raise_for_status()
            
            response_data = response.json()
            generated_text = response_data.get("response", "")
            
            if not generated_text:
                raise Exception("Empty response from LLM")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {str(e)}")
            raise
    
    async def _update_session_context(self, session_id: str, query: str, response: Dict[str, Any]):
        """Update session context for conversational interactions"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "query_count": 0,
                "conversation_history": []
            }
        
        session = self.active_sessions[session_id]
        session["last_activity"] = datetime.utcnow().isoformat()
        session["query_count"] += 1
        
        # Add to conversation history (keep last 10 exchanges)
        session["conversation_history"].append({
            "query": query,
            "response": response["response"],
            "timestamp": datetime.utcnow().isoformat(),
            "sources_count": len(response.get("sources", []))
        })
        
        # Keep only recent conversation history
        if len(session["conversation_history"]) > 10:
            session["conversation_history"] = session["conversation_history"][-10:]
        
        logger.info(f"💬 Updated session {session_id}: {session['query_count']} queries")

# Create global agent team instance
agent_team = RAGAgentTeam()
app = agent_team.app

# Main entry point for direct script execution
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting TA_V8 RAG Agent Team in standalone mode...")
    
    uvicorn.run(
        "rag_agent_team:app",
        host="0.0.0.0",
        port=8006,                         # Agent team coordination port
        workers=1,
        log_level="info",
        access_log=True,
        reload=False,
        server_header=False,
        date_header=True
    )