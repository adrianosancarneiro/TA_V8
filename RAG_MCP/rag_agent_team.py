#!/usr/bin/env python3
"""
# =============================================================================
# TA_V8 RAG AGENT TEAM - CRITIC-DRIVEN MULTI-AGENT SYSTEM
# =============================================================================
# Purpose: Orchestrate multiple specialized agents for iterative retrieval-refinement
#          with critic-driven quality control using LangGraph and AutoGen
#
# Architecture: Critic-Driven Multi-Agent System with iterative improvement:
# - Retriever Agent: Semantic search using MCP retrieval service
# - Critic Agent: Evaluates retrieved information sufficiency and quality
# - Refiner Agent: Suggests improvements and crafts final responses
#
# Flow Control: LangGraph manages conversation flow with round limits
# Agent Communication: Microsoft AutoGen for internal agent conversations
# Integration: Uses MCP microservices for chunking, embedding, and retrieval
# =============================================================================
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

# LangGraph and AutoGen imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    logging.warning(f"LangGraph and/or AutoGen not available: {e}. Install with: pip install langgraph autogen")

# Configure logging for comprehensive monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST/RESPONSE MODELS AND STATE MANAGEMENT
# ============================================================================

class QueryRequest(BaseModel):
    """Comprehensive query request model for critic-driven multi-agent processing"""
    query: str = Field(..., description="Natural language query from user")
    tenant_id: str = Field(..., description="Tenant identifier for secure data isolation")
    session_id: Optional[str] = Field(None, description="Session ID for conversational context")
    max_results: int = Field(5, description="Maximum number of results to retrieve")
    max_rounds: int = Field(3, description="Maximum rounds of critic-retriever iterations")
    include_sources: bool = Field(True, description="Whether to include source citations")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context and metadata")

class AgentRole(Enum):
    """Agent roles in the critic-driven system"""
    RETRIEVER = "retriever"
    CRITIC = "critic"
    REFINER = "refiner"

class ConversationState(BaseModel):
    """State management for LangGraph conversation flow"""
    query: str
    tenant_id: str
    current_round: int = 0
    max_rounds: int = 3
    retrieved_info: List[Dict[str, Any]] = []
    critic_feedback: List[str] = []
    refinement_suggestions: List[str] = []
    final_answer: Optional[str] = None
    is_sufficient: bool = False
    conversation_history: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

@dataclass
class Agent:
    """Agent configuration with role, capabilities, and tools"""
    name: str
    role: AgentRole
    system_prompt: str
    tools: List[str]
    capabilities: Optional[Dict[str, Any]] = None

@dataclass
class AgentResponse:
    """Comprehensive response from individual agent execution"""
    success: bool
    agent_name: str
    agent_role: AgentRole
    response_data: Dict[str, Any]
    metadata: Dict[str, Any]
    sources: List[Dict[str, Any]]
    processing_time: float

# ============================================================================
# MULTI-AGENT ORCHESTRATION CLASS
# ============================================================================

class RAGAgentTeam:
    """
    Critic-Driven Multi-Agent RAG System Orchestrator
    
    Implements iterative retrieval-refinement with critic evaluation:
    - Retriever Agent: Semantic search via MCP retrieval service
    - Critic Agent: Evaluates information sufficiency and quality  
    - Refiner Agent: Suggests improvements and crafts final responses
    
    Features:
    - LangGraph-controlled conversation flow with round limits
    - AutoGen-powered internal agent conversations
    - Iterative improvement until critic approval or max rounds
    - Integration with MCP microservices
    - Production-ready FastAPI endpoints
    """
    
    def __init__(self):
        """Initialize the critic-driven multi-agent RAG system"""
        logger.info("🤖 Initializing TA_V8 Critic-Driven RAG Agent Team...")
        
        # Dependency availability
        if not LANGGRAPH_AVAILABLE:
            logger.warning("⚠️ LangGraph/AutoGen not available - using simplified flow")
        self.langgraph_workflow = None
        self.autogen_agents = None

        # Service configuration - check for stdio vs HTTP transport
        self.mcp_transport = os.getenv("MCP_TRANSPORT", "http")
        
        if self.mcp_transport == "stdio":
            # MCP stdio service names for systemd services
            self.chunking_mcp_service = os.getenv("CHUNKING_MCP_SERVICE", "chunking-mcp")
            self.embedding_mcp_service = os.getenv("EMBEDDING_MCP_SERVICE", "embedding-mcp") 
            self.retrieval_mcp_service = os.getenv("RETRIEVAL_MCP_SERVICE", "retrieval-mcp")
            # Initialize MCP stdio clients
            self.mcp_clients = {}
        else:
            # HTTP service endpoints (updated ports for HTTP + SSE)
            self.chunking_mcp_url = os.getenv("CHUNKING_MCP_URL", "http://localhost:8001")
            self.embedding_mcp_url = os.getenv("EMBEDDING_MCP_URL", "http://localhost:8004")
            self.retrieval_mcp_url = os.getenv("RETRIEVAL_MCP_URL", "http://localhost:8003")
            
        self.vllm_url = os.getenv("VLLM_URL", "http://localhost:8000")

        # LLM configuration for vLLM
        self.model = os.getenv("DEFAULT_LLM", "openai/gpt-oss-20b")

        # HTTP client configuration
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # Initialize FastAPI application
        self.app = FastAPI(
            title="TA_V8 Critic-Driven RAG Agent Team",
            description="Multi-agent system with critic-driven iterative retrieval and refinement",
            version="8.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize specialized agents
        self._initialize_agents()
        
        # Initialize LangGraph workflow if available
        if LANGGRAPH_AVAILABLE:
            self._initialize_langgraph_workflow()
        
        # Initialize AutoGen agents if available
        if LANGGRAPH_AVAILABLE:
            self._initialize_autogen_agents()
        
        # Register API endpoints
        self._register_endpoints()
        
        # Session management
        self.active_sessions = {}
        
        logger.info("✅ Critic-Driven RAG Agent Team initialization completed")
    
    async def initialize(self):
        """Initialize the RAG Agent Team async components"""
        logger.info("🚀 Initializing RAG Agent Team async components")
        
        # Test MCP service connections
        await self._test_mcp_connections()
        
        logger.info("✅ RAG Agent Team async initialization completed")
    
    async def _test_mcp_connections(self):
        """Test connections to MCP services based on transport mode"""
        if self.mcp_transport == "stdio":
            # Test systemd services for stdio MCP transport
            import subprocess
            services = [
                ("Retrieval MCP", self.retrieval_mcp_service),
                ("Embedding MCP", self.embedding_mcp_service), 
                ("Chunking MCP", self.chunking_mcp_service)
            ]
            
            for service_name, systemd_service in services:
                try:
                    result = subprocess.run(
                        ["systemctl", "is-active", systemd_service],
                        capture_output=True, text=True, timeout=5
                    )
                    
                    if result.returncode == 0 and result.stdout.strip() == "active":
                        logger.info(f"✅ {service_name} systemd service is active")
                    else:
                        logger.warning(f"⚠️ {service_name} systemd service is not active: {result.stdout.strip()}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ {service_name} service connection failed: {str(e)}")
        else:
            # HTTP transport (legacy mode)
            services = [
                ("Retrieval MCP", self.retrieval_mcp_url),
                ("Embedding MCP", self.embedding_mcp_url), 
                ("Chunking MCP", self.chunking_mcp_url)
            ]
            
            for service_name, url in services:
                try:
                    response = await self.http_client.get(
                        f"{url}/health",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        logger.info(f"✅ {service_name} service healthy")
                    else:
                        logger.warning(f"⚠️ {service_name} service returned {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ {service_name} service connection failed: {str(e)}")
    
    def _initialize_agents(self):
        """Initialize specialized agents for critic-driven RAG system"""
        logger.info("🔧 Initializing critic-driven RAG agents...")
        
        self.agents = {
            "retriever": Agent(
                name="Retriever Agent",
                role=AgentRole.RETRIEVER,
                system_prompt="""You are an expert information retrieval specialist. Your primary responsibility is to use semantic search to find relevant information from the vector database (Qdrant) using the MCP retrieval service.

CORE RESPONSIBILITIES:
1. Query Processing: Analyze user questions to understand information needs
2. Semantic Search: Use the MCP retrieval service to search embedded chunks in Qdrant
3. Metadata Utilization: Leverage chunk metadata to enhance search accuracy
4. Result Collection: Gather comprehensive information from vector database

SEARCH METHODOLOGY:
- Use the exact user query for initial semantic search
- Utilize vector embeddings to find semantically similar content
- Consider metadata filters when available (date, source, category, etc.)
- Retrieve chunks with highest similarity scores

TOOL USAGE:
- Primary tool: MCP Retrieval Service (http://localhost:8003)
- Search vector database using semantic similarity
- Include metadata in search considerations
- Return ranked results with similarity scores

OUTPUT FORMAT:
Return retrieved information with source attribution, similarity scores, and metadata.""",
                tools=["mcp_retrieval_service"],
                capabilities={
                    "max_results": 10,
                    "similarity_threshold": 0.7,
                    "metadata_aware": True,
                    "semantic_search": True
                }
            ),
            
            "critic": Agent(
                name="Critic Agent", 
                role=AgentRole.CRITIC,
                system_prompt="""You are an expert information quality critic. Your role is to evaluate whether retrieved information sufficiently answers the user's question and make critical decisions about information adequacy.

CORE RESPONSIBILITIES:
1. Sufficiency Analysis: Determine if retrieved information adequately answers the user's question
2. Quality Assessment: Evaluate completeness, accuracy, and relevance of information
3. Decision Making: Decide if information is sufficient or if more retrieval is needed
4. Feedback Generation: Provide specific feedback on what's missing or inadequate

EVALUATION CRITERIA:
- Completeness: Does the information fully address all aspects of the question?
- Relevance: Is the information directly related to the user's query?
- Accuracy: Is the information factually correct and up-to-date?
- Clarity: Is the information clear and understandable?

DECISION PROCESS:
- SUFFICIENT: Information completely answers the question -> approve for final response
- INSUFFICIENT: Information gaps exist -> provide specific feedback for improvement
- UNCLEAR: Information is ambiguous -> request clarification or better sources

OUTPUT FORMAT:
Provide clear decision (SUFFICIENT/INSUFFICIENT/UNCLEAR) with detailed reasoning and specific improvement suggestions.""",
                tools=["quality_assessment", "decision_making"],
                capabilities={
                    "quality_threshold": 0.8,
                    "completeness_check": True,
                    "relevance_scoring": True,
                    "feedback_generation": True
                }
            ),
            
            "refiner": Agent(
                name="Refiner Agent",
                role=AgentRole.REFINER,
                system_prompt="""You are an expert query refinement and response crafting specialist. You have two primary modes of operation based on critic feedback.

MODE 1 - REFINEMENT (when critic finds information insufficient):
1. Query Improvement: Suggest alternative vector database queries
2. Search Strategy: Recommend different search paths and approaches  
3. LLM-Powered Rephrasing: Use language models to generate new query phrasings
4. User Interaction: Ask users for additional information when needed

REFINEMENT STRATEGIES:
- Synonym expansion: Suggest queries with related terms
- Perspective shifts: Try different angles or viewpoints
- Specificity adjustment: Make queries more specific or more general
- Context enhancement: Add relevant context to improve search

MODE 2 - RESPONSE CRAFTING (when critic approves information):
1. Final Answer Creation: Craft comprehensive, well-structured responses
2. Source Integration: Seamlessly integrate information from multiple sources
3. Citation Management: Provide proper source attribution
4. Quality Assurance: Ensure response completeness and accuracy

OUTPUT FORMAT:
- For refinement: Provide specific alternative queries and improvement strategies
- For final response: Deliver polished, comprehensive answer with proper citations""",
                tools=["query_refinement", "llm_rephrasing", "response_crafting", "citation_management"],
                capabilities={
                    "query_generation": True,
                    "llm_integration": True,
                    "response_crafting": True,
                    "citation_formatting": True,
                    "user_interaction": True
                }
            )
        }
        
        logger.info(f"✅ Initialized {len(self.agents)} specialized agents: {list(self.agents.keys())}")
    
    def _initialize_langgraph_workflow(self):
        """Initialize LangGraph workflow for conversation flow control"""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available - using simplified workflow")
            return
        
        try:
            # Create state graph for conversation flow
            workflow = StateGraph(ConversationState)
            
            # Add nodes for each agent and decision points
            workflow.add_node("retrieve", self._langgraph_retrieve_node)
            workflow.add_node("critique", self._langgraph_critique_node)
            workflow.add_node("refine", self._langgraph_refine_node)
            workflow.add_node("final_response", self._langgraph_final_response_node)
            
            # Define conversation flow edges
            workflow.set_entry_point("retrieve")
            
            # From retrieve -> critique
            workflow.add_edge("retrieve", "critique")
            
            # From critique -> conditional routing
            workflow.add_conditional_edges(
                "critique",
                self._should_continue_retrieval,
                {
                    "continue": "refine",      # Need more information
                    "finish": "final_response" # Information is sufficient
                }
            )
            
            # From refine -> retrieve (iterative improvement)
            workflow.add_edge("refine", "retrieve")
            
            # Final response -> end
            workflow.add_edge("final_response", END)
            
            # Compile workflow
            self.langgraph_workflow = workflow.compile(
                checkpointer=MemorySaver()
            )
            
            logger.info("✅ LangGraph workflow initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangGraph workflow: {str(e)}")
            self.langgraph_workflow = None
    
    def _initialize_autogen_agents(self):
        """Initialize AutoGen agents for internal agent conversations"""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("AutoGen not available - using direct agent calls")
            return
        
        try:
            # Configure LLM for AutoGen agents
            llm_config = {
                "config_list": [{
                    "model": self.model,
                    "base_url": f"{self.vllm_url}/v1",
                    "api_key": "vllm",  # vLLM doesn't need real API key for local deployment
                    "api_type": "openai"
                }],
                "temperature": 0.7,
                "timeout": 60
            }
            
            # Create AutoGen agents for internal conversations
            self.autogen_agents = {
                "retriever_agent": ConversableAgent(
                    name="RetrieverAgent",
                    system_message=self.agents["retriever"].system_prompt,
                    llm_config=llm_config,
                    human_input_mode="NEVER"
                ),
                
                "critic_agent": ConversableAgent(
                    name="CriticAgent", 
                    system_message=self.agents["critic"].system_prompt,
                    llm_config=llm_config,
                    human_input_mode="NEVER"
                ),
                
                "refiner_agent": ConversableAgent(
                    name="RefinerAgent",
                    system_message=self.agents["refiner"].system_prompt,
                    llm_config=llm_config,
                    human_input_mode="NEVER"
                )
            }
            
            logger.info("✅ AutoGen agents initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AutoGen agents: {str(e)}")
            self.autogen_agents = None
    
    def _register_endpoints(self):
        """Register API endpoints for multi-agent team coordination"""
        
        @self.app.post("/query", response_model=Dict[str, Any])
        async def process_query(request: QueryRequest):
            """Process queries through the critic-driven multi-agent RAG pipeline"""
            start_time = asyncio.get_event_loop().time()
            
            try:
                logger.info(f"🔄 Processing query for tenant: {request.tenant_id}")
                logger.info(f"📝 Query: {request.query}")
                
                # Initialize conversation state
                initial_state = ConversationState(
                    query=request.query,
                    tenant_id=request.tenant_id,
                    max_rounds=request.max_rounds,
                    metadata={
                        "session_id": request.session_id,
                        "max_results": request.max_results,
                        "include_sources": request.include_sources,
                        **request.context
                    }
                )
                
                # Process through LangGraph workflow if available
                if LANGGRAPH_AVAILABLE and self.langgraph_workflow:
                    final_state = await self._process_with_langgraph(initial_state)
                else:
                    # Fallback to simplified processing
                    final_state = await self._process_without_langgraph(initial_state)
                
                # Calculate total processing time
                total_time = asyncio.get_event_loop().time() - start_time
                
                # Compile comprehensive response
                complete_response = {
                    "success": True,
                    "tenant_id": request.tenant_id,
                    "query": request.query,
                    "response": final_state.final_answer or "Unable to generate response",
                    "sources": self._extract_sources_from_state(final_state),
                    "metadata": {
                        "total_processing_time": round(total_time, 3),
                        "rounds_completed": final_state.current_round,
                        "max_rounds": final_state.max_rounds,
                        "information_sufficient": final_state.is_sufficient,
                        "critic_feedback_count": len(final_state.critic_feedback),
                        "refinement_suggestions": len(final_state.refinement_suggestions),
                        "session_id": request.session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_used": self.model,
                        "pipeline_version": "8.0_critic_driven"
                    }
                }
                
                # Update session context
                if request.session_id:
                    await self._update_session_context(request.session_id, request.query, complete_response)
                
                logger.info(f"🎉 Query processed in {total_time:.2f}s with {final_state.current_round} rounds")
                return complete_response
                
            except Exception as e:
                logger.error(f"❌ Query processing failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
        
        @self.app.get("/health")
        async def health_check_route():
            """FastAPI health endpoint"""
            return await self.health_check()
    
    async def health_check(self) -> Dict[str, Any]:
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
                },
                "capabilities": {
                    "langgraph": LANGGRAPH_AVAILABLE,
                    "autogen": LANGGRAPH_AVAILABLE and self.autogen_agents is not None
                }
            }

            # Agent readiness
            for agent_name, agent in self.agents.items():
                health_status["agents"][agent_name] = {
                    "status": "ready",
                    "role": agent.role,
                    "tools": agent.tools,
                    "capabilities": agent.capabilities or {}
                }

            # Service connectivity based on transport mode
            if self.mcp_transport == "stdio":
                # Check systemd services status for stdio MCP services
                import subprocess
                services_to_check = [
                    ("chunking_mcp", self.chunking_mcp_service),
                    ("embedding_mcp", self.embedding_mcp_service),
                    ("retrieval_mcp", self.retrieval_mcp_service),
                ]
                
                for service_name, service_systemd_name in services_to_check:
                    try:
                        # Check if systemd service is active
                        result = subprocess.run(
                            ["systemctl", "is-active", service_systemd_name],
                            capture_output=True, text=True, timeout=5
                        )
                        
                        if result.returncode == 0 and result.stdout.strip() == "active":
                            health_status["services"][service_name] = {
                                "status": "healthy",
                                "transport": "stdio",
                                "systemd_service": service_systemd_name,
                                "response_time": None
                            }
                        else:
                            health_status["services"][service_name] = {
                                "status": "unhealthy",
                                "transport": "stdio", 
                                "systemd_service": service_systemd_name,
                                "error": f"Service not active: {result.stdout.strip()}"
                            }
                            healthy_services += 1
                            
                    except Exception as e:
                        health_status["services"][service_name] = {
                            "status": "error",
                            "transport": "stdio",
                            "systemd_service": service_systemd_name,
                            "error": str(e)
                        }
                        
                # Always check vLLM via HTTP
                try:
                    response = await self.http_client.get(f"{self.vllm_url}/v1/models", timeout=5.0)
                    response_time = (response.elapsed.total_seconds() * 1000 if response.elapsed else None)
                    
                    health_status["services"]["vllm"] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "transport": "http",
                        "url": self.vllm_url,
                        "response_time": response_time,
                        "status_code": response.status_code
                    }
                    
                    if response.status_code == 200:
                        healthy_services += 1
                        
                except Exception as e:
                    health_status["services"]["vllm"] = {
                        "status": "error",
                        "transport": "http", 
                        "url": self.vllm_url,
                        "error": str(e)
                    }
                    
            else:
                # HTTP transport mode (legacy)
                services_to_check = [
                    ("chunking_mcp", self.chunking_mcp_url),
                    ("embedding_mcp", self.embedding_mcp_url),
                    ("retrieval_mcp", self.retrieval_mcp_url),
                    ("vllm", self.vllm_url)
                ]

                for service_name, service_url in services_to_check:
                    try:
                        # Use different endpoint for vLLM health check
                        if service_name == "vllm":
                            health_endpoint = f"{service_url}/v1/models"  # vLLM models endpoint
                        else:
                            health_endpoint = f"{service_url}/health"
                        
                        response = await self.http_client.get(
                            health_endpoint,
                            timeout=5.0
                        )
                        response_time = (
                            response.elapsed.total_seconds() * 1000
                            if response.elapsed is not None else None
                        )

                        health_status["services"][service_name] = {
                            "status": "healthy" if response.status_code == 200 else "unhealthy",
                            "transport": "http",
                            "response_time_ms": response_time,
                            "url": service_url
                        }

                        if response.status_code == 200:
                            healthy_services += 1

                    except Exception as exc:
                        health_status["services"][service_name] = {
                            "status": "unhealthy",
                            "transport": "http",
                            "error": str(exc),
                            "url": service_url
                        }

            return health_status

        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    # ============================================================================
    # LANGGRAPH WORKFLOW METHODS
    # ============================================================================
    
    async def _process_with_langgraph(self, initial_state: ConversationState) -> ConversationState:
        """Process query using LangGraph workflow"""
        try:
            config = {"configurable": {"thread_id": f"conversation_{id(initial_state)}"}}
            
            # Run the workflow
            final_state = None
            async for state in self.langgraph_workflow.astream(initial_state, config):
                final_state = state
                
            return final_state or initial_state
            
        except Exception as e:
            logger.error(f"❌ LangGraph processing failed: {str(e)}")
            return await self._process_without_langgraph(initial_state)
    
    async def _process_without_langgraph(self, state: ConversationState) -> ConversationState:
        """Simplified processing without LangGraph"""
        logger.info("🔄 Using simplified critic-driven processing")
        
        for round_num in range(state.max_rounds):
            state.current_round = round_num + 1
            logger.info(f"🔄 Starting round {state.current_round}/{state.max_rounds}")
            
            # Step 1: Retrieve information
            retrieval_result = await self._execute_retriever_agent(state)
            if retrieval_result:
                state.retrieved_info.extend(retrieval_result.get("sources", []))
            
            # Step 2: Critic evaluation
            critic_result = await self._execute_critic_agent(state)
            if critic_result:
                is_sufficient = critic_result.get("is_sufficient", False)
                feedback = critic_result.get("feedback", "")
                
                if feedback:
                    state.critic_feedback.append(feedback)
                
                if is_sufficient:
                    state.is_sufficient = True
                    # Step 3: Generate final response
                    refiner_result = await self._execute_refiner_agent(state, mode="response")
                    if refiner_result:
                        state.final_answer = refiner_result.get("final_answer", "")
                    break
                else:
                    # Step 3: Get refinement suggestions
                    refiner_result = await self._execute_refiner_agent(state, mode="refinement")
                    if refiner_result:
                        suggestions = refiner_result.get("suggestions", [])
                        state.refinement_suggestions.extend(suggestions)
                        # Update query for next round
                        if suggestions:
                            state.query = suggestions[0]  # Use first suggestion
        
        return state
    
    async def _langgraph_retrieve_node(self, state: ConversationState) -> ConversationState:
        """LangGraph node for retrieval step"""
        logger.info(f"🔍 LangGraph Retrieval - Round {state.current_round}")
        
        result = await self._execute_retriever_agent(state)
        if result:
            state.retrieved_info.extend(result.get("sources", []))
        
        return state
    
    async def _langgraph_critique_node(self, state: ConversationState) -> ConversationState:
        """LangGraph node for critique step"""
        logger.info(f"⚖️ LangGraph Critique - Round {state.current_round}")
        
        result = await self._execute_critic_agent(state)
        if result:
            state.is_sufficient = result.get("is_sufficient", False)
            feedback = result.get("feedback", "")
            if feedback:
                state.critic_feedback.append(feedback)
        
        return state
    
    async def _langgraph_refine_node(self, state: ConversationState) -> ConversationState:
        """LangGraph node for refinement step"""
        logger.info(f"🔧 LangGraph Refinement - Round {state.current_round}")
        
        result = await self._execute_refiner_agent(state, mode="refinement")
        if result:
            suggestions = result.get("suggestions", [])
            state.refinement_suggestions.extend(suggestions)
            if suggestions:
                state.query = suggestions[0]  # Update query for next iteration
        
        state.current_round += 1
        return state
    
    async def _langgraph_final_response_node(self, state: ConversationState) -> ConversationState:
        """LangGraph node for final response generation"""
        logger.info("📝 LangGraph Final Response Generation")
        
        result = await self._execute_refiner_agent(state, mode="response")
        if result:
            state.final_answer = result.get("final_answer", "")
        
        return state
    
    def _should_continue_retrieval(self, state: ConversationState) -> str:
        """Decision function for LangGraph conditional routing"""
        # Check if we've reached max rounds
        if state.current_round >= state.max_rounds:
            logger.info(f"🛑 Max rounds ({state.max_rounds}) reached - finishing")
            return "finish"
        
        # Check if critic approved the information
        if state.is_sufficient:
            logger.info("✅ Critic approved information - finishing")
            return "finish"
        
        # Continue with refinement
        logger.info("🔄 Information insufficient - continuing refinement")
        return "continue"
    
    # ============================================================================
    # AGENT EXECUTION METHODS
    # ============================================================================
    
    async def _execute_retriever_agent(self, state: ConversationState) -> Optional[Dict[str, Any]]:
        """Execute retriever agent via MCP service"""
        try:
            retrieval_request = {
                "tenant_id": state.tenant_id,
                "collection": "knowledge_base",
                "query": {
                    "text": state.query,
                    "use_embedding": True
                },
                "top_k": 10,
                "filters": {}
            }
            
            logger.info(f"🔍 Retriever executing: '{state.query}'")
            
            response = await self.http_client.post(
                f"{self.retrieval_mcp_url}/mcp/execute",
                json=retrieval_request,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                hits = result.get("hits", [])
                sources = []
                
                for hit in hits:
                    source = {
                        "id": hit.get("id", ""),
                        "text": hit.get("text", ""),
                        "score": hit.get("score", 0),
                        "metadata": hit.get("metadata", {}),
                        "retrieval_timestamp": datetime.utcnow().isoformat()
                    }
                    sources.append(source)
                
                logger.info(f"✅ Retrieved {len(sources)} sources")
                return {"sources": sources}
            else:
                logger.error(f"❌ Retrieval failed: {response.status_code}")
                return None
                        
        except Exception as e:
            logger.error(f"❌ Retriever agent error: {str(e)}")
            return None
    
    async def _execute_critic_agent(self, state: ConversationState) -> Optional[Dict[str, Any]]:
        """Execute critic agent evaluation"""
        try:
            # Prepare evaluation context
            context = {
                "query": state.query,
                "retrieved_info": state.retrieved_info,
                "previous_feedback": state.critic_feedback,
                "current_round": state.current_round,
                "max_rounds": state.max_rounds
            }
            
            system_message = """You are a Critic Agent in a RAG system. Your job is to evaluate whether the retrieved information is sufficient to answer the user's query comprehensively.

Evaluation Criteria:
- Relevance: Does the information directly address the query?
- Completeness: Are all aspects of the query covered?
- Quality: Is the information accurate and detailed enough?
- Context: Is there enough context for a comprehensive answer?

Return JSON format:
{
    "is_sufficient": boolean,
    "feedback": "string explaining your evaluation",
    "missing_aspects": ["list of missing information if insufficient"]
}"""
            
            user_message = f"""Evaluate this retrieval for sufficiency:
Query: {context['query']}
Retrieved Information: {json.dumps(context['retrieved_info'][:5], indent=2)}  # Limit for token efficiency
Previous Feedback: {context['previous_feedback']}
Round: {context['current_round']}/{context['max_rounds']}

Provide your evaluation."""
            
            logger.info(f"⚖️ Critic evaluating round {state.current_round}")
            
            # Use vLLM for evaluation
            vllm_request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.3,  # Lower temperature for consistent evaluation
                "max_tokens": 500
            }

            response = await self.http_client.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=vllm_request,
                timeout=60.0
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                try:
                    evaluation = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    is_sufficient = "sufficient" in content.lower() or "good" in content.lower()
                    evaluation = {
                        "is_sufficient": is_sufficient,
                        "feedback": content
                    }
                
                is_sufficient = evaluation.get("is_sufficient", False)
                feedback = evaluation.get("feedback", "")
                
                logger.info(f"✅ Critic evaluation: {'Sufficient' if is_sufficient else 'Insufficient'}")
                return evaluation
            else:
                logger.error(f"❌ Critic evaluation failed: {response.status_code}")
                return {"is_sufficient": False, "feedback": "Evaluation failed"}
                        
        except Exception as e:
            logger.error(f"❌ Critic agent error: {str(e)}")
            return {"is_sufficient": False, "feedback": f"Evaluation error: {str(e)}"}
    
    async def _execute_refiner_agent(self, state: ConversationState, mode: str) -> Optional[Dict[str, Any]]:
        """Execute refiner agent in refinement or response mode"""
        try:
            context = {
                "query": state.query,
                "retrieved_info": state.retrieved_info,
                "critic_feedback": state.critic_feedback,
                "refinement_suggestions": state.refinement_suggestions,
                "mode": mode
            }
            
            if mode == "refinement":
                system_message = """You are a Refiner Agent in refinement mode. Based on the critic's feedback, suggest improved queries to retrieve better information.

Your task:
- Analyze the critic's feedback about missing or insufficient information
- Generate 2-3 refined queries that address the gaps
- Make queries more specific, targeted, or comprehensive as needed

Return JSON format:
{
    "suggestions": ["refined query 1", "refined query 2", "refined query 3"],
    "reasoning": "explanation of refinement strategy"
}"""
                
                user_message = f"""Original Query: {context['query']}
Critic Feedback: {context['critic_feedback'][-1] if context['critic_feedback'] else 'No feedback'}
Retrieved Info Summary: {len(context['retrieved_info'])} sources

Generate refined queries to address the identified gaps."""
                
            else:  # mode == "response"
                system_message = """You are a Refiner Agent in response mode. Generate a comprehensive, well-structured answer using the retrieved information.

Your task:
- Synthesize all retrieved information
- Address all aspects of the user's query
- Provide a clear, comprehensive response
- Use citations where appropriate

Return JSON format:
{
    "final_answer": "comprehensive response text",
    "sources_used": ["list of source identifiers used"]
}"""
                
                user_message = f"""Query: {context['query']}
Retrieved Information: {json.dumps(context['retrieved_info'][:10], indent=2)}
Critic Approved: Information is sufficient for response

Generate the final comprehensive answer."""
            
            logger.info(f"🔧 Refiner executing in {mode} mode")
            
            # Use vLLM for refinement/response
            vllm_request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }

            response = await self.http_client.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=vllm_request,
                timeout=90.0
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                try:
                    refinement_result = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback for non-JSON responses
                    refinement_result = {
                        "suggestions": [content] if mode == "refinement" else [],
                        "final_response": content if mode == "response" else "",
                        "raw_content": content
                    }
                
                if mode == "refinement":
                    suggestions = refinement_result.get("suggestions", [])
                    logger.info(f"✅ Generated {len(suggestions)} refined queries")
                else:
                    logger.info("✅ Generated final comprehensive response")
                
                return refinement_result
            else:
                logger.error(f"❌ Refiner {mode} failed: {response.status_code}")
                return None
                        
        except Exception as e:
            logger.error(f"❌ Refiner agent ({mode}) error: {str(e)}")
            return None
    
    # All agent execution is now handled by the critic-driven system above
    # The _execute_retriever_agent, _execute_critic_agent, and _execute_refiner_agent methods
    # implement the modern multi-agent approach with iterative improvement

    async def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using vLLM LLM service with OpenAI-compatible API"""
        try:
            vllm_request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.9,
                "stop": ["\\n\\nUser:", "\\n\\nHuman:"]
            }
            
            response = await self.http_client.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=vllm_request,
                timeout=60.0
            )
            response.raise_for_status()
            
            response_data = response.json()
            generated_text = response_data["choices"][0]["message"]["content"]
            
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

    def _extract_sources_from_state(self, state: ConversationState) -> List[Dict[str, Any]]:
        """Extract and format sources from conversation state"""
        sources = []
        for source_info in state.retrieved_info:
            if isinstance(source_info, dict):
                # Extract source information with proper formatting
                source = {
                    "id": source_info.get("id", ""),
                    "text": source_info.get("text", ""),
                    "similarity_score": source_info.get("score", 0.0),
                    "metadata": source_info.get("metadata", {})
                }
                sources.append(source)
        return sources

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