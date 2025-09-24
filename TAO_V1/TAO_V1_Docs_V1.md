# Team Agent Orchestrator (TAO) – Module Documentation

## 1\. Overview

The **Team Agent Orchestrator (TAO)** is the central coordinator of the Team Agent platform. It acts as a **Model Context Protocol (MCP)** gateway that standardizes access to various tools and services for AI agents[\[1\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TAO%20serves%20as%20the%20central,and%20express%20YAML%20upload%20paths). TAO manages multi-agent teams and their interactions with tools, enforces team member-specific tool permissions, and integrates domain knowledge into agent workflows. It provides a unified interface for other platform modules (Team Agent Builder and Team Agent Executor) to create, configure, and execute agent teams. Key capabilities include a flexible domain knowledge hierarchy, granular permission control per team member, and a comprehensive team-building workflow with both conversational “wizard” guidance and direct YAML configuration upload[\[1\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TAO%20serves%20as%20the%20central,and%20express%20YAML%20upload%20paths).

**Modularity and ROI:** TAO is designed as an independent, deployable module within the Team Agent platform. Each sub-component of TAO is a working piece of software that delivers value on its own (e.g. tool registry, orchestrator, semantic search), enabling iterative development and testing[\[2\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Each%20module%20is%20designed%20with,targets%20and%20owners%20to%20be)[\[3\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2A%20Multi,coverage). This modular approach ensures that TAO can be developed and improved incrementally while remaining compatible with the larger system.

## 2\. Architecture

### 2.1 High-Level Architecture

TAO’s architecture consists of several core submodules and supporting services that together facilitate team orchestration. At a high level, TAO includes:

* **MCP Server/Gateway (MCPManager):** An MCP-compatible server that exposes TAO’s functionality (tool invocation, team execution) via standardized interfaces[\[4\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20TAO%20Core%20MCPManager,Team%20Orchestrator). This gateway allows agents or external callers to request tool actions through TAO using a unified protocol.

* **TeamOrchestrator:** The central logic that orchestrates agent team operations. It loads team configurations, selects appropriate team members for tasks, and sequences multi-agent workflows[\[5\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=DomainResolver,Team%20Build%20Workflow)[\[6\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=1,enriched%20with%20member%20metadata).

* **MemberManager:** Manages individual agent team members, including their roles, tool permissions, and state[\[5\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=DomainResolver,Team%20Build%20Workflow). It ensures each member can only use allowed tools and maintains execution history per member.

* **ToolRegistry:** A registry of all available tools (local functions or remote services) that agents may use[\[4\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20TAO%20Core%20MCPManager,Team%20Orchestrator). It provides a lookup for tool adapters and coordinates with an **MCP Tool Registry Client** for dynamic discovery of new tools[\[7\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,The%20MCP%20Tool%20Registry%20Client).

* **Tool Adapters:** Concrete adapters implementing each tool type behind a common interface[\[8\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Tool%20Adapters%20VectorSearch,Tool%5D%20end). Adapters handle the actual invocation of tools – e.g. **VectorSearch**, **EmbeddingService**, **ChunkFetch**, **ExternalAPI**, and **TeamAsTool** adapters for semantic search, embeddings, document retrieval, external web APIs, and even treating another agent team as a tool[\[8\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Tool%20Adapters%20VectorSearch,Tool%5D%20end).

* **TenantManager & DomainResolver:** Utilities for multi-tenancy and domain knowledge integration. **TenantManager** handles tenant-specific configuration, and **DomainResolver** maps agent teams to relevant domain knowledge scopes (using Neo4j for domain hierarchy)[\[4\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20TAO%20Core%20MCPManager,Team%20Orchestrator).

* **BuildWorkflow (Team Build Workflow):** Manages the agent team creation process. It includes a **WizardEngine** for conversational setup, **YAMLValidator** for config file import, and state management for pause/resume of team building sessions[\[9\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Team%20Building%20WizardEngine,Step%20Processor%5D%20end). (This is closely related to the Team Agent Builder module, but key workflow logic lives in TAO).

* **Service Layer Clients:** Adapters for external services: a **QdrantClient** for vector database operations, **PostgresClient** and **Neo4jClient** for database access, and **OllamaClient** for interfacing with the LLM service (Ollama)[\[10\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Service%20Layer%20QdrantClient,Ollama%20Client%5D%20end).

**Inter-component Relationships:** The core components interact as follows – the TeamOrchestrator calls into the MemberManager to decide which agent handles a given task; the MemberManager in turn uses the ToolRegistry to retrieve the correct tool adapter[\[11\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TeamOrchestrator%20,2%20Component%20Communication%20Flow%20python). The BuildWorkflow uses the WizardEngine and YAML validator to guide or validate team configurations[\[12\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TeamOrchestrator%20,2%20Component%20Communication%20Flow). These relationships ensure a clear flow from high-level orchestration down to tool execution.

**Execution Flow:** When a request comes in (e.g. a query for an agent team to handle), TAO processes it in several stages[\[6\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=1,enriched%20with%20member%20metadata):

1. **Request Reception:** The MCP Server receives a request containing a team\_id (and possibly a specific member\_id or target agent).

2. **Team Identification:** The TeamOrchestrator loads the team configuration from the database for the given team\_id[\[6\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=1,enriched%20with%20member%20metadata). This includes team members, their roles, and allowed tools.

3. **Member Selection:** The MemberManager chooses the appropriate team member(s) to handle the request based on context or requested tool (if a specific agent isn’t pre-specified)[\[6\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=1,enriched%20with%20member%20metadata). Selection can be as simple as round-robin or based on which member’s skillset (tools) matches the task.

4. **Permission Check:** For the selected member(s), their tool permissions are retrieved and verified[\[13\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=3,Execution%20history%20logged%20per%20member). TAO ensures the agent has access to any tool it needs to call for this request; otherwise, the action is denied with a permission error.

5. **Tool Invocation:** The orchestrator (or the agent via the executor) invokes the needed tool through the ToolRegistry’s adapter, with the MemberManager adding any member-specific context to the call. The Model Context Protocol is used to standardize how tool inputs/outputs are structured[\[1\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TAO%20serves%20as%20the%20central,and%20express%20YAML%20upload%20paths).

6. **Execution Logging:** The outcome of the tool execution (success or failure, results, timestamps) is logged per team member[\[14\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=5.%20Tool%20execution%20with%20member,enriched%20with%20member%20metadata). TAO records this in both the relational database and the graph database for audit trail.

7. **Response Assembly:** The orchestrator collects the results (and any intermediate outputs) and enriches the final response with metadata (e.g. which member provided the answer, tool usage stats)[\[14\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=5.%20Tool%20execution%20with%20member,enriched%20with%20member%20metadata) before returning it to the requester (or to the Team Agent Executor for further handling).

This end-to-end flow underscores TAO’s role as the **“brain” of the multi-agent system**, coordinating between agents and tools while maintaining governance and context[\[15\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Description%3A%20The%20Orchestrator%20is%20the,and%20a%20set%20of%20agents%2Ftools)[\[16\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=builder%20component%20of%20the%20orchestrator,class%20that%20stores%20the%20prompt).

### 2.2 Submodule Overview Diagram

For a visual representation, TAO’s architecture can be thought of in layers: core orchestrator components, tool adapter layer, and underlying services. The original specification provides a mermaid diagram summarizing the submodules and their groupings[\[17\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20TAO%20Core%20MCPManager,Change%20Detector)[\[18\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Tool%20Adapters%20VectorSearch,Tool%5D%20end):

* **TAO Core:** MCPManager, ToolRegistry, TenantManager, DomainResolver, TeamOrchestrator, MemberManager, BuildWorkflow, VersionManager, ChangeDetector.

* **Team Building:** WizardEngine, YAMLValidator, StateManager, StepProcessor (workflow management).

* **Tool Adapters:** VectorSearch, EmbeddingService (BGE-M3), ChunkFetch, ExternalAPI, TeamAsTool.

* **Service Layer:** QdrantClient, PostgresClient, Neo4jClient, OllamaClient.

*(The VersionManager and ChangeDetector handle configuration versioning and change tracking – they monitor updates in configs or knowledge bases to invalidate caches or signal updates, ensuring the orchestrator always uses up-to-date information. These are part of TAO’s extensibility but are not detailed in this MVP scope.)*

## 3\. Submodule Specifications and Implementation Guides

Each submodule in TAO is implemented as an independent, testable unit with a clear purpose, its own return on investment (ROI) value, and a defined test plan[\[2\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Each%20module%20is%20designed%20with,targets%20and%20owners%20to%20be). This section breaks down each major subcomponent of TAO, detailing its responsibilities, design, and how to build and verify it. The folder structure in the codebase (shown below) reflects these components, with each submodule residing in a dedicated module or package[\[19\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%82%20%20%20%20,New%3A%20Team%20orchestration)[\[20\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%82%20%20%20%20,py):

src/tao/  
├── core/  
│   ├── server.py           \# MCP server/gateway  
│   ├── registry.py         \# ToolRegistry implementation  
│   ├── tenant\_manager.py   \# TenantManager (tenant configs)  
│   ├── domain\_resolver.py  \# DomainResolver (domain knowledge logic)  
│   ├── team\_orchestrator.py\# TeamOrchestrator logic  
│   ├── member\_manager.py   \# MemberManager logic  
│   └── ... (other core components)  
├── tools/  
│   ├── base.py             \# Base Tool adapter class  
│   ├── vector\_search.py    \# VectorSearch tool adapter  
│   ├── embedding.py        \# EmbeddingService adapter  
│   ├── chunk\_fetch.py      \# ChunkFetch adapter  
│   ├── external\_api.py     \# ExternalAPI adapter  
│   ├── team\_as\_tool.py     \# Team-as-Tool adapter  
│   └── ...  
├── workflows/  
│   ├── team\_builder.py     \# Team building workflow coordinator  
│   ├── wizard\_engine.py    \# Conversational wizard for team setup  
│   ├── yaml\_processor.py   \# YAML config import & validation  
│   ├── build\_state\_manager.py \# Handles pause/resume of builds  
│   └── ...  
├── adapters/  
│   ├── qdrant.py           \# Qdrant (vector DB) client  
│   ├── postgres.py         \# Postgres DB client  
│   ├── neo4j.py            \# Neo4j DB client  
│   └── ...  
└── ... (models, utils, etc.)

*(Simplified view of relevant directories; see project README for full structure.)*

Below, each submodule is described with its **purpose**, **ROI (value as a standalone unit)**, core **implementation details**, and **testing plan**.

### 3.1 TeamOrchestrator

**Purpose & Responsibilities:** The TeamOrchestrator orchestrates the overall workflow of an agent team. It is the entry point for executing a team against a user query or task[\[6\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=1,enriched%20with%20member%20metadata). The orchestrator loads the team configuration (including members and their tool sets), selects which agent(s) should handle the incoming task, and coordinates the sequence of actions the team takes. In advanced scenarios, it can construct a **LangGraph** – a directed acyclic graph of steps representing the plan for multi-agent problem solving[\[21\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Description%3A%20The%20Orchestrator%20is%20the,plan%2C%20without%20falling%20into%20cycles)[\[22\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,the%20conversation%20and%20tool%20usage). For MVP, the orchestrator handles a linear or simple branched sequence of agent actions (e.g. retrieve info then answer), ensuring correct order and passing outputs between steps[\[23\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,orchestrator%20is%20designed%20so%20we)[\[24\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,fail%20the%20whole%20task%20or). It also injects any tenant-specific or user-specific context into the process (for example, organizational policies or user profile data relevant to the query)[\[25\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,say%20it%E2%80%99s%20supposed).

**Standalone ROI:** The TeamOrchestrator by itself showcases automated task planning and coordination among multiple agents. Even as a standalone component, it can take a defined team configuration and execute a multi-step reasoning process, demonstrating the power of orchestrated AI agents[\[26\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=core%2Fregistry_client%2F%20Orchestrator%20%26%20LangGraph%20DAG,%E2%80%93%20End%20of%20Sep%202025). This provides immediate value by enabling complex workflows (e.g. a Q\&A flow with retrieval and answering agents). As the “brain” of the system, improvements to the orchestrator (like more sophisticated planning or parallel execution) directly increase the system’s problem-solving capability.

**Implementation Approach:** The orchestrator is implemented as a class (e.g. TeamOrchestrator in team\_orchestrator.py) that is initialized with connections to the database and other managers[\[27\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,members). Key methods include:

* get\_team(team\_id): Fetches the team configuration from Postgres (and optionally enriches with graph data) for use in execution[\[28\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20get_team,if%20not%20team_data%3A%20return%20None)[\[29\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=team_id%3Dteam_data%5B%27team_id%27%5D%2C%20tenant_id%3Dteam_data%5B%27tenant_id%27%5D%2C%20team_name%3Dteam_data%5B%27team_name%27%5D%2C%20status%3Dteam_data%5B%27status%27%5D%2C%20members%3D%5BAgentTeamMember%28,). This loads all member records associated with the team and returns an AgentTeam model instance.

* select\_member\_for\_task(team\_id, task\_type, required\_tools): Logic to pick the best team member for a given task or tool usage. For MVP this may simply filter members that have all required\_tools in their permissions and choose the first available[\[30\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,append%28member)[\[31\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,0). Future implementations might use load balancing or skill matching.

* execute\_with\_member(team\_id, member\_id, tool\_name, \*\*kwargs): Orchestrates invoking a particular tool via a specific team member[\[32\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20execute_with_member,member%20%3D%20await%20self.member_manager.get_member%28member_id)[\[33\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=if%20tool_name%20not%20in%20member,). This method first verifies the member exists and has permission for that tool, then delegates to the MemberManager to perform the actual execution. It wraps the call with logging for audit: generating an execution record before invocation and updating it with success/failure and results after[\[34\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,now%28%29)[\[35\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=return%20result).

**LangGraph and Multi-Step Plans:** In current scope, the orchestrator can handle a sequence of steps (e.g., one agent’s output feeding into another). The concept of **LangGraph DAG** is implemented internally by defining an ordered list of steps or a simple dependency graph in code[\[36\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=graph%20could%20be%20an%20agent%E2%80%99s,Agent%20B%20might%20wait)[\[37\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,single%20node). Each step knows which agent or tool to invoke and what input it needs (possibly the output of a prior step). The orchestrator ensures no cycles in this graph (aside from optional critic feedback loops treated as new cycles)[\[38\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=LangGraph%20Concept%3A%20We%20use%20a,which%20defines%20what)[\[39\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=using%20that%20info%5B46%5D%5B47%5D.%20,a%20full%20graph%20library%3B%20a). For example, if a team has a Retriever agent and a Solver agent, the orchestrator will plan: Step 1 – run Retriever with the user’s query; Step 2 – run Solver with the retrieved info to produce the answer[\[23\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,orchestrator%20is%20designed%20so%20we). This plan can be hard-coded for known team patterns in MVP, with the ability to extend to dynamic planning in future phases.

**Testing Plan:** The TeamOrchestrator is tested via unit tests for its methods and integration tests in the context of a running workflow[\[40\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Granular%20tool%20access%20control)[\[41\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Build%20session%20management%20Acceptance%20Criteria). Key test scenarios include:

* *Team Loading:* Given a known team in the test database, get\_team should return an object with correct members and attributes. Use a fixture to seed a sample team (with members/tools) and assert that the returned AgentTeam matches expected values.

* *Member Selection:* Create a team with multiple members having different tool permissions. For a required tool set (e.g. \["vector\_search"\]), ensure select\_member\_for\_task returns a member that has that tool. Test edge cases where no member qualifies (expect a warning or None return)[\[30\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,append%28member).

* *Tool Execution Flow:* Using a stub tool (or a real tool adapter in a test environment), test execute\_with\_member. Verify that it raises a PermissionError if the member lacks permission[\[42\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=if%20not%20member%3A%20raise%20ValueError%28f,not%20found), and that it calls MemberManager to execute the tool when permissions are valid. Use dependency injection or mocking for MemberManager to simulate a tool result, and check that log\_member\_execution and update\_member\_execution in Postgres are called with correct parameters[\[34\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,now%28%29)[\[43\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=except%20Exception%20as%20e%3A%20,update_member_execution%28%20execution_id%3Dexecution_id%2C%20status%3D%27failed%27%2C%20error%3Dstr%28e%29).

* *Integration (end-to-end):* Write an integration test where a full query flows through orchestrator: e.g., simulate a request to the orchestrator with a team of two agents (retriever & solver). Mock the retriever’s tool call to return some data and the solver’s step to return a final answer. Assert that orchestrator produces a combined result and logs both steps. (In practice, this is done in cooperation with the executor – see TAE integration, but orchestrator logic can be tested by simulating the executor’s role.)

### 3.2 MemberManager

**Purpose & Responsibilities:** The MemberManager handles all operations related to individual team members (agents). It is responsible for retrieving member details (including which tools they can use), executing tools on behalf of a member, and updating member-specific records (like execution history). Essentially, MemberManager encapsulates the logic of an “agent profile” – what the agent is allowed to do and tracking what they have done[\[44\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=class%20MemberManager%3A%20,members%20and%20their%20tool%20permissions)[\[45\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20get_member,member_id).

**Standalone ROI:** As its own unit, MemberManager provides **granular control over AI agent capabilities**. It enables scenarios where different agents have different permissions and roles. The ROI of this component is evident in enterprise contexts: it allows enforcing that a “FinanceBot” can only run tools related to financial data, whereas an “HRBot” is limited to HR knowledge. This isolation adds safety and interpretability to multi-agent systems. MemberManager can be reused in any multi-agent framework where role-based tool access is needed, making it valuable beyond the TAO module[\[46\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,LangGraph%20wizard%20engine)[\[47\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,tool%20permission%20enforcement).

**Implementation Details:** Implemented as MemberManager in member\_manager.py, initialized with a reference to the Postgres client and the ToolRegistry[\[48\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,their%20tool%20permissions). Key functions include:

* get\_member(member\_id): Fetches a member’s data and their allowed tools from the database[\[49\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20get_member,member_id)[\[50\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=tools%20%3D%20await%20self). It populates an AgentTeamMember model with fields like name, role, and tool permissions. A simple in-memory cache (member\_cache) may be used to avoid repeated DB hits for the same member[\[51\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=self.member_cache%20%3D%20).

* execute\_tool(member\_id, tool\_name, \*\*kwargs): Core method to execute a tool as a given member[\[52\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20execute_tool,get_member%28member_id%29%20if%20not%20member). It first ensures the member exists and has the specified tool in their permission list[\[53\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,). Then it retrieves the actual tool adapter from the ToolRegistry (tool\_registry.get\_tool(tool\_name))[\[54\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Tool%20%7Btool_name%7D%20not%20found). Before invoking the tool’s execute method, it injects a \_member\_context into the call (this context can include member ID, name, role, etc., allowing the tool or downstream logging to know which agent is calling)[\[55\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,member_role). Finally, it calls tool.execute(\*\*kwargs) asynchronously and captures the result[\[56\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,kwargs). After execution, it logs the event via postgres.log\_member\_tool\_execution – recording that this member used that tool, with success/failure status[\[57\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,get%28%27success%27%2C%20False%29). This logging supplements the orchestrator-level logging by focusing on the member-tool interaction.

* update\_member\_tools(member\_id, tool\_names): A management function to update which tools a member is allowed to use (e.g., when reconfiguring a team). This would update the Postgres member\_tools mapping table and refresh caches. (Implementation straightforward: not shown above, but likely iterates provided tool list, verifies tool existence, and updates DB accordingly)[\[58\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=if%20not%20self.tool_registry.has_tool%28tool_name%29%3A%20raise%20ValueError%28f,not%20found%20in%20registry)[\[59\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,255%29%20UNIQUE%20NOT%20NULL).

**Testing Plan:** The MemberManager is tested with a focus on permission enforcement and correct tool invocation[\[60\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=tests%2Fintegration%2Ftest_team_building,team_builder%20import%20TeamBuilder%2C%20BuildStep)[\[61\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%40pytest.mark.asyncio%20async%20def%20test_member_tool_permissions%28%29%3A%20,manager%20%3D%20MemberManager%28postgres%2C%20tool_registry). Test considerations include:

* *Member Retrieval:* Insert a dummy member in the DB with known tool permissions. Call get\_member and assert that the returned AgentTeamMember contains the expected list of tool\_permissions (matching what’s in the member\_tools table)[\[62\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=manager%20%3D%20MemberManager). Test caching by calling get\_member twice and (if possible) verifying the second call did not query the DB again (this can be done by spying on the Postgres client or checking a timestamp in the returned object).

* *Permission Enforcement:* Use a member with a restricted tool set. Attempt to execute\_tool with a tool not in their permissions and expect a PermissionError with an appropriate message[\[53\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,). Also test that a permitted tool proceeds to execution.

* *Tool Execution:* For a permitted tool, replace the actual tool adapter with a stub that returns a known result (e.g., monkeypatch ToolRegistry.get\_tool to return a dummy tool whose execute returns {"success": True, "output": "OK"} without doing real work). Then call execute\_tool and verify it returns that result, and that the Postgres client’s log\_member\_tool\_execution was called with success=True[\[56\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,kwargs). Simulate an exception in the tool (have the dummy raise) to ensure the method raises the error and does not catch it silently – and in this case, verify that a log entry with success=False is recorded.

* *Tool Update:* If implemented, test update\_member\_tools by adding and removing some tools for a member. After calling it, fetch the member again and confirm the permissions list changed accordingly. Ensure it cleans up old permissions in DB and adds new ones (the underlying DB constraint might enforce uniqueness per member-tool pair).

By testing these, we ensure the MemberManager reliably gates tool usage and accurately records each member’s actions.

### 3.3 ToolRegistry and Tool Adapters

**Purpose & Responsibilities:** The ToolRegistry serves as a **catalog and factory** for tools that agents can use. It holds definitions of each tool (name, type, connection info) and provides methods to retrieve a tool’s adapter or check if a tool exists[\[63\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L527%20,Tool%20%7Btool_name%7D%20not%20found)[\[64\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Tool%20%7Btool_name%7D%20not%20found). Tool adapters are concrete classes that know how to invoke a specific kind of tool (whether it’s an internal function, a database query, a web API call, or even another agent team acting as a tool). By abstracting tools behind adapters, the agents and orchestrator can invoke any tool with a uniform interface (e.g., tool.execute(input)), without worrying about transport details[\[65\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Tool%20Loader%20%26%20Transport%20Adapter,It)[\[66\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L260%20,function%20calls%2C%20as%20per%20the).

**Standalone ROI:** The ToolRegistry and its adapters enable **plug-and-play extensibility** of the system’s capabilities. As a standalone module, it allows new tools to be integrated by simply registering them (in a config or registry DB) and providing an adapter, without changing core logic. This has high ROI: it can be reused in other AI systems to manage tool plugins uniformly[\[67\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Tool%20Loader%20%26%20Transport%20Adapter,using%20the%20correct%20protocol%2Ftransport)[\[68\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Diagram%20%E2%80%93%20Tool%20Loader%20Context%3A,calls). For example, one could use this registry to let different AI models or services be called in a standardized way, demonstrating dynamic tool integration (e.g. adding a new API tool at runtime and agents immediately being able to use it)[\[69\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L254%20,11%5D.%20This%20allows%20the).

**Implementation Details:** The ToolRegistry might be implemented as part of registry.py in core, or conceptually split into a **Tool Loader** and an **in-memory Registry**[\[65\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Tool%20Loader%20%26%20Transport%20Adapter,It)[\[68\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Diagram%20%E2%80%93%20Tool%20Loader%20Context%3A,calls). Key aspects:

* **Tool Definitions:** Tools can be defined via configuration files (e.g. a tools.toml or YAML) or discovered via the MCP registry. Each tool has metadata like tool\_name (unique), tool\_type (e.g., "vector\_db", "embedding\_service", "rest\_api", "team\_tool"), endpoint or connection info, input/output schema, etc.[\[70\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L985%20mcp_mode%20VARCHAR,TEXT%2C%20input_schema%20JSONB%2C%20output_schema%20JSONB)[\[71\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=mcp_mode%20VARCHAR,JSONB%2C%20output_schema%20JSONB). In Postgres, there is a tools table capturing this info (with columns for name, mode, endpoint, schemas, etc.)[\[72\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%3D%3D%3D%20MCP%20Configuration%20%3D%3D%3D%20MCP_SERVER_HOST%3D0,MCP_SERVER_PORT%3D8100%20MCP_TRANSPORT_MODE%3Dstdio).

* **Loading Tools:** On startup, or when a team is initialized, the ToolRegistry (or a Tool Loader component) loads all relevant tool definitions. If using static config, it reads from file/DB. If dynamic, it can call an **MCP Tool Registry Client** to fetch the latest tool list from a registry service[\[73\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,11%5D.%20This%20allows%20the)[\[74\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Integration%20with%20Tool%20Loader%3A%20The,tool%20definitions%20from%20the%20registry). Each tool definition is then used to instantiate the appropriate adapter class. For example, a tool of type "vector\_search" will create a VectorSearchTool adapter, a tool of type "external\_api" might create a RESTAPIAdapter, etc.[\[75\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=When%20initializing%2C%20the%20Tool%20Loader,tool%20type%20and%20instantiate%20the)[\[76\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,the%20adapter). Adapters are stored in an in-memory map (e.g., self.tools dictionary keyed by tool\_name) for quick access[\[68\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Diagram%20%E2%80%93%20Tool%20Loader%20Context%3A,calls).

* **Tool Adapter Classes:** Under src/tao/tools/, each adapter subclass inherits from a common BaseTool interface (with at least an execute(\*\*kwargs) coroutine method). Adapters implement the transport logic:

* *VectorSearch Adapter:* Connects to the Qdrant vector database to perform semantic similarity searches over indexed data (e.g., find relevant documents or tools given an embedding). It likely uses the Qdrant client from adapters/qdrant.py to query the vector index[\[77\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=Tool%20Search%20Index%20,using%20the%20member_executions%20logging%20with).

* *EmbeddingService Adapter:* Calls the **BGE-M3 Embedding** microservice to get text embeddings[\[78\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L645%20,the%20adapter). For example, it might issue an HTTP request to the embedding server’s /embed endpoint with the text and return the vector. This allows agents to vectorize queries or content for semantic search.

* *ChunkFetch Adapter:* Retrieves document chunks or knowledge pieces from a data store (possibly Postgres or an object storage like MinIO) given an identifier or query. This might be used by a retriever agent to get the content of documents by ID.

* *ExternalAPI Adapter:* Handles calling third-party APIs or services. For instance, a tool named "WebSearch" could be an external REST API for web search; the adapter would perform an HTTP GET/POST to the API’s URL, inject query parameters, and return the response[\[79\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L245%20The%20Tool,9). It may manage auth headers or API keys as configured.

* *TeamAsTool Adapter:* A special adapter that allows an entire agent team to be invoked as if it were a single tool. Given a target team ID and an input, this adapter will forward the request to TAO (or TAE) to execute that team and return the result. Essentially, it wraps a call like “ask Team X this question” inside the tool interface, enabling **multi-team orchestration**. This is powerful for scaling out capabilities (one team can leverage another team’s expertise by calling it as a subroutine).

* **MCP Transport Integration:** Some tools might use MCP as their protocol. For example, if a tool is itself an AI model running with an MCP interface (like a local LLM on stdio), the adapter would use an MCP transport (via fastmcp library) to send a request and receive a response[\[80\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=)[\[81\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L270%20JSON%20from,py). The ToolRegistry’s design allows for different transport modes (HTTP, gRPC, stdio, etc.), configured per tool. In the tools table, fields like mcp\_mode and endpoint specify how to talk to the tool[\[72\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%3D%3D%3D%20MCP%20Configuration%20%3D%3D%3D%20MCP_SERVER_HOST%3D0,MCP_SERVER_PORT%3D8100%20MCP_TRANSPORT_MODE%3Dstdio). The adapter uses these to choose the method (e.g., starting a subprocess for stdio tools or making HTTP calls for REST tools).

**Usage Example:** If an agent wants to use a tool "WebSearch":  
1\. The agent (via MemberManager) calls tool \= tool\_registry.get\_tool("WebSearch").  
2\. The registry returns the ExternalAPIAdapter instance for WebSearch (created at load time).  
3\. The agent calls await tool.execute(query="How tall is Mount Everest?").  
4\. The adapter formats an HTTP request, e.g., GET http://api.websearch.com?q=How%20tall%20is%20Mount%20Everest (with any required API key), and gets the response JSON.  
5\. The adapter returns the answer (perhaps parsed into a simpler dict) to the agent. The agent can then use that result in its reasoning.

From the orchestrator’s perspective, it doesn’t matter if "WebSearch" was an internal function or an external API – the ToolRegistry abstracts that away[\[82\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,function%20calls%2C%20as%20per%20the)[\[68\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Diagram%20%E2%80%93%20Tool%20Loader%20Context%3A,calls).

**Testing Plan:** Testing of ToolRegistry and adapters involves both unit tests for individual adapters and integration tests with actual or mocked services:

* *ToolRegistry Initialization:* Provide a set of tool definitions (could be hard-coded in a test or use a temporary database). Invoke the tool loading routine and ensure that get\_tool("some\_tool") returns a non-null adapter for each defined tool. For each tool type, confirm the adapter is an instance of the correct class. If dynamic loading via MCP registry is used, mock the registry client to return a test list of tools and verify they’re loaded.

* *Adapter Execution (Unit):* For each adapter type, write a test for its execute method in isolation. This often means mocking the underlying service:

* For VectorSearch, mock the Qdrant client’s search method to return a known vector search result. Then call the adapter’s execute (with a sample query or vector) and check that the output matches expected format (e.g. a list of IDs or items) and that the Qdrant client was called with correct parameters.

* For EmbeddingService, similarly mock an HTTP or client call to the embedding server to return a known vector (e.g. \[0.1, 0.2, ...\]) and ensure the adapter returns that vector. Also test error handling (like the service not reachable).

* ExternalAPI adapter: simulate an API by mocking requests.get or the HTTP client. Provide a fake response (status code \+ JSON) and confirm the adapter returns the expected parsed result (and handles HTTP error codes properly).

* TeamAsTool adapter: this one can be tested by stubbing the orchestrator call. For instance, monkeypatch the orchestrator’s execute\_team(team\_id, input) method (which you would implement to run a team) to return a canned response. Then call the adapter’s execute (with some input) and ensure it returns that canned response and perhaps that it logged the cross-team call.

* *Integration Test:* Start a minimal TAO environment (maybe with an in-memory Qdrant and a dummy embedding server) via Docker or fixtures. Register a couple of tools (one of each type if possible). Then simulate an agent query that requires using each tool once. For example, an agent that searches a vector DB then calls an external API. Verify that each adapter was indeed invoked and that the overall result is assembled. This end-to-end ensures the ToolRegistry wiring is correct.

**Note:** Many adapter tests can be done with dependency injection (pass a fake client into the adapter) or by patching the specific network call. The goal is to not require the actual external services in unit tests, but we will have a separate integration test with real services (like hitting a test Qdrant instance, etc.) to ensure everything works in a deployed setting.

### 3.4 MCP Server / Gateway

**Purpose & Responsibilities:** The MCP Server (also referred to as MCP Gateway in TAO) is the entry point service that allows external clients (including other modules or possibly end-user applications) to interact with TAO’s functionality via the **Model Context Protocol**. MCP (Model Context Protocol) is a standardized interface for tool and model interactions; TAO’s server implements this so that all tool executions and orchestrations can be invoked in a uniform way[\[1\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TAO%20serves%20as%20the%20central,and%20express%20YAML%20upload%20paths). In practical terms, the MCP Server is an API (could be HTTP REST, WebSocket, or CLI/stdio based on configuration) that listens for requests such as “execute tool X with these inputs” or “run team Y on query Z” and routes them to the appropriate TAO component.

**Deployment Mode:** TAO’s MCP server can run in different modes as configured by environment variables – e.g., MCP\_TRANSPORT\_MODE can be "http" or "stdio"[\[72\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%3D%3D%3D%20MCP%20Configuration%20%3D%3D%3D%20MCP_SERVER_HOST%3D0,MCP_SERVER_PORT%3D8100%20MCP_TRANSPORT_MODE%3Dstdio). In HTTP mode, TAO likely runs a FastAPI or similar web server on a specified port (e.g. default 8100\)[\[72\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%3D%3D%3D%20MCP%20Configuration%20%3D%3D%3D%20MCP_SERVER_HOST%3D0,MCP_SERVER_PORT%3D8100%20MCP_TRANSPORT_MODE%3Dstdio), exposing endpoints. In stdio mode, it could integrate with other processes via stdin/stdout streams (useful if TAO is embedded as a subprocess tool itself). The default is typically HTTP for a standalone service deployment.

**Key Endpoints/Interfaces:** In HTTP mode, the MCP server might expose endpoints such as: \- POST /mcp – a generic endpoint to handle MCP-formatted requests (where the request body includes the tool or action and parameters).  
\- GET /health – a simple health check endpoint returning status (e.g. "ok") to indicate the service is running[\[83\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=neo4j%3A%20image%3A%20neo4j%3A5,ports)[\[84\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=cpus%3A%20%274%27%20memory%3A%2020G%20healthcheck%3A,10s%20retries%3A%205%20start_period%3A%2060s).  
\- GET /metrics – if metrics are not on a separate port, an endpoint to scrape Prometheus metrics (though in our case we start a separate metrics server thread; see Section 6).

Internally, the server translates incoming requests into calls to TeamOrchestrator or ToolRegistry. For example, a request might include a tool name and payload; the server would call MemberManager.execute\_tool() or orchestrator logic to fulfill it, then package the result back in an MCP response format.

**Multi-Tenancy & Routing:** The MCP server is multi-tenant aware. Many requests will include a tenant\_id or be authenticated in a way that the server can determine which tenant’s context to use. TAO ensures isolation by using the tenant\_id to scope database queries (only load teams and data for that tenant)[\[3\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2A%20Multi,coverage). If multiple tenants’ teams are active, the server handles concurrent requests separately, updating metrics per tenant/team as needed.

**Standalone ROI:** As a module, the MCP Server provides a **unified API layer** that could be reused to front any tool execution system with a consistent protocol. For instance, any AI or tool-serving system can adopt MCP and use this server implementation to handle requests. The ROI is a standardized integration point – once clients speak MCP, they can interact with a variety of tools through TAO without custom API for each tool. It also simplifies adding new tools: as long as they register with TAO, they automatically get an API via the MCP server.

**Implementation Details:** The server is likely implemented with an async framework. Given references to fastmcp[\[80\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=), TAO might use a library that simplifies building MCP servers on FastAPI or similar. We see in config that MCP\_SERVER\_HOST and MCP\_SERVER\_PORT are defined, and MCP\_TRANSPORT\_MODE can switch to stdio for command-line execution[\[72\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%3D%3D%3D%20MCP%20Configuration%20%3D%3D%3D%20MCP_SERVER_HOST%3D0,MCP_SERVER_PORT%3D8100%20MCP_TRANSPORT_MODE%3Dstdio). Also, fastmcp suggests the MCP specification 1.4 is used, ensuring compliance with that protocol standard.

The server’s responsibilities include: \- **Starting up** and listening on the specified host/port. Possibly it spawns in core/server.py by reading env vars (host, port, mode) and calling fastmcp server runner. \- **Request handling:** For each incoming request, parse the MCP envelope (which likely includes an action type and payload). Identify if it’s a tool invocation or a team query. Dispatch accordingly: \- If it’s a direct tool call (with a tool name and inputs), and maybe a member context, call MemberManager.execute\_tool or through orchestrator if needed. \- If it’s a team execution request (with team\_id and user query), call orchestrator’s main entry (e.g., TeamOrchestrator.run(team\_id, user\_query) which would invoke the multi-agent execution process). \- **Error handling:** Translate exceptions or validation errors into MCP error responses so that clients get structured error info (like permission denied, tool not found, etc.). For example, a PermissionError might become an MCP response with an error code and message.

**Health Check Endpoint:** A /health endpoint should quickly check the status of dependent services (database connections, etc.). Often this just returns 200 OK if the process is up; more advanced checks could ping the Postgres or Neo4j. In docker-compose, we see that the databases and other services have health checks defined[\[85\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=memory%3A%208G%20healthcheck%3A%20test%3A%20%5B%22CMD,30s%20timeout%3A%2010s%20retries%3A%203)[\[84\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=cpus%3A%20%274%27%20memory%3A%2020G%20healthcheck%3A,10s%20retries%3A%205%20start_period%3A%2060s). TAO’s own health could be monitored similarly by calling /health. In our environment scripts, a service\_health.sh script exists, which likely curls these endpoints[\[86\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%9C%E2%94%80%E2%94%80%20scripts%2F%20%E2%94%82%20%20,sh).

**Testing Plan:**  
\- *Unit Test API Handlers:* If using FastAPI or similar, utilize the test client to send sample requests: \- Send a health check request and expect a 200 response with body like {"status":"ok"}. \- Simulate a tool execution request: e.g., POST to /mcp with JSON {"action": "execute\_tool", "tool": "vector\_search", "params": {...}, "member\_id": "...", "team\_id": "..."} and ensure the response contains the expected result or an error if unauthorized. For testing, one can mock the MemberManager to avoid doing real tool calls. \- Simulate a team query request: e.g., {"action": "execute\_team", "team\_id": "...", "input": "User question?"} and verify it triggers orchestrator. This might require a stub orchestrator that returns a dummy answer so the test can assert on the output. \- *Authentication & Tenant Isolation:* If the MCP server uses an auth token or API key, include tests to ensure requests without auth are rejected (401) and with valid auth proceed. If multi-tenant, test that providing a tenant context only returns data for that tenant (this overlaps with permission control in Section 8). \- *Concurrency:* Use an async test (or stress test) to send multiple requests in parallel to ensure the server can handle concurrent orchestrations. The acceptance criteria might include supporting 100+ concurrent requests[\[87\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Acceptance%20Criteria%3A%20,50ms%20response%20for%20cached%20queries), so a load test can be done (though not typical in unit tests, this would be a separate performance test). Verify no cross-talk (one request’s data affecting another).

Ensuring the MCP server is robust is critical as it is the front-door of TAO in production.

### 3.5 Semantic Tool Search (Vector Index via Qdrant)

**Purpose:** The Semantic Tool Search subsystem provides intelligent recommendations or lookups of tools based on semantic similarity. All available tools (their names, descriptions, metadata) are indexed as vectors in a vector database (Qdrant), enabling *“find the best tool for a given need”* queries[\[77\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=Tool%20Search%20Index%20,using%20the%20member_executions%20logging%20with). This is useful when an agent or orchestrator needs to decide which tool to use for a given user query or subtask: instead of relying purely on static configuration, TAO can query this index to see which tool’s description is most relevant to the query context.

**How it Works:** Each tool has an embedding vector representation of its description or capabilities. These embeddings are generated using the EmbeddingService (BGE-M3) and stored in Qdrant (which is a vector DB that allows similarity search). When a new tool is registered, TAO (via the ToolRegistry or a registry client) will: \- Compute the embedding for the tool's description and upsert it into the Qdrant collection of tools (with the tool’s ID as metadata). \- Also store the tool’s details in Postgres (for retrieval of full info).

When an agent team is processing a query, the orchestrator or an agent can formulate a vector query representing the task at hand (for example, embedding of the user’s query or the agent’s current goal) and ask Qdrant for the top-N similar tool vectors[\[77\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=Tool%20Search%20Index%20,using%20the%20member_executions%20logging%20with). Qdrant returns the IDs of the most relevant tools. TAO then looks up those tool IDs in Postgres to get human-readable info (name, how to invoke, etc.)[\[88\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=request%20come%20the%20agent%20can,The%20MCP%20Tool%20Registry%20Client). This can either be presented to the agent as a suggestion (“These tools might be useful”) or automatically chosen if the confidence is high.

**Integration:** The Semantic Tool Search is implemented with a **QdrantClient** (in adapters/qdrant.py) and ties into both the tool registry (for indexing tools) and the orchestrator (for query-time search). It may maintain a collection called “tool\_index” in Qdrant. The **EmbeddingService** (BGE-M3) is used to encode any text to vectors – both tools’ descriptions and incoming queries. The sequence for a recommendation might be: 1\. Given a query text, call EmbeddingServiceAdapter.execute(text=query) to get an embedding vector. 2\. Call QdrantClient.search(vector) to search the tool\_index for similar items. 3\. Get back a list of tool IDs (with similarity scores). 4\. For each, fetch tool info from Postgres via ToolRegistry or direct DB query. 5\. Provide the list of candidate tools (ranked) to the orchestrator/agent.

**Use Cases:** This is especially valuable if dozens or hundreds of tools are available – the agent might not “know” all of them. The semantic index helps surface the most relevant tools dynamically[\[77\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=Tool%20Search%20Index%20,using%20the%20member_executions%20logging%20with). For MVP, the number of tools is moderate (tens to low-hundreds), which Qdrant can handle easily with millisecond latency[\[89\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=one%20call%20per%20needed%20tool,which%20Qdrant%20handles%20easily%20in)[\[90\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L1828%20tenant%20IDs,%28Some%20of). We assume one global tool index (possibly segmented by tenant if tools differ per tenant).

**Standalone ROI:** The semantic search index can be offered as a generic “Tool Recommendation Service” for any AI workflow that involves tool selection. Its standalone value is significant: it turns a manual configuration problem (deciding which tool to use) into an automated retrieval problem. This could be repurposed beyond Team Agent (e.g., recommending APIs to a developer based on a query). It demonstrates advanced capability (using embeddings for context matching) even if the initial system has a small registry[\[73\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,11%5D.%20This%20allows%20the)[\[66\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L260%20,function%20calls%2C%20as%20per%20the).

**Implementation Details:** \- **Qdrant Setup:** Docker compose runs a Qdrant container on port 6333 (HTTP)[\[91\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=qdrant%3A%20image%3A%20qdrant%2Fqdrant%3Alatest%20container_name%3A%20qdrant,.%2Fdata%2Fqdrant%3A%2Fqdrant%2Fstorage%3Az%20environment). TAO’s QdrantClient uses the Qdrant REST or gRPC API to perform operations. On initialization, TAO should ensure the collection (e.g., “tools”) exists, with an appropriate vector dimension matching the embedding model (e.g., 768 for BGE-M3), and configure metadata (store tool\_id as a payload).  
\- **Indexing Tools:** A method like index\_tool(tool: Tool) computes the embedding (via embedding adapter) and sends a Upsert request to Qdrant (with vector and payload including tool\_id and perhaps tool\_name for convenience). This is called whenever a new tool is added or an existing tool’s description changes. A periodic job or the ChangeDetector might ensure the index stays in sync[\[92\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,get%20the%20information%20in%20human).  
\- **Searching Tools:** search\_tools(query\_text) will embed the text and execute a similarity search (cosine similarity or dot product) in Qdrant, retrieving top results. It then returns the tool IDs and scores. We might set a similarity threshold or number of results (e.g., top 3 tools).  
\- **Using Search in Orchestrator:** During a team execution, if an agent is unsure which tool to pick, it could query this function. For example, if the user question is “What is the revenue of Company X last year?”, the orchestrator or an agent could query the tool index with that, which might return tools like “SEC\_FilingsAPI” or “FinancialReportDB” if those were registered, because their descriptions mention financial data. The orchestrator can then either automatically route the query to a relevant agent that has those tools, or advise the agent (if the agent prompt is designed to accept suggestions).

**Testing Plan:**  
\- *Indexing:* Use a known set of tools with distinct descriptions. Manually compute or have expected differences in embeddings (this might be tough without real model; one can stub the embedding service with a deterministic output). Index the tools via the Qdrant client and then directly query Qdrant (or check via Qdrant API) that the vectors were stored and the payloads contain correct tool IDs. Also test updating a tool (re-index to see if it overwrites or adds properly).  
\- *Search:* If using the real embedding model in tests is not feasible, stub the embedding process: e.g., override EmbeddingServiceAdapter.execute to return a vector that you control. For example, pretend we have two tools, "ToolA" and "ToolB", and give them fake 2D vectors like \[0.1, 0.9\] and \[0.9, 0.1\]. For a query vector \[0.2, 0.8\], the search should return ToolA as more similar. Test that search\_tools("query...") returns the expected tool ID ranking.  
\- *Integration test:* Run Qdrant (perhaps using a temporary test instance) and the real embedding model on a small scale. Feed a couple of tool descriptions to the system and ensure that an actual query returns a reasonable result. This is more of a system test; it can be done if the CI environment permits running those services.  
\- *Performance:* Though not a typical unit test, ensure that searching, even with say 100 tools, is fast. Qdrant is optimized for vectors, so it should be \<50ms for even larger sets[\[93\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,reduction%20in%20support%20tickets). Logging can be added to measure search time, which is checked in performance testing.

### 3.6 Audit & Execution Logging

**Purpose:** TAO includes comprehensive logging of actions for auditing, troubleshooting, and learning. Every tool execution by any team member is recorded with details such as which team and member executed what tool, when, with what outcome[\[34\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,now%28%29)[\[43\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=except%20Exception%20as%20e%3A%20,update_member_execution%28%20execution_id%3Dexecution_id%2C%20status%3D%27failed%27%2C%20error%3Dstr%28e%29). Additionally, team building actions and changes are logged. This audit trail serves multiple purposes: \- **Post-mortem analysis:** If a result was incorrect or a tool misbehaved, we can trace which agent/tool was involved. \- **Permission and compliance:** In regulated environments, one might need to show which data or tool an AI accessed (e.g., did it call an external API? did it access a certain domain knowledge?). \- **Training feedback:** Execution logs can be mined to improve agent strategies (though this is future work).

**Implementation:** Logging is done in both the relational database and Neo4j graph: \- In **Postgres**, the member\_executions table captures each execution event[\[94\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1140%20CREATE%20TABLE,member_id). Fields likely include an execution\_id, team\_id, member\_id, tool\_name, timestamp (started\_at, completed\_at), status (success/failed), result or error message (possibly truncated or stored as JSON). This table can be queried to get a history of what each agent has done. There may also be a user\_executions table to log actions initiated by end-users or by the system on behalf of a user (mentioned in permission control context; see Section 8). For team building, tables like team\_build\_sessions and team\_build\_steps log the sequence of steps and their status[\[95\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20team_build_sessions%20,tenant_id)[\[96\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20team_build_steps%20,step_name), which is part of audit for how a team configuration was created. \- In **Neo4j**, relationships such as (:Member)-\[:EXECUTED {timestamp, success, execution\_time\_ms}\]-\>(:Tool) are created for each execution[\[97\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28m%3AMember%29,tool%3ATool). This allows graph queries like “find all tools used by member X in the last week”[\[98\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Track%20member%20execution%20history,1%20Team%20Building%20Tests%20python). Also, (:Member)-\[:SUBMITTED\]-\>(:Query) and (:Query)-\[:USED\_CHUNK\]-\>(:Chunk) relationships track which queries were asked and what knowledge chunks were used to answer them[\[99\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Query%20tracking%20CREATE%20,). These graph relationships can be useful for analytics (e.g., to find frequently used knowledge or to see network of interactions between agents and data).

Additionally, **application logs** (via a logging library) record events. For example, structlog is used in the code, so actions like no eligible member found will log a warning event no\_eligible\_members with context[\[100\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=eligible_members). These logs (text logs) complement the structured DB logs for real-time debugging.

**Audit Access:** There could be an API or admin UI to retrieve these logs. TAO could expose endpoints like GET /teams/{team\_id}/executions or similar to list execution records. This is not detailed here but is a logical extension.

**Standalone ROI:** The logging subsystem ensures **accountability**. Even as a separate piece, the idea of member-specific execution logging can be reused in any system where actions by different roles need tracking. It provides immediate value by enabling audit trails and supporting permission enforcement (by checking logs, one can verify no unauthorized tool use occurred)[\[101\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,tool%20permission%20enforcement).

**Testing Plan:**  
\- *Database Logging:* In unit tests for orchestrator/member manager, verify that when execute\_with\_member runs successfully, a record is inserted in member\_executions with status 'success'. This might involve mocking the Postgres client’s log\_member\_execution to return an ID and update\_member\_execution to capture input – then assert it was called with status='success' or 'failed' accordingly[\[34\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,now%28%29)[\[43\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=except%20Exception%20as%20e%3A%20,update_member_execution%28%20execution_id%3Dexecution_id%2C%20status%3D%27failed%27%2C%20error%3Dstr%28e%29). We can also simulate failures and ensure the log updates with status 'failed' and an error message.  
\- *Graph Logging:* If Neo4j logging is done in parallel, we would test the Cypher queries or functions that create the EXECUTED relationships. For example, after a tool execution in a test, query Neo4j (via Neo4jClient) for (m:Member {id: member\_001})-\[:EXECUTED\]-\>(t:Tool {name: toolX}) and expect to find a relationship with success=true. Since Neo4j might be harder to verify in unit tests, this could be part of an integration test with a running Neo4j (or using an embedded test DB).  
\- *Log Volume & Performance:* Not a typical unit test, but we should consider that logging happens on every execution. Ensure the member\_executions table has appropriate indexes (e.g., index by team\_id or member\_id)[\[102\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20INDEX%20idx_teams_status%20ON%20agent_teams,tenant_id) so that queries don't slow down as it grows. Migration tests could ensure those indexes exist. Also, test that the logging doesn’t block the main flow – e.g., in an async context, writing to DB can happen quickly. If needed, the design might offload logging to a background task queue for performance; however, simplicity in MVP likely just writes synchronously.

By verifying logs thoroughly, we can trust that TAO’s operations are transparent and traceable, which is crucial for a system orchestrating autonomous agents.

## 4\. Data Persistence Schemas

TAO utilizes **PostgreSQL** for structured data (configurations, state, and logs) and **Neo4j** for graph data (domains, relationships between entities). Below are the schemas specific to TAO’s components:

### 4.1 PostgreSQL Schema

The Postgres schema covers multi-tenant configurations, team definitions, tool registry, and execution logs. Important tables include:

* **tenants:** Tenants using the system (each tenant corresponds to an organization or context).

* CREATE TABLE tenants (  
      tenant\_id VARCHAR(255) PRIMARY KEY,  
      tenant\_type VARCHAR(50) CHECK (tenant\_type IN ('Company','NGO','School','Government')),  
      name TEXT,  
      config JSONB    \-- Other tenant-specific settings  
  );

* **tools:** Registry of available tools (global across tenants, or optionally tenant-scoped).

* CREATE TABLE tools (  
      tool\_id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
      tool\_name VARCHAR(255) UNIQUE NOT NULL,  
      description TEXT,  
      mcp\_mode VARCHAR(50),      \-- e.g., 'http', 'grpc', 'stdio'  
      endpoint TEXT,             \-- URL or command to invoke the tool  
      input\_schema JSONB,  
      output\_schema JSONB  
  );

* Each tool has a unique name and details on how to call it. mcp\_mode and endpoint define integration details[\[72\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%3D%3D%3D%20MCP%20Configuration%20%3D%3D%3D%20MCP_SERVER_HOST%3D0,MCP_SERVER_PORT%3D8100%20MCP_TRANSPORT_MODE%3Dstdio).

* **agent\_teams:** Records for each agent team configuration.

* CREATE TABLE agent\_teams (  
      team\_id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
      tenant\_id VARCHAR(255) REFERENCES tenants(tenant\_id),  
      team\_name VARCHAR(255) NOT NULL,  
      status VARCHAR(50) DEFAULT 'active',   \-- e.g., 'active', 'building', 'archived'  
      description TEXT,  
      created\_at TIMESTAMP DEFAULT NOW(),  
      updated\_at TIMESTAMP DEFAULT NOW()  
  );

* A team references its tenant and has a status (which could indicate if it’s in creation, ready, etc.).

* **agent\_team\_members:** The agents (members) belonging to teams.

* CREATE TABLE agent\_team\_members (  
      member\_id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
      team\_id UUID REFERENCES agent\_teams(team\_id) ON DELETE CASCADE,  
      member\_name VARCHAR(255) NOT NULL,  
      member\_role VARCHAR(100),  
      persona TEXT,         \-- e.g., a description or prompt context for the agent  
      created\_at TIMESTAMP DEFAULT NOW()  
  );

* On deletion of a team, its members are deleted too. Each member can have a role (like “Analyst”, “Researcher”) and an optional persona that influences its behavior.

* **member\_tools:** Many-to-many relationship between members and tools to define permissions.

* CREATE TABLE member\_tools (  
      member\_id UUID REFERENCES agent\_team\_members(member\_id) ON DELETE CASCADE,  
      tool\_id UUID REFERENCES tools(tool\_id),  
      permission\_level VARCHAR(50) DEFAULT 'execute',  \-- e.g., execute/configure/admin  
      PRIMARY KEY (member\_id, tool\_id)  
  );

* This table lists exactly which tools each member can use[\[103\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1060%20CREATE%20TABLE,execute%2C%20configure%2C%20admin). permission\_level could be extended in future (for now, likely just 'execute' rights). The primary key ensures no duplicate entries, and cascading delete means if a member is removed, their permissions go too.

* **team\_domains:** (If using domain access control in relational model) Links teams to domains they have access to.

* CREATE TABLE team\_domains (  
      team\_id UUID REFERENCES agent\_teams(team\_id) ON DELETE CASCADE,  
      domain\_id VARCHAR(255) REFERENCES domains(domain\_id),  
      access\_level VARCHAR(50) DEFAULT 'read',  
      PRIMARY KEY (team\_id, domain\_id)  
  );

* This might mirror the graph relationships in SQL for quick checks. domains table (not fully shown above) holds domain knowledge identifiers configured per tenant[\[104\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1014%20CREATE%20TABLE,domain_id).

* **team\_build\_sessions:** Tracks ongoing or completed team-building wizard sessions.

* CREATE TABLE team\_build\_sessions (  
      session\_id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
      tenant\_id VARCHAR(255) REFERENCES tenants(tenant\_id),  
      team\_id UUID REFERENCES agent\_teams(team\_id),  
      status VARCHAR(20) NOT NULL,      \-- 'active','paused','completed'  
      current\_step VARCHAR(100),       \-- reference to step definitions  
      started\_at TIMESTAMP,  
      updated\_at TIMESTAMP  
  );

* When a user is interactively building a team (via TAB), a session is created to save progress. If paused, it can be resumed by looking at current\_step and status.

* **team\_build\_steps:** Stores each step taken in a build session (audit of team creation process).

* CREATE TABLE team\_build\_steps (  
      step\_id BIGSERIAL PRIMARY KEY,  
      session\_id UUID REFERENCES team\_build\_sessions(session\_id) ON DELETE CASCADE,  
      step\_name VARCHAR(100),  
      user\_input JSONB,  
      completed\_at TIMESTAMP  
  );

* Each record is one completed step (for example, “CHOOSE\_DOMAIN” step with the domains user selected). There is also likely a **team\_build\_step\_definitions** table enumerating possible steps and their order[\[96\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20team_build_steps%20,step_name).

* **member\_executions:** Logs of tool executions by members (audit trail)[\[105\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1140%20CREATE%20TABLE,member_id).

* CREATE TABLE member\_executions (  
      execution\_id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
      team\_id UUID REFERENCES agent\_teams(team\_id),  
      member\_id UUID REFERENCES agent\_team\_members(member\_id),  
      tool\_name VARCHAR(255),  
      started\_at TIMESTAMP,  
      finished\_at TIMESTAMP,  
      status VARCHAR(10),    \-- 'success' or 'failed'  
      result\_summary TEXT,   \-- optional short summary or error message  
      full\_result JSONB      \-- optional detailed result payload  
  );

* The member\_id and team\_id let us know who did what. We store timestamps and status, and possibly some details of the result (like output or an error trace). This table can grow large, so indexes on member\_id and team\_id are set for query performance[\[102\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20INDEX%20idx_teams_status%20ON%20agent_teams,tenant_id).

* **(Optional) user\_executions:** While not specified explicitly in the prior spec, the merge notes suggest tracking user actions as well[\[106\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=should%20have%20all%20the%20CRUD,execute%20an%20agent%20team%2C%20etc). If implemented, a table here would log when a user triggers builds or executions, along with their user\_id and action type.

* **chunks & related tables:** If TAO handles domain knowledge pieces, tables like documents and chunks might exist to store text chunks and their association to domains[\[107\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20chunks%20,tenant_id). E.g., chunks(chunk\_id PK, document\_id, tenant\_id, content TEXT, status, etc.) and linking tables to domains and queries:

* chunk\_domains(chunk\_id, domain\_id, relevance\_score) linking chunk to one or multiple domain tags[\[108\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1182%20CREATE%20TABLE,domain_id%29%2C%20relevance_score%20FLOAT).

* Possibly queries table for user queries and linking to chunks used (though query tracking is mainly in graph).

All foreign keys use proper constraints to maintain integrity. The database also has indexes for performance, e.g., on team status, member-team relationships, etc.[\[102\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20INDEX%20idx_teams_status%20ON%20agent_teams,tenant_id). For example, CREATE INDEX idx\_executions\_team ON member\_executions(team\_id) helps when retrieving all executions for a team.

**Migrations:** TAO includes migration scripts (see scripts/migrate\_db.sh) to set up or update these schemas[\[109\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=PostgreSQL%20migrations%20echo%20,d%20%24POSTGRES_DB%20%3C%3C%20%27EOF). The migration script drops some old tables (like deprecated \*\_config tables from earlier versions) and runs SQL files (001\_create\_base\_tables.sql, etc.) to build the new schema.

### 4.2 Neo4j Graph Schema

Neo4j is used to store the **domain knowledge graph** and relationships between tenants, teams, members, tools, queries, and knowledge chunks in a highly flexible way. This complements the relational data by efficiently representing hierarchical and many-to-many relationships that are cumbersome in SQL.

Key **Node labels** and their properties: \- **Tenant (id)** – Represents a tenant; unique constraint on Tenant.id[\[110\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Constraints%20CREATE%20CONSTRAINT%20tenant_unique,id%20IS%20UNIQUE). Could have properties like name, type, etc. \- **Domain (id)** – A domain of knowledge; unique constraint on Domain.id[\[111\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20domain_unique%20IF%20NOT,id%20IS%20UNIQUE). Properties: name, type (categorization), path (a hierarchical path string), level (depth in hierarchy), tenant\_id (to relate to Tenant), etc.[\[112\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Generic%20Domain%20nodes%20,name%3A%20%27Customer%20Portal%27%2C%20type%3A%20%27Application)[\[113\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28d%3ADomain%20,custom_field%3A%20%27value%27%7D%2C%20created_at%3A%20datetime). The flexible “path” allows nested domains (e.g. "/apps/customer\_portal/auth") and an explicit parent\_id to form a tree[\[114\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28d2%3ADomain%20,Feature%27%2C%20path%3A%20%27%2Fapps%2Fcustomer_portal%2Fauth%27%2C%20level%3A%202). \- **Team (id)** – An agent team; unique constraint on Team.id[\[115\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20team_unique%20IF%20NOT,id%20IS%20UNIQUE). Properties: name, status, tenant\_id, etc.[\[116\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Team%20node%20CREATE%20,). \- **Member (id)** – An agent (team member); unique constraint on Member.id[\[117\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20member_unique%20IF%20NOT,id%20IS%20UNIQUE). Properties: name, role, persona, team\_id, etc.[\[118\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Member%20nodes%20CREATE%20,oriented%20problem%20solver%27%2C%20created_at%3A%20datetime). \- **Tool (id)** – A tool; unique constraint on Tool.id[\[119\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20chunk_unique%20IF%20NOT,id%20IS%20UNIQUE). Properties: name, type (category like "search", "api"), description, etc.[\[120\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Tool%20nodes%20CREATE%20,). \- **Chunk (id)** – A chunk of knowledge or document piece; unique constraint Chunk.id[\[121\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20member_unique%20IF%20NOT,id%20IS%20UNIQUE). Properties: text\_snippet, position (if part of a doc), tenant\_id, created\_by\_team\_id, etc., plus some dynamic properties for learning (e.g., agent\_learn\_deprecated\_score, status)[\[122\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Chunk%20nodes%20CREATE%20,0)[\[123\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=text_snippet%3A%20%27First%20100%20chars%20of,current%27%2C%20status%3A%20%27active%27%2C%20created_at%3A%20datetime). \- **Query (id)** – (Optional node) A user or agent query. Could store the text of the question and timestamp[\[124\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Query%20tracking%20CREATE%20,).

Key **Relationship types**: \- **Tenant relationships:**  
\- (:Tenant)-\[:OWNS\_DOMAIN\]-\>(:Domain) – Tenant owns top-level domains[\[125\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Tenant%20relationships%20CREATE%20%28t%3ATenant%29,d%3ADomain).  
\- (:Tenant)-\[:HAS\_TEAM\]-\>(:Team) – Tenant has teams[\[126\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28t%3ATenant%29,team%3ATeam). This ties teams to their tenant.  
\- **Domain hierarchy:**  
\- (:Domain)-\[:HAS\_CHILD\]-\>(:Domain) – Links parent domain to child domain (forming a tree)[\[127\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Hierarchical%20domain%20structure%20,child%3ADomain). HAS\_CHILD may have a property like relationship\_type: 'hierarchical'[\[128\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Hierarchical%20domain%20structure%20,child%3ADomain). This allows arbitrary depth hierarchies (unlimited domain depth is supported[\[129\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,20)).  
\- **Team relationships:**  
\- (:Team)-\[:HAS\_MEMBER {joined\_at}\]-\>(:Member) – A team has members (with a timestamp of joining)[\[130\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Team%20relationships%20CREATE%20%28team%3ATeam%29,m%3AMember).  
\- (:Team)-\[:ACCESSES\_DOMAIN {access\_level}\]-\>(:Domain) – Team is allowed to access Domain (with read/write access level)[\[131\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28team%3ATeam%29,d%3ADomain). This is the graph counterpart to the SQL team\_domains and is useful for quickly finding which knowledge domains a team should consider.  
\- **Member relationships:**  
\- (:Member)-\[:CAN\_USE {permission\_level}\]-\>(:Tool) – Member can use a tool (with permission level, usually 'execute')[\[132\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Member%20relationships%20CREATE%20%28m%3AMember%29,tool%3ATool). This encodes each member’s tool permissions in the graph for quick queries (like “what tools can this role use?”)[\[133\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Find%20all%20tools%20a,tool%3ATool%29%20RETURN%20tool).  
\- (:Member)-\[:EXECUTED {timestamp, success, execution\_time\_ms}\]-\>(:Tool) – Member executed a tool at a certain time with success/failure[\[97\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28m%3AMember%29,tool%3ATool). Over time, a member node can have many EXECUTED relationships to various tools, one per execution event. These serve as a detailed audit trail in the graph.  
\- **Chunk relationships:**  
\- (:Chunk)-\[:BELONGS\_TO {relevance\_score}\]-\>(:Domain) – A chunk is primarily about a domain (score could denote confidence)[\[134\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Chunk%20relationships%20,d%3ADomain).  
\- (:Chunk)-\[:RELATED\_TO {relevance\_score}\]-\>(:Domain) – A chunk is also related to another domain (we allow multiple domain tagging)[\[135\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Multiple%20domain%20connections%20for,d2%3ADomain).  
\- (:Chunk)-\[:REFERENCES {reference\_type}\]-\>(:Domain) – Perhaps the chunk explicitly references a domain or concept (like an index)[\[136\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Multiple%20domain%20connections%20for,d2%3ADomain).  
These multiple relationship types indicate different strengths or types of association, giving flexibility in knowledge modeling (e.g., a paragraph can belong to one topic but also touch on others).  
\- **Query/Usage relationships:**  
\- (:Member)-\[:SUBMITTED\]-\>(:Query) – A member (or user acting via a member/agent) posed a query[\[137\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28m%3AMember%29,c%3AChunk).  
\- (:Query)-\[:USED\_CHUNK {score}\]-\>(:Chunk) – The query’s answer used a certain knowledge chunk (with some relevance score)[\[138\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28m%3AMember%29,c%3AChunk). This is useful for traceability of what knowledge was used to answer a query.

Constraints and indexes exist to ensure uniqueness and optimize lookups, e.g., an index on Domain.path for quick retrieval by path, and on Team.status to find active teams efficiently[\[139\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Indexes%20CREATE%20INDEX%20domain_type,type)[\[140\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20INDEX%20team_status%20IF%20NOT,status).

**Example Query:** To illustrate the graph’s utility, consider some Cypher queries: \- Get all tools a member can use:

MATCH (m:Member {id: $member\_id})-\[:CAN\_USE\]-\>(tool:Tool)  
RETURN tool.name;

\- Find all domains accessible by a team:

MATCH (t:Team {id: $team\_id})-\[:ACCESSES\_DOMAIN\]-\>(d:Domain)  
RETURN d.name;

\- Recent executions by a team’s members:

MATCH (t:Team {id: $team\_id})\<-\[:HAS\_TEAM\]-(ten:Tenant),  
      (t)-\[:HAS\_MEMBER\]-\>(m:Member)-\[e:EXECUTED\]-\>(tool:Tool)  
WHERE e.timestamp \> datetime() \- duration('P7D')  
RETURN m.name, tool.name, e.timestamp, e.success  
ORDER BY e.timestamp DESC;

(This finds all tools executed by members of the team in the last 7 days[\[98\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Track%20member%20execution%20history,1%20Team%20Building%20Tests%20python).)

**Maintenance:** The Neo4j schema is set up via automated migrations as well (see neo4j directory and possibly a cypher-shell script)[\[141\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Neo4j%20migrations%20echo%20,p%20%24NEO4J_PASSWORD)[\[142\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=cypher,NC). The migration ensures constraints and index creation (ASSERT ... IS UNIQUE, CREATE INDEX ...)[\[110\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Constraints%20CREATE%20CONSTRAINT%20tenant_unique,id%20IS%20UNIQUE)[\[143\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Indexes%20CREATE%20INDEX%20domain_type,type). After initial setup, TAO may also programmatically create nodes/relationships as needed when teams or domains are added. For example, when a new team is created, TAO could create a (Team) node for it and link it to the Tenant node and relevant Domain nodes (if any domain contexts are chosen for that team).

**Testing:** We don’t typically unit-test a schema, but we verify through integration tests that: \- Creating a new team via TAO results in the appropriate Neo4j nodes (Team, Member nodes and relationships). \- Domains created in Postgres (if also mirrored) appear as Domain nodes. \- Querying the graph returns expected relationships (like the test in section 7 for domain and permission queries, and ensuring performance of these queries is acceptable – e.g., domain traversal query returns quickly even with depth).

Overall, the combination of Postgres and Neo4j gives TAO both robust transactional storage and flexible graph insights, fulfilling the design of a generic, multi-domain knowledge support[\[144\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Phase%203%3A%20Domain%20Hierarchy%20%26,can%20belong%20to%20multiple%20domains)[\[129\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,20).

## 5\. Deployment and Environment Setup

TAO is built to run in a cloud-native environment using Docker containers and is configured via environment variables. This section outlines how to set up TAO along with its dependent services, manage configuration (.env files), and ensure health and stability of the deployment.

### 5.1 Docker Composition

A Docker Compose setup (see provided docker-compose-master.yml) defines all necessary services for a full TAO deployment[\[145\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=%23%20%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%20DATABASES%20%28CPU,postgres_user%20POSTGRES_PASSWORD%3A%20postgres_pass%20PGDATA%3A%20%2Fvar%2Flib%2Fpostgresql%2Fdata%2Fpgdata). The main components include:

* **PostgreSQL Database:** Stores TAO’s relational data. The compose file uses the official postgres:16-alpine image[\[146\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=postgres%3A%20image%3A%20postgres%3A16,Performance%20tuning%20for%2064GB%20RAM). It sets environment variables for database name, user, and password (e.g., POSTGRES\_DB=ta\_v8, POSTGRES\_USER=postgres\_user)[\[147\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=environment%3A%20POSTGRES_DB%3A%20ta_v8%20POSTGRES_USER%3A%20postgres_user,200%20POSTGRES_SHARED_BUFFERS%3A%204GB%20POSTGRES_EFFECTIVE_CACHE_SIZE%3A%2016GB). Data is persisted in a volume (./data/postgres). Health check is configured using pg\_isready to ensure Postgres is up before depending services start[\[85\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=memory%3A%208G%20healthcheck%3A%20test%3A%20%5B%22CMD,30s%20timeout%3A%2010s%20retries%3A%203). Performance-related env (shared buffers, etc.) is tuned for a high-end host, but defaults can be used or scaled down for smaller installations.

* **Qdrant Vector DB:** Provides semantic search capabilities. Compose pulls qdrant/qdrant:latest[\[91\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=qdrant%3A%20image%3A%20qdrant%2Fqdrant%3Alatest%20container_name%3A%20qdrant,.%2Fdata%2Fqdrant%3A%2Fqdrant%2Fstorage%3Az%20environment), mapping ports 6333 (HTTP) and 6334 (gRPC)[\[148\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=container_name%3A%20qdrant%20ports%3A%20,environment%3A%20QDRANT__SERVICE__HTTP_PORT%3A%206333%20QDRANT__SERVICE__GRPC_PORT%3A%206334). Data is stored in ./data/qdrant. A health check uses curl http://localhost:6333/health to verify Qdrant’s readiness[\[149\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=cpus%3A%20%274%27%20memory%3A%208G%20healthcheck%3A,10s%20retries%3A%205%20start_period%3A%2030s).

* **Neo4j Graph DB:** The graph database for domain and relationship data. The compose builds a custom image (or uses neo4j:5-community) with APOC plugin enabled[\[150\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=neo4j%3A%20build%3A%20context%3A%20,%227687%3A7687%22%20volumes)[\[151\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=,RAM%20NEO4J_server_memory_heap_initial__size%3A%204G%20NEO4J_server_memory_heap_max__size%3A%208G). Ports 7474 (HTTP UI) and 7687 (bolt protocol) are exposed[\[152\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=container_name%3A%20neo4j%20ports%3A%20,.%2Fneo4j%2Fimport%3A%2Fvar%2Flib%2Fneo4j%2Fimport). Environment variables set the auth (NEO4J\_AUTH=neo4j/\<password\>)[\[151\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=,RAM%20NEO4J_server_memory_heap_initial__size%3A%204G%20NEO4J_server_memory_heap_max__size%3A%208G) and configure memory usage for production loads. The health check hits http://localhost:7474 (which returns 200 if the web interface is up, indicating Neo4j started)[\[84\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=cpus%3A%20%274%27%20memory%3A%2020G%20healthcheck%3A,10s%20retries%3A%205%20start_period%3A%2060s).

* **MinIO (Object Storage):** (If used for document storage) A MinIO service is included for storing files and chunks[\[153\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=minio%3A%20image%3A%20minio%2Fminio%3Alatest%20container_name%3A%20minio,environment%3A%20MINIO_ROOT_USER%3A%20minioadmin%20MINIO_ROOT_PASSWORD%3A%20minioadmin). It listens on 9000/9001. TAO might use this for storing large documents or embeddings. In our context, chunks might be kept in Postgres or flat files; MinIO provides S3-compatible storage if needed for larger data.

* **Ollama (LLM Service):** This is the on-prem LLM server (Ollama) hosting models (like GPT-3, GPT-4, or the mentioned gpt-oss:20b). The compose builds a ta\_v8\_ollama container[\[154\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=ollama%3A%20build%3A%20context%3A%20,.%2Fmodels%2Follama%3A%2Froot%2F.ollama). It uses GPU (Nvidia runtime, reserving 1 GPU)[\[155\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=resources%3A%20reservations%3A%20devices%3A%20,stopped) and exposes port 11434 for the Ollama API[\[156\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=dockerfile%3A%20,0). TAO’s OllamaClient will call this service for language model completions or agent reasoning steps. A health check calls the /api/version endpoint of Ollama to ensure the model server is responsive[\[157\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=limits%3A%20memory%3A%2032G%20restart%3A%20unless,30s%20timeout%3A%2010s%20retries%3A%203).

* **Embedding Service (BGE-M3):** A custom container for the embedding model is included (called bge-m3). It likely runs a small web server on port 8080 to provide vectorization of text. It uses GPU as well if available. Health check is on http://localhost:8080/health[\[158\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=healthcheck%3A%20test%3A%20%5B%22CMD%22%2C%20%22curl%22%2C%20%22,10s%20retries%3A%205%20start_period%3A%2060s). TAO’s embedding tool adapter calls this service to get embeddings for semantic search.

* **TAO Service:** The TAO orchestrator itself would typically be another service in the compose, though in the provided snippet we don’t explicitly see it. It’s possible the TAO container (and similarly TAB/TAE) are defined elsewhere or to be added. We expect a service definition like:

* tao:  
    build: .  
    container\_name: tao  
    depends\_on:  
      \- postgres  
      \- neo4j  
      \- qdrant  
      \- ollama  
    environment:  
      \- MCP\_TRANSPORT\_MODE=http  
      \- MCP\_SERVER\_PORT=8100  
      \- POSTGRES\_HOST=postgres  
      \- POSTGRES\_USER=postgres\_user  
      \- POSTGRES\_PASSWORD=postgres\_pass  
      \- POSTGRES\_DB=ta\_v8  
      \- NEO4J\_URI=bolt://neo4j:7687  
      \- NEO4J\_USER=neo4j  
      \- NEO4J\_PASSWORD=\<password\>  
      \- QDRANT\_HOST=qdrant  
      \- OLLAMA\_URL=http://ollama:11434  
      \- WIZARD\_MODE\_ENABLED=true  
      \- EXPRESS\_MODE\_ENABLED=true  
    ports:  
      \- "8100:8100"  
    depends\_on:  
      postgres:  
        condition: service\_healthy  
      neo4j:  
        condition: service\_healthy  
      qdrant:  
        condition: service\_healthy  
    healthcheck:  
      test: \["CMD", "curl", "-f", "http://localhost:8100/health"\]  
      interval: 30s  
      retries: 3

* *(This is a representative example – actual compose may vary)*. This ensures TAO only starts after the databases are healthy[\[159\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1851%20neo4j%3A%20condition%3A,service_healthy%20qdrant%3A%20condition%3A%20service_started). The environment variables connect TAO to Postgres and Neo4j, and enable or disable features like Wizard mode (TAB integration for chat interface) or Express YAML mode.

* **Monitoring (optional):** A separate docker-compose-monitoring.yml can be used to launch Prometheus and Grafana for observing TAO’s metrics[\[160\]](file://file-DNQKKk5zsqCCwFKjiaKQP2#:~:text=services%3A%20,.%2Fmonitoring%2Fprometheus.yml%3A%2Fetc%2Fprometheus%2Fprometheus.yml). Prometheus would scrape TAO’s metrics endpoint, and Grafana provides dashboards (the compose file provisions a dashboard directory). Node exporter and GPU exporter are included for hardware metrics[\[161\]](file://file-DNQKKk5zsqCCwFKjiaKQP2#:~:text=,%2Fsys%3A%2Fhost%2Fsys%3Aro)[\[162\]](file://file-DNQKKk5zsqCCwFKjiaKQP2#:~:text=nvidia,stopped).

All these services share a network (by default compose network) so they can refer to each other by name (e.g., TAO can reach Postgres at postgres:5432). Data volumes ensure persistence across restarts for databases.

### 5.2 Configuration and .env Integration

The TAO module uses a .env file for configuration so that sensitive information and environment-specific settings are not hard-coded. A file like .env.example is provided with all required keys[\[163\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=3,config.yaml%20%E2%94%9C%E2%94%80%E2%94%80%20.venv). Key environment variables include:

* **Database Config:** POSTGRES\_HOST, POSTGRES\_USER, POSTGRES\_PASSWORD, POSTGRES\_DB – for connecting to Postgres; NEO4J\_URI, NEO4J\_USER, NEO4J\_PASSWORD – for Neo4j connection (e.g., bolt://neo4j:7687)[\[164\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=NEO4J_URI%3Dbolt%3A%2F%2Flocalhost%3A7687%20NEO4J_USER%3Dneo4j%20NEO4J_PASSWORD%3DpJnssz3khcLtn6T).

* **MCP Server Config:** MCP\_SERVER\_HOST (often 0.0.0.0 to listen on all interfaces), MCP\_SERVER\_PORT (e.g. 8100\) and MCP\_TRANSPORT\_MODE (http or stdio)[\[72\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%3D%3D%3D%20MCP%20Configuration%20%3D%3D%3D%20MCP_SERVER_HOST%3D0,MCP_SERVER_PORT%3D8100%20MCP_TRANSPORT_MODE%3Dstdio). For production, HTTP is common; stdio might be used if TAO were launched as a subprocess tool.

* **Feature Toggles:** WIZARD\_MODE\_ENABLED, EXPRESS\_MODE\_ENABLED booleans to turn on the conversational team builder and YAML import paths[\[165\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1840%20,EXPRESS_MODE_ENABLED%3Dtrue%20env_file). TAO might disable these if, for example, running purely as an executor without build capabilities.

* **External Service URLs:** e.g., EMBEDDING\_SERVICE\_URL=http://bge-m3:8080 to direct the embedding adapter where to call[\[166\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L652%20EMBEDDING_SERVICE_URL%3Dhttp%3A%2F%2Flocalhost%3A8000%29,Deployment%20%26%20Environment), or credentials for external APIs (like if a tool requires an API key, it might be set via env and read by the adapter).

* **Logging/Debug:** LOG\_LEVEL or similar to control verbosity. Not explicitly shown above, but likely present in defaults.

The .env file is loaded by TAO on startup (the code likely uses python-dotenv or simply reads os.environ for those values). The Docker Compose references this .env to populate container environment as well (via env\_file or direct mappings)[\[167\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=HEALTHCHECK%20,).

**Secrets Management:** Passwords (like the Neo4j password shown as pJnssz3khcLtn6T in examples[\[164\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=NEO4J_URI%3Dbolt%3A%2F%2Flocalhost%3A7687%20NEO4J_USER%3Dneo4j%20NEO4J_PASSWORD%3DpJnssz3khcLtn6T)) should ideally be stored in .env and not in version control. The example shows it plainly, but in a real deployment, one would change these to secure values. Compose allows using Docker secrets or environment files to avoid hardcoding them in the yaml.

### 5.3 Health Check Endpoints and Service Monitoring

TAO provides a basic health endpoint at GET /health (or /api/health depending on routing) for liveness checks. This returns a simple status (e.g., {"status": "ok"}) and possibly build/version info. The Docker Compose healthcheck for TAO calls this endpoint to auto-restart the container if it becomes unresponsive[\[159\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1851%20neo4j%3A%20condition%3A,service_healthy%20qdrant%3A%20condition%3A%20service_started). Additionally, TAO might check the health of dependencies: for example, an extended health response could include subchecks (database connectivity, etc.), but generally if the process is up and responding, we consider it healthy.

For deeper monitoring, Prometheus metrics (section 6\) are exposed. In the docker-compose-monitoring.yml, Prometheus is configured to scrape targets like the TAO service on a certain port (if TAO’s metrics are on port 9090 inside container, Prometheus can be set to scrape tao:9090)[\[168\]](file://file-DNQKKk5zsqCCwFKjiaKQP2#:~:text=prometheus%3A%20image%3A%20prom%2Fprometheus%3Alatest%20container_name%3A%20prometheus,prometheus_data%3A%2Fprometheus%20command). Grafana dashboards then visualize these metrics.

**Startup Order and Readiness:** The compose uses depends\_on.condition to ensure ordering – e.g., TAO waits for service\_healthy of the databases[\[159\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1851%20neo4j%3A%20condition%3A,service_healthy%20qdrant%3A%20condition%3A%20service_started). However, orchestrator might still need to handle transient unavailability (like if Neo4j is up but still warming up). Typically, TAO on startup will try to connect to Postgres/Neo4j; if it fails, it might retry or exit with error. Kubernetes (if used) would also rely on readiness probes (like hitting /health) to add TAO to the load balancer.

**Docker Images:** TAO’s Dockerfile (not shown) likely extends a Python base image, installs required dependencies (fastmcp, asyncpg, neo4j, etc.), and copies the TAO source. It would then set entrypoint to run the MCP server (e.g., python \-m tao.core.server or similar).

**Deployment Note:** For a production deployment, all these containers would be orchestrated via Kubernetes or Docker Swarm. The provided compose is mainly for development or proof-of-concept single-host deployment (with some high resource allocations as examples).

## 6\. Monitoring and Metrics

TAO exports a range of **Prometheus metrics** to facilitate monitoring of the system’s performance and usage. A built-in metrics server (via the prometheus\_client library) is started on a configurable port (default 9090\) to serve these metrics[\[169\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=def%20start_metrics_server,Modular%20Deliverables%20%26%20ROI). Prometheus can scrape this endpoint to collect metrics over time.

Key metric categories and examples:

* **System Info:**

* tao\_system (Prometheus Info metric) – contains static info tags like the TAO version and configured embedding model[\[170\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=System%20info%20system_info%20%3D%20Info,m3%27). For example, tao\_system{version="0.3.0", embedding\_model="BAAI/bge-m3"} 1 indicates the current software version and embedding model in use.

* **Team and Member Metrics:**

* tao\_active\_teams (Gauge) – Number of active teams currently in the system[\[171\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Team%20metrics%20active_teams%20%3D%20Gauge,status%27%5D). This could be set on startup or updated when teams are added/removed.

* tao\_team\_members (Gauge) – Total count of team members (agents) across all teams[\[171\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Team%20metrics%20active_teams%20%3D%20Gauge,status%27%5D).

* tao\_build\_sessions{status} (Gauge) – Count of team build sessions by status[\[172\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=active_teams%20%3D%20Gauge,status%27%5D). Labeled by status (e.g. active, paused, completed). This shows how many build processes are ongoing or awaiting input.

* **Execution Metrics:**

* tao\_member\_executions\_total{team\_id, member\_id, tool\_name, status} (Counter) – Cumulative count of tool executions by members[\[173\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Member%20metrics%20member_executions%20%3D%20Counter,). It’s labeled with team, member, tool, and status (success/failed). This allows slicing metrics per team or tool. For example, one can monitor how often a particular tool is used and how often it fails.

* tao\_member\_execution\_seconds{member\_id, tool\_name} (Histogram) – Distribution of execution times of tools per member[\[174\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=member_execution_time%20%3D%20Histogram,buc%20Retry%20A)[\[175\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=python%20src%2Ftao%2Futils%2Fmetrics.py%20%28continued%29%20buckets%3D,). This helps identify performance outliers (e.g., certain tools consistently take longer). The histogram buckets might include, for example, \[0.01s, 0.05s, 0.1s, 0.25s, ... up to 5s\] for tool execution duration.

* tao\_permission\_denials\_total{team\_id, member\_id, tool\_name} (Counter) – Counts how many times a member attempted to use a tool they’re not allowed to[\[176\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=permission_denials%20%3D%20Counter,team_id%27%2C%20%27member_id%27%2C%20%27tool_name%27%5D). A spike in this might indicate misconfiguration or an agent frequently trying disallowed actions.

* **Tool and Permission Metrics:**

* tao\_tool\_permissions\_total{tool\_name} (Gauge) – The total number of permissions granted for each tool[\[177\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Tool%20permission%20metrics%20tool_permissions%20%3D,tool_name%27%5D). Essentially how many members (across all teams) have access to a given tool. This indicates tool popularity or breadth of usage.

* (We might also track total number of tools available, but that can be inferred from tool\_permissions by summing or just using the tools table count).

* **Build Process Metrics:**

* tao\_build\_steps\_completed\_total{step\_name, mode} (Counter) – Counts how many times each build step was completed, labeled by step and mode (wizard vs express YAML)[\[178\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Build%20workflow%20metrics%20build_steps_completed%20%3D,mode%3A%20wizard%2Fexpress). This shows usage of the team builder. For instance, how often the YAML import is used versus the interactive wizard (mode), and if certain steps are frequently revisited.

* tao\_build\_session\_duration\_seconds (Histogram) – Distribution of durations of team build sessions (from start to completion)[\[179\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=build_session_duration%20%3D%20Histogram,). Buckets might be \[1min, 5min, 10min, 30min, 1h, 2h\]. This can reveal how long it typically takes to configure a team and if pause/resume is being utilized (very long durations likely mean sessions were paused).

* **Domain and Knowledge Metrics:**

* tao\_domain\_hierarchy\_depth (Histogram) – Distribution of domain hierarchy depths across tenants[\[180\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Domain%20metrics%20domain_hierarchy_depth%20%3D%20Histogram,). Buckets 1-10 (levels). Useful to see how complex the domain taxonomy is (e.g., most tenants have 3-level deep domain trees).

* tao\_chunks\_per\_domain (Histogram) – Distribution of how many chunks (knowledge pieces) each domain has[\[181\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Domain%20metrics%20domain_hierarchy_depth%20%3D%20Histogram,). Buckets ranging from 10 to 5000\. This helps identify domains that are very large (lots of content) versus small ones, which can correlate with performance of search or agent focusing.

All metrics use the prefix tao\_ to namespace them to this module. The metrics endpoint structure typically is plain text listing each metric and its labels. For example:

\# HELP tao\_member\_executions\_total Member tool executions  
\# TYPE tao\_member\_executions\_total counter  
tao\_member\_executions\_total{team\_id="team\_001", member\_id="member\_001", tool\_name="vector\_search", status="success"} 5  
tao\_member\_executions\_total{team\_id="team\_001", member\_id="member\_002", tool\_name="external\_api", status="failed"} 2  
...  
\# HELP tao\_member\_execution\_seconds Member execution time  
\# TYPE tao\_member\_execution\_seconds histogram  
tao\_member\_execution\_seconds\_bucket{member\_id="member\_001", tool\_name="vector\_search", le="0.05"} 3  
...  
tao\_member\_execution\_seconds\_sum{member\_id="member\_001", tool\_name="vector\_search"} 0.15  
tao\_member\_execution\_seconds\_count{member\_id="member\_001", tool\_name="vector\_search"} 5

*(The above is illustrative of how Prometheus text format looks with counters and histograms.)*

**Accessing Metrics:** If TAO’s metrics server runs on port 9090 inside the container, one must map it or allow Prometheus to scrape it internally. In our monitoring setup, Prometheus can scrape the container directly on the compose network. For example, in prometheus.yml, a job might be defined as:

\- job\_name: 'tao'  
  scrape\_interval: 15s  
  static\_configs:  
    \- targets: \['tao:9090'\]

This would instruct Prometheus to fetch metrics from TAO every 15 seconds. Grafana can then be used to plot these metrics over time. Pre-built dashboards could include charts like “Tool Usage by Team”, “Average Tool Execution Time”, “Active Build Sessions”, etc., leveraging the above metrics.

**Testing Metrics:** One can test that metrics are exposed by simply curling the metrics endpoint (curl http://localhost:9090/metrics from within container or appropriate port mapping) and checking that all the metric names appear. Unit tests can also verify that certain actions increment metrics: \- After executing a tool in a test, check that member\_executions\_total counter increased (Prom client allows inspecting counters in-process). \- After creating a team in a test, check that tao\_active\_teams gauge was incremented (if implemented to do so). \- Use the prometheus\_client registry in tests to get current metric values and assert correctness.

Metrics and monitoring ensure that when TAO is running in production, developers and operators can observe its behavior and health in real time, facilitating rapid response to issues and data-driven optimization decisions.

## 7\. Integration with TAB and TAE

TAO operates within the larger **Team Agent Platform**, alongside the Team Agent Builder (TAB) and Team Agent Executor (TAE) modules. It is designed to function independently, but also to interface smoothly with these modules. This section describes how TAO integrates with TAB and TAE, including input/output hand-offs and collaborative workflows.

### 7.1 Team Agent Builder (TAB) → TAO Integration

**TAB’s Role:** TAB is responsible for guiding users (typically tenant admins or experts) through the process of creating and configuring a multi-agent team. It provides a user-friendly interface, possibly a chat-based wizard or a form, to specify the team’s details – such as the problem domain, the roles needed, and the tools each agent should have[\[182\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,the%20tenants%20list%20of%20agents)[\[183\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=etc%20,done%20or%20something%20like%20that). TAB can operate in two modes: \- **Interactive Wizard Mode:** A step-by-step conversational flow where the user answers questions (which domain knowledge to use, what roles the agents should have, etc.).  
\- **Express YAML Mode:** The user supplies a full configuration in a YAML file. The TAB will validate it and only prompt the user if something is missing or incorrect[\[183\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=etc%20,done%20or%20something%20like%20that).

**TAO’s Support:** TAO implements the core logic for building teams, which TAB invokes: \- The **WizardEngine** (LangGraph Wizard) and **BuildWorkflow** in TAO handle the state machine of the interactive build[\[9\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Team%20Building%20WizardEngine,Step%20Processor%5D%20end). TAB’s UI calls TAO’s endpoints or methods to advance to the next step. For example, when a user selects a domain, TAB sends that input to TAO (via TeamBuilder.process\_step), which updates the session state and returns the next prompt or step info[\[184\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,)[\[185\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=). \- The **YAML Validator/Processor** in TAO is used by TAB to validate YAML submissions[\[9\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Team%20Building%20WizardEngine,Step%20Processor%5D%20end). When a user uploads a YAML, TAB passes it to TAO (perhaps an endpoint like POST /teams/validate\_yaml). TAO’s yaml\_processor checks for completeness and consistency, possibly using the **Configuration Validator** tool internally[\[183\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=etc%20,done%20or%20something%20like%20that). If the YAML is valid, TAO can directly create the team from it (persist to DB, set up graph links). If not, TAO returns errors or warnings to TAB, which then enters Wizard mode to fix the issues interactively[\[183\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=etc%20,done%20or%20something%20like%20that)[\[186\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,knowledge%20config%20or%20domain%20knowledge).

**Workflow Coordination:** TAO’s BuildWorkflow (often exposed via a TeamBuilder interface in code) handles session creation, pausing, and resuming: \- When TAB initiates a new team build, it calls TAO to start\_conversational\_build(tenant\_id, user\_id)[\[187\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%40pytest.mark.asyncio%20async%20def%20test_conversational_build_with_pause_resume%28%29%3A%20,postgres%2C%20wizard%2C%20validator). TAO creates a team\_build\_sessions record and returns a session ID and the first step (e.g., “CHOOSE\_DOMAIN”)[\[187\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%40pytest.mark.asyncio%20async%20def%20test_conversational_build_with_pause_resume%28%29%3A%20,postgres%2C%20wizard%2C%20validator)[\[188\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=assert%20session.status%20%3D%3D%20,CHOOSE_DOMAIN). TAB then knows which question to ask the user. \- As the user responds, TAB calls process\_step(session\_id, step, user\_input) on TAO[\[184\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,). TAO validates the input, updates its state, and returns the next step or indicates completion. For example, after choosing domain, next step might be “CHOOSE\_ROLES”[\[189\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=result%20%3D%20await%20builder,). \- The session can be paused (say the user stops mid-way). TAB can call pause\_build(session\_id), TAO marks the session paused[\[190\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=assert%20result). Later, resuming via resume\_build(session\_id) returns the current step so TAB can continue the conversation[\[191\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=). This is crucial for a good UX where the user can come back later without losing progress. \- Once all steps are completed, TAO finalizes the team: inserts team/members into the database, sets status to “active/done”, and closes the session.

TAB essentially is the *front-end*, while TAO is the *back-end logic* for team building. TAO ensures that whether via chat or YAML, the resulting team config is valid and stored. TAO also uses **Semantic Tool Search** during the build: For instance, in wizard mode, if user says "I need an agent to analyze text", TAO could suggest the "embedding" and "vector\_search" tools by querying Qdrant for relevant tools (the description “analyze text” might match those tool descriptions). TAB can present these suggestions to the user (“We recommend adding the vector search tool for this agent, do you agree?”).

**Integration Testing:** TAB and TAO integration is tested by simulating user flows: \- Test that starting a build session through TAB indeed creates a session in TAO and returns the first prompt. \- Follow through an example conversation to ensure at the end a team appears in TAO’s database with expected configuration. \- Test YAML upload path: feed an incomplete YAML to TAB, ensure TAO returns issues, then fix them and see TAO accept it and create the team. \- Test permission: Only users with appropriate rights (maybe “builder” role) can trigger build endpoints – see Section 8\.

### 7.2 Team Agent Executor (TAE) ↔ TAO Integration

**TAE’s Role:** TAE is the runtime executor that actually runs the agents (LLMs) in a team according to the plan orchestrated by TAO. It likely manages the agent loop, handles LLM prompts and responses for each agent, and ensures the agents communicate properly (possibly using a framework like Microsoft Autogen or LangChain)[\[192\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=approach,building%20an%20agent%20team%20and)[\[193\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Multi,of%20Sep%202025%20TBD%20core%2Fmulti_agent_executor). In short, TAE takes a configured team and a user query and produces the answer by simulating the conversation between agents and calling tools as needed.

**Interaction with TAO:** There are a couple of patterns for how TAO and TAE could collaborate: \- **TAO Orchestrator calling TAE:** In this pattern, TAO’s TeamOrchestrator, upon receiving a request to execute a team, will delegate to TAE’s Multi-Agent Team Executor to carry out the multi-turn conversation. TAO passes the team configuration and user query into TAE (possibly via an API or direct module call if in the same process). TAE then handles the detailed agent interactions (LLM prompts for each agent, including when an agent decides to use a tool). When a tool use is required, TAE can call back to TAO to execute that tool via the ToolRegistry (this could be done through the MCP gateway or directly calling TAO’s MemberManager). After all agents have done their part, TAE returns the final answer to TAO, which then returns it to the caller[\[194\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,This%20is%20an)[\[195\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Agent%20Executor%20,52). \- **TAO Integrated Execution:** Alternatively, TAO’s orchestrator might itself loop through agent steps, calling LLMs via OllamaClient and handling logic. However, given a separate module name “Team Agent Executor”, it’s likely that is a distinct component focusing on running the conversation, whereas TAO focuses on the planning and tool integration.

**Tool Calls via MCP:** If an agent in TAE needs to use a tool, one approach is that the agent’s logic issues a request to the MCP server (TAO) to run the tool (with the appropriate member\_id). For example, if using a ReAct style prompt, the agent might output an action like Tool: vector\_search\["query": "keyword"\]. TAE sees that and translates to an API call to TAO (MCP) to execute vector\_search for that team’s member. TAO processes it (checking permissions via MemberManager, calling the adapter) and returns the result. TAE then feeds that result back into the agent’s context as an observation. This handshake continues until agents finish[\[194\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,This%20is%20an)[\[196\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=fine,required%20nodes%20complete%2C%20the%20orchestrator). By using MCP, the interface remains uniform.

**Data Exchange:** TAO provides TAE with: \- The agent prompts/personas and tools (from the team config). TAE uses these to initialize agent instances. For example, TAE might create agent objects (with an initial system prompt constructed from member persona and available tools documentation). TAO can supply those prompts or at least the pieces to build them[\[197\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,for%20example)[\[198\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=with%20its%20persona%2Fprompt%20and%20be,the%20plan%20based%20on%20the). \- The execution DAG or sequence (if orchestrator pre-computed it). For MVP, if we assume a fixed sequence, TAO might simply inform TAE “if you have two agents, run Agent1 then Agent2”. In future, if TAO generates a LangGraph, it could pass a structured plan to TAE. However, since initial flows are straightforward (e.g., always retriever then solver), TAE might not need an explicit graph input beyond just knowing the roles.

TAE returns to TAO: \- The final answer content. \- Possibly intermediate outcomes or any flags (like if the Critic agent determined output was low quality and a loop was done). \- Optionally, logs of the conversation (could be stored via TAO for audit, but that might be heavy; maybe just references).

**Integration Steps:** A typical execution might be: 1\. **TAE triggers execution:** A user query comes into the system (via some UI or API hitting TAE). TAE identifies which team should handle it (maybe user explicitly chose a team, or a routing mechanism did). TAE then calls TAO: TeamOrchestrator.get\_team(team\_id) to get latest config, and possibly TeamOrchestrator.prepare\_execution(team\_id, query) which might set up any context (like retrieving relevant domain info or injecting tenant guidelines). 2\. **Initialize Agents:** TAO or TAE (depending on design) uses the config to initialize agent states. If it’s TAE’s responsibility, it will ask TAO for each member’s persona and tool list to construct prompts. TAO could also supply a formatted “Agent Prompt Template” including any tenant context (like “You are an agent on tenant X, which is about Y domain...”). 3\. **Run Agents:** TAE enters the loop: send user query to Agent1 (through Ollama or another model interface), get response. If response indicates a tool call, TAE uses TAO to execute it. TAO logs it. Then TAE gives the tool result back to agent or to next agent. Continue until final agent produces an answer. 4\. **Finalize:** TAE returns the answer to whoever requested (user or upstream system). TAO, having been involved in all tool calls, has logs of each and increments metrics accordingly.

**TAB/TAE Combined Workflow:** In a scenario where a user uses the system end-to-end: \- They build a team with TAB (powered by TAO in the background). \- Once the team is ready, they execute queries on it via TAE (with TAO handling tools). \- TAO ensures consistency: if after building the team, some tools changed or domains updated, TAO’s ChangeDetector/VersionManager might update relevant cached info so that TAE always runs with current data. For instance, if domain knowledge got new documents, TAO ensures any vector index is updated before execution.

**Testing Integration:** \- Simulate a full query: have TAO and TAE running (perhaps in a test or staging environment). Use a simple team (one agent that calls one tool). Send a query through TAE’s interface and verify the final answer. Check that TAO’s orchestrator was invoked and logs created. \- If TAE has an API, test that calling that API triggers the expected calls in TAO (this can be done by injecting a spy in TAO’s tool execution to see that it was called). \- Multi-agent test: e.g., set up two dummy agent logic (like one agent just outputs a fixed instruction to call a tool, second agent returns a fixed answer). Ensure TAO correctly handles multiple tool calls in one query.

In summary, TAO is the backbone that **prepares and supports execution** (via data, tools, context), while TAE is the engine that **drives the agent conversation**. TAO and TAE communicate through well-defined interfaces so that improvements to one (e.g. a smarter orchestrator in TAO or a more capable conversation loop in TAE) benefit the whole system without tight coupling.

## 8\. Security and Permission Control

Security in TAO operates on two levels: **user/tenant permissions** for performing certain actions, and **agent tool permissions** as already discussed. The system must ensure that only authorized users can build or execute agent teams, and that agents (team members) only use tools they are allowed to.

### 8.1 Tenant and User Permissions

Each action in the Team Agent platform can be gated by the user’s role or permission set: \- **Tenant Isolation:** Data in TAO (teams, domains, tools usage) is partitioned by tenant. A user from Tenant A should not be able to see or affect teams of Tenant B. TAO enforces this by scoping queries by tenant\_id. For example, when a user calls “create team”, the request includes their tenant, and TAO will attach that tenant\_id to the team and all subsequent queries filter by it. Multi-tenancy is supported inherently in the schema (almost every table has tenant\_id)[\[199\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20domains%20,domain_id)[\[200\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20agent_teams%20,255%29%20NOT%20NULL). \- **User Roles:** Within a tenant, different users might have roles like **Admin**, **Builder**, **Executor**, **Viewer**, etc. An Admin can do anything (create/update/delete teams, run queries), a Builder can create or modify teams, an Executor (or normal user) can only run queries on existing teams, and a Viewer might only view configurations or results. TAO should integrate with an authentication service (perhaps an upstream gateway or an auth token that includes roles). The auth.py module in TAO likely handles verifying a token and extracting the user’s identity and roles[\[201\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%82%20%20%20%20,New%3A%20Team%20orchestration)[\[202\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%82%20%20%20%20,py). \- **Permission Checks:** Before performing certain actions, TAO checks the user context: \- Creating a team: require role \>= Builder for that tenant. \- Deleting or updating a team config: require Admin or Builder. \- Executing a team (running a query): require that user has access to that team (maybe any authenticated tenant user can run, or a special role if needed). \- Viewing logs or configurations: perhaps restricted to Admin/Builder as well, since config can contain sensitive info like API keys or internal notes.

If a user lacks permission, TAO should reject the operation (HTTP 403 Forbidden or an MCP error response). These events can be logged (so attempted security breaches are recorded). The merge guidelines mention logging user\_executions as well[\[106\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=should%20have%20all%20the%20CRUD,execute%20an%20agent%20team%2C%20etc) – implying whenever a user triggers something, it’s logged along with their user\_id and outcome (success/fail). This is similar to an audit log of user actions, separate from agent actions.

**Execution Context Propagation:** When a user’s query is executed by a team, the context of “who asked” might be relevant: \- The orchestrator could attach the user’s ID or profile to the query context, enabling agents to tailor responses with that in mind (e.g., a profile says the user is in Finance dept, so answer finance questions in more detail). \- Also, if an agent is calling a tool that involves user data, it might use the user’s identity for access control (for instance, an internal HR database tool might only allow queries if the user has clearance).

For MVP, user context usage is minimal (maybe just for logging and simple personalization)[\[25\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,say%20it%E2%80%99s%20supposed), but future could integrate with identity systems.

### 8.2 Agent Tool Permissions (Recap)

On the agent side, MemberManager enforces that agents cannot use tools outside their allowance[\[53\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,). This is crucial to prevent a scenario where an agent “breaks character” and tries to use some powerful admin tool it shouldn’t. The PermissionError thrown in such cases ensures no actual call is made, and permission\_denials\_total metric captures the event[\[176\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=permission_denials%20%3D%20Counter,team_id%27%2C%20%27member_id%27%2C%20%27tool_name%27%5D). These rules are configured when building the team (the user explicitly assigns tools to each member). The system by design doesn’t give agents the ability to add new tools at runtime (they can only choose from what’s loaded in ToolRegistry and allowed to them). This is a form of sandboxing for AI behavior.

**Team-as-Tool Permissions:** If an agent team is offered as a tool to another team, permission control extends one level: the calling team’s agent must have permission to use the “TeamTool” (the proxy tool that represents the other team). Internally, that means only certain roles/teams are allowed to invoke certain other teams. TAO could enforce this by how it registers Team-as-Tool entries. For example, if Team B is to be callable by Team A, one must register a tool (say “TeamB\_Tool”) for Team A’s members. Only those members then have access. This prevents arbitrary cross-team calling unless configured.

### 8.3 Data Access and Domain Permissions

The domain access control ensures agents only retrieve or see knowledge chunks they are allowed to: \- The team\_domains (SQL) and ACCESSES\_DOMAIN (graph) relationships define which domains a team can access[\[203\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Team%20relationships%20CREATE%20%28team%3ATeam%29,m%3AMember)[\[204\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Member%20relationships%20CREATE%20%28m%3AMember%29,tool%3ATool). TAO’s DomainResolver will, when an agent tries to fetch knowledge (chunks), filter out chunks from domains not accessible. For example, if a team is only allowed “Customer Support” domain, and not “HR Policies”, queries to knowledge base will exclude HR documents. \- If an agent somehow tries to query an off-limits domain, DomainResolver should return nothing or an error. This is analogous to tool permission but for knowledge.

**User-level domain restrictions:** Potentially, if a user (tenant user) has restricted domain visibility, that could further filter. But likely, domain restrictions are set at team design time for now.

### 8.4 Authentication Mechanism

We expect an auth token (JWT or similar) to accompany requests to TAO’s API. TAO’s auth.py would validate this token (possibly via public key or a user service) and populate a context (user\_id, tenant\_id, roles). All TAO handlers then refer to this context to enforce checks. For example:

if action \== "create\_team":  
    if current\_user.role not in \['Admin','Builder'\]:  
        raise Unauthorized("Not allowed to create teams")  
    \# else proceed

For internal calls (like between TAO and TAE if on a private network), they might trust each other’s system-level credentials, or use service-to-service auth (e.g., TAE calling TAO with a special token).

**Audit of User Actions:** As mentioned, beyond just preventing unauthorized actions, TAO logs authorized actions too (who did what and when). A user\_executions or more aptly user\_actions table could store entries like: timestamp, user\_id, action\_type, target\_team\_id, status. This log, combined with the agent execution logs, gives a full picture (“User X ran Team Y which caused Agent Z to call Tool W”).

**Testing Security:** \- Attempt API calls with and without auth in a test: ensure the ones without auth fail. \- Test role enforcement by crafting tokens with different roles and calling endpoints. E.g., a Viewer token trying to create a team should get 403\. \- If possible, test that SQL injection or similar is mitigated (likely by using parameterized queries/ORM; not a direct concern in spec but good practice). \- Multi-tenant test: user from tenant A cannot access team from tenant B. In a controlled test, create teams under different tenant\_ids and simulate a request from the wrong tenant (perhaps by altering token tenant claim), TAO should not return data or should error out.

By having these layers of permission, TAO ensures both **front-door security** (only right users can use it in certain ways) and **internal safety** (agents are constrained to their intended tools and data).

## 9\. Scope, MVP Features, and Future Phases

This section delineates what features are included in the **Minimum Viable Product (MVP)** of TAO and what enhancements are planned for future phases.

### 9.1 MVP Scope

The MVP of TAO focuses on delivering a functional end-to-end system with core capabilities, while deferring more advanced or optional features. The MVP includes:

* **Multi-Tenant MCP Gateway:** A running TAO service that can handle requests for multiple tenants, with basic auth and isolation[\[3\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2A%20Multi,coverage).

* **Agent Team & Member Management:** Ability to define teams with multiple members, each with specific roles and tool permissions[\[3\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2A%20Multi,coverage). Creating, reading, and listing teams via API or builder is supported (update/delete might be limited or manual in MVP).

* **Tool Registry & Permissions:** A functioning ToolRegistry with a predefined set of tool adapters (vector search, embedding, chunk fetch, external API). Tools can be invoked uniformly, and member permission checks are enforced on each call[\[205\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Acceptance%20Criteria%3A%20,Complete%20audit%20trail%20per%20member).

* **Basic Team Orchestration:** The orchestrator can handle a straightforward sequence of agent interactions for at least one archetype (e.g., question answering with retrieval). It can select the right agent for a tool and gather results[\[3\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2A%20Multi,coverage).

* **Team Building Workflow (Wizard & YAML):** The interactive builder and YAML import are implemented as two ways to create agent teams[\[206\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Phase%202%3A%20Team%20Building%20Workflow,import%20validates%20and%20creates%20teams)[\[207\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,technical%20users). Pause/resume of the wizard is supported, with state persisted.

* **Domain Knowledge Integration:** A simple domain structure is in place (in Neo4j and Postgres). Teams can be linked to domains, and chunks can belong to domains[\[144\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Phase%203%3A%20Domain%20Hierarchy%20%26,can%20belong%20to%20multiple%20domains). The system uses this to scope what knowledge is considered but a full semantic search over chunks might not be fully utilized yet (MVP mainly sets the stage).

* **Audit Logging:** All tool executions by agents are logged in the database. Basic audit retrieval can be done via direct DB queries or simple API if available.

* **Dockerized Deployment:** The system can be deployed using Docker Compose with all services (as described). A developer or tester can bring up the entire stack in minutes[\[208\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Phase%204%3A%20Production%20Optimization%20,All%20metrics%20exposed%20to%20Prometheus)[\[87\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Acceptance%20Criteria%3A%20,50ms%20response%20for%20cached%20queries).

* **Monitoring:** Prometheus metrics are exposed for critical counts and durations, and a sample Grafana dashboard is available to monitor them[\[209\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Documentation%20covers%20all%20features).

* **Testing & Quality:** Unit tests cover at least 80% of core logic, and an integration test suite ensures the main user journeys (build team, run query) work reliably[\[205\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Acceptance%20Criteria%3A%20,Complete%20audit%20trail%20per%20member).

In terms of timeline from the spec: MVP corresponds roughly to Phase 1 and Phase 2 deliverables[\[3\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2A%20Multi,coverage)[\[206\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Phase%202%3A%20Team%20Building%20Workflow,import%20validates%20and%20creates%20teams), possibly a bit of Phase 3 if basic domain usage is included. Features like the Critic agent or parallel agent execution are **not** in MVP.

### 9.2 Future Phases and Enhancements

Beyond MVP, the following enhancements are planned or envisioned:

* **Advanced Multi-Agent Coordination:** Introduce parallel and more complex DAG execution in the orchestrator. For example, two agents working in parallel on subproblems, and orchestrator merging their results[\[210\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=the%20graph,the%20YAML%20Parser%2C%20the%20orchestrator)[\[211\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=parallel%20branches%20,as%20part). Manage synchronization and result combination. Also, allow orchestrating multiple teams at once (a multi-team scenario), though this is lower priority[\[212\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=%2A%20Multi,but%20that%E2%80%99s%20beyond%20initial%20scope).

* **“Critic” Feedback Loop:** Integrate the optional Critic agent that reviews answers and triggers a refinement loop[\[213\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=An%20AI%20%E2%80%9CCritic%E2%80%9D%20agent%20that,implementation%20steps%2C%20testing%20plan%2C%20and). Post-MVP, we can allow the orchestrator to call a Critic agent after the main answer is ready. If the Critic finds issues, orchestrator runs an improvement iteration (e.g., have the Solver agent revise the answer)[\[214\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=if%20the%20solver%20isn%E2%80%99t%20satisfied%29,with%20the%20question%20and%20the)[\[215\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,147). This may tie into agent prompt templates and orchestrator logic for iterative loops.

* **Automated Tool Discovery:** Currently, new tools must be registered manually or via a controlled MCP registry. A future feature is to allow agents to query an open tool repository at runtime (beyond the semantic search index). For instance, if the agent realizes none of its known tools suffice, it could ask TAO to find if any *other* tool is available (maybe through a marketplace of tools). This would involve expanding the MCP Tool Registry Client to perhaps fetch and load a tool on the fly[\[74\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Integration%20with%20Tool%20Loader%3A%20The,tool%20definitions%20from%20the%20registry)[\[216\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L780%204,we%20might%20implement%20a%20specific). This dynamic integration needs careful permission and security vetting (we don’t want an agent to load a dangerous tool unapproved).

* **Enhanced Domain Knowledge Integration:** Build more functionality around the domain graph:

* Use domain relationships to do smart context injection. For example, if a team is associated with a domain, automatically include a summary of recent chunks from that domain in agent’s context.

* Implement domain-specific memory or fine-tuning: agents could “learn” from past interactions by updating agent\_learn\_\* properties on nodes (like increasing a chunk’s deprecated score if it’s no longer relevant)[\[217\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,file%20or%20domain%20knowledge%20info)[\[218\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,If%20the%20TA%20bot).

* Possibly integrate a vector search on the chunks (like a Qdrant collection for documents) so that agents can retrieve info from large document sets semantically. MVP might have basic key-word retrieval; future would involve full vector-based knowledge search.

* **User Personalization & Profiles:** Introduce user profiles and preferences in the Tenant config (or separate user config). TAO could then ensure responses are personalized. E.g., a user’s preferred language or formality level could be stored and TAO would pass that to agents. Also, refine permission control so that even within a team, certain actions require a certain user clearance (maybe an agent might ask “User, do you have approval to run this tool?” if it’s sensitive).

* **Expanded Tool Suite:** Add more adapters, like database query tools, email or notification tools (for an agent to send a result somewhere), etc. Also integrate cloud APIs (AWS, Azure services as tools) for more powerful agent capabilities. Each new tool type would come with an adapter implementation and possibly UI in TAB to configure credentials.

* **Optimization and Scaling:** Improve performance and scalability:

* Support \>100 concurrent queries by optimizing I/O (e.g., more asynchronous operations, connection pooling, perhaps running multiple orchestrator instances behind a load balancer)[\[219\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Migration%20Scripts).

* Caching of frequently used results (e.g., results of recent tool calls or embeddings) to cut down latency for similar queries.

* Scale out vector DB and others if data grows (sharding Qdrant, clustering Neo4j).

* **Better UI/UX in TAB:** While not TAO’s direct responsibility, future phases would refine the Team Builder UI: more guidance, templates for common team setups, and maybe a library of pre-built agent teams that can be imported.

* **Auto-Documentation and Explanation:** TAO could generate documentation or graphs for a team configuration (since it knows all relationships). For example, when a team is built, output a summary: “Team Alpha (Solver, uses toolX and toolY; Retriever, uses toolZ)… focusing on domain Finance”. This could help users trust and verify the setup.

* **Continuous Learning:** The system could monitor outcomes and adjust configurations. If certain tools always fail or never get used, suggest removing them (or automatically do so in a new version). If an agent frequently passes a question to another agent, maybe that indicates a role change needed. These are speculative ideas for intelligent refinement of agent teams over time.

### 9.3 Timeline (Phases Recap)

To align with development milestones[\[220\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=12,Create%20teams%20with%20multiple%20members)[\[41\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Build%20session%20management%20Acceptance%20Criteria): \- **Phase 1 (MVP):** Core TAO infrastructure – multi-tenant support, tool registry, basic orchestration, test suite. *(Covered as MVP above.)*  
\- **Phase 2:** Team Builder integration – wizard interface, YAML import, pause/resume. This makes the system user-friendly and feature-complete for configuration.  
\- **Phase 3:** Domain and Knowledge Graph – advanced domain management, multi-domain chunk mapping, and ensuring the agents leverage it properly. End of this phase, TAO can support complex org structures and content.  
\- **Phase 4:** Production Hardening – full Docker/K8s deployment, monitoring, performance tuning, documentation completion. Ready the system for real-world usage by ensuring it’s observable and reliable under load.

Future phases beyond could tackle the advanced items like Critic agent and dynamic tool loading. One optional future addition is a **Cross-Team Collaboration**: enabling multiple teams to coordinate on a single query (beyond one team calling another as a tool, truly have two teams each with distinct expertise collaborating). This is complex and would be considered if a use-case demands it.

In conclusion, the TAO module as specified provides a robust foundation for orchestrating multi-agent systems. The MVP delivers immediate value by allowing creation and execution of agent teams with tool usage, while the planned future enhancements pave the way for a highly adaptive, intelligent orchestration platform that can handle evolving needs and scale. Each submodule of TAO is built to be independent and testable, ensuring that as features are added, they can be integrated without breaking the existing structure – maintaining the consistency and modularity of the Team Agent Platform.

---

[\[1\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TAO%20serves%20as%20the%20central,and%20express%20YAML%20upload%20paths) [\[3\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2A%20Multi,coverage) [\[4\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20TAO%20Core%20MCPManager,Team%20Orchestrator) [\[5\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=DomainResolver,Team%20Build%20Workflow) [\[6\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=1,enriched%20with%20member%20metadata) [\[8\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Tool%20Adapters%20VectorSearch,Tool%5D%20end) [\[9\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Team%20Building%20WizardEngine,Step%20Processor%5D%20end) [\[10\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Service%20Layer%20QdrantClient,Ollama%20Client%5D%20end) [\[11\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TeamOrchestrator%20,2%20Component%20Communication%20Flow%20python) [\[12\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=TeamOrchestrator%20,2%20Component%20Communication%20Flow) [\[13\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=3,Execution%20history%20logged%20per%20member) [\[14\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=5.%20Tool%20execution%20with%20member,enriched%20with%20member%20metadata) [\[17\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20TAO%20Core%20MCPManager,Change%20Detector) [\[18\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=subgraph%20Tool%20Adapters%20VectorSearch,Tool%5D%20end) [\[19\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%82%20%20%20%20,New%3A%20Team%20orchestration) [\[20\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%82%20%20%20%20,py) [\[27\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,members) [\[28\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20get_team,if%20not%20team_data%3A%20return%20None) [\[29\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=team_id%3Dteam_data%5B%27team_id%27%5D%2C%20tenant_id%3Dteam_data%5B%27tenant_id%27%5D%2C%20team_name%3Dteam_data%5B%27team_name%27%5D%2C%20status%3Dteam_data%5B%27status%27%5D%2C%20members%3D%5BAgentTeamMember%28,) [\[30\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,append%28member) [\[31\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,0) [\[32\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20execute_with_member,member%20%3D%20await%20self.member_manager.get_member%28member_id) [\[33\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=if%20tool_name%20not%20in%20member,) [\[34\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,now%28%29) [\[35\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=return%20result) [\[40\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Granular%20tool%20access%20control) [\[41\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Build%20session%20management%20Acceptance%20Criteria) [\[42\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=if%20not%20member%3A%20raise%20ValueError%28f,not%20found) [\[43\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=except%20Exception%20as%20e%3A%20,update_member_execution%28%20execution_id%3Dexecution_id%2C%20status%3D%27failed%27%2C%20error%3Dstr%28e%29) [\[44\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=class%20MemberManager%3A%20,members%20and%20their%20tool%20permissions) [\[45\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20get_member,member_id) [\[46\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,LangGraph%20wizard%20engine) [\[47\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,tool%20permission%20enforcement) [\[48\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,their%20tool%20permissions) [\[49\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20get_member,member_id) [\[50\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=tools%20%3D%20await%20self) [\[51\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=self.member_cache%20%3D%20) [\[52\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=async%20def%20execute_tool,get_member%28member_id%29%20if%20not%20member) [\[53\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,) [\[54\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Tool%20%7Btool_name%7D%20not%20found) [\[55\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,member_role) [\[56\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,kwargs) [\[57\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,get%28%27success%27%2C%20False%29) [\[58\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=if%20not%20self.tool_registry.has_tool%28tool_name%29%3A%20raise%20ValueError%28f,not%20found%20in%20registry) [\[59\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,255%29%20UNIQUE%20NOT%20NULL) [\[60\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=tests%2Fintegration%2Ftest_team_building,team_builder%20import%20TeamBuilder%2C%20BuildStep) [\[61\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%40pytest.mark.asyncio%20async%20def%20test_member_tool_permissions%28%29%3A%20,manager%20%3D%20MemberManager%28postgres%2C%20tool_registry) [\[62\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=manager%20%3D%20MemberManager) [\[63\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L527%20,Tool%20%7Btool_name%7D%20not%20found) [\[64\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Tool%20%7Btool_name%7D%20not%20found) [\[70\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L985%20mcp_mode%20VARCHAR,TEXT%2C%20input_schema%20JSONB%2C%20output_schema%20JSONB) [\[71\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=mcp_mode%20VARCHAR,JSONB%2C%20output_schema%20JSONB) [\[72\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%3D%3D%3D%20MCP%20Configuration%20%3D%3D%3D%20MCP_SERVER_HOST%3D0,MCP_SERVER_PORT%3D8100%20MCP_TRANSPORT_MODE%3Dstdio) [\[80\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=) [\[83\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=neo4j%3A%20image%3A%20neo4j%3A5,ports) [\[86\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%9C%E2%94%80%E2%94%80%20scripts%2F%20%E2%94%82%20%20,sh) [\[87\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Acceptance%20Criteria%3A%20,50ms%20response%20for%20cached%20queries) [\[93\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,reduction%20in%20support%20tickets) [\[94\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1140%20CREATE%20TABLE,member_id) [\[95\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20team_build_sessions%20,tenant_id) [\[96\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20team_build_steps%20,step_name) [\[97\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28m%3AMember%29,tool%3ATool) [\[98\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Track%20member%20execution%20history,1%20Team%20Building%20Tests%20python) [\[99\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Query%20tracking%20CREATE%20,) [\[100\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=eligible_members) [\[101\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,tool%20permission%20enforcement) [\[102\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20INDEX%20idx_teams_status%20ON%20agent_teams,tenant_id) [\[103\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1060%20CREATE%20TABLE,execute%2C%20configure%2C%20admin) [\[104\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1014%20CREATE%20TABLE,domain_id) [\[105\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1140%20CREATE%20TABLE,member_id) [\[107\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20chunks%20,tenant_id) [\[108\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1182%20CREATE%20TABLE,domain_id%29%2C%20relevance_score%20FLOAT) [\[109\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=PostgreSQL%20migrations%20echo%20,d%20%24POSTGRES_DB%20%3C%3C%20%27EOF) [\[110\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Constraints%20CREATE%20CONSTRAINT%20tenant_unique,id%20IS%20UNIQUE) [\[111\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20domain_unique%20IF%20NOT,id%20IS%20UNIQUE) [\[112\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Generic%20Domain%20nodes%20,name%3A%20%27Customer%20Portal%27%2C%20type%3A%20%27Application) [\[113\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28d%3ADomain%20,custom_field%3A%20%27value%27%7D%2C%20created_at%3A%20datetime) [\[114\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28d2%3ADomain%20,Feature%27%2C%20path%3A%20%27%2Fapps%2Fcustomer_portal%2Fauth%27%2C%20level%3A%202) [\[115\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20team_unique%20IF%20NOT,id%20IS%20UNIQUE) [\[116\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Team%20node%20CREATE%20,) [\[117\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20member_unique%20IF%20NOT,id%20IS%20UNIQUE) [\[118\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Member%20nodes%20CREATE%20,oriented%20problem%20solver%27%2C%20created_at%3A%20datetime) [\[119\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20chunk_unique%20IF%20NOT,id%20IS%20UNIQUE) [\[120\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Tool%20nodes%20CREATE%20,) [\[121\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20CONSTRAINT%20member_unique%20IF%20NOT,id%20IS%20UNIQUE) [\[122\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Chunk%20nodes%20CREATE%20,0) [\[123\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=text_snippet%3A%20%27First%20100%20chars%20of,current%27%2C%20status%3A%20%27active%27%2C%20created_at%3A%20datetime) [\[124\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Query%20tracking%20CREATE%20,) [\[125\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Tenant%20relationships%20CREATE%20%28t%3ATenant%29,d%3ADomain) [\[126\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28t%3ATenant%29,team%3ATeam) [\[127\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Hierarchical%20domain%20structure%20,child%3ADomain) [\[128\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Hierarchical%20domain%20structure%20,child%3ADomain) [\[129\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,20) [\[130\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Team%20relationships%20CREATE%20%28team%3ATeam%29,m%3AMember) [\[131\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28team%3ATeam%29,d%3ADomain) [\[132\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Member%20relationships%20CREATE%20%28m%3AMember%29,tool%3ATool) [\[133\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Find%20all%20tools%20a,tool%3ATool%29%20RETURN%20tool) [\[134\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Chunk%20relationships%20,d%3ADomain) [\[135\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Multiple%20domain%20connections%20for,d2%3ADomain) [\[136\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Multiple%20domain%20connections%20for,d2%3ADomain) [\[137\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28m%3AMember%29,c%3AChunk) [\[138\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20%28m%3AMember%29,c%3AChunk) [\[139\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Indexes%20CREATE%20INDEX%20domain_type,type) [\[140\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20INDEX%20team_status%20IF%20NOT,status) [\[141\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Neo4j%20migrations%20echo%20,p%20%24NEO4J_PASSWORD) [\[142\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=cypher,NC) [\[143\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Indexes%20CREATE%20INDEX%20domain_type,type) [\[144\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Phase%203%3A%20Domain%20Hierarchy%20%26,can%20belong%20to%20multiple%20domains) [\[159\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1851%20neo4j%3A%20condition%3A,service_healthy%20qdrant%3A%20condition%3A%20service_started) [\[163\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=3,config.yaml%20%E2%94%9C%E2%94%80%E2%94%80%20.venv) [\[164\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=NEO4J_URI%3Dbolt%3A%2F%2Flocalhost%3A7687%20NEO4J_USER%3Dneo4j%20NEO4J_PASSWORD%3DpJnssz3khcLtn6T) [\[165\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=match%20at%20L1840%20,EXPRESS_MODE_ENABLED%3Dtrue%20env_file) [\[167\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=HEALTHCHECK%20,) [\[169\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=def%20start_metrics_server,Modular%20Deliverables%20%26%20ROI) [\[170\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=System%20info%20system_info%20%3D%20Info,m3%27) [\[171\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Team%20metrics%20active_teams%20%3D%20Gauge,status%27%5D) [\[172\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=active_teams%20%3D%20Gauge,status%27%5D) [\[173\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Member%20metrics%20member_executions%20%3D%20Counter,) [\[174\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=member_execution_time%20%3D%20Histogram,buc%20Retry%20A) [\[175\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=python%20src%2Ftao%2Futils%2Fmetrics.py%20%28continued%29%20buckets%3D,) [\[176\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=permission_denials%20%3D%20Counter,team_id%27%2C%20%27member_id%27%2C%20%27tool_name%27%5D) [\[177\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Tool%20permission%20metrics%20tool_permissions%20%3D,tool_name%27%5D) [\[178\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Build%20workflow%20metrics%20build_steps_completed%20%3D,mode%3A%20wizard%2Fexpress) [\[179\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=build_session_duration%20%3D%20Histogram,) [\[180\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Domain%20metrics%20domain_hierarchy_depth%20%3D%20Histogram,) [\[181\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Domain%20metrics%20domain_hierarchy_depth%20%3D%20Histogram,) [\[184\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,) [\[185\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=) [\[187\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%40pytest.mark.asyncio%20async%20def%20test_conversational_build_with_pause_resume%28%29%3A%20,postgres%2C%20wizard%2C%20validator) [\[188\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=assert%20session.status%20%3D%3D%20,CHOOSE_DOMAIN) [\[189\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=result%20%3D%20await%20builder,) [\[190\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=assert%20result) [\[191\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=) [\[199\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20domains%20,domain_id) [\[200\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=CREATE%20TABLE%20agent_teams%20,255%29%20NOT%20NULL) [\[201\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%82%20%20%20%20,New%3A%20Team%20orchestration) [\[202\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%E2%94%82%20%20%20%20,py) [\[203\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Team%20relationships%20CREATE%20%28team%3ATeam%29,m%3AMember) [\[204\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=%2F%2F%20Member%20relationships%20CREATE%20%28m%3AMember%29,tool%3ATool) [\[205\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Acceptance%20Criteria%3A%20,Complete%20audit%20trail%20per%20member) [\[206\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Phase%202%3A%20Team%20Building%20Workflow,import%20validates%20and%20creates%20teams) [\[207\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,technical%20users) [\[208\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=Phase%204%3A%20Production%20Optimization%20,All%20metrics%20exposed%20to%20Prometheus) [\[209\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Documentation%20covers%20all%20features) [\[219\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=,Migration%20Scripts) [\[220\]](file://file-7AcmdqLaDNW1fbH9XhHSqH#:~:text=12,Create%20teams%20with%20multiple%20members) TA\_V1\_3\_Opus\_4\_1.docx

[file://file-7AcmdqLaDNW1fbH9XhHSqH](file://file-7AcmdqLaDNW1fbH9XhHSqH)

[\[2\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Each%20module%20is%20designed%20with,targets%20and%20owners%20to%20be) [\[15\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Description%3A%20The%20Orchestrator%20is%20the,and%20a%20set%20of%20agents%2Ftools) [\[16\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=builder%20component%20of%20the%20orchestrator,class%20that%20stores%20the%20prompt) [\[21\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Description%3A%20The%20Orchestrator%20is%20the,plan%2C%20without%20falling%20into%20cycles) [\[22\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,the%20conversation%20and%20tool%20usage) [\[23\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,orchestrator%20is%20designed%20so%20we) [\[24\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,fail%20the%20whole%20task%20or) [\[25\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,say%20it%E2%80%99s%20supposed) [\[26\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=core%2Fregistry_client%2F%20Orchestrator%20%26%20LangGraph%20DAG,%E2%80%93%20End%20of%20Sep%202025) [\[36\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=graph%20could%20be%20an%20agent%E2%80%99s,Agent%20B%20might%20wait) [\[37\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,single%20node) [\[38\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=LangGraph%20Concept%3A%20We%20use%20a,which%20defines%20what) [\[39\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=using%20that%20info%5B46%5D%5B47%5D.%20,a%20full%20graph%20library%3B%20a) [\[65\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Tool%20Loader%20%26%20Transport%20Adapter,It) [\[66\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L260%20,function%20calls%2C%20as%20per%20the) [\[67\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Tool%20Loader%20%26%20Transport%20Adapter,using%20the%20correct%20protocol%2Ftransport) [\[68\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Diagram%20%E2%80%93%20Tool%20Loader%20Context%3A,calls) [\[69\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L254%20,11%5D.%20This%20allows%20the) [\[73\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,11%5D.%20This%20allows%20the) [\[74\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Integration%20with%20Tool%20Loader%3A%20The,tool%20definitions%20from%20the%20registry) [\[75\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=When%20initializing%2C%20the%20Tool%20Loader,tool%20type%20and%20instantiate%20the) [\[76\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,the%20adapter) [\[78\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L645%20,the%20adapter) [\[79\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L245%20The%20Tool,9) [\[81\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L270%20JSON%20from,py) [\[82\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,function%20calls%2C%20as%20per%20the) [\[89\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=one%20call%20per%20needed%20tool,which%20Qdrant%20handles%20easily%20in) [\[90\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L1828%20tenant%20IDs,%28Some%20of) [\[166\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L652%20EMBEDDING_SERVICE_URL%3Dhttp%3A%2F%2Flocalhost%3A8000%29,Deployment%20%26%20Environment) [\[192\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=approach,building%20an%20agent%20team%20and) [\[193\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Multi,of%20Sep%202025%20TBD%20core%2Fmulti_agent_executor) [\[194\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,This%20is%20an) [\[195\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=Agent%20Executor%20,52) [\[196\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=fine,required%20nodes%20complete%2C%20the%20orchestrator) [\[197\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,for%20example) [\[198\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=with%20its%20persona%2Fprompt%20and%20be,the%20plan%20based%20on%20the) [\[210\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=the%20graph,the%20YAML%20Parser%2C%20the%20orchestrator) [\[211\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=parallel%20branches%20,as%20part) [\[212\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=%2A%20Multi,but%20that%E2%80%99s%20beyond%20initial%20scope) [\[213\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=An%20AI%20%E2%80%9CCritic%E2%80%9D%20agent%20that,implementation%20steps%2C%20testing%20plan%2C%20and) [\[214\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=if%20the%20solver%20isn%E2%80%99t%20satisfied%29,with%20the%20question%20and%20the) [\[215\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=,147) [\[216\]](file://file-RzjMEPckxLsqAd5D47TJdW#:~:text=match%20at%20L780%204,we%20might%20implement%20a%20specific) TA\_V1\_4\_Gpt5.docx

[file://file-RzjMEPckxLsqAd5D47TJdW](file://file-RzjMEPckxLsqAd5D47TJdW)

[\[7\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,The%20MCP%20Tool%20Registry%20Client) [\[77\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=Tool%20Search%20Index%20,using%20the%20member_executions%20logging%20with) [\[88\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=request%20come%20the%20agent%20can,The%20MCP%20Tool%20Registry%20Client) [\[92\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,get%20the%20information%20in%20human) [\[106\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=should%20have%20all%20the%20CRUD,execute%20an%20agent%20team%2C%20etc) [\[182\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,the%20tenants%20list%20of%20agents) [\[183\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=etc%20,done%20or%20something%20like%20that) [\[186\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,knowledge%20config%20or%20domain%20knowledge) [\[217\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,file%20or%20domain%20knowledge%20info) [\[218\]](file://file-NyBSYtBuJqRobCvTLCezUb#:~:text=,If%20the%20TA%20bot) \_\_Final\_01\_Merge\_strategy.docx

[file://file-NyBSYtBuJqRobCvTLCezUb](file://file-NyBSYtBuJqRobCvTLCezUb)

[\[84\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=cpus%3A%20%274%27%20memory%3A%2020G%20healthcheck%3A,10s%20retries%3A%205%20start_period%3A%2060s) [\[85\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=memory%3A%208G%20healthcheck%3A%20test%3A%20%5B%22CMD,30s%20timeout%3A%2010s%20retries%3A%203) [\[91\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=qdrant%3A%20image%3A%20qdrant%2Fqdrant%3Alatest%20container_name%3A%20qdrant,.%2Fdata%2Fqdrant%3A%2Fqdrant%2Fstorage%3Az%20environment) [\[145\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=%23%20%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%20DATABASES%20%28CPU,postgres_user%20POSTGRES_PASSWORD%3A%20postgres_pass%20PGDATA%3A%20%2Fvar%2Flib%2Fpostgresql%2Fdata%2Fpgdata) [\[146\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=postgres%3A%20image%3A%20postgres%3A16,Performance%20tuning%20for%2064GB%20RAM) [\[147\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=environment%3A%20POSTGRES_DB%3A%20ta_v8%20POSTGRES_USER%3A%20postgres_user,200%20POSTGRES_SHARED_BUFFERS%3A%204GB%20POSTGRES_EFFECTIVE_CACHE_SIZE%3A%2016GB) [\[148\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=container_name%3A%20qdrant%20ports%3A%20,environment%3A%20QDRANT__SERVICE__HTTP_PORT%3A%206333%20QDRANT__SERVICE__GRPC_PORT%3A%206334) [\[149\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=cpus%3A%20%274%27%20memory%3A%208G%20healthcheck%3A,10s%20retries%3A%205%20start_period%3A%2030s) [\[150\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=neo4j%3A%20build%3A%20context%3A%20,%227687%3A7687%22%20volumes) [\[151\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=,RAM%20NEO4J_server_memory_heap_initial__size%3A%204G%20NEO4J_server_memory_heap_max__size%3A%208G) [\[152\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=container_name%3A%20neo4j%20ports%3A%20,.%2Fneo4j%2Fimport%3A%2Fvar%2Flib%2Fneo4j%2Fimport) [\[153\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=minio%3A%20image%3A%20minio%2Fminio%3Alatest%20container_name%3A%20minio,environment%3A%20MINIO_ROOT_USER%3A%20minioadmin%20MINIO_ROOT_PASSWORD%3A%20minioadmin) [\[154\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=ollama%3A%20build%3A%20context%3A%20,.%2Fmodels%2Follama%3A%2Froot%2F.ollama) [\[155\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=resources%3A%20reservations%3A%20devices%3A%20,stopped) [\[156\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=dockerfile%3A%20,0) [\[157\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=limits%3A%20memory%3A%2032G%20restart%3A%20unless,30s%20timeout%3A%2010s%20retries%3A%203) [\[158\]](file://file-Av5tPFUyyVQk9Asf8HsAq3#:~:text=healthcheck%3A%20test%3A%20%5B%22CMD%22%2C%20%22curl%22%2C%20%22,10s%20retries%3A%205%20start_period%3A%2060s) docker-compose-master.yml

[file://file-Av5tPFUyyVQk9Asf8HsAq3](file://file-Av5tPFUyyVQk9Asf8HsAq3)

[\[160\]](file://file-DNQKKk5zsqCCwFKjiaKQP2#:~:text=services%3A%20,.%2Fmonitoring%2Fprometheus.yml%3A%2Fetc%2Fprometheus%2Fprometheus.yml) [\[161\]](file://file-DNQKKk5zsqCCwFKjiaKQP2#:~:text=,%2Fsys%3A%2Fhost%2Fsys%3Aro) [\[162\]](file://file-DNQKKk5zsqCCwFKjiaKQP2#:~:text=nvidia,stopped) [\[168\]](file://file-DNQKKk5zsqCCwFKjiaKQP2#:~:text=prometheus%3A%20image%3A%20prom%2Fprometheus%3Alatest%20container_name%3A%20prometheus,prometheus_data%3A%2Fprometheus%20command) docker-compose-monitoring.yml

[file://file-DNQKKk5zsqCCwFKjiaKQP2](file://file-DNQKKk5zsqCCwFKjiaKQP2)