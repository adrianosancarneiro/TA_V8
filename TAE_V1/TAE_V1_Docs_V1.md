# Team Agent Executor (TAE) – Module Documentation

## 1\. Overview

The Team Agent Executor (TAE) is the runtime engine responsible for **executing multi-agent teams of AI agents** in the Team Agent platform. Once a team of agents is configured (via the Team Agent Builder, TAB) and orchestrated (via the Team Agent Orchestrator, TAO), TAE takes over to actually run the agents’ conversation loops and produce answers to user queries. In essence, TAE manages the **LLM interactions** for each agent in a team according to the plan and context provided by TAO. It handles the flow of messages between agents, calls external tools on behalf of agents when needed, and collects the final result to return to the user or upstream system. TAE’s key capabilities include coordinating multi-turn dialogues among specialized agents, integrating tool usage seamlessly via TAO’s Model Context Protocol (MCP) interface, and maintaining per-agent conversational context or memory.

**Modularity and ROI:** TAE is designed as an independent module within the Team Agent platform, focusing solely on execution. Each sub-component of TAE is a distinct piece of software that provides value on its own (for example, the tool-calling client or memory manager could be reused in other LLM applications). This modular approach means TAE can be developed, tested, and improved in isolation while remaining compatible with TAO and TAB. As a standalone component, TAE demonstrates the **power of orchestrated multi-agent reasoning**: even outside the full platform, it can take a team definition and a query and drive a complex interactive reasoning process to solve the query. This delivers immediate ROI by enabling advanced behaviors (like an agent reasoning and retrieving information in multiple steps) that a single prompt/LLM alone might not achieve. TAE’s design also ensures that improvements in agent conversation techniques (such as better prompt strategies or memory use) can be incorporated without requiring changes to how teams are built or orchestrated in TAO.

## 2\. Architecture

### 2.1 High-Level Architecture

TAE’s architecture consists of several core submodules and supporting services that together enable multi-agent execution. At a high level, **TAE includes the following submodules**:

* **ExecutorEngine:** The central **multi-agent conversation orchestrator**. It receives a user query and a team configuration, then manages the turn-by-turn execution of agents according to a plan (sequence or graph of steps). The ExecutorEngine handles the looping logic for agents (including ReAct-style thought→action→observation loops) and delegates tasks like prompt generation or tool calls to other submodules.

* **AgentRuntime:** The LLM interaction layer for agents. For each agent turn, AgentRuntime builds the prompt (injecting the agent’s persona, available tools, and any context) and calls the language model to get the agent’s response. It then interprets the response (e.g., does it contain a tool request or a final answer?) and hands control back to the ExecutorEngine. This layer abstracts the details of prompt templates and model API calls.

* **ExecutionPlanner:** A planning component that can construct a **directed acyclic graph (DAG) of execution steps** (a “LangGraph”) for the agent team. In the MVP, the ExecutionPlanner might simply define a fixed sequence of agent roles (e.g. Agent1 then Agent2), but it lays the groundwork for more complex or dynamic planning in the future (such as branching or conditional steps). The ExecutorEngine uses the plan from this submodule to know which agent(s) to activate in what order.

* **AgentMemoryManager:** Manages each agent’s contextual memory and recall. This includes tracking the conversation history (recent messages) for context, and potentially interfacing with long-term memory stores (like a vector database) to enable an agent to remember information across queries. In MVP, this may be a simple in-memory log of past interactions for prompt inclusion, but the design allows extension to more sophisticated memory recall or knowledge base integration for agents.

* **ToolCaller:** A utility that handles **invoking external tools via TAO’s MCP interface** whenever an agent decides to use a tool. When the ExecutorEngine or an agent’s response indicates a tool action (e.g. call a vector search or an API), the ToolCaller formats the request and calls out to TAO’s ToolRegistry (through the MCP server or direct API) to execute that tool. It then returns the tool’s result (observation) back to the agent’s context. This submodule abstracts away the details of how tools are executed (HTTP call, gRPC, etc.) so that the rest of TAE can treat tool use as a simple function call.

* **CriticFeedbackEngine:** An optional feedback module that can **validate or critique the agents’ output**. This could be used after an answer is formulated (or during intermediate steps) to assess quality, coherence, or compliance with guidelines. For example, a “Critic” agent (possibly another LLM prompt) could be invoked to judge if the answer seems correct or safe; if not, TAE could trigger a revision loop (asking agents to refine their answers). In the MVP, this engine may be a stub or off by default (since adding a critic agent is an advanced feature), but the architecture includes it for future extensibility.

* **ExecutionContextLoader:** A component responsible for **loading all necessary context and configuration** for a given execution. When a query comes in for a team, the ExecutionContextLoader fetches the team’s latest configuration (agents, roles, tool permissions) – either by querying the databases (Postgres and Neo4j) or by calling TAO’s APIs (e.g., TeamOrchestrator.get\_team). It also loads any relevant tenant-specific settings or domain knowledge metadata associated with the team. Essentially, this submodule prepares everything the ExecutorEngine and AgentRuntime need to know about the team and its context *before* the conversation starts. By centralizing this, we ensure TAE always runs with up-to-date information (for instance, if domain knowledge was updated or a tool’s details changed, the loader will retrieve the latest data).

In addition to these core submodules, TAE relies on several **supporting services** and clients:

* **LLM Service Client:** TAE uses a language model to power the agents. In our deployment, this is **Ollama** serving the gpt-oss:20b model (running in a container), accessed via an API endpoint. The AgentRuntime will call this service (or an equivalent OpenAI API, etc., configurable by environment variables) to get completions for agent prompts. The LLM client details (URL, model name) are set via environment (e.g. OLLAMA\_URL, OLLAMA\_MODEL) and can be swapped out without affecting the rest of TAE logic.

* **Database Clients:** TAE may directly use the **Postgres database** (which stores team configs, tool definitions, and execution logs) and the **Neo4j graph database** (which stores domain relationships and possibly tool/agent graphs) to retrieve context. These connections use credentials and URIs from environment variables (e.g. POSTGRES\_HOST, NEO4J\_URI) and allow TAE to function even if it needs to read or write data without always going through TAO. In practice, most config is loaded via TAO’s provided interfaces, but direct DB access is available as a fallback or for performance caching (e.g., loading a team config in one query instead of multiple API calls).

* **Vector Index (Qdrant) Client:** For any semantic search or memory recall, TAE can utilize the Qdrant vector database. In TAO, Qdrant is used for the Semantic Tool Search index[\[1\]](file://file-RMU5LSSYyRXPrC4xyjpz1r#:~:text=3,Index%20via%20Qdrant), but TAE’s memory manager could also use it to store and retrieve encoded conversational context or long-term agent knowledge. The Qdrant service is part of the platform stack (accessible via QDRANT\_HOST:QDRANT\_PORT), and ExecutionContextLoader or AgentMemoryManager could query it if needed to fetch relevant context vectors (e.g., finding related documents or past interactions). In MVP, TAE does not perform vector searches on its own (that is typically done by an agent using a tool call to TAO), but future memory features may leverage this.

The relationships between these components can be thought of in layers or groups:

* **TAE Core Execution:** ExecutorEngine, ExecutionPlanner, AgentRuntime, ToolCaller, and (optionally) CriticFeedbackEngine form the core engine that drives the multi-agent conversation. The ExecutorEngine uses the planner to determine which agent(s) go in what order, calls AgentRuntime for each agent’s LLM output, and invokes ToolCaller whenever an agent needs to use a tool. The CriticFeedbackEngine, if enabled, wraps around the process to provide a final validation step.

* **Context & Memory:** ExecutionContextLoader and AgentMemoryManager handle context preparation and memory. The loader pulls static configuration (team setup, domain info, etc.) at the start of execution, while the memory manager tracks dynamic context (messages exchanged, any learned info) during execution. These ensure each agent has the relevant information needed at its turn – from initial persona and tools, to what was said in previous turns.

* **External Services Layer:** TAE interfaces with several external or underlying services: the **LLM server** (Ollama or others) for generating agent responses, the **TAO/MCP server** for executing tools and obtaining config data, the **Postgres** and **Neo4j** databases for persistent data, and the **Qdrant** vector store for semantic similarity queries (if used). This layer is shared with TAO in the sense that both modules connect to the same infrastructure components. For example, TAO’s MCP Gateway might be running at http://tao:8100 and TAE will call that, and both TAO and TAE point to the same Postgres DB instance. TAE is configured via environment variables to know where to reach these services (see Section 5.2).

**Execution Flow:** When a user (or external caller) triggers a team execution, TAE goes through a series of steps to produce the final result:

1. **Query Reception:** The process starts with a query and a target team\_id. This could come via an API call to TAE (e.g., a REST endpoint or function call in the app). TAE identifies which team to run – often the user specifies this, otherwise a routing layer would decide.

2. **Context Loading:** TAE invokes the ExecutionContextLoader to fetch the team’s configuration and context. This involves retrieving the list of team members (agents), each agent’s role/persona and allowed tools, the tenant’s configuration (if any general guidelines or profile), and any domain knowledge references the team is associated with. For example, if the team is linked to a “Financial Reports” domain, the loader might fetch a summary or at least the ID of that domain (which the agents might use via tools). This step ensures TAE has a complete picture of the team and environment before starting. The loader may call TAO.TeamOrchestrator.get\_team(team\_id) to get this info in one go, and TAO may perform preparatory tasks like ensuring relevant indexes are warm or injecting tenant guidelines (via a prepare\_execution routine).

3. **Plan Initialization:** Next, TAE determines the execution plan. In MVP, this might be as simple as deciding a fixed order for the agents (for instance, always let Agent1 speak first, then Agent2). If an ExecutionPlanner is in use, TAE calls it with the team config and query to obtain a plan (which could be an ordered list of agent IDs or a graph of steps). For example, in a two-agent team with roles "Retriever" and "Solver", the planner might output a plan: Step1 – use Retriever agent, Step2 – then use Solver agent. The plan could also encode that if any tool usage is needed, those are inserted as additional steps (though currently tool calls are handled dynamically within an agent’s turn). In future, the plan could be more complex (conditional branches, parallel steps), but MVP assumes a straightforward sequential or loop structure.

4. **Agent Loop Execution:** The ExecutorEngine now enters a loop or sequence to actually run the agents as per the plan. For each step or each agent’s turn:

5. TAE (via AgentRuntime) **constructs the prompt** for the active agent. This prompt typically includes a **system message** defining the agent’s persona/role and listing its available tools and instructions, along with the **conversation history** or context so far (e.g., the user’s query or prior agent responses), and possibly a directive on the output format (for example, instructing the agent how to denote a tool action).

6. TAE then calls the LLM service to get the agent’s **response**.

7. When the response comes back, AgentRuntime **parses the output** to see if the agent is requesting a tool/action or providing a final answer.

   * If it’s a tool action (for example, the agent outputs something like: Action: vector\_search\["keyword"\]), the ExecutorEngine invokes the ToolCaller to execute that action. This results in an **observation** (tool result) returned from TAO. TAE then inserts that observation into the context (as if the environment spoke back to the agent). The ExecutorEngine may loop back to let the *same agent* continue reasoning with the new information (this is typical in a ReAct pattern where one agent iteratively uses tools), or move to the next agent depending on the intended flow.

   * If the agent’s response is a **final answer** (e.g., it doesn’t indicate any further action needed), then that may mark the end of that agent’s turn. In a single-agent scenario, that means the task is complete. In a multi-agent scenario, the ExecutorEngine will take that answer and either pass it as input to the next agent (if another agent is supposed to refine or use it) or decide that it is the final output if that was the last agent in the plan.

8. The ExecutorEngine repeats this process according to the plan: for example, if Agent1 (Retriever) just provided some data (possibly via a tool), it then triggers Agent2 (Solver) with that data to get a final answer. Each agent thus gets a chance to contribute: one might fetch information, another might analyze it and answer.

9. **Critic Feedback (optional):** If a CriticFeedbackEngine is enabled, after the nominal final answer is produced, TAE can perform an extra validation step. For MVP, this might be as simple as checking certain heuristics (like “does the answer look empty or irrelevant?”) and possibly appending a note or marking the answer low confidence. In more advanced usage, TAE could spawn a “critic agent” (another LLM prompt) to review the answer. If the critic determines the answer is unsatisfactory, TAE could loop back into execution – for instance, prompt the solver agent to try again with additional hints, or involve another agent (if available) to verify facts. This is not implemented in the initial version except perhaps logging the critique, but the architecture allows inserting this step before finalizing the result.

10. **Finalization:** TAE assembles the final result to return to the caller. This is usually the final agent’s answer (possibly post-processed or formatted). TAE also collects any metadata that should accompany the answer, such as which tools were used or which agents contributed. Much of the **execution trace is logged by TAO** throughout the process – every tool call was recorded in TAO’s databases – so TAE doesn’t need to duplicate that. However, TAE may log the overall conversation or outcome in a high-level execution log (e.g., creating a user\_execution record indicating user X ran team Y at time Z and it succeeded). Finally, TAE returns the answer (and possibly the execution log ID or reference) to the user or upstream system that made the request.

This end-to-end flow highlights how TAE serves as the **“executor” or runtime** for the multi-agent system, managing the dialog and calls needed to solve queries. TAO remains in the loop mainly for tool execution and providing configuration, ensuring governance (permissions) and up-to-date knowledge, while TAE focuses on the **step-by-step reasoning and interaction between agents** and the LLM.

### 2.2 Submodule Overview Diagram

For a visual representation, we can group TAE’s submodules and their interactions in layers (similar to TAO’s architecture):

* **TAE Core:** **ExecutorEngine**, **ExecutionPlanner**, **AgentRuntime**, **ToolCaller**, **CriticFeedbackEngine**. These implement the primary logic of running a team execution. The ExecutorEngine uses the planner to figure out the order of execution and orchestrates calls to AgentRuntime for LLM outputs. When an agent needs an external action, ExecutorEngine invokes the ToolCaller. If configured, CriticFeedbackEngine wraps around to evaluate the result. These components together enable TAE to conduct a multi-agent conversation and tool-using workflow.

* **Context & Memory:** **ExecutionContextLoader**, **AgentMemoryManager**. The ExecutionContextLoader prepares initial state (team info, etc.), and the AgentMemoryManager keeps track of conversational state and any long-term memory resources. They ensure agents have the right context (from static config to dynamic message history) at each step of execution.

* **External Services & Integration:** **TAO (MCP Server)**, **LLM Service (Ollama)**, **Databases (Postgres & Neo4j)**, **Vector Store (Qdrant)**. TAE interacts with TAO’s MCP interface to perform tool actions and can call TAO for config or domain info. It uses the LLM service to generate agent responses. Postgres and Neo4j are tapped for configuration data and logging (TAE may read/write as needed), and Qdrant is available for any semantic search or memory embedding operations. These services run as separate containers or processes and TAE is configured to reach them via network calls.

*(In the original design spec, a Mermaid diagram depicts these groupings and interactions; conceptually, one can imagine TAE’s core loop (ExecutorEngine) in the middle, calling out to an LLM on one side (via AgentRuntime) and to TAO/ToolRegistry on the other side (via ToolCaller), with context coming in from the loader and memory manager.)*

To ground these abstractions, the next section provides detailed specifications and implementation guidance for each submodule, including how they work, why they provide standalone value, key methods or classes, and test strategies.

## 3\. Submodule Specifications and Implementation Guides

Each submodule in TAE is implemented as an independent, testable unit with a clear purpose, its own ROI as a standalone component, and a defined testing plan. The codebase is organized such that each submodule resides in its own module or package (for example, a executor\_engine.py for the ExecutorEngine, a agent\_runtime/ package for AgentRuntime, etc.). This modular structure not only enforces separation of concerns but also allows us to build and verify each piece in isolation before integrating them together. Below, we describe each major submodule, including how it works and how to implement it.

### 3.1 ExecutorEngine

**Purpose & Responsibilities:** The **ExecutorEngine** orchestrates the overall workflow of running an agent team for a given query. It serves as the entry point in TAE for executing a team, analogous to how TAO’s TeamOrchestrator plans the high-level team behavior. The ExecutorEngine’s job is to take a loaded team configuration and a user’s request, then manage the sequence of agent invocations and tool calls required to fulfill that request. It ensures that agents speak in the correct order (or loop appropriately), that any actions an agent decides to take (like using a tool) are carried out and fed back into the loop, and that the process terminates with an answer or a graceful failure. In advanced usage, the ExecutorEngine can work with complex execution graphs (via the ExecutionPlanner) to coordinate multiple agents, possibly with branching or parallelism. However, in the MVP it handles a mostly linear sequence or simple loop of agent turns – for example, agent A then agent B, or agent A looping until done – making sure each step’s output is passed as input to the next as needed. The ExecutorEngine is effectively the “conductor” of the multi-agent conversation.

**Standalone ROI:** As a standalone component, the ExecutorEngine encapsulates the logic of multi-step, multi-agent reasoning. Even outside the Team Agent platform, one could use an ExecutorEngine with a predefined team config to achieve automated problem-solving that goes beyond a single prompt. For instance, it can demonstrate how an agent can iteratively use tools and refine answers (ReAct pattern), or how two specialized agents can collaborate (one finds information, another uses it to answer). This provides immediate value: it showcases orchestration of LLM agents for complex tasks, which is useful in any application needing reasoning steps (e.g. a QA system that first searches documents then answers). Because it is modular, the ExecutorEngine could be integrated into other AI orchestration frameworks or extended to orchestrate non-LLM tasks as well, delivering multi-agent coordination capabilities as a reusable service.

**Implementation Approach:** We implement ExecutorEngine as a class (e.g., ExecutorEngine in executor\_engine.py) that is initialized with references to key collaborators like the AgentRuntime, ToolCaller, etc. It also needs access to a planning capability and context loader (either by constructing/owning those or via injection). Key methods and workflow in this class include:

* **execute\_team(team\_id, user\_query, user\_id=None) \-\> str**: This is the main method to execute a given team on a new query. It orchestrates the entire flow:

* Load context via ExecutionContextLoader (fetch team config, etc.).

* Initialize an execution plan via ExecutionPlanner (if using; otherwise determine order).

* Loop through agents as per the plan (or for a single-agent, loop until done). For each agent turn, prepare the prompt and get output via AgentRuntime.

* If the agent’s output indicates a tool action, call self.tool\_caller.call\_tool(...) and inject the result into the agent’s context (possibly by updating memory or passing as next prompt input).

* Manage the control flow: decide whether to give control to the next agent, repeat the same agent (if it’s continuing a ReAct loop), or finish if answer is reached.

* Optionally invoke CriticFeedbackEngine at the end to validate the answer.

* Return the final answer (and possibly any additional metadata).

* **\_process\_agent\_turn(agent, input\_message) \-\> (output\_message, done\_flag)**: A helper that runs one agent’s turn. It uses AgentRuntime to build the prompt from the current context (which includes the agent’s persona and the input\_message – which could be the user’s query or another agent’s prior output). It then gets the agent’s response. If the response is an action, it triggers the ToolCaller and returns the tool result (setting done\_flag=False so the loop knows the conversation continues). If the response is a final answer, it returns that with done\_flag=True. This function abstracts the logic for a single step, making the main loop clearer and also facilitating testing (we can unit test \_process\_agent\_turn by simulating various agent outputs).

* **\_handle\_tool\_result(agent, tool\_action) \-\> str**: Another helper to take a parsed tool request (e.g. an object with tool name and parameters), call the ToolCaller, and format the result as a message that can be given to the agent as an observation. For example, if tool\_action says “use vector\_search for 'climate data'”, this method calls TAO (via ToolCaller) and gets back a result like *“Found 3 relevant documents about climate.”*. It then might format it as: "Observation: Found 3 relevant documents about climate." which AgentRuntime can include in the next prompt.

* **\_log\_execution(user\_id, team\_id, status, final\_answer)** (optional): Records the execution event in the database (user\_executions log), including whether it was successful and perhaps summary info. In MVP, this could simply call TAO or a shared logging utility to insert a record. This ensures auditability (linking user and team with the query run).

The ExecutorEngine, in essence, implements a state machine for the conversation: it manages which agent is “active”, tracks the conversation state (likely via AgentMemoryManager), and knows when to stop.

**Example – Single-Agent ReAct Loop:** To illustrate the logic, consider a team with a single agent that can use tools (the ReAct pattern). The ExecutorEngine would keep calling the same agent in a loop until the agent signals it has a final answer. Pseudocode for this might look like:

\# Pseudocode: Single-agent ReAct execution loop  
agent \= team.agents\[0\]  
conversation \= \[\]  \# to hold message history  
done \= False  
while not done:  
    prompt \= agent\_runtime.build\_prompt(agent, conversation)  
    response \= agent\_runtime.get\_response(prompt)  
    action \= agent\_runtime.parse\_action(response)  
    if action is not None:  
        \# Agent wants to use a tool  
        result \= tool\_caller.call\_tool(agent.id, action.name, action.params)  
        \# Add the tool result as an observation for the agent  
        observation\_msg \= f"Observation: {result}"  
        conversation.append({"agent": "tool", "message": observation\_msg})  
        \# Loop continues with same agent using new info  
    else:  
        \# Final answer provided  
        final\_answer \= response  
        done \= True

In this example, conversation holds the messages exchanged (the agent’s thoughts or prior tool observations). The loop continues until done is True, meaning the agent decided to give an answer. Each iteration, the agent’s prompt includes any new Observation from the last tool it used.

**Example – Multi-Agent Sequential Loop:** Now consider two agents, A and B, where A’s role is to gather info (maybe via tool) and B’s role is to produce the final answer. The ExecutorEngine would orchestrate a handoff from A to B:

\# Pseudocode: Two-agent sequential execution  
conversation \= \[\]  
\# Agent A turn  
prompt\_A \= agent\_runtime.build\_prompt(agent\_A, conversation, user\_query)  
response\_A \= agent\_runtime.get\_response(prompt\_A)  
action\_A \= agent\_runtime.parse\_action(response\_A)  
if action\_A:  
    result \= tool\_caller.call\_tool(agent\_A.id, action\_A.name, action\_A.params)  
    conversation.append({"agent": "tool", "message": f"Observation: {result}"})  
    \# After using a tool, Agent A might provide an answer or not; for this pattern, assume A's job was just to fetch info  
    \# Perhaps Agent A always ends after providing info (maybe the Observation itself is the info needed).  
\# Take whatever Agent A’s final output was (if any) as input to Agent B  
input\_for\_B \= response\_A if not action\_A else result  \# e.g., if A returned some data or just used a tool  
prompt\_B \= agent\_runtime.build\_prompt(agent\_B, conversation, input\_for\_B)  
response\_B \= agent\_runtime.get\_response(prompt\_B)  
\# Agent B presumably gives final answer (or could also invoke tools, which would be handled similarly)  
final\_answer \= response\_B

In practice, Agent A might output something like “I have retrieved the data you need.” after using the tool, and that could be passed to Agent B. The ExecutorEngine would manage these transitions and ensure each agent’s context is set correctly (Agent B’s prompt might include Agent A’s output as part of the conversation history).

**Note:** The above pseudocode is simplified. The real implementation would include checks, loops (Agent B could potentially call tools too), and use the AgentMemoryManager to accumulate conversation history rather than a simple list as shown.

**Testing Plan:** We will test the ExecutorEngine with a variety of scenarios to ensure robustness:

* *Basic single-agent execution:* Create a dummy agent (no tools) that directly answers a query. Use a stub AgentRuntime that returns a fixed answer for a known prompt. Verify that execute\_team returns that answer and doesn’t enter any unnecessary loops.

* *Tool invocation flow:* Configure a single agent that is supposed to use a tool. In the test, stub the AgentRuntime to first output an “Action” (tool request) and then, on the next call (after receiving observation), output a final answer. Also stub ToolCaller to return a predetermined observation for the tool. Assert that the final answer is returned and that ToolCaller was invoked exactly once with the correct tool name and parameters. Also verify that the observation from ToolCaller was indeed included in the prompt of the second AgentRuntime call (this can be done by examining a log or the stubbed AgentRuntime’s input).

* *Multi-agent handoff:* Simulate two agents, A and B. Stub AgentRuntime such that:

* For Agent A, given the user query, it outputs some text (or possibly an action and then text).

* For Agent B, given A’s output, it returns a final answer. Use a stub ToolCaller if Agent A is supposed to use a tool. Then run execute\_team and assert that the final answer is as expected. Check the sequence: ensure AgentRuntime was called for A then for B in order. We can also verify intermediate states via logs or by enhancing the ExecutorEngine to optionally record the transcript for testing purposes.

* *Permission and error handling:* Although permission checks are primarily TAO’s responsibility, simulate a case where ToolCaller raises a PermissionError (meaning the agent tried to use a tool it wasn’t allowed). The ExecutorEngine should catch this (or the ToolCaller could return an error result to the agent). We might decide to either terminate execution with an error message or let the agent handle the “permission denied” as an observation. For test, ensure that a PermissionError does not crash the ExecutorEngine – it should handle it (perhaps by logging and breaking out with an error state). Similarly, test behavior if the LLM service is unreachable or returns an invalid response – ExecutorEngine should timeout or abort gracefully.

* *Integration test (end-to-end in-memory):* Using a lightweight in-memory setup, combine a simple AgentRuntime, ToolCaller, and actual tool logic to run a trivial scenario. For example, define a fake tool “echo” that just returns what you send it. Configure an agent that if it sees a certain keyword in the query, it will request the echo tool (simulate via AgentRuntime logic), otherwise it answers directly. Run ExecutorEngine with this and verify the control flow: when the keyword is present, it indeed calls the tool and uses the response; when not, it returns directly. This ensures the branching logic in ExecutorEngine works.

Through these tests, we validate that the ExecutorEngine correctly sequences agent interactions, handles tool calls and agent outputs, and stops under the right conditions.

### 3.2 AgentRuntime

**Purpose & Responsibilities:** The **AgentRuntime** submodule handles all interactions with the language model on behalf of an agent. Its responsibilities include **prompt construction, model invocation, and response parsing** for each agent turn. Essentially, AgentRuntime is the layer that translates an agent’s state and context into a prompt string (or structured input) for the LLM, calls the LLM (Ollama or other API) to get the completion, and then interprets the completion to determine the agent’s intended action. By centralizing this logic, we make it easy to adjust prompting strategies or swap out the LLM backend without affecting the ExecutorEngine’s orchestration logic.

Key tasks for AgentRuntime: \- **Prompt Template Composition:** It builds a prompt that injects the agent’s persona/role, the current conversation or relevant context, and instructions for how the agent should respond. This often includes a system message like: *“You are Agent Alice, a financial analyst. You have the following tools: \[VectorSearch, Calculator\]. When needed, you will use the format Action: \<Tool\>(\<params\>) to use a tool. Answer the user’s question to the best of your ability.”* plus the dialogue so far. If domain knowledge or tenant info is available, that can be injected as well (e.g., *“Organization policy: always ensure data is from 2021 or later.”*). \- **Model Invocation:** It sends the prompt to the LLM via the configured client (for example, an HTTP call to Ollama’s API with the prompt and model name) and awaits a response. This might involve handling streaming versus non-streaming outputs, but typically we get a complete response text. \- **Response Handling:** Once the LLM responds, AgentRuntime must parse it to determine if the agent provided an answer or requested an action. We might use conventions in the prompt to help with this. For instance, we can instruct the agent to begin a tool request with a special token or format like "Action:". The AgentRuntime can then search the response for that pattern. If found, it will extract the tool name and parameters (converting them perhaps from a JSON or bracketed format into a structured ToolAction object). If no such pattern is present, it assumes the response is an answer to be given to the user (or to the next agent).

By encapsulating these steps, AgentRuntime makes it easy to tweak the agent prompt templates or parsing rules independently of the rest of the system. It also serves as a potential integration point for different LLMs or frameworks – e.g., in the future we might use a LangChain agent or a different prompting style, which would primarily affect this component.

**Standalone ROI:** AgentRuntime as a standalone unit provides a reusable interface to LLMs for agent behaviors. For example, it can be used in any scenario where you need to programmatically generate prompts based on some template and context, and interpret the results. It essentially acts as a mini “agent framework.” A developer could use AgentRuntime outside of TAE to quickly prototype an agent that can decide on tool usage from its output format. Additionally, by isolating model calls, this component could be swapped to use different models (e.g., OpenAI GPT-4, or local models) by just changing the client implementation and templates. Thus, AgentRuntime delivers ROI by making LLM interactions more systematic and testable (with clear input-output), which is valuable in any LLM-driven application.

**Implementation Details:** We implement AgentRuntime likely as a class (e.g. AgentRuntime in a module like agent\_runtime.py or package) that is initialized with configurations such as the prompt templates and the LLM client to use. The class might have internal templates for system messages and format instructions, which could be loaded from a configuration file or defined as constants.

Key functions in AgentRuntime include:

* **build\_prompt(agent, conversation\_history, user\_input=None) \-\> str:** Constructs the full prompt for the agent’s next turn. It will typically do something like:

* Start with a **System Prompt** that includes the agent’s persona/role description and instructions. For example:

* You are \*\*{agent.name}\*\*, a {agent.role}. Your persona: {agent.persona}.  
  Tools available: {list of tools and usage format}.  
  If you need to use a tool, output in the format: Action: \<ToolName\>(\<parameters\>).  
  Otherwise, provide your answer directly.

* (The exact wording can be refined for better LLM performance.)

* Include any **context** such as tenant or domain guidelines. For instance, if ExecutionContextLoader provides a tenant note (“All financial data must be in USD.”), the system prompt can append that as a rule the agent should follow.

* Append the **conversation history** in a suitable format. If using a chat format, we might prepend each message with speaker labels. E.g.,

* User: {the original user question}  
  Agent1: {Agent1's previous response (if any)}  
  Agent2: {Agent2's previous response (if any)}  
  ...

* But in a turn-based prompting approach, we might not include full history every time due to token limits; instead, we might include the last relevant exchanges or a summary via AgentMemoryManager.

* Finally, include the **current user’s query or the input for this agent** as a prompt. In a multi-agent chain, the “user” for Agent B might effectively be Agent A’s output. So we might do: User: {some input}\\n{Agent.name}: and let the model complete the agent’s response. In other cases, we call the model with a single turn prompt and get a full answer.

The output of build\_prompt is a string ready to be sent to the LLM. We may maintain multiple templates (for example, one for the first agent vs subsequent agents, or different styles for different roles), but a unified builder method can handle variations based on agent metadata.

* **get\_response(prompt: str) \-\> str:** Sends the prompt to the LLM and returns the raw response text. Internally, this will use the LLM client configured (which could be an HTTP POST request to Ollama’s API at OLLAMA\_URL with the prompt and model specified). We may need to handle streaming responses; for MVP, we can wait for the full completion. This method should also incorporate basic error handling (timeout management, retries if necessary, and error logging if the model fails to return a valid response).

* **parse\_response(response: str) \-\> Tuple\[Optional\[ToolAction\], Optional\[str\]\]:** Analyzes the model’s response to see if it’s requesting a tool or giving an answer. One design is to look for a specific prefix or pattern. If our prompt instructions say to output tool use as Action: ToolName(args), we can search for "Action:" in the response. If found:

* Extract the tool name and parameter string between parentheses. We might define a simple grammar: e.g., Action: VectorSearch("climate change").

* Create a ToolAction object (a small dataclass or dict) like ToolAction(name="VectorSearch", params={"query": "climate change"}).

* Return (tool\_action, None) to indicate an action was parsed. If no "Action:" prefix is found (or whatever convention we choose, like the agent might output Tool: or some JSON), then we assume the response is an answer. We then return (None, answer\_text) where answer\_text is the full response (or possibly we might strip any agent signature). We will likely refine this parsing with more robust methods in future (like requiring the agent to output a JSON blob or a \<TAG\> for actions, etc.), but string parsing is sufficient for MVP given controlled prompt structure.

* **format\_observation(observation: str) \-\> str:** (Optional helper) Formats a tool’s result into a string that can be appended to the prompt for the next call. This might just prepend "Observation:" as we did in our examples, or whatever format the agent was told to expect for results.

**Prompt Template Example:** Below is an example stub of how the AgentRuntime might compose a prompt with context injection:

\# Example prompt template composition  
system\_instructions \= (  
    f"You are {agent.name}, a {agent.role}.\\n"  
    f"Persona: {agent.persona}\\n"  
    f"Tools available: {', '.join(\[t.name for t in agent.tools\])}.\\n"  
    f"When you use a tool, respond with: Action: \<ToolName\>(\<parameters\>)\\n"  
    f"Otherwise, provide a direct answer.\\n"  
)  
\# Add tenant or domain context if available  
if tenant\_guidelines:  
    system\_instructions \+= f"(Tenant guideline: {tenant\_guidelines})\\n"  
if domain\_info:  
    system\_instructions \+= f"(Domain context: {domain\_info})\\n"

conversation\_history \= ""  
for msg in conversation\_history\_list:  
    speaker \= msg\["agent"\]  
    content \= msg\["message"\]  
    conversation\_history \+= f"{speaker}: {content}\\n"

user\_part \= f"User: {user\_input}\\n{agent.name}:"

prompt \= system\_instructions \+ conversation\_history \+ user\_part

This would yield a prompt like:

You are Agent Alice, a Data Analyst.  
Persona: An expert in financial data analysis, who is helpful and precise.  
Tools available: VectorSearch, Calculator.  
When you use a tool, respond with: Action: \<ToolName\>(\<parameters\>)  
Otherwise, provide a direct answer.  
(Tenant guideline: All monetary values should be in USD.)  
(Domain context: Financial Q3 Reports Database)

User: What were our company's earnings growth in Q3 compared to Q2?  
Agent Alice:

AgentRuntime would send this to the LLM. Suppose the model returns:

Action: VectorSearch("Q3 earnings vs Q2")

The AgentRuntime parse\_response identifies this as a tool request (ToolAction(name="VectorSearch", params={"query": "Q3 earnings vs Q2"})). The ExecutorEngine then calls the ToolCaller, gets an observation, and then AgentRuntime will be used again to formulate the next prompt including:

Agent Alice: I will search for "Q3 earnings vs Q2".  
Action: VectorSearch("Q3 earnings vs Q2")  
Observation: Found a report stating Q3 earnings grew 5% over Q2.  
Agent Alice:

And the model might then answer:

Our company's Q3 earnings were 5% higher than in Q2.

This illustrates prompt building with context and the cycle of action/observation.

**Testing Plan:** To test AgentRuntime thoroughly:

* *Prompt assembly tests:* Given a known agent config (with persona "test persona" and tools \["ToolA", "ToolB"\]) and some conversation history, call build\_prompt and inspect the string. Verify that it contains the agent’s name, persona, the tool names, and any provided guidelines or domain context. We might design the prompt format and then assert that important markers (like the "Tools available:" line or the format instruction) are present. These tests ensure we don’t accidentally omit crucial info.

* *Model invocation tests:* Here we can stub the LLM client. Instead of calling a real model, we inject a fake get\_response implementation that returns a pre-canned string (or echo’s part of the prompt back). For example, one test could set the LLM client to return "Hello world." for any prompt, and we verify get\_response indeed returns "Hello world." (ensuring that the integration with the HTTP client is correct in a real scenario could be done with a integration test, but in unit tests a stub suffices).

* *Response parsing tests:* These are crucial:

* Test that when given a string like "Action: SomeTool(\\"param\\")", the parse\_response correctly yields a ToolAction with name "SomeTool" and params containing "param". Try variations: different spacing, single vs double quotes in parameters, etc. We should also test a case with multiple parameters if our format allows (e.g. Action: APICall("arg1", "arg2")).

* Test that when given a normal answer string (no "Action:"), parse\_response returns (None, answer) and that the answer text is intact.

* Test edge cases: if the model response is empty or just whitespace, parse\_response should handle gracefully (maybe treat as answer \= "" or throw an error that ExecutorEngine can catch). If the model response includes an "Action:" but malformed (like no closing parenthesis), ensure our parser does not crash – it might return no action and treat it as answer, or raise an error that we handle upstream.

* If we plan to support a JSON format in future, test that detection as well (for now, sticking to simple prefix detection).

* *Integration test with ExecutorEngine:* Combine a real AgentRuntime (pointing to a stub LLM that behaves deterministically) with a minimal ExecutorEngine to see end-to-end. For instance, configure the stub LLM to:

* On first prompt, return "Action: Echo(\\"hello\\")" (pretending there's a tool named Echo).

* On second prompt (after observation), return "Final answer.". Then verify that the ExecutorEngine \+ AgentRuntime together result in "Final answer." as output, and that ToolCaller (stubbed to just echo back the input) was called with "hello". This test ensures AgentRuntime’s parsing and prompt building align with ExecutorEngine’s usage.

AgentRuntime is relatively easy to unit test since it’s mostly pure functions (build string, parse string) and an external call. By covering these, we ensure that agents will be prompted correctly and their outputs correctly understood, which is critical for the multi-agent loop to function.

### 3.3 ExecutionPlanner

**Purpose & Responsibilities:** The **ExecutionPlanner** is responsible for creating a structured plan (sequence or DAG of steps) for the execution of a team of agents. While the ExecutorEngine can hard-code a simple order of agent turns, the planner provides a layer of abstraction that decides *which agent should act when* and *in what order tools or subtasks should be executed*. In advanced scenarios, this could mean constructing a **LangGraph** – a graph of agent nodes and tool nodes that represents the flow of information and decisions. For example, if a user query might need parallel searches or a conditional step, the planner would capture that logic. In the MVP context, the ExecutionPlanner’s role is limited: it might simply output a list of agent IDs in the order they should be invoked, or a trivial graph with a start-\>Agent1-\>Agent2-\>end. Essentially, it formalizes what would otherwise be implicit in code, which will make it easier to extend in the future.

**Standalone ROI:** As a standalone unit, ExecutionPlanner demonstrates automated reasoning about task structure. Even on its own, a planner can be given a problem and produce a multi-step solution outline (which could then be executed by some executor). This separation of planning from execution is a common design in complex systems (think of a planner that outputs a plan and an executor that follows it). The ROI is that different planning strategies can be experimented with independently. For instance, one could plug in a more sophisticated planner (maybe an LLM that reads the team’s capabilities and the query, and suggests a plan) without changing the ExecutorEngine. Moreover, ExecutionPlanner could be reused in other contexts where tasks need to be broken into sub-tasks (even outside multi-agent scenarios). In our platform, it adds value by making the orchestrator (ExecutorEngine/TAO combination) more flexible and by enabling more complex workflows down the line.

**Implementation Structure:** We can implement ExecutionPlanner as a class (e.g., ExecutionPlanner in execution\_planner.py). Its interface might be something like plan\_execution(team\_config, query) \-\> Plan. The Plan could be a simple object that contains an ordered list of steps or agents for MVP, but is envisioned to be a DAG structure in the future. We might define a PlanStep data class that can represent either an agent invocation or a tool invocation, with fields like agent\_id (or tool name), and perhaps pointers to next steps.

In MVP, an extremely simple implementation might be: \- If the team has N agents, the plan is just \[Agent1, Agent2, ..., AgentN\] in that order. (We assume the team config might even specify an intended order or a leader agent, but lacking that, we just use the list as given). \- If an agent is designated as “solo” (like no other agent needed), the plan would just be \[Agent\]. \- If in the future we add a “Critic” agent role in the team definition (some team might explicitly have a member who is a Critic), the planner might decide to put that agent at the end of the sequence, after the main answer is formulated.

For slightly smarter planning, the ExecutionPlanner could consider the query content or agent tool capabilities. For example, if the query is a simple factual Q, maybe only the Q\&A agent is needed (skip the retrieval agent), but if it requires info retrieval, ensure the retrieval agent is invoked first. However, that level of dynamic decision might be beyond MVP scope – MVP will likely assume the team is built such that every agent has a purpose in the sequence.

**LangGraph-based DAG:** The mention of LangGraph suggests a possible approach where the planner can produce a DAG structure understood by a LangChain or similar framework. We might not implement that fully now, but we can lay groundwork. Perhaps the Plan could include parallel branches or multiple cycles. For now: \- We ensure our Plan representation can later be expanded. For example, define Plan as:

class Plan:  
    steps: List\[PlanStep\]  \# sequential for now  
    \# In future, could include a graph adjacency list or a more complex structure.

\- Each PlanStep might have an agent\_id and maybe a depends\_on list for future DAG (to indicate this step should only run after certain others complete).

**Integration with ExecutorEngine:** ExecutorEngine will use the ExecutionPlanner at the start of execution. For MVP, if planner.plan\_execution(team, query) returns a list of agent IDs, the ExecutorEngine can iterate through them in order. If an agent in the middle needs to loop with a tool, that is handled at execution time, not necessarily in the plan (the plan is more about the high-level order of agents, not every micro-step).

**Testing Plan for ExecutionPlanner:**

* *Simple team ordering:* Given a fake team config with 3 agents, ensure plan\_execution returns a Plan listing those 3 agents in the correct order. If the team config has an order field or roles, test that it respects that (e.g. if team roles are like “Planner, Solver, Critic”, maybe the plan should list Planner then Solver, and exclude Critic if not enabled – depending on design).

* *Edge cases:* Team with 1 agent → plan is just \[that agent\]. Team with 0 agents (shouldn’t happen because a team should have at least one member) → maybe return an empty plan or throw an exception (test that behavior).

* *Conditional logic test:* If we include any logic that depends on query or domain, test those. For example, if we implement “if any agent has a tool requiring domain X and query mentions X, put that agent first”, we’d craft a scenario to verify it. Initially, likely no such logic.

* *Future structure placeholder:* If we have a notion of parallel tasks (not in MVP but possibly stubbed), we could include a disabled code path and a test for it (for instance, a special query or config that triggers a Plan with two branches). This could be as simple as making sure our Plan can hold multiple sequences in parallel for later. But that might be overkill for MVP testing.

Overall, ExecutionPlanner in MVP will be straightforward. The test focus will be to ensure it correctly interprets whatever fields in team config indicate execution flow. Because the team config is largely static (list of agents with roles), the planner might largely be static as well. The main value of including it now is to signal how we will extend planning capabilities later.

*(In summary, ExecutionPlanner is minimal in MVP, so its testing and complexity are low. We ensure it doesn’t conflict with the ExecutorEngine’s expectations and that it can be bypassed or enhanced without altering ExecutorEngine code.)*

### 3.4 AgentMemoryManager

**Purpose & Responsibilities:** The **AgentMemoryManager** handles the memory and context tracking for agents during execution. This has two aspects: 1\. **Short-term memory (per execution):** Keeping track of the ongoing conversation history so that agents have context of what has already been said or found. For example, if an agent used a tool and we got an observation, that needs to be remembered and possibly included in the prompt for the next agent or next loop iteration. Also, if multiple turns occur (like agent A said something, then agent B, then back to A), we need to retain those messages. 2\. **Long-term memory (across executions or additional knowledge):** Storing and retrieving information that an agent might have learned in past sessions, or general knowledge that was too large to fit in the prompt. This could involve vector embeddings and semantic search for relevant past data.

In MVP, the primary role of AgentMemoryManager is to manage **conversation state** during a single execution. It can be as simple as an object that appends messages to a list and can retrieve the recent N messages for prompt inclusion. However, by designing it as a separate module, we allow for future enhancements like: \- Automatic summarization of earlier parts of conversation if it gets long (to avoid context window issues). \- Storing key facts from the conversation in a vector store such that if an agent forgets something said 10 turns ago, it can be reminded by searching that store. \- Implementing “episodic memory” for agents: e.g., after finishing an execution, some summary could be saved to a knowledge base so that next time the agent can recall it.

AgentMemoryManager may also integrate with the platform’s knowledge storage: \- It might interface with **Neo4j** to attach information gleaned to certain nodes (the TAO design mentioned possibly updating graph nodes with agent learnings). \- It could use **Qdrant** to store embeddings of conversation chunks for later retrieval (like a semantic memory of what was discussed).

**Standalone ROI:** As a standalone component, AgentMemoryManager can be reused in any conversational agent context to add memory capabilities. For instance, if one is building a chatbot, plugging in this memory manager can immediately give the bot the ability to reference prior parts of the conversation or maintain state. It also abstracts away the details of how memory is stored (in-memory list vs external DB vs vector store). This means it can be independently improved (e.g., swap an in-memory list with a sophisticated memory retrieval algorithm) without affecting the core execution logic. In enterprise scenarios, having a modular memory manager is valuable – you could tailor it to store conversation logs in a secure database or apply retention policies, all without altering how the agent logic operates.

**Implementation Approach:** We might implement AgentMemoryManager as a class (AgentMemoryManager in agent\_memory.py) which is instantiated at the start of an execution (or per agent). There are a couple of design options: \- A single MemoryManager instance for the whole team execution, which tracks all messages with tags as to which agent said it or if it was a tool observation. \- Or one instance per agent, which tracks what that agent has “seen”. However, since all agents in a team indirectly share context (Agent B should know what Agent A said if it’s relevant), it might be simpler to have one conversation log accessible to all.

For simplicity, consider a single manager with methods like: \- add\_message(sender, content): Records a message. The sender might be an agent name/ID or "User" or "Tool" to indicate the source. \- get\_recent\_messages(n): Returns the last n messages in some formatted way. \- summarize\_long\_history() (future): If more than X messages, generate a summary and prune detail. \- clear() or start\_new() for resetting state when a new execution begins.

Additionally, for persistent memory, methods like: \- store\_memory(agent, data): Save something learned to a database or vector index. \- retrieve\_memory(agent, query): Search the long-term memory for relevant info.

MVP likely won’t implement persistent storage beyond perhaps appending to Neo4j relationships that an agent used certain info. But we can plan: for example, if an agent finds an important piece of info (like “Earnings grew 5%”), maybe mark that in a structured form somewhere. This is speculative and probably out of MVP scope.

**Use in Execution:** The ExecutorEngine/AgentRuntime will use AgentMemoryManager when building prompts: \- Instead of passing a raw list of messages around (as we did in pseudocode), the ExecutorEngine can query MemoryManager for the conversation so far. \- For example, conversation\_history \= memory\_manager.get\_recent\_messages(10) could return the last 10 exchanges formatted appropriately to include in AgentRuntime.build\_prompt. \- After each step, the ExecutorEngine calls memory\_manager.add\_message(speaker, content) for the agent’s output and similarly for any observations.

If we implement memory summarization, MemoryManager might automatically summarize once the conversation exceeds a certain length and keep a summary in place of raw messages for older parts.

**Data Schema for Memory:** If we choose to persist memory, one approach is using Qdrant: \- We could dedicate a **Qdrant collection per agent or per team**. For instance, collection name "team\_{team\_id}\_memory". \- When storing, we embed the text of a conversation turn or a summary of it and store with metadata like agent, timestamp. \- Then retrieve\_memory might embed the new query or a context string and do a similarity search. \- This is akin to how one might do long-term memory retrieval by semantic similarity. \- In MVP, we likely won’t go this far, but we mention it as future.

Alternatively, use **Postgres**: \- A table agent\_memories(agent\_id, content, vector) where vector is an embedding. Could use the pgvector extension. \- But since Qdrant is already set up for vectors, that is easier.

**Testing Plan:**

* *Adding and retrieving messages:* Use MemoryManager in isolation. Call add\_message a few times (simulate a short dialogue), then get\_recent\_messages(n) and ensure it returns the last n in correct order. If formatting is applied (like prefixing with speaker labels), check that format is correct. For example, if messages added were \[("User", "Hello"), ("Agent1", "Hi")\], and we ask for recent 2, we should get something like \["User: Hello", "Agent1: Hi"\] or a structured form that AgentRuntime can easily incorporate.

* *Overflow handling:* If we set a max memory length (not required in MVP but we might cap to avoid huge prompts), simulate adding more messages than the cap and see if the oldest get dropped or summarized. For MVP, if no cap, then just ensure it can handle a reasonably large number of messages without performance issue (this is trivial if using a list).

* *Long-term memory stub:* If any functionality is included (even as stub) to store memory externally, test that calling it doesn’t error. For example, if store\_memory is supposed to insert into a DB, we could mock the DB call and ensure it’s invoked.

* *Integration tests:* Use MemoryManager with ExecutorEngine and AgentRuntime in a test:

* Ensure that after each agent turn, the conversation memory has been updated. This could be done by injecting a hook or by checking the memory content after execution.

* A specific test scenario: the user asks two questions in one session (if we allow multi-query sessions). The memory manager might persist from question1 to question2 if not reset. In MVP, probably each query triggers a fresh execution context and memory is reset. We can test that starting a new execution either uses a new MemoryManager instance or a cleared state, to avoid leakage between queries (unless that’s a feature we want).

* If we plan for user follow-up in same session (like a conversation with the team), then we do want to persist memory across queries. That might be more of a TAB or UI concern, but something to keep in mind.

By testing these, we verify that context is properly accumulated and made available to agents, which is vital for correctness (e.g., agent B needs to see what agent A found, etc.). Although MVP memory is basic, establishing these tests helps when we upgrade memory features later.

### 3.5 ToolCaller

**Purpose & Responsibilities:** The **ToolCaller** submodule is responsible for executing external tool actions on behalf of agents by interfacing with TAO’s tool execution APIs. In the system’s design, agents themselves do not directly call external services; instead, when an agent decides to use a tool, that request is funneled through the Model Context Protocol (MCP) to the TAO module, which actually executes the tool (since TAO houses the ToolRegistry and adapters). ToolCaller encapsulates all the logic needed to make that call and retrieve the result. Its main responsibilities: \- Take a structured tool action request (for example, a tool name and parameters that an agent wants to run, plus information on which agent is requesting it). \- Format this request according to the interface that TAO expects (could be an HTTP API call or an RPC invocation). \- Handle the response from TAO, including success or error. If successful, return the tool’s output in a form that can be inserted into the conversation (often plain text). If an error or permission issue occurs, decide how to propagate that (it might throw an exception, or return a special error message for the agent).

Because TAO enforces tool permissions and logs each execution, ToolCaller in TAE doesn’t need to implement permission checks itself – it just makes the call in the context of a specific agent and tool, and TAO will respond with either the result or an error if not allowed. However, ToolCaller should be aware of how to present errors back to the agent if needed (e.g., if TAO says “permission denied”, maybe present that as an observation or raise an exception that the ExecutorEngine catches and handles by stopping the execution).

**Standalone ROI:** As an independent piece, ToolCaller can be reused in any scenario where a process (or agent) needs to call a set of external tools in a standardized way. It essentially acts as an **API client for the ToolRegistry**. If separated, it could be used by a different agent framework to leverage the same TAO tool ecosystem without using the rest of TAE. For example, if someone wanted to write a simple script that just calls tools by name through TAO, they could use this module. It hides the details of network calls and protocol, providing an easy method like execute\_tool(team\_id, member\_id, tool\_name, params) that reliably triggers the right action. This abstraction provides immediate value by simplifying how developers integrate tool usage – they don’t need to know the details of TAO’s endpoints or response formats.

**Implementation Details:** We implement ToolCaller likely as a simple class or utility (e.g., ToolCaller in tool\_caller.py). It will need configuration for how to reach TAO: \- We might have an environment variable like TAO\_API\_URL or separate ones for host/port (if TAO runs at a known internal address). For instance, if TAO’s MCP server runs on HTTP at http://tao:8100, ToolCaller can default to that, or allow configuration injection. \- Also possibly an authentication token if TAO requires auth on its API (though within a closed network maybe not needed). \- Perhaps a mode flag: if MCP\_TRANSPORT\_MODE is set to "http" (the likely default), ToolCaller will do HTTP calls. If it were "grpc" or "stdio", different logic could be used. MVP likely uses HTTP.

**Operation:** \- **Method call\_tool(member\_id, tool\_name, params) \-\> str:** This would construct a request to TAO. We need to include context of which agent (member) is calling the tool, because TAO uses that to check permissions and to log which team member executed the tool. TAO’s API might be something like POST /teams/{team\_id}/members/{member\_id}/tools/{tool\_name} with a JSON body of parameters. Or possibly POST /execute\_tool with all these in the body. We will assume some reasonable API shape; since we control both sides, we could define it. \- For example, body might be {"tool": "vector\_search", "params": {"query": "climate change"}, "member\_id": "agent1\_uuid", "team\_id": "team\_uuid"}. \- TAO will perform the action (e.g. call the VectorSearch adapter) and respond with a result, maybe {"result": "Found X documents ..."} or just plaintext result. \- ToolCaller then returns the result as a string. Possibly it might strip any JSON wrapper if TAO returns structured data and we just need text.

* Error handling:

* If the HTTP request fails (connection error, timeout), ToolCaller should raise an exception or retry a bit. The ExecutorEngine can catch exceptions to abort gracefully.

* If TAO returns an error status (like 403 for permission or 500 for tool failure), ToolCaller can either:

  * Translate that into a Python exception (e.g., PermissionError for 403, a generic ToolExecutionError for others).

  * Or return a special result string like "Error: permission denied". But returning as a normal result might confuse the agent vs throwing an exception which the ExecutorEngine can decide how to handle (maybe by injecting "Observation: \[Tool denied\]" to the agent).

* Considering clarity, raising an exception up to ExecutorEngine might be best. Then ExecutorEngine could decide to stop execution and, for MVP, maybe return an error message to user or just stop. Alternatively, we could catch it and inject as observation to let the agent handle it. But that may cause the agent to freak out, depending on prompt. Simpler: abort on tool failure for now.

**Integration with TAO:** According to integration design, when an agent requests a tool, TAE (via ToolCaller) should call TAO’s MCP interface. In practice, we might call the same endpoints that an external caller would for tool execution. The TAO module likely exposes an HTTP API (MCP Server) where one can POST a Tool request; we ensure to use the correct endpoint and include the agent’s identity. The Model Context Protocol’s idea is to have a unified way to call any tool with standardized input/output, which simplifies ToolCaller’s job – it mostly just passes through parameters.

**Testing Plan:**

* *Direct call success:* In a test environment, we might not have a live TAO server, so we simulate it. Use a library like requests but point to a dummy server or monkeypatch requests.post in tests. For example, monkeypatch ToolCaller.\_post to return a predefined response. Then test that call\_tool("member123", "echo\_tool", {"text": "hello"}) returns "hello" if that’s the defined dummy result. Essentially, ensure the request is being formed correctly: we could capture the URL and payload the ToolCaller tries to send. (Using dependency injection: maybe ToolCaller can be initialized with a client object that we can fake for tests.)

* *Permission error handling:* Simulate a 403 response from TAO. For instance, our dummy requests.post returns status\_code 403 with message "Forbidden". Verify that ToolCaller raises a PermissionError (we’d implement it to do so). Similarly simulate a 500 or timeout to see that it raises a generic exception. We want to ensure it doesn’t return a normal result in these cases because upstream should be alerted to the failure.

* *Integration with ExecutorEngine:* We can integrate a stub ToolCaller in an execution test. For example, stub ToolCaller such that when called with tool "X", it returns a known string. Then run ExecutorEngine with an agent that will request tool "X". Verify the final output includes the effect of that tool result. This indirectly tests that ExecutorEngine and AgentRuntime are using ToolCaller correctly. (We already planned such integration tests under ExecutorEngine’s plan.)

* *Real integration (optional):* If possible in a staging environment, spin up TAO’s server and register a simple test tool (like a ping tool). Then let ToolCaller call it for real and verify the response flows. This is more of an integration test that might be done when both modules are in place.

Since ToolCaller is relatively thin (calls an API and returns result), much of its testing is about handling various outcomes properly. With these tests, we ensure that when an agent asks for a tool, the system will actually get the result or appropriately handle failure, which is crucial for trust in the tool-using capability of agents.

### 3.6 CriticFeedbackEngine

**Purpose & Responsibilities:** The **CriticFeedbackEngine** provides a mechanism to **evaluate and possibly improve the output of the agents** before finalizing a response. Its role is somewhat analogous to a reviewer or quality control step that can catch mistakes, low-quality answers, or policy violations. In practice, this might be implemented as an additional agent (a "Critic" LLM) or a set of heuristic checks. The CriticFeedbackEngine can either simply give a verdict (pass/fail) on the final answer or generate feedback/suggestions for improvement.

For MVP, the CriticFeedbackEngine is likely **optional and minimal**: \- It might not be enabled by default, or might just log that a critic step could happen. \- If implemented, maybe it checks if the final answer is non-empty and relevant to the question (a naive check to avoid an agent just saying "I don't know" incorrectly). \- Alternatively, it might not be implemented at all in the first version, but we include the design so that we can add it later.

In future iterations, CriticFeedbackEngine could: \- Use another LLM prompt like: *“You are a Critic Agent. The user asked X, the team answered Y. Evaluate the answer for correctness and completeness. If it's not good, suggest a reason.”* The output could be used to either adjust the answer or trigger another round of reasoning. \- Enforce compliance: e.g., check the answer doesn't contain disallowed content or confidential info. This might be more rule-based or involve calling an external moderation API.

**Standalone ROI:** Even on its own, a Critic/feedback mechanism is valuable. Many AI applications benefit from a second-pass review of outputs. The CriticFeedbackEngine can be applied to any LLM output to improve reliability – for example, to filter out hallucinations or to provide an explanation. In the Team Agent context, it can be reused across different teams to maintain quality standards. So isolating it as a module means one could take it and use it in a simpler single-agent scenario just as well to double-check answers. It also demonstrates the pattern of **Agent \<\> Critic** which is a known approach to boost accuracy (two AI agents, one to solve and one to critique).

**Implementation Outline:** We can implement CriticFeedbackEngine as a class (e.g., CriticFeedbackEngine in critic\_feedback.py) that might have a method like evaluate\_response(query, final\_answer, context) \-\> Feedback. The Feedback could be a simple boolean pass/fail or a message.

For MVP, possibilities: \- If we want to include something, a simple rule-based approach: \- If final\_answer contains certain keywords that indicate uncertainty (like "I am not sure", "I cannot find"), and maybe if the domain expects a concrete answer, we flag that. \- If final\_answer is very short or doesn't address the query (we could compare overlap of words between question and answer; if none, maybe irrelevant). \- If there's a known ground truth (not in general, but for testing we might have known Q\&A pairs to test the critic). \- These are brittle, but MVP could just note potential issues. \- If hooking an LLM: \- Possibly call the same LLM with a prompt to rate the answer. But using the same model to critique can be iffy; ideally a specialized or larger model might do better. For MVP, perhaps not.

* Optionally, integrate a **human-in-the-loop mode**: Critic could simply tag it and require a human review if uncertain (not likely needed in initial internal setting, but for future safety, might consider it).

**Workflow with ExecutorEngine:** If integrated, after ExecutorEngine obtains the final answer from the last agent, it would call critic\_engine.evaluate(query, answer, conversation\_log): \- If the feedback says "OK", then continue to return answer. \- If feedback says "Not OK" and perhaps provides a suggestion, the ExecutorEngine could decide to do one of: \- Run another iteration: possibly prompt one of the agents (maybe the final agent) with the critique to try to improve. For example, if Critic says "The answer might be missing data from Q3", the ExecutorEngine could feed that back to the solver agent. \- Or attach the critique as part of the result (maybe as a disclaimer). \- Or simply log it and still return the answer but mark it as possibly low quality. \- For MVP, likely we’d just log or ignore the critique due to complexity of re-looping.

We should also ensure if Critic is optional, the system runs fine without it (maybe ExecutorEngine has a flag or if critic\_engine is None, just skip).

**Testing Plan:**

* *Heuristic evaluation tests:* If using simple rules, write tests where we feed a query and answer:

* Query: "What is 2+2?" Answer: "It’s 5." Critic might detect incorrectness if it has some knowledge or if we simply know in test to expect fail. But we likely won't have ground truth, so maybe skip correctness logic unless trivial domain.

* Answer: "I don't know." Possibly Critic should flag this because maybe the system expects some attempt (though "I don't know" might be valid if team truly can't know).

* Answer unrelated: Query: "What’s the weather?" Answer: "I like pizza." We could have Critic measure similarity (very low similarity, so flag). We’d verify that evaluate\_response returns a Feedback indicating an issue for these cases, and a pass for obviously good ones.

* *Critic as LLM test (if implemented):* If we decide to do an LLM prompt for critique, we can stub the LLM like we do with AgentRuntime. E.g., have CriticFeedbackEngine call an LLM client with "Evaluate this answer", and stub it to return "Answer seems fine." for certain inputs and "Answer is incorrect" for others. Then check that we interpret "incorrect" as fail. This might be too much detail given MVP likely not doing LLM-based critic.

* *Integration test:* If Critic is plugged in, test a full run where the critic is triggered. For example, set up a scenario where agent returns an obviously low-quality answer. Make the CriticFeedbackEngine mark it as bad, and perhaps configure ExecutorEngine to do one retry. We can stub AgentRuntime such that on second try the agent gives a better answer. Then see if final output is improved. This is a complex multi-step test but would demonstrate critic loop working. If not implementing the loop, we might just test that even if Critic flags something, in MVP we still return the original answer (but perhaps with a note). Possibly we won’t implement the loop for MVP, so integration might just ensure it logs feedback.

Since CriticFeedbackEngine is likely minimal in MVP, tests will be lightweight. But designing them helps to ensure that if we flip the switch to use it, it doesn’t break the flow.

### 3.7 ExecutionContextLoader

**Purpose & Responsibilities:** The **ExecutionContextLoader** is in charge of gathering all necessary configuration and context data needed to start an execution of a team. Its primary function is to load the **team definition** from persistent storage (or via TAO) so that TAE knows the composition of the team (members, roles, tools, etc.). Additionally, it can fetch related context like: \- **Tenant information:** e.g., tenant-wide settings or guidelines that should apply. If each tenant organization has specific rules, those might be stored in a tenant config (like in Postgres tenants table as a JSONB config). The loader would retrieve that if applicable. \- **Domain knowledge links:** Teams can be associated with domain knowledge (in TAO’s model, an agent team can link to zero or more domain knowledge configs). The loader might fetch identifiers or descriptions of those knowledge domains. It might even proactively fetch some key data (perhaps, for example, retrieving a summary or relevant facts). However, typically retrieval of actual data is done by an agent via tools at runtime, not pre-loaded, unless we have a strategy of providing a brief context upfront (like “This team is connected to Finance Reports domain – general context: ...”). \- **Tool definitions:** It can ensure we have details about the tools each agent can use. The basic info (tool name, maybe an ID) is likely part of team config or a join table (e.g., agent\_tool\_permissions). But if we want to give agents a natural language description of what each tool does (for better prompting), the loader might query the tools table (as defined in TAO’s schema) to get each tool’s description and usage info. This can then be passed to AgentRuntime for constructing the prompt (embedding a brief help like “Tool VectorSearch: searches documents by keyword”). \- **Initial state or memory:** If there is any stored memory or previous state associated with this team, loader could gather it. For example, if the team was paused mid-execution (in some future scenario) or if there are default “starting messages” or a shared long-term context, loader would fetch that. MVP probably doesn’t have that scenario.

In summary, ExecutionContextLoader centralizes all the data fetching needed so that once it’s done, the ExecutorEngine has everything required to proceed without further database calls mid-run.

**Standalone ROI:** The ExecutionContextLoader can be seen as a general configuration loader utility. As a standalone, you could use it in scenarios like: loading a team config to display it in a UI, or verifying a config by loading and inspecting it. It provides ROI by abstracting how and where config data is stored – the rest of TAE just asks this loader for a team’s info and doesn’t care if it came from a local cache, a database query, or an API call to TAO. This means the loader could be reused in any service that needs to resolve a team by ID into a concrete configuration. It also allows for caching strategies in one place: e.g., it could cache team configs in memory for quick subsequent loads (useful if many queries hit the same team repeatedly). Independent testing of the loader ensures that if there are any issues in team data (missing fields, etc.), we catch them at load time rather than during execution.

**Implementation Details:** We can implement it as ExecutionContextLoader class (in execution\_context\_loader.py). Likely usage is load\_team\_context(team\_id) \-\> TeamContext. TeamContext can be a data structure we define that holds: \- Team metadata (team name, etc., though not strictly needed for execution aside from logging). \- A list of agent member objects each containing: member\_id, name, role, persona (prompt template segment), allowed tools list (could be list of tool objects or just names). \- Tenant info (if needed): e.g., tenant\_id and any tenant-specific instructions or config values. \- Domain info: maybe a list of domain descriptors (like {"domain\_name": "...", "domain\_id": "...", "description": "..."}). \- Possibly a mapping of tool\_name to tool description (if we want to quickly get descriptions for prompts). \- We might also include things like any global tools the team can use (not tied to a single agent, but in our design, tools are per agent).

**Data sources:** \- **Postgres:** We will join agent\_teams, agent\_team\_members, and probably a linking table for tools (maybe agent\_member\_tools). The TAO documentation suggests that each member’s tools might be recorded somewhere (it mentions retrieving member’s allowed tools from DB). If such a table exists, loader would join or do a separate query to get those tool IDs, then join with tools table to get details. \- **Neo4j:** If domain knowledge relationships are stored in Neo4j (they indicated a mapping between agent teams and domains), we might either: \- Use Neo4jClient to query for domains related to this team node. \- Or TAO might have already stored these in Postgres as well (like an agent\_team\_domains table). \- For MVP, it might be simpler if TAO duplicates essential info in Postgres. But if not, loader can query Neo4j (with Cypher) to get domain names or IDs. \- **TAO API alternative:** Instead of direct DB access, the loader could call a TAO API like GET /teams/{team\_id}/full which returns the team config (with members and their tools, and references to domains). TAO’s TeamOrchestrator likely has an internal get\_team method and possibly an API for it. Using that would ensure we don’t bypass any TAO logic (like if TAO attaches default tools or does any transformation). \- The strategy might be to reduce coupling by calling TAO rather than the DB. However, since both are internal, performance might be better going to DB directly. We could implement either or even allow both by config (like a SOURCE=api vs SOURCE=db mode). \- **Cache:** We could incorporate a simple cache keyed by team\_id to avoid hitting DB repeatedly if the same team is executed often. Since team configs don’t change frequently during execution (and if they do, TAO’s VersionManager would presumably inform or we might bust cache on updates), caching is safe for short intervals. MVP could skip caching or just load fresh each time for simplicity.

**Testing Plan:**

* *Database loading tests:* These would require either a connection to a test database or better, we abstract the DB client so we can inject a fake. For example, if ExecutionContextLoader uses a postgres\_client internally, in tests we supply a stub that returns predetermined rows for queries. Simulate:

* One agent team with two members. Stub the DB calls:

  * Query to agent\_teams returns team info.

  * Query to agent\_team\_members returns two members (with their IDs, names, roles, persona).

  * Query to agent\_tool permissions returns, say, Member1 has ToolX, Member2 has ToolY and ToolZ (just by IDs).

  * Query to tools table returns details for ToolX, ToolY, ToolZ. Then call load\_team\_context(team\_id) and verify:

  * The returned TeamContext has exactly 2 agent entries, with correct names, roles, persona text matching the stub.

  * Each agent’s tools list contains the expected tool names or descriptors.

  * If we expect tenant or domain info, stub those too: e.g., a query to tenants returns a guideline "All data confidential", ensure TeamContext.tenant\_guideline \== that string.

  * Domain: stub that team is associated with domain "Finance", verify loader returns that in context. This ensures the loader correctly merges data from multiple sources.

* *API loading tests:* If we implement an API call path, we can stub the TAO API client. E.g., fake a response JSON from GET /teams/{id} and test that loader parses it to TeamContext. This might be simpler: if TAO API already returns a nice JSON with members and tools, loader mostly maps it to our internal representation.

* *Edge cases:* Test behavior for nonexistent team\_id (DB returns no rows) – loader should throw an exception or return None (we need to decide; likely throw). Test a team with no members (shouldn’t happen normally, but if so, handle gracefully). Test a member with no tools (that’s possible if an agent doesn’t need external tools); ensure it doesn’t break (just an empty list for that agent’s tools).

* *Integration test:* Once integrated, we could test that ExecutionContextLoader \+ AgentRuntime work together: For example, after loader fetches a context, ensure that when AgentRuntime builds a prompt, it indeed has access to the persona and tool list from context. This is more of an indirect test. Or we simply trust unit tests for loader and integration tests in ExecutorEngine which inherently rely on loader to provide correct data (if something was wrong, the execution test might fail due to missing persona or tools, etc.).

By thoroughly testing the loader, we ensure that the rest of TAE always receives correct and complete team info. This is vital because any omissions (like forgetting to load allowed tools or persona) could lead to an agent either not knowing it can use a tool or losing its identity in prompt – which would degrade performance significantly.

## 4\. Data Persistence and Schema

TAE itself is mostly an execution layer and does not introduce a lot of new persistent data structures; it leverages TAO’s existing databases for config and logging. However, there are a few data aspects to highlight, particularly regarding **execution logging** and **agent memory persistence**.

### 4.1 Execution Logging Schema

**User Executions:** In support of auditing and permission enforcement, the platform uses a concept of logging each execution initiated by a user. TAO’s design references a user\_executions table, which records whenever a user (or a system component on behalf of a user) runs an agent team. The ExecutionContextLoader or ExecutorEngine in TAE will typically trigger the creation of such a log entry at the start of an execution, and update it at the end:

CREATE TABLE user\_executions (  
    execution\_id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
    team\_id UUID REFERENCES agent\_teams(team\_id),  
    user\_id UUID,  \-- The user who initiated; could be null for system or scheduled exec  
    started\_at TIMESTAMP DEFAULT NOW(),  
    completed\_at TIMESTAMP,   
    status VARCHAR(20),  \-- e.g. 'success', 'failed', 'aborted'  
    query\_text TEXT,   
    final\_answer TEXT  \-- (optional) maybe store the answer or a summary  
);

When TAE receives a request, it can log a new row with status='running'. On normal completion, it updates the row with completed\_at and status='success', along with maybe storing the final\_answer for record. If an error occurred (exception, permission denied, etc.), it would update status='failed' and possibly store an error message.

This table allows administrators to later see *who ran what, when*, and whether it succeeded. It’s also useful for usage analytics (e.g., how often a particular team is executed, average runtime if we store durations).

**Agent/Tool Execution Logs:** TAE relies on TAO to log the granular steps: \- The member\_executions table in TAO logs each tool use by an agent. It includes fields like execution\_id (linking to user\_executions perhaps, or its own id), member\_id, tool, timestamps and result snippet. TAO populates this whenever TAE (ToolCaller) calls a tool. TAE doesn’t directly write to this table but should be aware that TAO is handling it. \- In the Neo4j graph, TAO also logs relationships like (Member)-\[:EXECUTED\]-\>(Tool) with properties. Again, TAE doesn’t manage that, but as part of data persistence it’s worth noting that each tool call from TAE results in a graph update for auditing and analytics.

**Execution Trace Storage:** If we wanted to store the full conversation transcript of an execution for later review (e.g., debugging or fine-tuning), we could: \- Save it in a text or JSON form in object storage (MinIO). For instance, after execution, push a JSON with the sequence of messages to an S3 bucket, and store a reference (URL or key) in the user\_executions table. That way we don’t bloat the database with large text. \- Alternatively, a separate Postgres table execution\_messages(execution\_id, seq, sender, content) to store each message. This would allow SQL queries over conversations but could get large quickly. For MVP, we do not store every message, as it can be verbose and is not strictly needed. We rely on tool logs for detail of what happened. However, if debugging a specific incident, having transcripts might be useful, so we consider it for future additions (with caution re: storage size).

### 4.2 Agent Memory Persistence

If we implement persistent long-term memory for agents, that would entail additional data structures: \- **Vector Store (Qdrant):** Likely, we will use Qdrant collections to store embeddings of text that agents should remember. For example, we could create one collection per agent (or per team) to store memories. Each vector could have metadata including agent\_id, and we can use a common collection for all memories with agent\_id as filter tag. \- **Memory embedding:** When an agent finishes an execution, we might embed key facts from the conversation and upsert them into Qdrant. The AgentMemoryManager would handle that (using the embedding service BGE-M3 as needed). \- **Memory retrieval:** Before or during execution, AgentMemoryManager could query Qdrant for relevant past info given the current query and provide it to AgentRuntime as additional context (e.g., “Previously, this question was answered with ...” or “Reminder: last quarter’s earnings were X.”). \- **Postgres Memory Table:** Alternatively or additionally, we could have a table like:

CREATE TABLE agent\_longterm\_memory (  
    memory\_id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
    agent\_id UUID REFERENCES agent\_team\_members(member\_id),  
    content TEXT,  
    vector VECTOR(1536)  \-- if using pgvector for semantic search  
);

Each row stores a piece of knowledge the agent learned or was given. We’d use either pgvector or an external vector DB for similarity search on content. In MVP, we do not populate this, but designing it now means we know where to store things later. \- **Neo4j Knowledge Graph:** Another angle: If agents are supposed to “learn” relationships, Neo4j could be used to represent what agents have learned. The TAO doc hints at possibly updating properties like agent\_learn\_\* on graph nodes. For example, if an agent finds that a certain document chunk was useful, we might mark it in the graph as such (increase a score). This is more TAO’s domain (since TAO deals with domain chunks and Neo4j), but TAE could feed back info (like “these chunks were used in answering”). For now, TAO already captures chunk usage via relationships like (Query)-\[:USED\_CHUNK\]-\>(Chunk), so that is indirectly a memory of what was used to answer each query.

In summary, **MVP Data Persistence:** \- Leverage TAO’s agent\_teams, agent\_team\_members, tools, etc., for config (read-only from TAE perspective). \- Use user\_executions for logging at execution level (TAE/TAO combined). \- TAO automatically uses member\_executions and graph for detailed logging. \- Not storing conversation transcripts or long-term memory in MVP (but structure is prepared to add these later).

We ensure all new tables or fields introduced are properly integrated into migrations and have indices where appropriate. For example, on user\_executions, we’d index team\_id and user\_id for querying usage by team or user, and perhaps a composite index on (user\_id, team\_id) if needed for permission auditing (“which users ran which teams”). We’d also likely index started\_at for time-range queries (to find executions in last week).

Finally, **cleanup strategy:** Over time, logs grow. The system might implement retention policies (e.g., keep execution logs for X days). This again is more an operational concern, but we note it for completeness: perhaps an archive job moves old user\_executions to a history table or deletes them after a year, etc., to manage DB size. In MVP, no specific cleanup, but the design should not explode in short term (text fields are moderate if not storing full answers excessively).

## 5\. Deployment and Environment Setup

TAE is designed to run as a cloud-native service, containerized and configurable via environment variables. It operates alongside TAO and TAB in the platform deployment. This section covers how TAE is deployed, key environment configurations, container setup, health checks, and monitoring.

### 5.1 Containerization and Service Deployment

**Container Image:** TAE is packaged as a Docker container (for example, using a Python base image). The Dockerfile for TAE would: \- Copy the TAE module code. \- Install dependencies (e.g., requests for API calls, any DB drivers like psycopg2 for Postgres, neo4j Python driver if needed, qdrant-client, etc., and possibly fastapi or similar if we expose an HTTP server). \- Set an entrypoint to launch the TAE service (this could be a simple process that listens for execution requests, possibly via HTTP API, or even a CLI triggered by TAO, depending on design).

**Service Mode:** There are two possible modes for TAE to operate: \- **As an API service:** TAE could expose a REST (or gRPC) API where external callers (like a web UI backend or an orchestrator) send a request to execute a team. For example, a FastAPI app with an endpoint POST /execute that takes team\_id and query (and maybe user\_id) in JSON, then internally runs the ExecutorEngine and returns the answer. This makes TAE a standalone microservice. In this mode, we define a port (e.g., 8200\) for TAE’s server. \- **As a library called by TAO:** Alternatively, TAO could directly invoke TAE’s logic in-process (if they are part of one larger application or if TAE is a pip installable module). But given we want decoupled modules, the service approach is more likely.

We will assume TAE runs an HTTP service. So, in docker-compose deployment, we would have something like:

services:  
  tae:  
    build: ./tae  \# path to TAE Dockerfile  
    container\_name: tae  
    depends\_on:  
      \- tao  
      \- postgres  
      \- neo4j  
      \- qdrant  
      \- ollama  
    environment:  
      \- MCP\_TRANSPORT\_MODE=http  
      \- MCP\_SERVER\_HOST=tao  
      \- MCP\_SERVER\_PORT=8100  
      \- POSTGRES\_HOST=postgres  
      \- POSTGRES\_PORT=5432  
      \- POSTGRES\_DB=ta\_v8  
      \- POSTGRES\_USER=postgres\_user  
      \- POSTGRES\_PASSWORD=postgres\_pass  
      \- NEO4J\_URI=bolt://neo4j:7687  
      \- NEO4J\_USER=neo4j  
      \- NEO4J\_PASSWORD=\<password\>  
      \- QDRANT\_URL=http://qdrant:6333  
      \- OLLAMA\_URL=http://ollama:11434  
      \- OLLAMA\_MODEL=gpt-oss:20b  
      \- TAE\_SERVER\_PORT=8200  
    ports:  
      \- "8200:8200"  
    restart: unless-stopped

This hypothetical snippet shows how TAE would be configured in a compose file: \- It depends on TAO and the various services to be up. \- MCP\_SERVER\_HOST and MCP\_SERVER\_PORT tell it how to reach TAO’s MCP gateway (for tool calls). \- DB and other service credentials are provided as needed. \- We map port 8200 for external access (if needed; if only internal, we might not expose it outside the docker network). \- TAE could also be scaled horizontally (multiple replicas) if needed, since it’s stateless except for ephemeral execution state (which could be fine if using a load balancer with sticky sessions or just letting any instance handle any request and all sharing the same DB).

**Startup Dependencies:** TAE should ideally only start after TAO and databases are ready, since it might immediately try to serve requests or at least verify connectivity. We use Docker Compose’s depends\_on and perhaps healthcheck wait conditions: \- E.g., don’t start TAE until Postgres healthcheck is healthy (so that ExecutionContextLoader can query without error). \- Also, ensure TAO is up (though TAE could start and simply fail tool calls until TAO is up; but better to wait).

### 5.2 Configuration via Environment Variables

TAE is configured through environment variables to allow flexibility across deployment environments. Key environment variables include:

* **Database Configuration:**

* POSTGRES\_HOST, POSTGRES\_PORT: Hostname (likely the service name e.g. postgres) and port (5432) for the Postgres instance containing team and tool data.

* POSTGRES\_DB, POSTGRES\_USER, POSTGRES\_PASSWORD: Credentials for connecting to the database. TAE needs read access to team config tables, and write access to execution log tables. These are set up in the .env (for example, ta\_v8 database, user postgres\_user as in our compose).

* Alternatively, a single DATABASE\_URL can be used (as seen in the setup script) which TAE can parse to connect.

* **Neo4j Configuration:**

* NEO4J\_URI: e.g. bolt://neo4j:7687 (pointing to the Neo4j service’s Bolt port).

* NEO4J\_USER, NEO4J\_PASSWORD: credentials for Neo4j (matching what’s in compose, e.g., neo4j/pJnssz3khcLtn6T).

* If TAE reads domain info or logs to Neo4j, it uses these. If MVP TAE doesn’t directly talk to Neo4j (leaving that to TAO), these could be optional. But we include them so that if ExecutionContextLoader wants to query Neo4j for domain descriptions, it can.

* **Vector DB (Qdrant):**

* QDRANT\_URL: Base URL for Qdrant’s HTTP API (e.g., http://qdrant:6333).

* QDRANT\_COLLECTION: (optional) default collection name for semantic tool search or memory. TAO uses ta\_v8 as collection in environment. TAE might use the same for any tool suggestions or memory queries.

* If we only call Qdrant via TAO (like TAO does tool search), TAE might not use this directly. But if memory search is implemented, TAE would need these.

* **LLM Service:**

* OLLAMA\_URL: URL of the Ollama server (in compose it's http://ta\_v8\_ollama:11434 internally, or http://localhost:11434 if testing locally).

* OLLAMA\_MODEL: The default model to use (e.g., gpt-oss:20b). AgentRuntime might use this if it’s not specified per agent. Possibly if we allow different models per agent, this could be overridden by agent config (not MVP).

* Alternatively, if using OpenAI or others, environment keys like OPENAI\_API\_KEY would be used (not in current stack, but mention that the system can be configured to point to a different LLM service by environment).

* LLM\_BASE\_URL is also shown in env (maybe redundant with OLLAMA\_URL).

* **MCP/TAO Connection:**

* MCP\_TRANSPORT\_MODE: This tells TAE how to communicate with TAO for tool calls. By default, http (meaning use REST calls). If set to grpc or stdio, TAE would use different logic (these modes might not be implemented initially, but environment is prepared).

* MCP\_SERVER\_HOST, MCP\_SERVER\_PORT: e.g., tao and 8100 as in compose (TAO’s service name and port). ToolCaller will construct TAO’s URL from these (like http://tao:8100/api/mcp etc. depending on endpoint).

* There might also be an MCP\_API\_KEY or auth token if TAO’s endpoints are secured; not in current config (assuming internal network and/or tokenless internal APIs), but we allow adding such a variable if needed.

* **TAE Service Config:**

* TAE\_SERVER\_PORT: The port on which TAE’s own API runs (e.g., 8200). This could default, but making it configurable is useful.

* Logging and debug flags: e.g., LOG\_LEVEL=info/debug to control verbosity.

* ENABLE\_CRITIC: a boolean flag to turn on the CriticFeedbackEngine. MVP could leave it off by default. If set, ExecutorEngine will invoke the critic step.

* MAX\_STEPS: perhaps an integer to cap how many loops an agent can do (to prevent infinite loops). Could default to, say, 5\. If an agent doesn’t conclude by then, TAE will break out. Configurable via env if needed.

* **Tool/Feature Toggles:**

* If certain tools or integrations are optional, we could have env flags. For example, ENABLE\_NEWS\_API\_TOOL=false if that tool not configured. But since TAO controls tool registry, TAE might not need to know.

* USE\_CACHE=true/false: whether ExecutionContextLoader should cache team configs in memory.

**Secrets Management:** The .env approach and environment variables are how we supply passwords (like DB and Neo4j). In production, one might use Docker secrets or vault rather than plain env. The documentation notes that secrets should be handled via environment files or Docker secrets to avoid hardcoding. For now, we document using environment, but one can easily swap to secrets in docker-compose (noting in docs that sensitive values can come from Docker secrets for improved security).

**Example .env file:** from the setup script, we saw:

POSTGRES\_HOST=localhost  
POSTGRES\_PORT=5432  
...  
NEO4J\_URI=bolt://localhost:7687  
NEO4J\_USER=neo4j  
NEO4J\_PASSWORD=...  
...  
OLLAMA\_URL=http://localhost:11434  
...  
EMBEDDING\_URL=http://localhost:8080  
EMBEDDING\_MODEL=BAAI/bge-m3

In our container context, localhost wouldn’t apply (since inside container, it’s not host’s localhost). So in deployment, those are set to service names (postgres, neo4j, etc.). The environment variables for embedding might not be directly used by TAE in MVP, but if AgentMemoryManager wanted to call the embedding service (BGE-M3) to embed a piece of text, it would use EMBEDDING\_URL and EMBEDDING\_MODEL. Currently, TAO likely uses those for indexing tools or documents, but TAE could also use them if memory or summarization is implemented.

### 5.3 Health Checks and Service Monitoring

**Health Check Endpoint:** TAE will expose a simple health check URL (commonly /health or /ping). If using FastAPI or similar, we implement an endpoint that returns status 200 and maybe a JSON like {"status": "ok"}. This endpoint does minimal work (optionally could check that it can reach dependencies like the DB, but usually just indicates the service is up). In Docker Compose, we configure the healthcheck to curl this endpoint:

healthcheck:  
  test: \["CMD", "curl", "-f", "http://localhost:8200/health"\]  
  interval: 30s  
  timeout: 5s  
  retries: 3

This ensures orchestration systems know if TAE is alive. The provided health-monitor.sh script can be extended to include checking tae service along with others. (In that script, currently services are only the infra ones, but we can add "tae" to the list).

**Readiness Check:** If TAE has a startup phase (like pre-loading some models or data), a separate readiness endpoint could be used to signal when it’s ready to accept traffic. MVP probably doesn’t need a heavy warm-up (the LLM model runs in Ollama outside TAE, so TAE itself should start quickly). So health and readiness might be the same.

**Prometheus Metrics Endpoint:** Following the pattern used in TAO (which exposed metrics on port 9090 or similar), TAE can integrate with the Prometheus Python client to expose metrics at /metrics. We can run the HTTP server on a given port (if same as the main API, then /metrics route, or a separate port if needed). We include the prometheus\_client library and define some counters/gauges: \- tae\_executions\_total (counter): increment every time an execution is run (maybe separate labels for success vs failure). \- tae\_execution\_duration\_seconds (histogram): measure how long each execution takes from start to end. \- tae\_toolcalls\_total (counter): count how many tool calls TAE made (though TAO also counts them, but we can count from TAE perspective too). \- tae\_critic\_flagged\_total (counter): if Critic flags outputs as bad. \- tae\_active\_executions (gauge): how many executions are currently in progress (increment when start, decrement when done) – though typically each request is short, but if we have concurrency, that gauge is useful.

These metrics allow monitoring system load and performance. Prometheus can be configured (in prometheus.yml) to scrape the TAE service at the given endpoint and interval. It would look similar to TAO’s job:

\- job\_name: 'tae'  
  scrape\_interval: 15s  
  static\_configs:  
    \- targets: \['tae:8200'\]  
      metrics\_path: '/metrics'

(This assumes our container is named tae on network and using port 8200 for metrics – if using same port for both API and metrics.)

**Grafana Dashboards:** One could create dashboards to visualize: \- Queries per minute (from tae\_executions\_total). \- Average execution time. \- Number of tool calls per query (TAO metrics or we can compute as tao\_member\_executions\_total / tae\_executions\_total perhaps). \- Success vs failure count (if we label metrics by status or rely on logs for that). \- If Critic is used, how often it flags outputs.

**Logging:** TAE will produce logs for operational insight: \- Use a structured logging library (like Python logging or structlog). \- Log an INFO when an execution starts (with team\_id, user\_id maybe, and query truncated). \- Log when an execution ends, with status and duration. \- Log warnings or errors if something goes wrong (tool call fails, etc.). \- Possibly debug logs for each step (like each agent’s response) when debug mode is on, which helps during development but might be turned off in production due to verbosity (or logged at debug level so not collected normally).

These logs can be collected by Docker (STDOUT/STDERR) and aggregated if using ELK/EFK stack. The main script mentioned a health-monitor.log for that monitor script, but TAE’s own logs might be separate. In a production environment, we might integrate with a logging service or volume mount a log file if needed.

**High Performance Considerations:** The deployment is set up to use GPU for the Ollama and BGE-M3 containers. TAE itself doesn’t require GPU; it’s CPU-bound for orchestration and I/O. It should run with minimal CPU and memory (except what’s needed for Python runtime and any caching). We can limit its resources in Compose if needed (though compared to LLM and DB, it’s lightweight).

**Security:** If TAE’s API is exposed outside (e.g., to a web client or external service), ensure authentication. Possibly behind an API gateway or at least require a token on /execute calls. In an internal setting, maybe only internal trusted services call it. For completeness: \- Integration with an Auth service: TAO doc mentioned verifying tokens in auth.py. If the user request comes in via an API gateway that attaches a user identity (JWT etc.), TAE might need to parse that and pass user\_id to ExecutionContextLoader and to TAO calls. If TAO expects an auth token or user context, TAE should forward it (maybe TAO can re-check permissions based on user\_id or we log user\_executions as mentioned). \- In MVP, we assume trusted environment or that the UI calling TAE has already authenticated the user and provides user\_id in the request payload.

**Scaling and Load:** Each TAE instance can handle multiple sequential requests, but heavy concurrency might be limited by how many LLM calls can run concurrently (which ties to Ollama’s capacity). If many queries come in at once: \- TAE could queue them or run in threads (the LLM calls could be async awaited). \- The Ollama server was likely configured with a max concurrent model usage (e.g., it might queue internally or run one at a time if model is large). \- We can run multiple TAE replicas to distribute load if needed. Just ensure they share DB and talk to the same TAO. TAO itself might then become a bottleneck if one (we can also scale TAO possibly, though TAO also touches DB which is single unless clustered). \- The design is modular so we can horizontally scale the stateless parts (TAE, TAO’s API if stateless aside from DB).

**Startup**: The setup-high-performance-stack.sh covers deploying infra. After running it, one would run TAO, TAB, TAE containers (perhaps via docker-compose up). We should ensure in documentation that **the order**: first launch DB and related infra (the script does that), then launch TAO, then TAE. If any caching or seed needed, mention it. But presumably everything picks up from the DB contents which could be seeded with initial tool and team data (maybe via migrations or the builder module).

By following these deployment guidelines and environment configurations, the TAE module can be reliably deployed and managed in production-like environments, and operators can monitor its health and performance easily.

## 6\. Integration with TAO and TAB

TAE does not operate in isolation; it works closely with the other modules of the Team Agent platform. Here we outline how TAE integrates with **TAO (Team Agent Orchestrator)** and **TAB (Team Agent Builder)** to support end-to-end functionality, including the inputs it expects and outputs it produces.

### 6.1 Workflow Integration from Team Building to Execution

A typical user journey involves using TAB to create a team, then using TAE (through some interface) to execute that team on queries. The integration points are:

* **After Team Building (TAB → TAO → DB):** When a user finishes building a team using TAB’s wizard or YAML upload, TAO stores the new team configuration in the database (Postgres and Neo4j). At this point, the team’s status might be marked as “active” or “ready”. TAE itself is not directly involved in creation, but it relies on the output: the newly created team config now available for execution. TAB and TAO ensure any errors in config are resolved before finalizing, so TAE should be able to load a clean config.

* **Execution Trigger (User → TAE):** When the user wants to run the team, they will typically select the team and submit a query. This request goes to TAE (for instance, via a UI calling TAE’s API). The request should include the team\_id of the team to execute, the user\_query text, and some form of user identity or auth token if applicable. For example:

* POST /execute   
  {  
    "team\_id": "123e4567-e89b-12d3-a456-426614174000",  
    "user\_id": "u-001",   
    "query": "What were our Q3 earnings compared to Q2?"  
  }

* The user\_id might be obtained from an auth token by the frontend or gateway and passed along so that TAE knows who initiated this (for logging and permission).

* **Permission Check (TAO authorization):** Upon receiving the request, TAE/ExecutionContextLoader can consult TAO or the database to ensure this user has rights to execute this team. TAO’s permission model indicates that perhaps any authenticated user of the tenant can run queries (role “Executor”) unless more restrictions are in place. In practice, TAE could either:

* Call a TAO endpoint like GET /teams/{team\_id}/access?user\_id=... to verify, or

* Rely on the gateway to have enforced that (if roles are embedded in token, the gateway might not even route the call if user isn’t allowed).

* Or attempt to load the team; if the user isn’t allowed, TAO’s get\_team might throw an error or return filtered data. For completeness, we ensure that either at ExecutionContextLoader or just before execution, if user lacks permission, we abort with HTTP 403\. This check leverages the roles defined (Admin, Builder, Executor, etc.). In MVP, we might assume all users who can send queries have execution rights for their tenant’s teams (especially if no fine-grained user-team mapping yet).

* **Fetching Configuration (TAE → TAO/DB):** TAE uses ExecutionContextLoader to get the team config. As discussed, this might directly query Postgres for team and members (with tenant isolation by tenant\_id to ensure, for security, that even if a malicious user tries a team\_id from another tenant, it finds nothing). TAO enforces that in queries by including tenant filters. If TAE calls TAO’s API, TAO will ensure the requesting user/tenant only gets their team (likely using the auth context).

* Assuming a well-behaved scenario, loader returns the team’s structure and context.

* **Orchestration Plan (TAO vs TAE):** Now, either TAO or TAE orchestrates the actual multi-agent flow:

* **Pattern 1: TAO calls TAE as a subprocess:** In one conceptual pattern, TAO itself, upon an execution request, could delegate to TAE. For instance, if an API call first hits TAO’s MCP gateway with a query, TAO might then internally call TAE’s executor to handle agent conversation. However, given our separation, the likely approach is:

* **Pattern 2: Direct TAE handling:** The UI or client calls TAE directly (bypassing TAO for the query execution step). This means TAE is the primary entry point for execution. TAE will call TAO *only* when it needs to use a tool or possibly to log something. This aligns with our architecture: TAO is the brain for configuration and tool governance, TAE is the muscle for running the conversation.

* **Multi-Agent Conversation (TAE ↔ TAO):** Once the execution starts, TAE runs the loop:

* When an agent’s turn comes and it outputs a tool request, TAE’s ToolCaller calls TAO’s MCP interface to execute the tool. This is a synchronous call that will yield a result. TAO upon receiving this call:

  * Checks the member’s permission for that tool (MemberManager in TAO).

  * Executes the adapter for the tool (could involve calling vector DB, etc.).

  * Logs the execution (in Postgres and Neo4j).

  * Returns the result to TAE.

* TAE receives the tool result and continues the conversation. During this, TAO is effectively servicing all tool requests but is not dictating conversation flow (TAE is).

* If an agent tries to use a tool it’s not allowed, TAO will respond with an error (e.g., 403 or a structured error). TAE’s ToolCaller will propagate this. TAE might then decide to abort the conversation because something went wrong (likely the agent’s strategy failed). Possibly in the future, TAE could catch this and tell the agent “that tool is not available” and let it continue differently, but MVP likely just stops or returns an error to user (with an apology or such).

* **Data Exchange:** The integration defines what data TAE gets from TAO at startup:

* **Agent Personas and Tools:** TAO (via DB or API) provides each agent’s persona (prompt), role, and tool list to TAE. TAE uses these to construct prompts. TAO might also provide a pre-formatted prompt snippet, but likely TAE builds it.

* **Execution Plan:** If TAO had advanced planning, it could send a plan. In MVP, TAO doesn’t send an explicit plan; TAE either has a hardcoded sequence or computes it. In the future, if TAO’s TeamOrchestrator could produce a LangGraph for complex queries, it might pass that structure to TAE (maybe as part of the prepare\_execution call). Currently, we assume sequential plan embedded in how TAE was coded (e.g., always agent1 then agent2).

* **Domain context:** TAO could supply relevant context if it has any pre-retrieved info for this query. For example, TAO might know which knowledge domain the team is linked to and possibly pre-fetch related chunks or summary. However, given the design, it’s more likely an agent will retrieve when needed, so TAO doesn’t push any content upfront except generic info like “Domain: Finance Reports”.

* In a call like TeamOrchestrator.prepare\_execution(team\_id, query), TAO could do tasks like:

  * Ensure vector indices are updated (via ChangeDetector/VersionManager).

  * Possibly identify which agent should start (maybe not needed if order fixed).

  * Potentially fetch some top relevant info (though more likely not in MVP).

  * Then return nothing or some context to TAE. In MVP, we might skip this extra step for simplicity.

* **Completing Execution (TAE → TAO):** Once TAE obtains a final answer from the agents:

* It returns that to the user (via API response).

* It also should inform TAO of completion. At minimum, TAO knows tool calls that happened, but TAO might not know that the entire query finished successfully. We might update the user\_executions log (which could be done via TAO or directly if TAE has DB access).

* If we wanted, we could notify TAO’s orchestrator for any post-processing, but not really necessary. TAO’s involvement per execution is done when the last tool was called (or if no tools were used, TAO might not even know an execution happened except via user\_executions log).

* TAO might want to record the final outcome (for learning or for cross-team calls). For instance, if Team A was used as a tool by Team B (TeamAsTool scenario), TAO would call TAE to run Team A and then TAO would take the result and feed to Team B. In that scenario, TAE returns answer to TAO instead of directly to user. But that’s essentially an internal user being TAO itself. We can see TAO and TAE can nest like that:

  * TeamAsTool: TAO’s TeamAsTool adapter, when invoked, might do something like result \= requests.post(TAE\_URL, {"team\_id": other\_team, "query": subquery}). TAE executes and returns, TAO then returns that to the original agent as if it were a normal tool result. This is an advanced integration but one the design explicitly considers. We ensure TAE’s API and permission model allow this (maybe TAO uses an internal token or has override to run any team as needed).

  * This means TAE should handle being called by TAO (service-to-service) just as well as by a user request. Likely it’s the same endpoint; TAO would supply team\_id and query (and perhaps a special user or service credential to authenticate itself).

  * TAE in that case logs that Team X was executed (it might mark user\_id as something like “system” or the team/tool that triggered it).

* **TAB integration:** TAE doesn’t directly integrate with TAB’s runtime, but indirectly:

* TAB after building a team might prompt the user "Team is ready, would you like to run a test query now?" If so, the UI would call TAE.

* If there’s a scenario where building and executing are combined (like a “try it now” feature in builder), the front-end would just call TAE under the hood. There’s no direct code connection between TAB and TAE except both use TAO’s data.

* One area of overlap: Both TAB and TAE might need to **load team configurations**. TAB likely uses TAO’s builder APIs to gradually build the config. Once done, TAO already has it in DB, so TAE just loads from DB. There’s no need for TAE to interact with TAB during execution.

* In the overall system, the user flow is:

  1. Use TAB to make team (which calls TAO’s endpoints).

  2. Then call TAE to execute (which uses TAO’s data, calls TAO’s tools).

  3. The results could be displayed in a UI. If the user isn’t happy, they might go back to TAB to adjust the team (which is an edit operation calling TAO’s update endpoints).

  4. Then run again via TAE. This iterative loop is supported by clear separation: TAO/TAB manage config, TAE runs it.

* **Multi-Tenancy Isolation:** TAO’s TenantManager ensures each tenant’s teams and data are separate. TAE must respect that by:

* Always including the tenant context when querying data (e.g., ExecutionContextLoader filters by tenant\_id).

* Not mixing data between tenants. For example, if user from tenant A somehow passes a team\_id of tenant B, the loader either returns none or error.

* Tool calls via TAO are also scoped: TAO will check that the member (with certain tenant) is using a tool that belongs to that tenant’s context (tools might be global, but any data retrieval like vector search will likely only search that tenant’s vectors due to how data is partitioned).

* In logging, user\_executions should have tenant\_id (we could include that for quick filtering).

* These measures ensure one tenant’s execution can’t access another’s domain info or tools (unless explicitly allowed via some cross-tenant tool, but that’s not typical).

* **Role-based Behavior:** TAO’s roles might influence integration:

* If a user is just an “Executor” role (cannot modify teams), they can only call TAE (run queries). If they attempt to call TAB’s build endpoints, they’d get denied.

* A “Builder” can do both build and execute. But maybe they use a different interface.

* The system could enforce that only “Executor” or “Admin” can call TAE. If needed, an auth layer would check user role on the /execute call. We mention that in security considerations.

* **Execution Handoff and Collaboration:** The integration also envisions potential **collaboration between teams** (future). For example, one team’s agent might have a tool to ask another team (team-as-tool). This essentially daisy-chains TAO and TAE:

* Agent in Team A says "Tool: TeamBAgent('query X')".

* ToolCaller in TAE (Team A’s context) sees this as an external team call. It could either:

  * Recognize the tool name as a special "TeamTool" and call TAE recursively (via TAO). Possibly TAO’s ToolRegistry has an adapter "TeamAsTool" that does exactly that: calls TAE for the specified team.

* That means TAE can be invoked mid-execution by TAO. The result returns, and TAE (Team A’s execution) continues.

* This is advanced usage but supported by the design. The main requirement is that TAO’s registry knows which team ID corresponds to that “TeamAsTool” name and how to map parameters. Also that user permissions allow Team A to call Team B as a tool (maybe only if same tenant and allowed).

* From TAE’s perspective, it doesn’t know the difference – it just gets a call to run team B. Possibly the user\_id might be TAO or the original user depending how we trace context. Ideally, user context flows so that if Team A’s user was John, and Team A triggers Team B, the logs still attribute to John (or perhaps mark it as system on behalf of John).

### 6.2 Integration Summary with TAO

To summarize TAE \<-\> TAO interactions: \- **Initial data load:** TAE fetches team config from TAO’s domain (DB or API). \- **Tool calls:** TAE calls TAO’s MCP endpoints to perform each external action. \- **Logging:** TAO logs each action and possibly the execution; TAE may also update logs, possibly by calling TAO or DB. \- **End result:** TAO isn’t directly notified of final answer content (unless we want to store it or feed it somewhere), but TAO can infer an execution completed if no more tool calls are coming and perhaps TAE could call an endpoint like POST /teams/{team\_id}/execution\_done mainly to update status or metrics. Not strictly needed; metrics can be updated by TAE itself. \- **Change management:** TAO’s ChangeDetector might update domain knowledge indexes in background. If, say, a new document was added to a domain right before execution, TAO could ensure the vector index is updated so that when TAE’s agent calls the vector\_search tool, it finds the new data. This integration is transparent to TAE – TAE just calls the tool, TAO ensures the tool is up-to-date. If TAO had cached an old state of knowledge, ChangeDetector invalidates that cache.

### 6.3 Integration with Team Agent Builder (TAB)

While TAE doesn’t directly call TAB, their integration is about **user flow and data consistency**: \- TAB ensures the team configuration is complete and valid, using TAO’s BuildWorkflow (WizardEngine, YAML validator). Once finished, the team is in the database. \- TAB likely sets the team’s status to “active” when done. TAE’s ExecutionContextLoader could filter to only load teams where status \= active (so it doesn’t accidentally run an incomplete team). So if a team is in "building" state (maybe user hasn’t finished config), TAE might refuse to run it. \- If a user attempts to execute a team that is not yet built (status not active), TAE should respond with an error (“Team configuration not finalized”). This scenario would only happen if user somehow calls the API out of order; the UI would normally prevent it by not showing that team as executable. \- **Combined UI example:** Suppose there is a single web UI that allows building and querying. After building, the UI might call something like: 1\. POST /teams (via TAB) \-\> creates team. 2\. POST /teams/{id}/complete if needed (or it knows build done). 3\. Then POST /execute on that team (via TAE). The UI should wait for step 1 to complete and confirm success before step 3\. If build is async or multi-step, the UI will do it synchronously anyway with user interacting, so presumably by the time they click "run query", team exists.

* **Error handling across modules:** If TAE tries to load a team and something’s missing (maybe a rare case, e.g., a tool that was referenced no longer exists due to an update), it might fail. These configuration mismatches should ideally be caught in TAO (like TAO’s VersionManager might notice a config referencing a removed tool and could fix/notify). In execution, if such an error arises, TAE can log it and maybe return a message to user, and the admin would need to fix config. But such situations are expected to be rare given the controlled build process.

* If user edits a team via TAB while an execution is happening, what then? Possibly:

* If they remove an agent or tool mid-run, the current execution wouldn’t know; it loaded at start. We might lock team editing while execution is in progress (but since executions are short, that might not be necessary).

* At least, concurrent changes are unlikely. If it did happen, worst-case the current execution uses outdated config (like calling a tool that was just removed – TAO might still allow it if the permission entry exists until commit of deletion).

* After execution, if team changed, next execution uses updated config (maybe with differences).

* TAO’s config versioning could help ensure consistency (e.g., each execution could note a config version and if a change happens, it increments version). TAE could log which version it executed. If needed, we can ignore this complexity for MVP.

In essence, TAE integration is primarily about **using TAO for any data and actions** it doesn’t handle itself, and aligning with TAB in terms of using a valid team. The modular design makes each part simpler: \- TAB handles creation (and doesn’t need to worry about execution logic). \- TAE handles execution (and doesn’t worry about how team came to be). \- TAO stands in the middle as the reference for team configuration and the gateway to tools and knowledge.

By clearly defining these integration points, we ensure that improvements in one module (like a more complex team build flow or more advanced orchestration logic in TAO) do not break the others; they simply communicate through the agreed-upon interfaces (database records and API calls).

## 7\. Security and Permission Control

Security for the Team Agent Executor revolves around two aspects: **user access control** and **safe agent behavior**. While TAO covers much of the permission logic (especially regarding who can build or run teams and what tools agents can use), TAE must enforce and respect those controls in practice during execution.

### 7.1 Execution Access Control

**User Roles and Execution Rights:** As described earlier, each user in the system has a role that dictates what they can do. Generally: \- **Admin** and **Builder** roles can do everything (including run executions). \- **Executor** (normal user) role is typically allowed to execute queries on existing teams but not modify them. \- **Viewer** might not even be allowed to execute, only view results or configs.

TAO’s API or the gateway should include the user’s role info, usually via an auth token. TAE should ensure that: \- It does not allow unauthorized execution. For instance, if a Viewer tries to call the execute endpoint directly, TAE should reject with 403 Forbidden. \- If a user tries to execute a team they don’t belong to (different tenant or not shared with them), that’s also forbidden. The ExecutionContextLoader will likely not find the team (due to tenant isolation) or TAO API will deny it.

In practice, if the system uses a JWT with claims, TAE’s server can decode it (or rely on an API gateway to do so) and get user\_id and roles. We then implement a simple check: if role is not in {Admin, Builder, Executor}, respond with 403\. Also, if an Executor user’s tenant doesn’t match the team’s tenant, respond 403 (though our DB query would return nothing, we can interpret that as unauthorized if team not found but might exist in another tenant).

**Logging and Audit:** Whenever an execution is initiated, we log the user and team. This not only helps with debugging and system monitoring but also provides an audit trail for security: \- If a user does something malicious or heavy, we know who triggered it. \- The user\_executions table serves as an audit log as mentioned. We ensure to populate user\_id and success/failure. Security reviews can then check this log to ensure no unauthorized access (e.g., see if any user tried to run a team they shouldn’t). \- We could also log in text logs events like “User X (role=Executor) started execution of Team Y” and “Completed execution of Team Y for User X”.

**Data Privacy and Tenant Isolation:** TAE must ensure that data from one tenant’s context does not leak to another: \- ExecutionContextLoader queries are always filtered by tenant\_id. E.g., SELECT \* FROM agent\_teams WHERE team\_id \= ? AND tenant\_id \= ?. The tenant\_id for the user could come from the auth context. Alternatively, we derive it if user\_id has tenant embedded or we do a join through users table. Possibly simpler: each team has tenant\_id, and each user is associated with a tenant; our system likely encodes that, so we can enforce at query or check after load. \- Similarly, when ToolCaller requests a tool via TAO, TAO will ensure the call is executed in the context of the correct tenant (like a vector search will only search that tenant’s collection because the tool adapter uses tenant\_id as part of query or separate index per tenant). \- The Neo4j domain graph is partitioned by tenant (likely by separate subgraphs or labels). If TAE or TAO queries Neo4j, they include tenant filter.

**Preventing Unauthorized Tool Use:** This is mostly handled by TAO’s MemberManager (tool permission check). However, TAE should not circumvent it: \- E.g., TAE should always call TAO to execute a tool, never call external service APIs directly even if it technically could (that would bypass permission checks/logging). We design such that all tool usage goes through TAO’s controlled interface. \- This ensures if an agent somehow tried to use a disallowed tool (maybe by outputting an action for a tool not in its list), TAO will catch it and not execute it. TAE should then handle that gracefully (stop or inform user). \- It’s also a safety measure to not let the agent’s raw output cause arbitrary actions. Because by routing through TAO, we only execute known tool commands. If an agent output some injection attempt or weird string, TAO’s parsing and permission system will mitigate any harm (worst case, TAO will say “unknown tool” or “not allowed”).

**Safe Agent Outputs:** There’s a risk that agents might produce answers that are inappropriate (toxic, leaking info, etc.). CriticFeedbackEngine (if implemented) is one way to handle that by reviewing outputs. \- For MVP, we assume our chosen model (gpt-oss 20B) and domain limited usage reduce this risk somewhat, but it’s still possible. \- We could integrate a content filter as a tool or an output filter. For instance, after final answer, run it through a moderation model or regex for disallowed content. If found, either redact or refuse. \- If we had such a filter, it would be another step before returning to user. Possibly Critic could double as a safety checker. This was not explicitly required, but it’s worth noting for completeness as a future addition for security compliance (especially if this platform is used in enterprise, they might want assurance no confidential info leaks or no harassment, etc.). \- For now, maybe rely on the model’s internal guardrails and the fact that it’s not internet-connected (no external tool except those explicitly allowed).

**Encryption & Transport Security:** In a typical internal deployment, services communicate over a docker network (which is not exposed publicly). If needed, one might secure the traffic: \- For external API calls (like if TAE API is accessible by a web client), use HTTPS via a reverse proxy (e.g., if behind an Nginx or an API Gateway that terminates TLS). \- Service-to-service calls (TAE to TAO) are within the cluster; we might not encrypt those on local network, but if zero trust needed, one could use mutual TLS or something between services. That adds complexity; presumably not needed as they run in same environment.

**Dependency Security:** TAE relies on LLM and tool outputs. If a tool returns something malicious (like an external API might return a huge payload or some injection attempt), TAO’s ToolRegistry should sanitize it or limit it. TAE will just treat it as text. We might consider limiting sizes (like if a tool returns a very large text, maybe truncate before feeding to agent to avoid prompt overflow). \- Also ensure that binary data from tools (if any) is handled appropriately (most tools likely return text or structured data which we treat as text to the agent). \- If a tool failed with an exception, TAO likely catches and returns an error message; TAE should not break on processing that.

**Resource Limits:** Malicious or overly complex queries could make agents loop or consume a lot of time/tokens: \- We set a max turns or token limit to avoid infinite loop. For example, if an agent keeps insisting on using tools repeatedly without end, we stop after e.g. 5 loops, and perhaps give up with a message. This prevents exhaustion of resources. (We may implement this via MAX\_STEPS env or similar). \- Similarly, the LLM (Ollama/gpt-oss) likely has a max token setting (could be defined in its model config, maybe 2048 or so). That inherently limits how long an answer can get or how long it can monologue. We also instruct agents to be concise if possible. \- Each tool call should be relatively quick (embedding, search, etc. are optimized; but if a tool could be slow like external API waiting, TAO might have a timeout. TAE can also have a global timeout for tool calls – e.g. if no response in 30s, abort with error to avoid hanging forever). \- Possibly an overall execution timeout (like if after 60 seconds no final answer, stop). The user might prefer a partial or error response than waiting indefinitely. This can be implemented via async and timeouts around the main loop.

**Deployment Security:** On the deployment side, ensure the TAE container and others are not accessible to unauthorized networks. E.g. in docker-compose, if we run everything on a closed server, only the UI or API gateway should expose ports to outside. TAE’s port might not need to be open to the internet, only to the UI backend (which could be same network). \- If multi-tenant contexts require data separation beyond DB (some might want separate instances per tenant), one could deploy multiple TAO/TAE stacks, one per tenant. But our design is multi-tenant in one stack with proper isolation, which is more efficient and still secure via schema separation.

In summary, TAE enforces a **defense-in-depth** approach: \- It relies on TAO’s robust permission checks for tool usage and team access. \- It logs all actions for traceability. \- It applies runtime limits to guard against misuse or runaway processes. \- It ensures each step of the execution respects the intended boundaries (both in terms of which data can be accessed and how long/hard it can try).

By doing so, the system ensures that only authorized users can execute agent teams, that those executions cannot breach data from other tenants or bypass tool restrictions, and that any attempt to do so is recorded and can be audited.

## 8\. Scope and Future Roadmap

This section delineates what features are included in the **Minimum Viable Product (MVP)** for the Team Agent Executor and what enhancements are planned for future phases. This clarity helps set expectations for the first release and provides a vision for how TAE can evolve.

### 8.1 MVP Features

The MVP implementation of TAE focuses on delivering the core functionality necessary to execute multi-agent teams in a controlled, useful manner. Key MVP capabilities include:

* **Sequential Multi-Agent Execution:** TAE can handle teams of agents in a predetermined sequence. For example, if a team has a Retriever agent and then a Solver agent, TAE will run the Retriever to gather info (via tools) and then pass the results to the Solver to produce the final answer. Dynamic or conditional branching is minimal; the flow is mostly linear or a simple fixed loop.

* **ReAct-Style Single Agent Loop:** TAE supports an agent (or agent role) performing reasoning with tool use iteratively until it reaches an answer. This covers cases where one agent can handle a query by itself but needs to use external tools (e.g., search, calculator) multiple times. MVP allows that looping, but with a safety cap on iterations to prevent infinite loops.

* **Tool Integration via TAO:** All tool usage by agents is fully integrated. Agents can call any tool that has been assigned to them, and those calls are executed through TAO’s MCP interface with permission checks and logging. MVP includes support for tools like vector search, embedding, document fetch, web API calls (if configured), etc., as long as they’re registered in TAO. There is no need for manual intervention; the agent’s outputs like Action: ToolName(...) trigger actual tool runs and return results seamlessly.

* **Basic Prompt Composition:** Each agent’s prompt is constructed using its persona, role description, list of available tools (with simple descriptions), and the context of the conversation. MVP’s prompts might be somewhat static or template-based, but they ensure the agent is aware of who it is, what it knows (domain context), and how to use tools. The injection of tenant or domain guidelines is included if such guidelines exist (e.g., adding a line “Follow finance department guidelines.” in the system prompt if applicable).

* **Memory (Conversation Context):** During an execution, TAE keeps track of the conversation so far (what each agent said, and what observations were obtained from tools). This short-term memory is included in subsequent prompts to maintain continuity. For MVP, the memory manager is simple (no long-term memory), but it prevents the agents from repeating work or forgetting the user’s question. It may not yet summarize or compress context, but it will include at least the recent history within token limits.

* **Logging and Audit:** MVP implements logging of executions and tool calls. The user\_executions table will be populated with each run, and TAO’s member\_executions logs each tool usage. This means we have an audit trail of what happened. In case of errors, those are recorded too (e.g., if a tool failed or permission was denied, it’s logged). Operators or developers can inspect these logs to debug or analyze usage patterns.

* **Health and Metrics:** The service exposes a health check and basic Prometheus metrics. So in MVP deployment, one can verify TAE is running and see metrics like number of executions. This ensures that from DevOps perspective, TAE is not a black box – it can be monitored and managed.

* **Integration with TAO/TAB:** The MVP completes the end-to-end pipeline: a team built in TAO (via TAB UI) can be executed by TAE and produce an answer that is returned to the user (or calling system). Permissions set in TAO (which user can run which team, which tools an agent can use, which domains are linked) are all respected during execution. Essentially, all components work together to solve a user’s query with minimal friction.

* **Performance considerations:** While not heavily optimized in MVP, the design should handle typical queries in a reasonable time. With one or two LLM calls (maybe a few more if tools are used) per query, and with local infrastructure (LLM on GPU, vector DB in-memory), we expect responses in seconds for moderate complexity tasks. The MVP will not address extreme performance tuning, but it will be structured to allow parallelism or scale-out if needed (for example, nothing in design prevents running multiple queries at once, aside from model constraints).

* **Error handling and user feedback:** MVP will handle common error scenarios gracefully:

* If an agent cannot find an answer or outputs something unhelpful, the system will still return whatever it got (perhaps including an “I’m sorry I cannot find that” if that’s what agent said). The Critic agent might not intervene in MVP to fix it.

* If a tool fails (e.g., external API down) or permission denied, TAE will catch the exception and likely return an error message to the user (“The agent could not complete the request due to a tool error.” or similar). It won’t crash or hang indefinitely.

* If the user query is outside the scope or too vague, agents might struggle; MVP doesn’t include advanced clarification or asking user follow-up – they either do their best or say they can’t solve it. (Future might allow multi-turn user-agent dialog, not in initial scope.)

In summary, the MVP provides a functional multi-agent execution environment that covers the most common use case: executing a pre-configured team to answer a single user query, using tools as needed. It ensures the system’s value can be demonstrated (e.g., a retrieval agent pulling info and a solver agent giving a refined answer, all automatically) with reliability and security.

### 8.2 Future Enhancements

Beyond the MVP, there are several enhancements planned to increase the capabilities, efficiency, and intelligence of the Team Agent Executor:

* **Dynamic Agent Orchestration:** In the future, we want to support dynamic decision-making about which agent to invoke when, rather than a fixed sequence. This could involve:

* A **Planner Agent** or an upgraded ExecutionPlanner that, given a query, decides the order of agent involvement or even if some agents should be skipped. For example, if a query doesn’t require external info, maybe skip the Retriever and go straight to the Solver. This could be done with an LLM analyzing the query, or rules based on query keywords.

* **Conditional flows:** e.g., if agent A’s result meets a criterion, go to agent B; otherwise maybe loop back or call agent C. Representing the plan as a DAG or state machine that can handle branching. This is more complex, akin to workflow orchestration.

* **Parallel execution:** If there are independent subtasks, run two agents at the same time. For instance, one agent summarizes one document while another summarizes a different document concurrently, then a third agent combines results. This requires threading or async and would push complexity in collecting and merging results. But it can speed up responses for certain tasks (massively parallel search, etc.).

* **Multi-turn Conversations with User:** Currently, TAE handles one user query at a time. In future, we might allow interactive conversations where the user and agents go back-and-forth:

* This means maintaining a session context across multiple queries (the user can ask a follow-up question and the agents remember the previous discussion).

* We’d extend AgentMemoryManager to persist the conversation context between calls (maybe in a cache or DB).

* Possibly the UI would route each user message to TAE, which would treat the user’s prior messages as part of conversation history.

* This changes the execution pattern slightly: rather than one query \= one answer, it becomes more chat-like. TAE’s design with memory and loop could accommodate that, but we’d need to allow the conversation to continue until user stops (and maybe incorporate user as an agent in the loop conceptually).

* This would greatly enhance use cases like a Q\&A session or an interactive assistant style usage, beyond single questions.

* **Critic and Self-Healing Loops:** We plan to fully implement the CriticFeedbackEngine and possibly an automated self-correction mechanism:

* A **Critic Agent** (which could be one of the team members designated as a “Critic” role, or an external agent not visible to user) will evaluate intermediate or final outputs. For example, after an answer is formed, the Critic agent might say “this answer might be missing details from Q3 report”.

* Based on critic feedback, TAE could loop back to agents to refine the answer. This is essentially a feedback loop where the system tries to improve its answer autonomously.

* Implementation might involve giving the critic’s comments to the solver agent and asking it to revise the answer. Or if the critic found an error, maybe trigger the retrieval agent to get more data.

* This enhancement aims to increase answer quality and correctness without user needing to prompt again. It’s an advanced technique (similar to Chain-of-Thought with reflection).

* The Critic can also enforce policy: if answer is disallowed content, the Critic could veto it and force a sanitized version or a refusal message.

* This feature will require careful prompt design to ensure the critic is effective and doesn’t cause infinite loops of its own. Possibly limit to one revision cycle unless improvement is clearly possible.

* **Learning and Adaptation:** In later phases, TAE (with TAO’s help) could incorporate learning from past executions:

* **Adaptive Memory:** After many queries, the system might identify common patterns or frequently used information. We could have an offline process to fine-tune agent prompts or store FAQs. But online, we could:

  * Adjust an agent’s persona or knowledge base slightly based on feedback. For instance, if an agent always ends up using the same document chunk for certain questions, perhaps integrate that chunk into its persona or make it default context.

  * Score successes and failures: The logs could be analyzed to see where agents often fail or ask for help, and then improve config (maybe certain tools should be added that are missing, or an agent’s instructions could be clarified).

* **Tool Discovery:** A future feature mentioned is allowing agents to discover new tools at runtime if their current tools don’t suffice. For example, an agent could query a “tool marketplace” (maybe through a semantic search in a registry of all tools) to see if something outside its allowed list could help. This is tricky because of permission, but we might implement a supervised mode where the system suggests to an admin “Agent X wanted a tool Y it doesn’t have, consider adding it.” Or possibly automatically grant it if policy allows.

* **Auto-configuration adjustments:** TAO and TAE together could monitor execution patterns and adjust team configurations. E.g., if one agent never gets used in queries, maybe flag that to remove that agent or if a new skill is needed, flag to add an agent. These would likely be semi-automated in future (with admin approval).

* **Scalability & High Availability:** As usage grows, we’ll improve TAE’s ability to handle high throughput:

* Implement asynchronous execution: currently, each execution is handled in a single thread/process. We could utilize Python asyncio if calling LLM and waiting, so we can handle other requests in parallel (provided the LLM backend supports concurrency).

* Scale horizontally by running multiple TAE instances behind a load balancer. The stateless nature (each execution only depends on DB and external services) makes it straightforward to load-balance. Future deployment might put TAE behind an API gateway that distributes requests to multiple replicas. We’ll need to ensure logs and such are aggregated properly in that scenario.

* Possibly implement a job queue for executions if we expect spikes – e.g., user requests goes into a queue that workers pick up. This might not be needed unless queries are extremely heavy or we want to smooth usage of limited resources like GPUs.

* **User Interface & Experience improvements:** Although not directly part of TAE’s backend code, future phases might involve changes that affect how TAE is used:

* Better error messages: e.g., if something fails, provide the user a more informative yet safe message, possibly with suggestions (“I couldn’t find data on Q3, maybe the knowledge base needs updating.”).

* Partial progress streaming: If using an LLM that can stream tokens, TAE could start streaming the answer to the client before the full generation is done (like how ChatGPT streams answers). For that, we’d open a WebSocket or use HTTP chunked responses. This is a UX improvement for long answers. TAE would orchestrate similarly but feed out partial outputs when available.

* Interactivity: Possibly allow the user to correct an agent mid-execution. This is complex (basically human-in-the-loop on tool usage), likely not in near-term roadmap, but conceptually possible if UI could display agent thinking and let user intervene. Our design currently hides agent reasoning from user (just gives final answer), but advanced UX could reveal steps.

* **Cross-Team Collaboration:** A far-future idea (as mentioned in TAO’s doc) is multiple teams working together on a query. For TAE, this would mean orchestrating not just agents of one team, but agents across teams, possibly in parallel or sequence. For instance, a Finance team and a Marketing team both contribute an answer to a cross-domain question. This is largely speculative and would require both TAO and TAE to support inter-team communication protocols (maybe treat one team as a tool for another as we have, or a higher-level orchestrator that uses multiple TAE instances).

* This might manifest as an extension where TAE can spin up sub-executions for other teams (which we already handle via TeamAsTool concept), but on a bigger scale if two teams actively dialog. Likely not immediate roadmap, but an interesting direction if the platform expands.

* **Different Agent Modalities:** Currently agents communicate in text. Future enhancements might include agents that produce images, or code, etc., depending on tools. TAE might have to handle binary outputs (like if an agent’s tool returns a chart image, TAE could pass that to the front-end). This might just be an extension of tool handling (TAO could return a URL to an image and TAE would pass it along or embed it).

* Also, integration with voice (speaking agents) or other modalities would mostly be external to TAE, but TAE might coordinate e.g., generating a text answer then a text-to-speech tool could be a part of pipeline as a final step.

In conclusion, the future roadmap for TAE is aimed at making it **more intelligent, flexible, and scalable**. The MVP establishes a solid foundation with core features, and subsequent phases will enhance the system’s ability to handle more complex scenarios and to continuously improve its performance and output quality. By maintaining the modular design, each enhancement (be it in planning, memory, or collaboration) can be incorporated incrementally without overhauling the entire system, ensuring that TAE can evolve in tandem with advances in multi-agent AI capabilities.

---

[\[1\]](file://file-RMU5LSSYyRXPrC4xyjpz1r#:~:text=3,Index%20via%20Qdrant) Team Agent Orchestrator (TAO) – Module Documentation.docx

[file://file-RMU5LSSYyRXPrC4xyjpz1r](file://file-RMU5LSSYyRXPrC4xyjpz1r)