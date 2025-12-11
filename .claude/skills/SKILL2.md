# AGENT_ARCHITECTURES.skill.md

## Building Autonomous AI Agents: Memory, Planning, Tool Use, and Multi-Agent Systems

**Version**: 1.0
**Domain**: AI Systems, Autonomous Agents, LLM Applications
**Prerequisites**: LLM basics, programming fundamentals
**Complexity**: Advanced

---

## 1. EXECUTIVE SUMMARY

AI agents are systems that use LLMs to autonomously perceive, reason, plan, and act to achieve goals. Unlike chatbots (reactive), agents are proactive - they maintain state, use tools, and pursue objectives over multiple steps.

**Core Components**: Perception, Memory, Reasoning, Planning, Action, Learning

**Key Insight**: Agents are not smarter than their underlying LLM - they're more capable through better structure, memory, and tool access.

---

## 2. AGENT FUNDAMENTALS

### 2.1 The Agent Loop

```
while goal not achieved:
    1. Perceive: Gather information (user input, tool output, memory)
    2. Think: Reason about current state and goal
    3. Plan: Decide on next action(s)
    4. Act: Execute action (call tool, generate response)
    5. Observe: Process action result
    6. Update: Modify memory, state, beliefs
```

### 2.2 Agent vs. Pipeline

**Pipeline**: Fixed sequence of steps
**Agent**: Dynamic decision-making about what to do next

```python
# Pipeline
def pipeline(query):
    docs = retrieve(query)
    context = summarize(docs)
    answer = generate(context, query)
    return answer

# Agent
def agent(goal):
    while not done:
        action = decide_next_action(state, goal, memory)
        result = execute(action)
        update_state(result)
    return final_result
```

### 2.3 Levels of Agency

1. **No Agency**: Direct prompt → response
2. **Tool Use**: LLM decides which tools to call
3. **Planning**: Multi-step plans toward goals
4. **Learning**: Improves from experience
5. **Multi-Agent**: Coordinates with other agents

---

## 3. MEMORY SYSTEMS

### 3.1 Memory Types

**Working Memory**: Current context window
- What the agent is currently thinking about
- Limited by context length
- Cleared between tasks

**Short-term Memory**: Recent interactions
- Last N conversation turns
- Recent observations
- Temporary scratchpad

**Long-term Memory**: Persistent knowledge
- Past experiences
- Learned facts
- User preferences
- Episodic memories (what happened)
- Semantic memories (what things mean)

### 3.2 Memory Implementation Patterns

**In-Context Memory**:
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": memory_summary},  # injected memory
    {"role": "user", "content": current_query},
]
```

**Vector Database Memory**:
```python
# Store
embedding = embed(experience)
vector_db.upsert(id, embedding, metadata)

# Retrieve
query_embedding = embed(current_context)
relevant_memories = vector_db.query(query_embedding, top_k=5)

# Inject
context = format_memories(relevant_memories) + current_query
```

**Structured Memory**:
```python
memory = {
    "facts": [...],           # Known information
    "episodes": [...],        # Past events
    "preferences": {...},     # User preferences
    "skills": [...],          # Learned procedures
}
```

### 3.3 Memory Operations

**Encoding**: Convert experience to storable form
**Storage**: Save to persistence layer
**Retrieval**: Find relevant memories
**Consolidation**: Compress/summarize old memories
**Forgetting**: Remove irrelevant memories

### 3.4 Retrieval Strategies

**Recency**: Most recent memories first
**Relevance**: Semantic similarity to current context
**Importance**: Memories marked as significant
**Hybrid**: Combine multiple signals

```python
def retrieve_memories(query, k=10):
    # Get candidates by relevance
    candidates = vector_search(query, k=50)

    # Re-rank by combined score
    scored = []
    for mem in candidates:
        score = (
            0.5 * mem.relevance_score +
            0.3 * recency_score(mem.timestamp) +
            0.2 * mem.importance
        )
        scored.append((mem, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
```

### 3.5 Memory Challenges

**Context Overflow**: Too much memory for context window
- Solution: Summarize, prioritize, hierarchical retrieval

**Staleness**: Outdated information
- Solution: Timestamps, periodic refresh, contradiction detection

**Hallucination Injection**: False memories
- Solution: Source tracking, confidence scores, verification

---

## 4. REASONING PATTERNS

### 4.1 Chain of Thought (CoT)

Think step by step before answering:

```
User: If I have 3 apples and buy 2 more, then give away 1, how many do I have?

Agent: Let me think step by step.
1. Start with 3 apples
2. Buy 2 more: 3 + 2 = 5 apples
3. Give away 1: 5 - 1 = 4 apples

Therefore, I have 4 apples.
```

### 4.2 ReAct (Reasoning + Acting)

Interleave reasoning with actions:

```
Thought: I need to find the capital of France.
Action: search("capital of France")
Observation: Paris is the capital of France.
Thought: Now I know the answer.
Action: respond("The capital of France is Paris.")
```

### 4.3 Tree of Thoughts (ToT)

Explore multiple reasoning paths:

```
Problem: [complex problem]

Branch 1:
  Thought 1.1: [approach A]
  Thought 1.2: [continue A]
  Evaluation: 0.6

Branch 2:
  Thought 2.1: [approach B]
  Thought 2.2: [continue B]
  Evaluation: 0.8  ← select this branch

Branch 3:
  Thought 3.1: [approach C]
  Evaluation: 0.3  ← prune
```

### 4.4 Self-Consistency

Generate multiple solutions, vote on answer:

```python
answers = []
for _ in range(5):
    answer = generate_with_cot(problem)
    answers.append(answer)

final_answer = majority_vote(answers)
```

### 4.5 Reflexion

Reflect on failures and retry:

```
Attempt 1:
  Action: [action]
  Result: [failure]
  Reflection: The action failed because [reason]. I should instead [improvement].

Attempt 2:
  Action: [improved action based on reflection]
  Result: [success]
```

### 4.6 Scratchpad

Use explicit working memory:

```
<scratchpad>
Goal: Book a flight from NYC to LA on December 1st
Known:
- Departure: NYC
- Destination: LA
- Date: Dec 1
Unknown:
- Available flights
- Prices
Next step: Search for flights
</scratchpad>

Action: search_flights(from="NYC", to="LA", date="2024-12-01")
```

---

## 5. PLANNING

### 5.1 Plan-and-Execute

Generate full plan, then execute:

```python
def plan_and_execute(goal):
    # Planning phase
    plan = llm.generate(f"Create a step-by-step plan to: {goal}")
    steps = parse_plan(plan)

    # Execution phase
    results = []
    for step in steps:
        result = execute_step(step)
        results.append(result)
        if result.failed:
            # Re-plan from here
            new_plan = replan(goal, steps, results)
            steps = new_plan

    return synthesize_results(results)
```

### 5.2 Hierarchical Planning

High-level plan → detailed sub-plans:

```
Goal: Build a web app

High-level plan:
1. Design architecture
2. Build backend
3. Build frontend
4. Deploy

Sub-plan for "Build backend":
1.1. Set up database schema
1.2. Create API endpoints
1.3. Implement authentication
1.4. Write tests
```

### 5.3 Adaptive Planning

Re-plan based on observations:

```python
def adaptive_agent(goal):
    plan = initial_plan(goal)

    while not goal_achieved:
        step = plan.next_step()
        result = execute(step)

        if result.unexpected():
            # Re-evaluate and potentially re-plan
            plan = revise_plan(plan, result, goal)
```

### 5.4 Planning with Constraints

```
Goal: [objective]

Constraints:
- Budget: $100
- Time: 2 hours
- Tools available: [list]
- Must not: [restrictions]

Plan: [plan that satisfies constraints]
```

---

## 6. TOOL USE

### 6.1 Tool Definition

```python
tools = [
    {
        "name": "search",
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {
            "expression": {"type": "string", "description": "Math expression"}
        }
    }
]
```

### 6.2 Function Calling

```python
response = llm.chat(
    messages=[{"role": "user", "content": "What's 15% of 200?"}],
    tools=tools,
    tool_choice="auto"
)

if response.tool_calls:
    for call in response.tool_calls:
        result = execute_tool(call.name, call.arguments)
        # Feed result back to LLM
```

### 6.3 Tool Categories

**Information Retrieval**:
- Web search
- Database queries
- File reading
- API calls

**Computation**:
- Calculator
- Code execution
- Data analysis

**Action**:
- Send email
- Create file
- API mutations
- System commands

**Communication**:
- Human handoff
- Multi-agent messaging

### 6.4 Tool Selection Strategies

**LLM-based**: Let the model choose
**Rule-based**: Pattern matching on intent
**Hybrid**: Rules for obvious cases, LLM for ambiguous

### 6.5 Tool Use Best Practices

1. **Clear descriptions**: Unambiguous tool purpose
2. **Examples**: Show when to use each tool
3. **Error handling**: What to do when tools fail
4. **Confirmation**: High-stakes actions need user approval
5. **Logging**: Track all tool calls for debugging

---

## 7. MULTI-AGENT SYSTEMS

### 7.1 Why Multiple Agents?

- **Specialization**: Different agents for different tasks
- **Parallelism**: Work on multiple things simultaneously
- **Diversity**: Multiple perspectives on problems
- **Modularity**: Easier to develop and debug

### 7.2 Communication Patterns

**Hierarchical**: Manager agent delegates to workers
```
Manager
├── Researcher Agent
├── Writer Agent
└── Reviewer Agent
```

**Peer-to-Peer**: Agents communicate directly
```
Agent A ↔ Agent B ↔ Agent C
```

**Blackboard**: Shared knowledge base
```
Agent A → Blackboard ← Agent B
              ↑
          Agent C
```

### 7.3 Coordination Mechanisms

**Turn-taking**: Agents speak in order
**Voting**: Majority decides
**Consensus**: All must agree
**Market**: Bid on tasks

### 7.4 Multi-Agent Debate

Agents argue different positions:

```
Agent 1 (Pro): [argument for X]
Agent 2 (Con): [argument against X]
Agent 1: [rebuttal]
Agent 2: [counter-rebuttal]
Judge: [synthesize and decide]
```

### 7.5 Actor-Critic Pattern

```
Actor Agent: Generates actions/content
Critic Agent: Evaluates and provides feedback
Actor Agent: Revises based on feedback
[repeat until quality threshold]
```

### 7.6 Multi-Agent Frameworks

- **AutoGen** (Microsoft): Conversational agents
- **CrewAI**: Role-based agent teams
- **LangGraph**: Graph-based agent orchestration

---

## 8. EVALUATION AND SAFETY

### 8.1 Agent Evaluation Metrics

**Task Success Rate**: Did it achieve the goal?
**Efficiency**: Steps/tokens/time to completion
**Cost**: API calls, compute resources
**Safety**: Harmful actions attempted
**User Satisfaction**: Subjective quality

### 8.2 Benchmarks

- **WebArena**: Web navigation tasks
- **GAIA**: General AI assistants
- **AgentBench**: Multi-domain agent tasks
- **ToolBench**: Tool use evaluation

### 8.3 Safety Considerations

**Action Verification**: Confirm before irreversible actions
**Sandboxing**: Limit agent's actual capabilities
**Human-in-the-Loop**: Require approval for high-stakes
**Monitoring**: Log all actions for review
**Kill Switch**: Ability to stop agent immediately

### 8.4 Common Failure Modes

**Infinite Loops**: Agent keeps retrying failed action
- Solution: Max iteration limits, loop detection

**Goal Drift**: Agent pursues different goal than intended
- Solution: Goal verification, regular check-ins

**Excessive Tool Use**: Calls tools unnecessarily
- Solution: Better prompting, tool use limits

**Confidently Wrong**: Agent is certain but incorrect
- Solution: Uncertainty estimation, verification steps

---

## 9. IMPLEMENTATION PATTERNS

### 9.1 Basic Agent Structure

```python
class Agent:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def run(self, goal):
        self.memory.add("goal", goal)

        while not self.is_done():
            # Get relevant context
            context = self.memory.retrieve(goal)

            # Decide next action
            action = self.llm.decide(context, self.tools)

            # Execute action
            if action.type == "tool":
                result = self.execute_tool(action)
                self.memory.add("observation", result)
            elif action.type == "respond":
                return action.content

        return self.synthesize_result()
```

### 9.2 LangChain Agent

```python
from langchain.agents import create_react_agent
from langchain.tools import Tool

tools = [
    Tool(name="Search", func=search, description="Search the web"),
    Tool(name="Calculate", func=calc, description="Do math"),
]

agent = create_react_agent(llm, tools, prompt)
result = agent.invoke({"input": "What is 15% of the US population?"})
```

### 9.3 Custom Agent Loop

```python
async def agent_loop(goal, max_steps=10):
    history = []

    for step in range(max_steps):
        # Build prompt with history
        prompt = build_prompt(goal, history)

        # Get LLM response
        response = await llm.generate(prompt)

        # Parse action
        action = parse_action(response)

        if action.type == "finish":
            return action.result

        # Execute tool
        observation = await execute_tool(action.tool, action.input)

        # Add to history
        history.append({
            "thought": response.thought,
            "action": action,
            "observation": observation
        })

    return "Max steps reached without completion"
```

---

## 10. ADVANCED PATTERNS

### 10.1 Cognitive Architecture

```
┌─────────────────────────────────────┐
│           Executive Control          │
│  (Goal management, attention, etc.)  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         Working Memory               │
│  (Current context, active goals)     │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌───▼───┐  ┌───▼───┐
│Episodic│ │Semantic│ │Procedural│
│Memory  │ │Memory  │ │Memory    │
│(Events)│ │(Facts) │ │(Skills)  │
└────────┘ └────────┘ └──────────┘
```

### 10.2 Meta-Learning Agent

Agent that improves its own prompts/strategies:

```python
def meta_learning_loop():
    strategy = initial_strategy()

    for task in tasks:
        result = agent.run(task, strategy)

        # Evaluate performance
        score = evaluate(result, task.expected)

        # Reflect on strategy
        reflection = llm.reflect(strategy, result, score)

        # Update strategy
        strategy = llm.improve_strategy(strategy, reflection)
```

### 10.3 World Model Agent

Agent maintains model of environment:

```python
class WorldModelAgent:
    def __init__(self):
        self.world_model = {}

    def act(self, goal):
        # Simulate actions in world model
        best_action = None
        best_outcome = -inf

        for action in possible_actions:
            predicted_state = self.simulate(action)
            value = self.evaluate_state(predicted_state, goal)
            if value > best_outcome:
                best_action = action
                best_outcome = value

        # Execute best action
        result = self.execute(best_action)

        # Update world model
        self.world_model.update(result)

        return result
```

---

## 11. DEPLOYMENT CONSIDERATIONS

### 11.1 Latency

- Cache common queries
- Parallel tool execution
- Streaming responses
- Speculative execution

### 11.2 Cost

- Token budget per task
- Model selection (cheap for simple tasks)
- Caching and deduplication
- Early stopping

### 11.3 Reliability

- Retry logic with backoff
- Fallback strategies
- Graceful degradation
- Health monitoring

### 11.4 Observability

- Log all agent actions
- Trace multi-step reasoning
- Monitor success rates
- Alert on anomalies

---

## 12. REFERENCES

**Papers**:
- Yao et al. (2022). ReAct: Synergizing Reasoning and Acting
- Shinn et al. (2023). Reflexion
- Yao et al. (2023). Tree of Thoughts
- Park et al. (2023). Generative Agents (Stanford)

**Frameworks**:
- LangChain: https://langchain.com
- AutoGen: https://microsoft.github.io/autogen/
- CrewAI: https://crewai.com

---

## 13. VERSION HISTORY

- **v1.0** (2024-11): Initial comprehensive version

---

*The most common mistake in agent design is over-engineering. Start with the simplest loop that could work, then add complexity only when needed.*
