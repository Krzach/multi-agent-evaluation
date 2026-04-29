import operator
from typing import Any, Dict, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from frameworks.base import BaseAgentFramework
from frameworks.langgraph.nodes import retrieve_node, generate_node

# Define the state for the Single-Hop RAG LangGraph agent
class AgentState(TypedDict):
    # Annotated with operator.add so messages append instead of overwrite
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str

class LangGraphAgent(BaseAgentFramework):
    def __init__(self, llm_model: Any, retriever: Any = None, metrics_tracker: Any = None):
        self.llm = llm_model
        self.retriever = retriever
        self.metrics_tracker = metrics_tracker
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Node wrappers to inject dependencies
        def run_retriever(state: AgentState):
            return retrieve_node(state, self.retriever)
            
        def run_generator(state: AgentState):
            return generate_node(state, self.llm)
        
        # Add nodes
        workflow.add_node("retrieve", run_retriever)
        workflow.add_node("generate", run_generator)
        
        # Define edges (Linear flow for single-hop RAG: Start -> Retrieve -> Generate -> End)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
        
    def run_task(self, task_input: str, **kwargs) -> Dict[str, Any]:
        # Start time tracking
        if self.metrics_tracker:
            self.metrics_tracker.start_task()
            
        inputs = {"messages": [HumanMessage(content=task_input)]}
        
        # Pass callbacks via config to track LangChain/LangGraph events
        config = {}
        if self.metrics_tracker and hasattr(self.metrics_tracker, 'get_langchain_callbacks'):
            config["callbacks"] = self.metrics_tracker.get_langchain_callbacks()
            
        output = self.graph.invoke(inputs, config=config)
        
        if self.metrics_tracker:
            self.metrics_tracker.end_task()
            
        return {
            "response": output["messages"][-1].content,
            "full_state": output
        }
