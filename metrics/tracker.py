import time
import json
from pathlib import Path
from typing import Dict, Any, Callable
from .cost import CostTracker
from .time import TimeTracker

class MetricsTracker:
    """
    A centralized tracker for collecting metrics across different multi-agent frameworks.
    Provides framework-specific integration points (e.g., LangChain callbacks, AutoGen hooks).
    """
    def __init__(self):
        self.cost = CostTracker()
        self.time = TimeTracker()
        # Add collaboration and qualitative trackers later
        
    def start_task(self):
        self.time.start_task()
        
    def end_task(self):
        self.time.end_task()

    def record_llm_call(self, duration: float, input_tokens: int, output_tokens: int):
        """
        A generic method to record LLM usage. 
        Framework adapters can call this directly if they don't support callbacks.
        """
        self.time.add_agent_response_time(duration)
        self.cost.add_tokens(input_tokens, output_tokens)
        
    # --- Framework Specific Integrations ---
        
    def get_langchain_callbacks(self) -> list:
        """
        Returns LangChain callback handlers for LangGraph.
        """
        try:
            from langchain_core.callbacks import BaseCallbackHandler
            
            class LangGraphMetricsCallback(BaseCallbackHandler):
                def __init__(self, tracker: 'MetricsTracker'):
                    self.tracker = tracker
                    self.agent_start_times = {}
                    
                def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
                    run_id = kwargs.get('run_id')
                    if run_id:
                        self.agent_start_times[run_id] = time.time()
                    
                def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
                    run_id = kwargs.get('run_id')
                    if run_id and run_id in self.agent_start_times:
                        duration = time.time() - self.agent_start_times[run_id]
                        
                        input_tokens = 0
                        output_tokens = 0
                        
                        if response.llm_output and 'token_usage' in response.llm_output:
                            usage = response.llm_output['token_usage']
                            input_tokens = usage.get('prompt_tokens', 0)
                            output_tokens = usage.get('completion_tokens', 0)
                            
                        self.tracker.record_llm_call(duration, input_tokens, output_tokens)
                        
            return [LangGraphMetricsCallback(self)]
        except ImportError:
            return []

    def register_autogen_hooks(self, agent: Any):
        """
        Registers hooks for AutoGen agents to track metrics.
        AutoGen provides `register_hook` for events like `process_message_before_send` 
        or you can wrap the client's `create` method.
        """
        # This is a conceptual implementation. AutoGen's tracking is often done
        # by inspecting `agent.client.actual_usage_summary`.
        pass
        
    def get_spadellm_interceptors(self) -> list:
        """
        Returns interceptors or middleware for SpadeLLM.
        """
        # Conceptual implementation for SpadeLLM tracking.
        pass

    def summarize_conversation_log(self, log_path: str) -> Dict[str, Any]:
        """Summarize a MAS conversation log JSONL file for collaboration metrics."""
        path = Path(log_path)
        if not path.exists():
            return {
                "log_path": log_path,
                "exists": False,
                "total_events": 0,
                "agent_outputs": 0,
                "passes": 0,
                "actions": 0,
                "errors": 0,
            }

        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

        event_records = [r for r in records if r.get("record_type") == "event"]
        return {
            "log_path": str(path),
            "exists": True,
            "total_records": len(records),
            "total_events": len(event_records),
            "agent_outputs": sum(1 for r in event_records if r.get("event_type") == "agent_output"),
            "passes": sum(1 for r in event_records if r.get("event_type") == "pass"),
            "actions": sum(1 for r in event_records if r.get("event_type") == "action"),
            "errors": sum(1 for r in event_records if r.get("event_type") == "error"),
        }
        
    def get_summary(self) -> Dict[str, Any]:
        return {
            "cost": self.cost.get_summary(),
            "time": self.time.get_summary()
        }