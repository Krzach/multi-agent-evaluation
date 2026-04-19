import time
from typing import Dict, Any
from .cost import CostTracker
from .time import TimeTracker

class MetricsTracker:
    def __init__(self):
        self.cost = CostTracker()
        self.time = TimeTracker()
        # Add collaboration and qualitative trackers later
        
    def start_task(self):
        self.time.start_task()
        
    def end_task(self):
        self.time.end_task()
        
    def get_langchain_callbacks(self) -> list:
        """
        Returns a list of LangChain callback handlers to automatically track 
        tokens, costs, and times during LangGraph execution.
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
                        self.tracker.time.add_agent_response_time(duration)
                        
                    # Track tokens if available in response
                    if response.llm_output and 'token_usage' in response.llm_output:
                        usage = response.llm_output['token_usage']
                        self.tracker.cost.add_tokens(
                            input_tokens=usage.get('prompt_tokens', 0),
                            output_tokens=usage.get('completion_tokens', 0)
                        )
                        
            return [LangGraphMetricsCallback(self)]
        except ImportError:
            # Fallback if langchain isn't installed
            return []
        
    def get_summary(self) -> Dict[str, Any]:
        return {
            "cost": self.cost.get_summary(),
            "time": self.time.get_summary()
        }
