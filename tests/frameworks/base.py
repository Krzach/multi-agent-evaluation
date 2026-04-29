from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgentFramework(ABC):
    @abstractmethod
    def run_task(self, task_input: str, **kwargs) -> Dict[str, Any]:
        """
        Runs a task through the agent framework.
        
        Args:
            task_input: The prompt or task description.
            **kwargs: Additional framework-specific arguments.
            
        Returns:
            A dictionary containing the final response and any framework-specific metadata.
        """
        pass
