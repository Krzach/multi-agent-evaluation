from datasets import load_dataset
from typing import List, Dict, Any

class HumanEvalDataset:
    """
    Utility class to load and access the HumanEval benchmark dataset.
    """
    def __init__(self, split: str = "test"):
        self._dataset = load_dataset("openai_humaneval", split=split)

    def get_tasks(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves tasks from the HumanEval dataset.
        
        Args:
            limit: Maximum number of samples to return.
            
        Returns:
            A list of dictionary objects, where each object represents a coding task.
        """
        tasks = list(self._dataset)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks
