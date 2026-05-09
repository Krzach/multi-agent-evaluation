from datasets import load_dataset
from typing import List, Dict, Any

class AetherCodeDataset:
    """
    Utility class to load and access the AetherCode benchmark dataset.
    """
    def __init__(self, split: str = "test", difficulty: str = None):
        self.split = split
        self.difficulty = difficulty

    def get_tasks(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves tasks from the AetherCode dataset without fully downloading the dataset upfront.
        
        Args:
            limit: Maximum number of samples to return.
            
        Returns:
            A list of dictionary objects, where each object represents a coding task.
        """
        try:
            # Enable streaming so we only download the items we need
            dataset = load_dataset("m-a-p/AetherCode", 'v1_2024', split=self.split, streaming=True)
        except Exception:
            dataset = load_dataset("m-a-p/AetherCode", 'v1_2024', split="train", streaming=True)
            
        tasks = []
        for item in dataset:
            if self.difficulty is not None and item.get("difficulty") != self.difficulty:
                continue
            
            tasks.append(item)
            if limit is not None and len(tasks) >= limit:
                break

        return tasks