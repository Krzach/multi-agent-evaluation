import json
from pathlib import Path
from typing import List, Dict, Any

class MultiAgentBenchDataset:
    """
    Utility class to load a subset of the MultiAgentBench tasks.
    """
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self._data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Loads the JSON dataset from disk."""
        if not self.data_path.exists():
            print(f"Warning: Dataset not found at {self.data_path}")
            return []
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_tasks(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves a subset of tasks.
        """
        filtered_data = self._data
        if limit is not None:
            filtered_data = filtered_data[:limit]
        return filtered_data
