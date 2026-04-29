import json
from pathlib import Path
from typing import List, Dict, Any

class GaiaDataset:
    """
    Utility class to load and filter a subset of the GAIA benchmark dataset.
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

    def get_subset(self, level: int = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves a subset of the dataset.
        
        Args:
            level: The GAIA difficulty level (e.g., 1 for single-hop, 2 for multi-hop).
            limit: Maximum number of samples to return.
            
        Returns:
            A list of dictionary objects representing the tasks.
        """
        filtered_data = self._data

        if level is not None:
            filtered_data = [item for item in filtered_data if item.get("Level") == level]

        if limit is not None:
            filtered_data = filtered_data[:limit]

        return filtered_data

    def get_single_hop_tasks(self, limit: int = None) -> List[Dict[str, Any]]:
        """Convenience method for retrieving Level 1 (single-hop) tasks."""
        return self.get_subset(level=1, limit=limit)

    def get_multi_hop_tasks(self, limit: int = None) -> List[Dict[str, Any]]:
        """Convenience method for retrieving Level 2 (multi-hop) tasks."""
        return self.get_subset(level=2, limit=limit)
