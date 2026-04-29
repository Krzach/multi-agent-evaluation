from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""
    @abstractmethod
    def evaluate(self, dataset: List[Dict[str, Any]]):
        """Evaluate the benchmark on the given dataset."""
        pass