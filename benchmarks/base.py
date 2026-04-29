from abc import ABC, abstractmethod
from typing import List, Dict, Any
from coding_scenario.base import CodingMASBase
import logging
class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""
    def __init__(self, mas_instance: CodingMASBase):
        self.mas = mas_instance
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def evaluate(self, dataset: List[Dict[str, Any]]):
        """Evaluate the benchmark on the given dataset."""
        pass