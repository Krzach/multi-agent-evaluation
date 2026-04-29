import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from benchmarks.human_eval.dataset import HumanEvalDataset
from benchmarks.human_eval.runner import HumanEvalRunner
from frameworks.langgraph.agent import LangGraphAgent
from metrics.tracker import MetricsTracker

class MockCodingLLM:
    """A mock LLM for testing HumanEval coding tasks."""
    def invoke(self, messages, **kwargs):
        time.sleep(0.1)
        # Check if the prompt asks for a specific task
        prompt = messages[0].content if messages else ""
        if "def has_close_elements" in prompt:
            # Provide a correct implementation for task 0
            code = """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
"""
            # To mock the AIMessage
            from langchain_core.messages import AIMessage
            return AIMessage(content=f"```python\n{code}\n```")
        
        from langchain_core.messages import AIMessage
        return AIMessage(content="def unknown_function():\n    pass")

class TestHumanEval(unittest.TestCase):
    def test_human_eval_flow(self):
        # 1. Load a single task from HumanEval (Task 0)
        # Note: In a real run, this requires internet access to download from HuggingFace
        # We'll mock the dataset output to ensure the test runs offline
        mock_dataset = [{
            "task_id": "HumanEval/0",
            "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
            "entry_point": "has_close_elements",
            "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
        }]
        
        # 2. Setup Agent and Tracker
        tracker = MetricsTracker()
        llm = MockCodingLLM()
        agent = LangGraphAgent(llm_model=llm, metrics_tracker=tracker)
        
        # 3. Run Evaluation
        runner = HumanEvalRunner(mas_instance=agent)
        results = runner.evaluate(dataset=mock_dataset)
        
        # 4. Assertions
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertEqual(result["task_id"], "HumanEval/0")
        self.assertEqual(result["correctness"], 1.0)
        self.assertEqual(result["error"], "")
        
        # Check metrics
        metrics = tracker.get_summary()
        self.assertGreaterEqual(metrics["time"]["total_task_time_seconds"], 0.1)
        
        print("\n=== HumanEval Test Passed ===")
        print("Final Correctness:", result["correctness"])

if __name__ == "__main__":
    unittest.main()
