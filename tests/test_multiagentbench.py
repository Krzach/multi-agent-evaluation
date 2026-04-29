import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from benchmarks.multiagentbench.dataset import MultiAgentBenchDataset
from benchmarks.multiagentbench.runner import MultiAgentBenchRunner
from coding_scenario.langchain_mas import WorkflowState, LangchainCodingMAS

class MockMASSystem:
    """Mock the LangchainCodingMAS specifically to return outputs compatible with MAB assertions."""
    def answer(self, query):
        if "csv" in query.lower():
            output = "90.0"
            code = "import csv; print(90.0)"
        elif "palindrome" in query.lower():
            output = "True"
            code = "print(True)"
        elif "fibonacci" in query.lower():
            output = "0, 1, 1, 2, 3, 5, 8, 13, 21, 34"
            code = "print('0, 1, 1, 2, 3, 5, 8, 13, 21, 34')"
        elif "dictionary" in query.lower():
            output = "3"
            code = "print(3)"
        else:
            output = "Error"
            code = ""
            
        return {
            "final_answer": "Here is the computed answer.",
            "execution_output": output,
            "writer_code": code,
            "attempt": 1,
            "safeguard_allowed": True
        }

class TestMultiAgentBench(unittest.TestCase):
    def setUp(self):
        # Point to the subset JSON we just created
        self.data_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'benchmarks', 
            'data', 
            'multiagentbench_subset.json'
        )
        self.dataset = MultiAgentBenchDataset(self.data_path)

    def test_multiagentbench_evaluation(self):
        # 1. Load tasks
        tasks = self.dataset.get_tasks()
        self.assertEqual(len(tasks), 4)
        
        # 2. Setup Mock MAS (replace with LangchainCodingMAS in production)
        mock_mas = MockMASSystem()
        
        # 3. Setup Runner
        runner = MultiAgentBenchRunner(mas_instance=mock_mas)
        
        # 4. Evaluate
        results = runner.evaluate(tasks)
        
        # 5. Assertions
        for result in results:
            self.assertEqual(result["correctness"], 1.0)
            self.assertIn("collaboration_metrics", result)
            self.assertEqual(result["collaboration_metrics"]["attempts"], 1)

if __name__ == '__main__':
    unittest.main()
