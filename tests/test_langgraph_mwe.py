import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from langchain_core.messages import AIMessage
from frameworks.langgraph.agent import LangGraphAgent
from metrics.tracker import MetricsTracker

class MockLLM:
    """A mock LLM for testing without API calls."""
    def invoke(self, messages, **kwargs):
        # Simulate LLM processing time
        time.sleep(0.1)
        return AIMessage(content="This is a dummy response for GAIA evaluation.")

class TestLangGraphMWE(unittest.TestCase):
    def test_minimal_working_example(self):
        # 1. Initialize metrics tracker
        tracker = MetricsTracker()
        
        # 2. Initialize agent with a mock LLM and the tracker
        llm = MockLLM()
        agent = LangGraphAgent(llm_model=llm, metrics_tracker=tracker)
        
        # 3. Run a dummy GAIA task
        task_input = "What is the capital of France?"
        result = agent.run_task(task_input)
        
        # 4. Assertions on output
        self.assertIn("response", result)
        self.assertEqual(result["response"], "This is a dummy response for GAIA evaluation.")
        
        # 5. Assertions on metrics
        metrics_summary = tracker.get_summary()
        
        # Task time should be > 0.1s due to our sleep
        self.assertGreaterEqual(metrics_summary["time"]["total_task_time_seconds"], 0.1)
        
        print("\n=== MWE Test Passed ===")
        print("Metrics Summary:", metrics_summary)

if __name__ == "__main__":
    unittest.main()
