import sys
import os
import time
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from frameworks.langgraph.agent import LangGraphAgent
from metrics.tracker import MetricsTracker

class MockLLM:
    """A mock LLM for testing without API calls."""
    def invoke(self, messages, **kwargs):
        time.sleep(0.1)
        
        # Check if context was passed in the system prompt
        sys_msg = next((m for m in messages if m.type == "system"), None)
        if sys_msg and "Mock retrieved document" in sys_msg.content:
            return AIMessage(content="Based on the context, the capital of France is Paris.")
            
        return AIMessage(content="This is a dummy response.")

class MockRetriever:
    """A mock retriever for testing without vector databases."""
    def invoke(self, query):
        return [Document(page_content=f"Mock retrieved document for query: '{query}'. The capital of France is Paris.")]

class TestSingleHopRAG(unittest.TestCase):
    def test_single_hop_flow(self):
        # 1. Initialize metrics and mocks
        tracker = MetricsTracker()
        llm = MockLLM()
        retriever = MockRetriever()
        
        # 2. Initialize agent
        agent = LangGraphAgent(llm_model=llm, retriever=retriever, metrics_tracker=tracker)
        
        # 3. Run task
        task_input = "What is the capital of France?"
        result = agent.run_task(task_input)
        
        # 4. Verify flow
        self.assertIn("response", result)
        self.assertIn("Paris", result["response"])
        
        # Verify the state contains the context
        state = result["full_state"]
        self.assertIn("context", state)
        self.assertIn("Mock retrieved document", state["context"])
        
        # Verify the messages list contains the Human query and the AI response
        messages = state["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, task_input)
        
        # 5. Verify metrics
        metrics = tracker.get_summary()
        self.assertGreaterEqual(metrics["time"]["total_task_time_seconds"], 0.1)
        
        print("\n=== Single-Hop RAG Test Passed ===")
        print("Final Response:", result["response"])

if __name__ == "__main__":
    unittest.main()
