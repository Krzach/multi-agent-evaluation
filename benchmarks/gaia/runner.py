import logging
from typing import List, Dict, Any

class GaiaRunner:
    """
    Integrates with MASEval to run the GAIA benchmark across different agent frameworks.
    """
    def __init__(self, agent_framework):
        self.agent = agent_framework
        self.logger = logging.getLogger(__name__)
        
    def evaluate(self, dataset: List[Dict[str, Any]]):
        """
        Loops through the GAIA dataset, passes questions to the agent, 
        and collects correctness metrics.
        """
        results = []
        for item in dataset:
            question = item.get('Question', '')
            expected_answer = item.get('Final answer', '')
            task_level = item.get('Level', 'Unknown')
            
            # Execute agent framework
            output = self.agent.run_task(task_input=question)
            
            # For a real MASEval run, you would use maseval.evaluate() here.
            # This is a naive correctness check for demonstration.
            # RAG answers are often verbose, so we just check if the exact answer string is in the response.
            is_correct = expected_answer.lower() in output['response'].lower()
            correctness_score = 1.0 if is_correct else 0.0
            
            results.append({
                "task_id": item.get("task_id", "Unknown"),
                "level": task_level,
                "question": question,
                "expected_answer": expected_answer,
                "agent_response": output['response'],
                "correctness": correctness_score,
                "metrics": self.agent.metrics_tracker.get_summary() if self.agent.metrics_tracker else None
            })
            
        return results
