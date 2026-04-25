import logging
from typing import List, Dict, Any

class HumanEvalRunner:
    """
    Integrates the HumanEval benchmark to evaluate coding tasks across different agent frameworks.
    """
    def __init__(self, agent_framework):
        self.agent = agent_framework
        self.logger = logging.getLogger(__name__)

    def evaluate(self, dataset: List[Dict[str, Any]]):
        """
        Loops through the HumanEval dataset, passes the prompt to the agent, 
        and evaluates the correctness by executing the test cases.
        """
        results = []
        for item in dataset:
            task_id = item.get("task_id", "Unknown")
            prompt = item.get("prompt", "")
            test_code = item.get("test", "")
            entry_point = item.get("entry_point", "")
            
            # The agent's task is to complete the code given the prompt
            task_input = (
                f"Complete the following Python code:\n\n{prompt}\n\n"
                "Only output the Python code that completes the function. "
                "Do not include explanations or markdown formatting if possible."
            )
            
            # Execute agent framework
            output = self.agent.run_task(task_input=task_input)
            generated_code = output['response']
            
            # Basic cleanup: if the agent outputs markdown code blocks, extract the code
            clean_code = generated_code
            if "```python" in clean_code:
                clean_code = clean_code.split("```python")[1].split("```")[0]
            elif "```" in clean_code:
                clean_code = clean_code.split("```")[1].split("```")[0]
                
            # Combine the original prompt, the generated completion, and the test cases
            # HumanEval test format relies on `check(entry_point)` being called at the end.
            full_execution_code = f"{prompt}\n{clean_code}\n\n{test_code}\ncheck({entry_point})"
            
            # Evaluate correctness
            # WARNING: Using `exec` executes untrusted code. In a production evaluation system, 
            # this MUST be executed in an isolated, sandboxed environment (e.g., Docker container).
            is_correct = False
            error_message = ""
            try:
                exec_globals = {}
                exec(full_execution_code, exec_globals)
                is_correct = True
            except Exception as e:
                is_correct = False
                error_message = type(e).__name__ + ": " + str(e)
            
            results.append({
                "task_id": task_id,
                "prompt": prompt,
                "generated_code": clean_code,
                "correctness": 1.0 if is_correct else 0.0,
                "error": error_message,
                "metrics": self.agent.metrics_tracker.get_summary() if self.agent.metrics_tracker else None
            })
            
        return results
