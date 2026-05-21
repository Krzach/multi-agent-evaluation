import logging
import time
import textwrap
from typing import List, Dict, Any

from benchmarks.base import BenchmarkRunner
from metrics.conversation_log_metrics import build_conversation_log_metrics_envelope


class HumanEvalRunner(BenchmarkRunner):
    """
    Integrates the HumanEval benchmark to evaluate coding tasks across different agent frameworks.
    """

    def evaluate(self, dataset: List[Dict[str, Any]]):
        """
        Loops through the HumanEval dataset, passes the prompt to the MAS,
        and evaluates the correctness by executing the test cases.
        """
        results = []
        for item in dataset:
            task_id = item.get("task_id", "Unknown")
            prompt = item.get("prompt", "")
            test_code = item.get("test", "")
            entry_point = item.get("entry_point", "")
            
            # The MAS's task is to complete the code given the prompt
            task_input = (
                f"Complete the following Python code:\n\n{prompt}\n\n"
                "Only output the Python code that completes the function. "
                "Do not include explanations or markdown formatting if possible."
            )
            
            # Start timer for Total Task Completion Time
            start_time = time.perf_counter()
            
            # Execute MAS
            mas_output = self.mas.answer(task_input)
            generated_code = mas_output.get("writer_code", "")
            attempts = mas_output.get("attempt", 0)
            safeguard_allowed = mas_output.get("safeguard_allowed", True)
            tu = mas_output.get("token_usage") or {}
            token_usage = {
                "prompt_tokens": int(tu.get("prompt_tokens", 0)),
                "completion_tokens": int(tu.get("completion_tokens", 0)),
                "total_tokens": int(
                    tu.get(
                        "total_tokens",
                        int(tu.get("prompt_tokens", 0))
                        + int(tu.get("completion_tokens", 0)),
                    )
                ),
            }
            conversation_log_path = mas_output.get("conversation_log_path")

            # End timer
            end_time = time.perf_counter()
            total_task_time = end_time - start_time

            conversation_log_metrics = build_conversation_log_metrics_envelope(
                mas_output,
                total_task_time,
                framework_class_name=self.mas.__class__.__name__,
            )

            # Collaboration metric: prefer measured inter-agent messages from log, fallback to heuristic.
            measured_messages = int(
                conversation_log_metrics
                .get("aggregate_agents", {})
                .get("messages_between_agents", 0)
            )
            total_messages = measured_messages
            
            
            # Basic cleanup: if the agent outputs markdown code blocks, extract the code
            clean_code = generated_code
            if "```python" in clean_code:
                clean_code = clean_code.split("```python")[1].split("```")[0]
            elif "```" in clean_code:
                clean_code = clean_code.split("```")[1].split("```")[0]
                
            clean_code = clean_code.strip()

            # Check if the agent has regenerated the full function
            if f"def {entry_point}" in clean_code:
                # Agent returned the full function definition.
                # We will use this code directly, but we need to make sure imports from the prompt are included.
                prompt_lines = prompt.split('\n')
                import_lines = [line for line in prompt_lines if line.strip().startswith('import ') or line.strip().startswith('from ')]
                imports_from_prompt = '\n'.join(import_lines)
                
                # The full code is the imports from the prompt plus the agent's code
                full_execution_code = f"{imports_from_prompt}\n{clean_code}\n\n{test_code}\ncheck({entry_point})"
            else:
                # Agent returned just the body of the function.
                # We need to indent it correctly and combine it with the prompt.
                
                # Fix for first-line indentation issue
                clean_lines = clean_code.split('\n')
                if clean_lines and len(clean_lines[0]) - len(clean_lines[0].lstrip()) == 0 and len(clean_lines) > 1:
                    min_indent = float('inf')
                    for line in clean_lines[1:]:
                        if line.strip():
                            min_indent = min(min_indent, len(line) - len(line.lstrip()))
                    
                    if min_indent > 0 and min_indent != float('inf'):
                        clean_lines[0] = ' ' * min_indent + clean_lines[0]
                        clean_code = '\n'.join(clean_lines)

                # Standardize indentation
                if clean_code:
                    clean_code = textwrap.indent(textwrap.dedent(clean_code), "    ")

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
                "conversation_log_path": conversation_log_path,
                "mas_framework": self.mas.__class__.__name__,
                "conversation_log_metrics": conversation_log_metrics,

                # Qualitative Metrics
                "correctness": 1.0 if is_correct else 0.0,
                "error": error_message,
                
                # Cost Metrics
                "cost_metrics": {
                    "input_tokens": token_usage["prompt_tokens"],
                    "output_tokens": token_usage["completion_tokens"],
                    "total_tokens": token_usage["total_tokens"],
                },
                
                # Time Metrics
                "time_metrics": {
                    "total_task_completion_time_seconds": total_task_time
                },
                
                # Agent Collaboration Metrics
                "collaboration_metrics": {
                    "conversation_iterations": attempts,
                    "messages_between_agents": total_messages,
                    "activated_agents": 3, 
                    "safeguard_blocked": not safeguard_allowed
                }
            })
            
        return results