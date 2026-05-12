import logging
import time
import subprocess
import tempfile
import os
import shutil
from typing import List, Dict, Any

from benchmarks.base import BenchmarkRunner
from metrics.conversation_log_metrics import build_conversation_log_metrics_envelope


class AetherCodeRunner(BenchmarkRunner):
    """
    Integrates the AetherCode benchmark to evaluate coding tasks across different agent frameworks.
    """

    def evaluate(self, dataset: List[Dict[str, Any]]):
        """
        Loops through the AetherCode dataset, passes the prompt to the MAS,
        and evaluates the correctness by compiling the generated python code and running the provided cpp checker.
        """
        results = []
        for item in dataset:
            # Based on AetherCode schema, problems are competitive programming tasks
            task_id = str(item.get("id", "Unknown"))
            prompt = item.get("description", "")
            checker_cpp = item.get("checker", "")
            test_cases = item.get("test_cases", [])
            
            # The MAS's task is to complete the code given the prompt
            task_input = (
                f"Write a Python script to solve the following problem:\n\n{prompt}\n\n"
                "Only output the Python code. Do not include explanations or markdown formatting if possible."
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
                
            # Evaluate correctness
            is_correct = False
            error_message = ""
            debug_info = [] # Store test case execution info
            
            if not prompt:
                error_message = f"Dataset schema error. Could not find 'description' field. Available keys: {list(item.keys())}"
            else:
                try:
                    # Verify syntax
                    compile(clean_code, '<string>', 'exec')
                    is_correct = True
                    error_message = "Code compiled successfully"
                    
                    if not test_cases:
                        error_message += " (Execution skipped due to lack of automated test cases in dataset)"
                    else:
                                
                        with tempfile.TemporaryDirectory() as tmpdir:
                            solution_path = os.path.join(tmpdir, "solution.py")
                            with open(solution_path, "w") as f:
                                f.write(clean_code)

                            # If a cpp checker is provided, compile and run it
                            checker_exe = None
                            if checker_cpp:
                                checker_path = os.path.join(tmpdir, "checker.cpp")
                                # Download testlib.h to the temp directory
                                testlib_path = os.path.join(tmpdir, "testlib.h")
                                try:
                                    subprocess.run(["wget", "-q", "-O", testlib_path, "https://raw.githubusercontent.com/MikeMirzayanov/testlib/master/testlib.h"])
                                except Exception as e:
                                    pass

                                checker_exe = os.path.join(tmpdir, "checker")
                                with open(checker_path, "w") as f:
                                    f.write(checker_cpp)
                                try:
                                    compile_result = subprocess.run(
                                        ["g++", "-O2", checker_path, "-o", checker_exe],
                                        capture_output=True, text=True, timeout=15
                                    )
                                    if compile_result.returncode != 0:
                                        checker_exe = None
                                        error_message = f"Checker compilation failed. Error: {compile_result.stderr}"
                                        is_correct = False
                                except Exception as e:
                                    checker_exe = None
                                    error_message = f"Checker compilation error: {e}"
                                    is_correct = False

                            if is_correct:
                                # Run against test cases
                                passed_all = True
                                failed_reason = ""
                                for i, tc in enumerate(test_cases):
                                    tc_input = tc.get("input", "")
                                    tc_output = tc.get("output", "")
                                    
                                    in_file = os.path.join(tmpdir, f"test_{i}.in")
                                    out_file = os.path.join(tmpdir, f"test_{i}.out")
                                    ans_file = os.path.join(tmpdir, f"test_{i}.ans")
                                    
                                    with open(in_file, "w") as f: f.write(tc_input)
                                    with open(ans_file, "w") as f: f.write(tc_output)
                                    
                                    test_info = {
                                        "test_index": i,
                                        "input": tc_input[:1000] + ("..." if len(tc_input) > 1000 else ""),
                                        "expected_output": tc_output[:1000] + ("..." if len(tc_output) > 1000 else ""),
                                        "actual_output": "",
                                        "status": "PASS",
                                        "checker_output": ""
                                    }
                                    
                                    try:
                                        with open(in_file, "r") as f_in, open(out_file, "w") as f_out:
                                            # Run python solution
                                            subprocess.run(
                                                ["python3", solution_path],
                                                stdin=f_in, stdout=f_out, stderr=subprocess.PIPE,
                                                text=True, timeout=5
                                            )
                                        # Read generated output
                                        with open(out_file, "r") as f:
                                            actual_out = f.read()
                                            test_info["actual_output"] = actual_out[:1000] + ("..." if len(actual_out) > 1000 else "")
                                            
                                    except subprocess.TimeoutExpired:
                                        test_info["status"] = "FAIL (Timeout)"
                                        test_info["actual_output"] = "[Execution Timed Out]"
                                        passed_all = False
                                        failed_reason = f"Test {i} timed out."
                                        debug_info.append(test_info)
                                        break
                                    except Exception as e:
                                        test_info["status"] = f"FAIL (Crash: {e})"
                                        passed_all = False
                                        failed_reason = f"Test {i} crashed: {e}"
                                        debug_info.append(test_info)
                                        break
                                        
                                    if checker_exe:
                                        # Run testlib checker: ./checker <input> <output> <answer>
                                        # Note: For testlib checkers, we usually pass input, output, answer.
                                        # So here: in_file is test.in, out_file is the output generated by the python script,
                                        # ans_file is the expected output from the dataset.
                                        chk_res = subprocess.run(
                                            [checker_exe, in_file, out_file, ans_file],
                                            capture_output=True, text=True
                                        )
                                        test_info["checker_output"] = chk_res.stderr.strip()
                                        # Testlib standard:
                                        # Exit code 0 means OK
                                        if chk_res.returncode != 0:
                                            test_info["status"] = "FAIL (Checker)"
                                            passed_all = False
                                            failed_reason = f"Test {i} failed checker: {chk_res.stderr.strip()}"
                                            debug_info.append(test_info)
                                            break
                                    else:
                                        # Strict token matching fallback if no checker
                                        with open(out_file, "r") as f: user_out = f.read().strip().split()
                                        expected_out = tc_output.strip().split()
                                        if user_out != expected_out:
                                            test_info["status"] = "FAIL (Mismatch)"
                                            test_info["checker_output"] = "Token mismatch fallback checker"
                                            passed_all = False
                                            failed_reason = f"Test {i} output mismatch."
                                            debug_info.append(test_info)
                                            break
                                            
                                    debug_info.append(test_info)

                                if passed_all:
                                    is_correct = True
                                    error_message = ""
                                else:
                                    is_correct = False
                                    error_message = failed_reason
                                    
                except Exception as e:
                    is_correct = False
                    error_message = type(e).__name__ + ": " + str(e)
            
            res_dict = {
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
            }
            
            if debug_info:
                res_dict["test_executions"] = debug_info
                
            results.append(res_dict)
            
        return results