import logging
import time
import re
import string
from typing import List, Dict, Any

class HotpotQARunner:
    """
    Integrates the HotpotQA benchmark to evaluate the multi-agent framework on multi-hop RAG tasks.
    """
    def __init__(self, agent_framework):
        self.agent = agent_framework
        self.logger = logging.getLogger(__name__)

    def _normalize_answer(self, s):
        """
        Standard SQuAD/HotpotQA evaluation metric normalization.
        Lower text and remove punctuation, articles, and extra whitespace.
        """
        if not isinstance(s, str):
            s = str(s)
            
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _exact_match_score(self, prediction, ground_truth):
        return (self._normalize_answer(prediction) == self._normalize_answer(ground_truth))

    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        
        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = sum(1 for token in prediction_tokens if token in ground_truth_tokens)
        
        if num_same == 0:
            return 0
            
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def evaluate(self, dataset: List[Dict[str, Any]]):
        """
        Loops through the HotpotQA dataset, passes the question and documents to the agent, 
        and evaluates the correctness of the generated answer.
        """
        results = []
        for item in dataset:
            task_id = item.get("task_id", "Unknown")
            question = item.get("question", "")
            expected_answer = item.get("expected_answer", "")
            documents = item.get("documents", [])
            
            # Format the input for a multi-hop RAG agent.
            # We provide the agent with all the distractor documents upfront 
            # and ask it to find the answer.
            doc_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
            
            task_input = (
                "You are an expert Question Answering system.\n"
                "Answer the following question using ONLY the provided documents. "
                "Your final answer should be extremely concise (usually a single word, name, or short phrase). "
                f"Question: {question}\n\n"
                f"Documents:\n{doc_str}"
            )
            
            # Start timer for Total Task Completion Time
            start_time = time.time()
            
            # Execute agent framework. We use the same fallback structure as HumanEval.
            if hasattr(self.agent, 'answer'):
                mas_output = self.agent.answer(task_input)
                generated_answer = mas_output.get("final_answer", "")
                # If final_answer is empty (e.g. safeguard blocked), fallback to writer
                if not generated_answer.strip():
                    generated_answer = mas_output.get("writer_interpretation", "")
                attempts = mas_output.get("attempt", 0)
                safeguard_allowed = mas_output.get("safeguard_allowed", True)
                token_usage = mas_output.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            else:
                mas_output = self.agent.run_task(task_input=task_input)
                generated_answer = mas_output.get('response', '')
                attempts = 0
                safeguard_allowed = True
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            # End timer
            end_time = time.time()
            total_task_time = end_time - start_time
            
            # Calculate Collaboration Metrics (Assuming typical 4 messages per attempt)
            messages_per_attempt = 4
            base_messages = 2 
            total_messages = base_messages + ((attempts + 1) * messages_per_attempt)
            
            # Calculate Standard RAG QA Metrics
            em_score = 1.0 if self._exact_match_score(generated_answer, expected_answer) else 0.0
            
            # If Exact Match is 0, we check if the exact answer is simply contained within a verbose response.
            # (e.g. "The answer is 1999" contains "1999"). This is a softer exact match often used for chat agents.
            soft_em = 0.0
            if not em_score:
                norm_gen = self._normalize_answer(generated_answer)
                norm_exp = self._normalize_answer(expected_answer)
                if norm_exp in norm_gen and len(norm_exp) > 0:
                    soft_em = 1.0
                    
            final_correctness = max(em_score, soft_em)
            f1 = self._f1_score(generated_answer, expected_answer)
            
            results.append({
                "task_id": task_id,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                
                # Qualitative Metrics
                "correctness": final_correctness, # 1.0 if correct, 0.0 otherwise
                "f1_score": f1,
                
                # Cost Metrics
                "cost_metrics": {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0)
                },
                
                # Time Metrics
                "time_metrics": {
                    "total_task_completion_time_seconds": total_task_time
                },
                
                # Agent Collaboration Metrics
                "collaboration_metrics": {
                    "conversation_iterations": attempts,
                    "messages_between_agents": total_messages,
                    "activated_agents": 3 if hasattr(self.agent, 'answer') else 1, 
                    "safeguard_blocked": not safeguard_allowed
                }
            })
            
        return results