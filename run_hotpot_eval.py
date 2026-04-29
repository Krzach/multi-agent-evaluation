import os
import json
import time
from dotenv import load_dotenv

from coding_scenario.langchain_mas import CommanderWriterSafeguardSystem
from benchmarks.hotpot_qa.dataset import HotpotQADataset
from benchmarks.hotpot_qa.runner import HotpotQARunner

def main():
    # Load environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment variables before running.")
        return

    print("Initializing HotpotQA (Multi-Hop RAG) Evaluation...")

    # 1. Load the dataset (from huggingface datasets)
    try:
        dataset = HotpotQADataset(split="validation")
        # Limit to 5 tasks for a quick test run.
        tasks = dataset.get_tasks(limit=5)
        print(f"Loaded {len(tasks)} multi-hop RAG tasks from HotpotQA.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Make sure you have installed 'datasets': pip install datasets")
        return

    # 2. Initialize the real Agent System
    model_id = "gpt-4o-mini" 
    max_iterations = 3
    print(f"Initializing CommanderWriterSafeguardSystem with model: {model_id} (max iterations: {max_iterations})")
    
    # We use the same generic CommanderWriterSafeguardSystem for RAG.
    # A true RAG agent would have access to a tool to retrieve context, 
    # but for HotpotQA "distractor" setting, we provide the context directly in the prompt.
    agent = CommanderWriterSafeguardSystem(model_id=model_id, max_iterations=max_iterations)

    # 3. Initialize the Runner
    runner = HotpotQARunner(agent_framework=agent)

    # 4. Run the Evaluation
    print("Starting evaluation (this may take a few minutes)...")
    results = runner.evaluate(tasks)

    # 5. Output the results
    print("\n" + "="*50)
    print("EVALUATION RESULTS (HOTPOTQA)")
    print("="*50)
    
    total_correct = 0
    total_time = 0
    total_messages = 0
    total_tokens = 0
    total_f1 = 0.0
    
    for res in results:
        task_id = res['task_id']
        correct = res['correctness'] == 1.0
        
        if correct:
            total_correct += 1
            
        time_taken = res['time_metrics']['total_task_completion_time_seconds']
        messages = res['collaboration_metrics']['messages_between_agents']
        tokens = res['cost_metrics']['total_tokens']
        f1 = res['f1_score']
        
        total_time += time_taken
        total_messages += messages
        total_tokens += tokens
        total_f1 += f1
            
        print(f"\nTask ID: {task_id}")
        print(f"Question: {res['question']}")
        print(f"Expected Answer: {res['expected_answer']}")
        print(f"Generated Answer: {res['generated_answer'][:150]}...")
        print(f"Correctness: {'PASSED' if correct else 'FAILED'} (F1: {f1:.2f})")
            
        print(f"Time Taken: {time_taken:.2f}s")
        print(f"Total Tokens: {tokens}")
        print(f"Conversation Turns: {res['collaboration_metrics']['conversation_iterations']}")
        print("-" * 30)

    accuracy = (total_correct / len(tasks)) * 100 if tasks else 0
    avg_f1 = (total_f1 / len(tasks)) * 100 if tasks else 0
    avg_time = total_time / len(tasks) if tasks else 0
    avg_messages = total_messages / len(tasks) if tasks else 0
    avg_tokens = total_tokens / len(tasks) if tasks else 0
    
    print(f"\n--- SUMMARY ---")
    print(f"Final Accuracy (Soft Exact Match): {total_correct}/{len(tasks)} ({accuracy:.1f}%)")
    print(f"Average F1 Score: {avg_f1:.1f}%")
    print(f"Average Task Time: {avg_time:.2f}s")
    print(f"Average Tokens per Task: {avg_tokens:.1f}")
    print(f"Average Messages per Task: {avg_messages:.1f}")
    
    # Save detailed results to a JSON file
    output_file = "hotpotqa_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
