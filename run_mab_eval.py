import os
import json
from dotenv import load_dotenv

from coding_scenario.langchain_mas import LangchainCodingMAS
from benchmarks.multiagentbench.dataset import MultiAgentBenchDataset
from benchmarks.multiagentbench.runner import MultiAgentBenchRunner

def main():
    # Load environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment variables before running.")
        return

    print("Initializing MultiAgentBench Evaluation...")

    # 1. Load the dataset (using the 4 tasks we created)
    dataset_path = os.path.join("benchmarks", "data", "multiagentbench_subset.json")
    dataset = MultiAgentBenchDataset(dataset_path)
    tasks = dataset.get_tasks()
    
    print(f"Loaded {len(tasks)} tasks.")

    # 2. Initialize the real Agent System
    # You can change the model_id to "gpt-4" or others if you prefer
    model_id = "gpt-5.4" 
    max_iterations = 3
    print(f"Initializing LangchainCodingMAS with model: {model_id} (max iterations: {max_iterations})")
    
    agent = LangchainCodingMAS(model_id=model_id, max_iterations=max_iterations)

    # 3. Initialize the Runner
    runner = MultiAgentBenchRunner(mas_instance=agent)

    # 4. Run the Evaluation
    print("Starting evaluation (this may take a few minutes)...")
    results = runner.evaluate(tasks)

    # 5. Output the results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    total_correct = 0
    total_time = 0
    total_messages = 0
    total_tokens = 0
    
    for res in results:
        task_id = res['task_id']
        correct = res['correctness'] == 1.0
        
        if correct:
            total_correct += 1
            
        time_taken = res['time_metrics']['total_task_completion_time_seconds']
        messages = res['collaboration_metrics']['messages_between_agents']
        tokens = res['cost_metrics']['total_tokens']
        
        total_time += time_taken
        total_messages += messages
        total_tokens += tokens
            
        print(f"\nTask ID: {task_id}")
        print(f"Correctness: {'PASSED' if correct else 'FAILED'}")
        if not correct:
            print(f"Error: {res['error']}")
            
        print(f"Time Taken: {time_taken:.2f}s")
        print(f"Total Tokens: {tokens} (Prompt: {res['cost_metrics']['input_tokens']}, Completion: {res['cost_metrics']['output_tokens']})")
        print(f"Conversation Turns: {res['collaboration_metrics']['conversation_iterations']}")
        print(f"Messages Exchanged: {messages}")
        print(f"Safeguard Blocked: {res['collaboration_metrics']['safeguard_blocked']}")
        print(f"Final Answer Preview: {res['final_answer'][:100]}...")
        print("-" * 30)

    accuracy = (total_correct / len(tasks)) * 100 if tasks else 0
    avg_time = total_time / len(tasks) if tasks else 0
    avg_messages = total_messages / len(tasks) if tasks else 0
    avg_tokens = total_tokens / len(tasks) if tasks else 0
    
    print(f"\n--- SUMMARY ---")
    print(f"Final Accuracy: {total_correct}/{len(tasks)} ({accuracy:.1f}%)")
    print(f"Average Task Time: {avg_time:.2f}s")
    print(f"Average Tokens per Task: {avg_tokens:.1f}")
    print(f"Average Messages per Task: {avg_messages:.1f}")
    
    # Save detailed results to a JSON file
    output_file = "mab_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
