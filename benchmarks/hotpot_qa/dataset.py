from datasets import load_dataset
from typing import List, Dict, Any

class HotpotQADataset:
    """
    Utility class to load and access the HotpotQA dataset for multi-hop RAG evaluation.
    We use the 'distractor' setting which contains the question, answer, and 
    a set of context paragraphs (some relevant, some distractors).
    """
    def __init__(self, split: str = "validation"):
        # We use 'validation' because the 'test' set answers are often kept private 
        # in academic datasets, and 'validation' is the standard evaluation target.
        # We use the 'distractor' setting as it's the standard for RAG evaluation.
        self._dataset = load_dataset("hotpot_qa", "distractor", split=split)

    def get_tasks(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves tasks from the HotpotQA dataset.
        
        Args:
            limit: Maximum number of samples to return.
            
        Returns:
            A list of dictionary objects, where each object represents a RAG QA task.
        """
        tasks = list(self._dataset)
        if limit is not None:
            tasks = tasks[:limit]
            
        formatted_tasks = []
        for task in tasks:
            # HotpotQA context format: a dict with 'title' (list of strings) 
            # and 'sentences' (list of lists of strings).
            # We flatten this into a list of document strings for easier consumption by the agent.
            documents = []
            for title, sentences in zip(task['context']['title'], task['context']['sentences']):
                # Join the sentences into a single paragraph
                paragraph = " ".join(sentences)
                # Prefix with the title to simulate a real document
                documents.append(f"Title: {title}\nContent: {paragraph}")
                
            formatted_tasks.append({
                "task_id": task["id"],
                "question": task["question"],
                "expected_answer": task["answer"],
                "supporting_facts": task["supporting_facts"], # List of titles/sentence indices that actually support the answer
                "documents": documents # All provided context documents
            })
            
        return formatted_tasks
