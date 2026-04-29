"""Shared abstract interface for coding multi-agent systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypedDict
from langchain_core.callbacks import BaseCallbackHandler

class CodingMASBase(ABC):
    """Abstract base for Commander–Writer–Safeguard style coding MAS implementations.

    Subclasses (LangGraph, AutoGen, SPADE, etc.) must implement ``answer``.

    The ``answer`` return value is a dict shaped for benchmark runners (e.g. HumanEval):
    at minimum ``writer_code``, ``attempt``, ``safeguard_allowed``, ``final_answer``;
    LangChain implementations may add ``token_usage`` with keys
    ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
    """

    def __init__(self, model_id: str, max_iterations: int) -> None:
        self.model_id = model_id
        self.max_iterations = max_iterations
        self.memory: List[str] = []

    @abstractmethod
    def answer(self, query: str) -> Dict[str, Any]:
        """Run the full workflow for ``query`` and return the final workflow state dict."""
        ...

class TokenTrackingCallback(BaseCallbackHandler):
    """Callback to track token usage across all LLM calls."""
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            self.total_tokens += usage.get('total_tokens', 0)
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)

class WorkflowState(TypedDict):
    user_query: str
    memory: List[str]
    max_iterations: int
    attempt: int
    commander_context: str
    writer_code: str
    writer_notes: str
    safeguard_allowed: bool
    safeguard_reason: str
    requires_execution: bool
    execution_output: str
    execution_error: str
    writer_interpretation: str
    final_answer: str
    finished: bool
