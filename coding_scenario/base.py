"""Shared abstract interface and logging utilities for coding MAS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict
from uuid import uuid4

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
        self._log_session_id: Optional[str] = None
        self._log_path: Optional[Path] = None
        self._log_event_index: int = 0

    @abstractmethod
    def answer(self, query: str) -> Dict[str, Any]:
        """Run the full workflow for ``query`` and return the final workflow state dict."""
        ...

    @property
    def conversation_log_path(self) -> Optional[str]:
        """Absolute path to the latest conversation log file."""
        if self._log_path is None:
            return None
        return str(self._log_path)

    def begin_conversation_log(self, user_query: str) -> str:
        """Create a timestamped conversation log and write header metadata."""
        timestamp = datetime.now(timezone.utc)
        title_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs") / "mas_conversations"
        log_dir.mkdir(parents=True, exist_ok=True)

        self._log_session_id = str(uuid4())
        self._log_event_index = 0
        self._log_path = log_dir / f"mas_conversation_{title_timestamp}_{self._log_session_id[:8]}.jsonl"

        self._write_log_line(
            {
                "record_type": "session_start",
                "session_id": self._log_session_id,
                "timestamp": timestamp.isoformat(),
                "schema_version": "1.0.0",
                "framework": self.__class__.__name__,
                "model_id": self.model_id,
                "max_iterations": self.max_iterations,
                "user_query": user_query,
            }
        )
        return str(self._log_path)

    def log_conversation_event(
        self,
        *,
        event_type: Literal["agent_output", "action", "pass", "state", "error", "final"],
        actor: str,
        target: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one structured event into the active conversation log."""
        if self._log_path is None or self._log_session_id is None:
            return

        self._log_event_index += 1
        self._write_log_line(
            {
                "record_type": "event",
                "schema_version": "1.0.0",
                "session_id": self._log_session_id,
                "event_index": self._log_event_index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "actor": actor,
                "target": target,
                "payload": payload or {},
            }
        )

    def end_conversation_log(self, final_answer: str) -> Optional[str]:
        """Close the current conversation log session."""
        if self._log_path is None or self._log_session_id is None:
            return None

        self._write_log_line(
            {
                "record_type": "session_end",
                "schema_version": "1.0.0",
                "session_id": self._log_session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_events": self._log_event_index,
                "final_answer_preview": final_answer[:500],
            }
        )
        return str(self._log_path)

    def load_conversation_log(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load a conversation log file (JSONL) as a list of records."""
        log_path = Path(path) if path else self._log_path
        if log_path is None or not log_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def _write_log_line(self, record: Dict[str, Any]) -> None:
        if self._log_path is None:
            return
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

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
