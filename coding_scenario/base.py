"""Shared abstract interface and logging utilities for coding MAS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict
from uuid import uuid4

# --- Conversation log taxonomy (overhead attribution) ---

Phase = Literal["coordination", "generation", "execution", "finalization"]

PHASE_COORDINATION: Phase = "coordination"
PHASE_GENERATION: Phase = "generation"
PHASE_EXECUTION: Phase = "execution"
PHASE_FINALIZATION: Phase = "finalization"

# ``answer()`` return dict: total orchestration gap in ms (LangGraph node gaps vs AutoGen step gaps).
ORCHESTRATION_GAP_MS_KEY = "orchestration_gap_ms"

# Stable event_name values shared across LangChain / AutoGen / mock SPADE
EVENT_RECEIVE_TASK = "receive_task"
EVENT_PASS_TO_WRITER = "pass_to_writer"
EVENT_GENERATE_CODE = "generate_code"
EVENT_SEND_CODE_TO_SAFEGUARD = "send_code_to_safeguard"
EVENT_SAFEGUARD_RULE_BLOCK = "safeguard_rule_block"
EVENT_SAFEGUARD_REVIEW = "safeguard_review"
EVENT_REDIRECT_WRITER = "redirect_writer"
EVENT_DECIDE_EXECUTION = "decide_execution"
EVENT_EXECUTE_CODE = "execute_code"
EVENT_SKIP_EXECUTION = "skip_execution"
EVENT_WRITER_INTERPRET = "writer_interpret"
EVENT_CONCLUDE = "conclude"


def default_phase_for_event(
    event_name: str, event_type: str, actor: str
) -> Phase:
    """Infer phase when caller does not set ``phase`` explicitly."""
    if event_name in (EVENT_EXECUTE_CODE, EVENT_SKIP_EXECUTION):
        return PHASE_EXECUTION
    if event_name == EVENT_GENERATE_CODE or event_name == EVENT_WRITER_INTERPRET:
        return PHASE_GENERATION
    if event_name == EVENT_CONCLUDE:
        return PHASE_FINALIZATION
    if actor == "Writer" and event_type == "agent_output":
        return PHASE_GENERATION
    return PHASE_COORDINATION

class CodingMASBase(ABC):
    """Abstract base for Commander–Writer–Safeguard style coding MAS implementations.

    Subclasses (LangGraph, AutoGen, SPADE, etc.) must implement ``answer``.

    The ``answer`` return value is a dict shaped for benchmark runners (e.g. HumanEval):
    at minimum ``writer_code``, ``attempt``, ``safeguard_allowed``, ``final_answer``;
    LangChain / AutoGen implementations may add ``token_usage`` (``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``) and ``orchestration_gap_ms`` (see
    ``ORCHESTRATION_GAP_MS_KEY``).
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

        record: Dict[str, Any] = {
            "record_type": "session_start",
            "session_id": self._log_session_id,
            "timestamp": timestamp.isoformat(),
            "schema_version": "1.1.0",
            "framework": self.__class__.__name__,
            "framework_mode": "production",
            "model_id": self.model_id,
            "max_iterations": self.max_iterations,
            "user_query": user_query,
        }
        self._write_log_line(record)
        return str(self._log_path)

    def begin_conversation_log_with_mode(
        self, user_query: str, *, framework_mode: str = "production"
    ) -> str:
        """Like ``begin_conversation_log`` but records ``framework_mode`` (e.g. mock)."""
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
                "schema_version": "1.1.0",
                "framework": self.__class__.__name__,
                "framework_mode": framework_mode,
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
        event_name: Optional[str] = None,
        phase: Optional[Phase] = None,
        duration_ms: Optional[float] = None,
        llm_api_duration_ms: Optional[float] = None,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """Append one structured event into the active conversation log.

        ``duration_ms`` is total wall time for the step. When an LLM provider call is part of
        the step, set ``llm_api_duration_ms`` to the time spent inside that call so metrics can
        separate step wall time outside the LLM call from API latency.
        """
        if self._log_path is None or self._log_session_id is None:
            return

        resolved_name = event_name or (payload or {}).get("step")
        resolved_phase = phase
        if resolved_phase is None and isinstance(resolved_name, str):
            resolved_phase = default_phase_for_event(
                resolved_name, event_type, actor
            )

        self._log_event_index += 1
        event_record: Dict[str, Any] = {
            "record_type": "event",
            "schema_version": "1.1.0",
            "session_id": self._log_session_id,
            "event_index": self._log_event_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "actor": actor,
            "target": target,
            "payload": payload or {},
        }
        if isinstance(resolved_name, str) and resolved_name:
            event_record["event_name"] = resolved_name
        if resolved_phase is not None:
            event_record["phase"] = resolved_phase
        if duration_ms is not None:
            event_record["duration_ms"] = duration_ms
        if llm_api_duration_ms is not None:
            event_record["llm_api_duration_ms"] = llm_api_duration_ms
        if token_usage is not None:
            event_record["token_usage"] = {
                "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(token_usage.get("completion_tokens", 0)),
                "total_tokens": int(token_usage.get("total_tokens", 0)),
            }
        self._write_log_line(event_record)

    def end_conversation_log(
        self,
        final_answer: str,
        *,
        mas_total_duration_ms: Optional[float] = None,
    ) -> Optional[str]:
        """Close the current conversation log session."""
        if self._log_path is None or self._log_session_id is None:
            return None

        end_rec: Dict[str, Any] = {
            "record_type": "session_end",
            "schema_version": "1.1.0",
            "session_id": self._log_session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_events": self._log_event_index,
            "final_answer_preview": final_answer[:500],
        }
        if mas_total_duration_ms is not None:
            end_rec["mas_total_duration_ms"] = mas_total_duration_ms
        self._write_log_line(end_rec)
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
