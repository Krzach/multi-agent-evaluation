from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

from .collaboration import CollaborationTracker
from .cost import CostTracker
from .time import TimeTracker


class LangGraphMetricsCallback:
    """LangChain callback handler that records token/time metrics."""

    def __init__(self, tracker: "MetricsTracker"):
        self._tracker = tracker
        self._agent_start_times: dict[Any, float] = {}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._agent_start_times[run_id] = perf_counter()

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        run_id = kwargs.get("run_id")
        if run_id is None:
            return None

        started_at = self._agent_start_times.pop(run_id, None)
        if started_at is None:
            return None

        duration = perf_counter() - started_at
        token_usage: Dict[str, Any] = {}
        if getattr(response, "llm_output", None) and "token_usage" in response.llm_output:
            token_usage = response.llm_output.get("token_usage") or {}

        self._tracker.record_llm_call(
            duration=duration,
            input_tokens=int(token_usage.get("prompt_tokens", 0)),
            output_tokens=int(token_usage.get("completion_tokens", 0)),
        )
        return None


@dataclass
class ConversationLogSummary:
    log_path: str
    exists: bool
    total_records: int = 0
    total_events: int = 0
    agent_outputs: int = 0
    passes: int = 0
    actions: int = 0
    errors: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "log_path": self.log_path,
            "exists": self.exists,
            "total_records": self.total_records,
            "total_events": self.total_events,
            "agent_outputs": self.agent_outputs,
            "passes": self.passes,
            "actions": self.actions,
            "errors": self.errors,
        }


@dataclass
class MetricsTracker:
    """
    A centralized tracker for collecting metrics across different multi-agent frameworks.
    Provides framework-specific integration points (e.g., LangChain callbacks, AutoGen hooks).
    """

    cost: CostTracker = field(default_factory=CostTracker)
    time: TimeTracker = field(default_factory=TimeTracker)
    collaboration: CollaborationTracker = field(default_factory=CollaborationTracker)

    def start_task(self) -> None:
        self.time.start_task()

    def end_task(self) -> None:
        self.time.end_task()

    def record_llm_call(self, duration: float, input_tokens: int, output_tokens: int) -> None:
        """
        A generic method to record LLM usage.
        Framework adapters can call this directly if they don't support callbacks.
        """
        self.time.add_agent_response_time(duration)
        self.cost.add_tokens(input_tokens, output_tokens)

    # --- Framework Specific Integrations ---

    def get_langchain_callbacks(self) -> list:
        """
        Returns LangChain callback handlers for LangGraph.
        """
        try:
            from langchain_core.callbacks import BaseCallbackHandler

            class _LangGraphMetricsCallback(BaseCallbackHandler, LangGraphMetricsCallback):
                def __init__(self, tracker: "MetricsTracker"):
                    LangGraphMetricsCallback.__init__(self, tracker)

            return [_LangGraphMetricsCallback(self)]
        except ImportError:
            return []

    def register_autogen_hooks(self, agent: Any) -> None:
        """
        Registers hooks for AutoGen agents to track metrics.
        AutoGen provides `register_hook` for events like `process_message_before_send`
        or you can wrap the client's `create` method.
        """
        # This is a conceptual implementation. AutoGen's tracking is often done
        # by inspecting `agent.client.actual_usage_summary`.
        pass

    def get_spadellm_interceptors(self) -> list:
        """
        Returns interceptors or middleware for SpadeLLM.
        """
        # Conceptual implementation for SpadeLLM tracking.
        return []

    def summarize_communication_overhead(
        self,
        log_path: Optional[str],
        mas_task_seconds: float,
    ) -> Dict[str, Any]:
        """Coordination overhead from conversation log + wall time (see metrics/communication_overhead)."""
        from metrics.communication_overhead import compute_communication_overhead

        return compute_communication_overhead(
            log_path=log_path,
            mas_task_seconds=mas_task_seconds,
        )

    def summarize_conversation_log(self, log_path: str) -> Dict[str, Any]:
        """Summarize a MAS conversation log JSONL file for collaboration metrics."""
        path = Path(log_path)
        if not path.exists():
            return ConversationLogSummary(log_path=log_path, exists=False).as_dict()

        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

        event_records = [r for r in records if r.get("record_type") == "event"]
        return ConversationLogSummary(
            log_path=str(path),
            exists=True,
            total_records=len(records),
            total_events=len(event_records),
            agent_outputs=sum(1 for r in event_records if r.get("event_type") == "agent_output"),
            passes=sum(1 for r in event_records if r.get("event_type") == "pass"),
            actions=sum(1 for r in event_records if r.get("event_type") == "action"),
            errors=sum(1 for r in event_records if r.get("event_type") == "error"),
        ).as_dict()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "cost": self.cost.get_summary(),
            "time": self.time.get_summary(),
            "collaboration": self.collaboration.get_summary(),
        }
