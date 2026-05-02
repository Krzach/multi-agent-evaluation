"""Timing splits derived from MAS conversation logs (JSONL).

Each event records total step wall time (``duration_ms``) and optionally the portion spent
inside the provider call (``llm_api_duration_ms``). The remainder
(``duration_ms - llm_api_duration_ms``) is **time outside the LLM call** for that step
(prompt assembly, parsing, graph state updates)—exposed in JSON as ``time_duration_ms`` /
``time_seconds``, not "overhead".

**Between-node** gaps for the LangGraph coding MAS come from ``TimeBetweenNodesCallback``
(``coding_scenario/langchain/callbacks/time_between_nodes.py``): elapsed time between the
end of one graph node runnable and the start of the next, using ``langgraph_node`` callback
metadata. Other MAS builds omit ``langgraph_between_nodes_duration_ms`` on ``answer()``;
those gaps are reported as **0** here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

KNOWN_AGENTS = ("commander", "writer", "safeguard")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def agent_bucket(actor: Any) -> str:
    """Map log ``actor`` string to aggregation key: commander | writer | safeguard | other."""
    if not isinstance(actor, str):
        return "other"
    key = actor.strip().lower()
    if key in KNOWN_AGENTS:
        return key
    return "other"


@dataclass
class ConversationLogReader:
    log_path: Path

    @classmethod
    def from_path(cls, log_path: str) -> "ConversationLogReader":
        return cls(log_path=Path(log_path))

    def records(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


@dataclass
class EventMetricsAggregator:
    records: List[Dict[str, Any]]

    @property
    def events(self) -> List[Dict[str, Any]]:
        return [record for record in self.records if record.get("record_type") == "event"]

    @staticmethod
    def _token_total(usage: Any) -> int:
        if not isinstance(usage, dict):
            return 0
        tt = _safe_int(usage.get("total_tokens", 0))
        if tt:
            return tt
        return _safe_int(usage.get("prompt_tokens", 0)) + _safe_int(usage.get("completion_tokens", 0))

    def llm_api_duration_ms_for_event(self, event: Dict[str, Any]) -> float:
        explicit = event.get("llm_api_duration_ms")
        if isinstance(explicit, (int, float)) and float(explicit) >= 0:
            return float(explicit)
        usage = event.get("token_usage")
        if self._token_total(usage) > 0:
            wall = event.get("duration_ms")
            if isinstance(wall, (int, float)):
                return float(wall)
        return 0.0

    def time_duration_ms_for_event(self, event: Dict[str, Any]) -> float:
        """Milliseconds of step wall time not attributed to the LLM API call for this event."""
        wall = event.get("duration_ms")
        if not isinstance(wall, (int, float)):
            return 0.0
        return max(0.0, float(wall) - self.llm_api_duration_ms_for_event(event))

    def sum_wall_duration_ms(self, agent: Optional[str] = None) -> float:
        total = 0.0
        for event in self.events:
            if agent is not None and agent_bucket(event.get("actor")) != agent:
                continue
            duration_ms = event.get("duration_ms")
            if isinstance(duration_ms, (int, float)):
                total += float(duration_ms)
        return total

    def sum_time_duration_ms(self, agent: Optional[str] = None) -> float:
        total = 0.0
        for event in self.events:
            if agent is not None and agent_bucket(event.get("actor")) != agent:
                continue
            total += self.time_duration_ms_for_event(event)
        return total

    def sum_llm_api_duration_ms(self, agent: Optional[str] = None) -> float:
        total = 0.0
        for event in self.events:
            if agent is not None and agent_bucket(event.get("actor")) != agent:
                continue
            total += self.llm_api_duration_ms_for_event(event)
        return total

    def sum_tokens(self, agent: Optional[str] = None) -> Dict[str, int]:
        prompt_tokens = completion_tokens = total_tokens = 0
        for event in self.events:
            if agent is not None and agent_bucket(event.get("actor")) != agent:
                continue
            usage = event.get("token_usage")
            if not isinstance(usage, dict):
                continue
            prompt_tokens += _safe_int(usage.get("prompt_tokens", 0))
            completion_tokens += _safe_int(usage.get("completion_tokens", 0))
            total_tokens += _safe_int(usage.get("total_tokens", 0))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens if total_tokens else prompt_tokens + completion_tokens,
        }

    def aggregate(self) -> Dict[str, Any]:
        by_agent: Dict[str, Any] = {}
        for ag in (*KNOWN_AGENTS, "other"):
            by_agent[ag] = {
                "time_duration_ms": self.sum_time_duration_ms(ag),
                "llm_api_duration_ms": self.sum_llm_api_duration_ms(ag),
                "tokens": self.sum_tokens(ag),
            }
        return {
            "by_agent": by_agent,
            "total_event_duration_ms": self.sum_wall_duration_ms(),
            "total_time_duration_ms": self.sum_time_duration_ms(),
            "total_llm_api_duration_ms": self.sum_llm_api_duration_ms(),
        }


@dataclass
class LogMetricsCalculator:
    records: List[Dict[str, Any]]
    mas_task_seconds: float
    between_nodes_duration_ms: float = 0.0

    def _session_wall_duration_ms(self) -> Optional[float]:
        for record in reversed(self.records):
            if record.get("record_type") == "session_end":
                value = record.get("mas_total_duration_ms")
                if isinstance(value, (int, float)):
                    return float(value)
                return None
        return None

    def _task_wall_denominator_seconds(
        self, aggregate: Dict[str, Any], wall_duration_ms: Optional[float]
    ) -> float:
        if self.mas_task_seconds > 0:
            return self.mas_task_seconds
        if wall_duration_ms:
            return wall_duration_ms / 1000.0
        return float(aggregate.get("total_event_duration_ms", 0.0)) / 1000.0

    def build_task_wall_attribution(
        self, aggregate: Dict[str, Any], wall_duration_ms: Optional[float]
    ) -> Dict[str, Any]:
        """
        Summarize how logged durations relate to overall task wall time: between-node gaps,
        total LLM API time, and per-agent time-outside-LLM vs LLM splits with shares of the wall.
        """
        denom_s = self._task_wall_denominator_seconds(aggregate, wall_duration_ms)
        total_llm_ms = float(aggregate.get("total_llm_api_duration_ms", 0.0))
        between_ms = float(self.between_nodes_duration_ms)
        between_s = between_ms / 1000.0
        between_share = (between_s / denom_s) if denom_s > 0 else 0.0

        by_agent = aggregate.get("by_agent") or {}
        per_agent: Dict[str, Any] = {}
        for ag in KNOWN_AGENTS:
            block = by_agent.get(ag) or {}
            time_ms = float(block.get("time_duration_ms", 0.0))
            llm_ms = float(block.get("llm_api_duration_ms", 0.0))
            time_s = time_ms / 1000.0
            llm_s = llm_ms / 1000.0
            per_agent[ag] = {
                "time_seconds": time_s,
                "llm_api_time_seconds": llm_s,
                "time_share_of_task_wall": (time_s / denom_s) if denom_s > 0 else 0.0,
                "llm_api_share_of_task_wall": (llm_s / denom_s) if denom_s > 0 else 0.0,
                "tokens": block.get("tokens", {}),
            }

        other_block = by_agent.get("other") or {}
        other_time_ms = float(other_block.get("time_duration_ms", 0.0))
        other_llm_ms = float(other_block.get("llm_api_duration_ms", 0.0))
        per_agent["other"] = {
            "time_seconds": other_time_ms / 1000.0,
            "llm_api_time_seconds": other_llm_ms / 1000.0,
            "time_share_of_task_wall": (
                (other_time_ms / 1000.0) / denom_s if denom_s > 0 else 0.0
            ),
            "llm_api_share_of_task_wall": (
                (other_llm_ms / 1000.0) / denom_s if denom_s > 0 else 0.0
            ),
            "tokens": other_block.get("tokens", {}),
        }

        return {
            "task_wall_denominator_seconds": denom_s,
            "total_llm_api_time_seconds": total_llm_ms / 1000.0,
            "between_nodes_time_seconds": between_s,
            "between_nodes_share_of_task_wall": between_share,
            "between_nodes_measurement_source": (
                "langgraph_callback_perf_counter when provided by MAS output; else 0"
            ),
            "per_agent": per_agent,
            "session_end_wall_duration_ms": wall_duration_ms,
        }

    def compute(self) -> Dict[str, Any]:
        aggregate = EventMetricsAggregator(self.records).aggregate()
        aggregate["between_nodes_duration_ms"] = float(self.between_nodes_duration_ms)
        wall_duration_ms = self._session_wall_duration_ms()
        return {
            "task_wall_attribution": self.build_task_wall_attribution(aggregate, wall_duration_ms),
            "aggregate_agents": aggregate,
        }


def load_conversation_log_records(log_path: str) -> List[Dict[str, Any]]:
    return ConversationLogReader.from_path(log_path).records()


def aggregate_agent_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return EventMetricsAggregator(records).aggregate()


def compute_conversation_log_metrics(
    *,
    log_path: Optional[str],
    mas_task_seconds: float,
    between_nodes_duration_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Build per-agent duration/token aggregates plus task-wall attribution from a log path."""
    records = load_conversation_log_records(log_path) if log_path else []
    bn_ms = float(between_nodes_duration_ms) if isinstance(between_nodes_duration_ms, (int, float)) else 0.0
    calculation = LogMetricsCalculator(
        records=records,
        mas_task_seconds=mas_task_seconds,
        between_nodes_duration_ms=max(0.0, bn_ms),
    ).compute()
    return {"log_path": log_path, **calculation}


def build_conversation_log_metrics_envelope(
    mas_output: Dict[str, Any],
    mas_task_seconds: float,
) -> Dict[str, Any]:
    """Benchmark helper: metrics dict from MAS ``answer()`` output plus runner wall time."""
    raw_bn = mas_output.get("langgraph_between_nodes_duration_ms")
    between_ms: Optional[float] = None
    if isinstance(raw_bn, (int, float)):
        between_ms = float(raw_bn)
    return compute_conversation_log_metrics(
        log_path=mas_output.get("conversation_log_path"),
        mas_task_seconds=mas_task_seconds,
        between_nodes_duration_ms=between_ms,
    )
