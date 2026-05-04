"""Timing splits derived from MAS conversation logs (JSONL).

Each event records total step wall time (``duration_ms``) and optionally the portion spent
inside the provider call (``llm_api_duration_ms``). The remainder
(``duration_ms - llm_api_duration_ms``) is **time outside the LLM call** for that step
(prompt assembly, parsing, graph state updates)—exposed in JSON as ``time_duration_ms`` and
``time_share_of_task_wall`` on each ``aggregate_agents.by_agent`` entry, not "overhead".

**Orchestration gap** totals use ``orchestration_gap_ms`` on ``answer()`` (see
``coding_scenario.base.ORCHESTRATION_GAP_MS_KEY``); legacy logs may still read
``langgraph_between_nodes_duration_ms``. For **LangGraph**, the value is from
``TimeBetweenNodesCallback``: gaps between ``on_chain_end`` and the next ``on_chain_start``
with ``langgraph_node`` metadata. For **AutoGen**, it is **manual** step-boundary gaps inside
GraphFlow step agents. ``between_nodes_measurement_source`` in task-wall attribution reflects
the MAS class when the benchmark passes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from coding_scenario.base import ORCHESTRATION_GAP_MS_KEY

KNOWN_AGENTS = ("commander", "writer", "safeguard")

_LEGACY_ORCHESTRATION_GAP_MS_KEY = "langgraph_between_nodes_duration_ms"


def _orchestration_gap_ms_from_mas_output(mas_output: Dict[str, Any]) -> Optional[float]:
    for key in (ORCHESTRATION_GAP_MS_KEY, _LEGACY_ORCHESTRATION_GAP_MS_KEY):
        raw = mas_output.get(key)
        if isinstance(raw, (int, float)):
            return float(raw)
    return None


def _between_nodes_measurement_source(framework_class_name: Optional[str]) -> str:
    """Human-readable provenance for ``orchestration_gap_ms`` on ``answer()``."""
    if framework_class_name == "LangchainCodingMAS":
        return (
            "LangGraph TimeBetweenNodesCallback: wall time between on_chain_end and the next "
            "on_chain_start for runs with langgraph_node metadata (orchestration gaps only)."
        )
    if framework_class_name == "AutoGenCodingMAS":
        return (
            "MAS custom step timer: perf_counter gaps between _mark_step_end and the next logged "
            "step in GraphFlow agents; not LangGraph nodes or AutoGen core spans."
        )
    return (
        "MAS-reported orchestration_gap_ms (or legacy langgraph_between_nodes_duration_ms); "
        "semantics depend on the MAS implementation (pass framework_class_name for a precise label)."
    )


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

    def agent_message_count(self) -> int:
        """Count explicit inter-agent message events in the log."""
        count = 0
        for event in self.events:
            actor = event.get("actor")
            target = event.get("target")
            if isinstance(actor, str) and isinstance(target, str) and actor and target:
                count += 1
        return count

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
            "messages_between_agents": self.agent_message_count(),
        }


@dataclass
class LogMetricsCalculator:
    records: List[Dict[str, Any]]
    mas_task_seconds: float
    between_nodes_duration_ms: float = 0.0
    between_nodes_measurement_source: str = (
        "MAS-reported orchestration_gap_ms when provided; semantics vary by implementation."
    )

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

    def _apply_agent_shares_of_task_wall(self, aggregate: Dict[str, Any], denom_s: float) -> None:
        """Mutates ``aggregate["by_agent"]`` in place: add share-of-task-wall for each bucket."""
        by_agent = aggregate.get("by_agent")
        if not isinstance(by_agent, dict):
            return
        for _ag, block in by_agent.items():
            if not isinstance(block, dict):
                continue
            time_ms = float(block.get("time_duration_ms", 0.0))
            llm_ms = float(block.get("llm_api_duration_ms", 0.0))
            time_s = time_ms / 1000.0
            llm_s = llm_ms / 1000.0
            block["time_share_of_task_wall"] = (time_s / denom_s) if denom_s > 0 else 0.0
            block["llm_api_share_of_task_wall"] = (llm_s / denom_s) if denom_s > 0 else 0.0

    def build_task_wall_attribution(
        self, aggregate: Dict[str, Any], wall_duration_ms: Optional[float]
    ) -> Dict[str, Any]:
        """
        Task-level fields only: denominator, total LLM time, between-node gap vs task wall.
        Per-agent durations and shares live under ``aggregate_agents.by_agent`` only.
        """
        denom_s = self._task_wall_denominator_seconds(aggregate, wall_duration_ms)
        total_llm_ms = float(aggregate.get("total_llm_api_duration_ms", 0.0))
        between_ms = float(self.between_nodes_duration_ms)
        between_s = between_ms / 1000.0
        between_share = (between_s / denom_s) if denom_s > 0 else 0.0

        return {
            "task_wall_denominator_seconds": denom_s,
            "total_llm_api_time_seconds": total_llm_ms / 1000.0,
            "between_nodes_time_seconds": between_s,
            "between_nodes_share_of_task_wall": between_share,
            "between_nodes_measurement_source": self.between_nodes_measurement_source,
            "session_end_wall_duration_ms": wall_duration_ms,
        }

    def compute(self) -> Dict[str, Any]:
        aggregate = EventMetricsAggregator(self.records).aggregate()
        aggregate["between_nodes_duration_ms"] = float(self.between_nodes_duration_ms)
        wall_duration_ms = self._session_wall_duration_ms()
        denom_s = self._task_wall_denominator_seconds(aggregate, wall_duration_ms)
        self._apply_agent_shares_of_task_wall(aggregate, denom_s)
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
    between_nodes_measurement_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build per-agent duration/token aggregates plus task-wall attribution from a log path."""
    records = load_conversation_log_records(log_path) if log_path else []
    bn_ms = float(between_nodes_duration_ms) if isinstance(between_nodes_duration_ms, (int, float)) else 0.0
    bn_src = between_nodes_measurement_source or (
        "MAS-reported orchestration_gap_ms when provided; semantics vary by implementation."
    )
    calculation = LogMetricsCalculator(
        records=records,
        mas_task_seconds=mas_task_seconds,
        between_nodes_duration_ms=max(0.0, bn_ms),
        between_nodes_measurement_source=bn_src,
    ).compute()
    return {"log_path": log_path, **calculation}


def build_conversation_log_metrics_envelope(
    mas_output: Dict[str, Any],
    mas_task_seconds: float,
    *,
    framework_class_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Benchmark helper: metrics dict from MAS ``answer()`` output plus runner wall time."""
    between_ms = _orchestration_gap_ms_from_mas_output(mas_output)
    return compute_conversation_log_metrics(
        log_path=mas_output.get("conversation_log_path"),
        mas_task_seconds=mas_task_seconds,
        between_nodes_duration_ms=between_ms,
        between_nodes_measurement_source=_between_nodes_measurement_source(framework_class_name),
    )
