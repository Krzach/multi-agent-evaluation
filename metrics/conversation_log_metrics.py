"""Timing splits derived from MAS conversation logs (JSONL).

Each event records total step wall time (``duration_ms``) and optionally the portion spent
inside the provider call (``llm_api_duration_ms``). The remainder
(``duration_ms - llm_api_duration_ms``) is **time outside the LLM call** for that step
(prompt assembly, parsing, graph state updates)—exposed in JSON as ``time_duration_ms`` and
``time_share_of_task_wall`` on each ``aggregate_agents.by_agent`` entry, not "overhead".

**Wall-time decomposition** (``wall_time_decomposition``): ``task_wall_ms`` minus summed
logged **LLM** time and **local tool** time (``execute_code`` step durations) yields
``residual_orchestration_ms`` (gaps between steps, logging, framework work not inside those
spans). **Per-step residual** (``per_step_residual_overhead``) is per event
``max(0, duration_ms - llm_api_duration_ms - tool_ms)`` where ``tool_ms`` is the full step wall
time for ``execute_code`` events only.

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
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from coding_scenario.base import EVENT_EXECUTE_CODE, ORCHESTRATION_GAP_MS_KEY

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


def _percentile_linear(values: Sequence[float], pct: float) -> float:
    """``pct`` in [0, 100]. Linear interpolation between closest ranks."""
    xs = [float(x) for x in values if isinstance(x, (int, float))]
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    y = sorted(xs)
    k = (len(y) - 1) * (pct / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f >= c:
        return y[f]
    return y[f] * (c - k) + y[c] * (k - f)


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

    def sum_tool_execution_duration_ms(self) -> float:
        """Wall time attributed to local ``execute_code`` steps (no LLM in that span)."""
        total = 0.0
        for event in self.events:
            if event.get("event_name") != EVENT_EXECUTE_CODE:
                continue
            d = event.get("duration_ms")
            if isinstance(d, (int, float)):
                total += float(d)
        return total

    def per_step_residual_overhead_stats(self) -> Dict[str, Any]:
        """Per logged event: ``max(0, duration - llm - tool)`` where tool is full wall for execute_code."""
        residuals: List[float] = []
        for event in self.events:
            wall = event.get("duration_ms")
            if not isinstance(wall, (int, float)):
                continue
            wf = float(wall)
            llm = self.llm_api_duration_ms_for_event(event)
            tool_ms = wf if event.get("event_name") == EVENT_EXECUTE_CODE else 0.0
            residuals.append(max(0.0, wf - llm - tool_ms))
        n = len(residuals)
        if n == 0:
            return {
                "events_with_duration_count": 0,
                "total_step_residual_ms": 0.0,
                "mean_step_residual_ms": 0.0,
                "median_step_residual_ms": 0.0,
                "p95_step_residual_ms": 0.0,
                "max_step_residual_ms": 0.0,
            }
        total_r = float(sum(residuals))
        return {
            "events_with_duration_count": n,
            "total_step_residual_ms": total_r,
            "mean_step_residual_ms": total_r / n,
            "median_step_residual_ms": float(statistics.median(residuals)),
            "p95_step_residual_ms": _percentile_linear(residuals, 95.0),
            "max_step_residual_ms": float(max(residuals)),
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
        aggregator = EventMetricsAggregator(self.records)
        aggregate = aggregator.aggregate()
        aggregate["between_nodes_duration_ms"] = float(self.between_nodes_duration_ms)
        wall_duration_ms = self._session_wall_duration_ms()
        denom_s = self._task_wall_denominator_seconds(aggregate, wall_duration_ms)
        self._apply_agent_shares_of_task_wall(aggregate, denom_s)

        total_llm_ms = float(aggregate.get("total_llm_api_duration_ms", 0.0))
        total_tool_ms = aggregator.sum_tool_execution_duration_ms()
        per_step = aggregator.per_step_residual_overhead_stats()

        task_wall_ms = 0.0
        task_wall_source = "none"
        if self.mas_task_seconds > 0:
            task_wall_ms = self.mas_task_seconds * 1000.0
            task_wall_source = "runner_mas_task_seconds"
        elif wall_duration_ms is not None and wall_duration_ms > 0:
            task_wall_ms = float(wall_duration_ms)
            task_wall_source = "session_end_mas_total_duration_ms"
        else:
            ev_sum = float(aggregate.get("total_event_duration_ms", 0.0))
            if ev_sum > 0:
                task_wall_ms = ev_sum
                task_wall_source = "sum_event_duration_ms_fallback"

        residual_ms = max(0.0, task_wall_ms - total_llm_ms - total_tool_ms)
        residual_share = (residual_ms / task_wall_ms) if task_wall_ms > 0 else 0.0
        wall_time_decomposition: Dict[str, Any] = {
            "task_wall_ms": task_wall_ms,
            "task_wall_source": task_wall_source,
            "measured_llm_api_ms": total_llm_ms,
            "measured_tool_execution_ms": total_tool_ms,
            "residual_orchestration_ms": residual_ms,
            "residual_orchestration_share_of_task_wall": residual_share,
        }

        return {
            "task_wall_attribution": self.build_task_wall_attribution(aggregate, wall_duration_ms),
            "aggregate_agents": aggregate,
            "wall_time_decomposition": wall_time_decomposition,
            "per_step_residual_overhead": per_step,
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
