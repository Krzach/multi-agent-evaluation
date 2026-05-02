"""Communication overhead metrics from MAS conversation logs."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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

    def sum_duration_ms(self, phase: Optional[str] = None) -> float:
        total = 0.0
        for event in self.events:
            if phase is not None and event.get("phase") != phase:
                continue
            duration_ms = event.get("duration_ms")
            if isinstance(duration_ms, (int, float)):
                total += float(duration_ms)
        return total

    def sum_tokens(self, phase: Optional[str] = None) -> Dict[str, int]:
        prompt_tokens = completion_tokens = total_tokens = 0
        for event in self.events:
            if phase is not None and event.get("phase") != phase:
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
        return {
            "coordination_duration_ms": self.sum_duration_ms("coordination"),
            "generation_duration_ms": self.sum_duration_ms("generation"),
            "execution_duration_ms": self.sum_duration_ms("execution"),
            "finalization_duration_ms": self.sum_duration_ms("finalization"),
            "coordination_tokens": self.sum_tokens("coordination"),
            "generation_tokens": self.sum_tokens("generation"),
            "execution_tokens": self.sum_tokens("execution"),
            "finalization_tokens": self.sum_tokens("finalization"),
            "total_event_duration_ms": self.sum_duration_ms(),
        }


@dataclass
class OverheadCalculator:
    records: List[Dict[str, Any]]
    mas_task_seconds: float

    def _session_wall_duration_ms(self) -> Optional[float]:
        for record in reversed(self.records):
            if record.get("record_type") == "session_end":
                value = record.get("mas_total_duration_ms")
                if isinstance(value, (int, float)):
                    return float(value)
                return None
        return None

    def _derive_wall_denom_seconds(self, aggregate_phases: Dict[str, Any], wall_duration_ms: Optional[float]) -> float:
        if self.mas_task_seconds > 0:
            return self.mas_task_seconds
        if wall_duration_ms:
            return wall_duration_ms / 1000.0
        return float(aggregate_phases.get("total_event_duration_ms", 0.0)) / 1000.0

    def _compute_metric_a(self, aggregate_phases: Dict[str, Any], wall_duration_ms: Optional[float]) -> Dict[str, Any]:
        coordination_seconds = float(aggregate_phases["coordination_duration_ms"]) / 1000.0
        denominator_seconds = self._derive_wall_denom_seconds(aggregate_phases, wall_duration_ms)
        coordination_share = (coordination_seconds / denominator_seconds) if denominator_seconds > 0 else 0.0
        return {
            "coordination_time_seconds": coordination_seconds,
            "coordination_wall_share_of_task": coordination_share,
            "coordination_tokens": aggregate_phases["coordination_tokens"],
            "phase_durations_ms": {
                "coordination": aggregate_phases["coordination_duration_ms"],
                "generation": aggregate_phases["generation_duration_ms"],
                "execution": aggregate_phases["execution_duration_ms"],
                "finalization": aggregate_phases["finalization_duration_ms"],
            },
            "session_wall_duration_ms": wall_duration_ms,
        }

    def compute(self) -> Dict[str, Any]:
        aggregate_phases = EventMetricsAggregator(self.records).aggregate()
        wall_duration_ms = self._session_wall_duration_ms()
        return {
            "metric_coordination_overhead": self._compute_metric_a(aggregate_phases, wall_duration_ms),
            "aggregate_phases": aggregate_phases,
        }


def load_conversation_log_records(log_path: str) -> List[Dict[str, Any]]:
    return ConversationLogReader.from_path(log_path).records()


def aggregate_phase_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return EventMetricsAggregator(records).aggregate()


def compute_communication_overhead(
    *,
    log_path: Optional[str],
    mas_task_seconds: float,
) -> Dict[str, Any]:
    """
    Coordination overhead: phase ``coordination`` time/tokens and share of task wall time.
    """
    records = load_conversation_log_records(log_path) if log_path else []
    calculation = OverheadCalculator(
        records=records,
        mas_task_seconds=mas_task_seconds,
    ).compute()
    return {"log_path": log_path, **calculation}


def build_overhead_envelope(
    mas_output: Dict[str, Any],
    mas_task_seconds: float,
) -> Dict[str, Any]:
    """Convenience for benchmark runners: coordination overhead from MAS output + wall time."""
    return compute_communication_overhead(
        log_path=mas_output.get("conversation_log_path"),
        mas_task_seconds=mas_task_seconds,
    )
