from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter


@dataclass
class TimeTracker:
    """Tracks task duration and per-call response timings."""

    task_start_time: float | None = None
    task_end_time: float | None = None
    agent_response_times: list[float] = field(default_factory=list)

    def start_task(self) -> None:
        self.task_start_time = perf_counter()
        self.task_end_time = None

    def end_task(self) -> None:
        self.task_end_time = perf_counter()

    def add_agent_response_time(self, duration_seconds: float) -> None:
        self.agent_response_times.append(float(duration_seconds))

    def reset(self) -> None:
        self.task_start_time = None
        self.task_end_time = None
        self.agent_response_times.clear()

    def get_total_task_time(self) -> float:
        if self.task_start_time is None or self.task_end_time is None:
            return 0.0
        return self.task_end_time - self.task_start_time

    def get_average_agent_response_time(self) -> float:
        if not self.agent_response_times:
            return 0.0
        return mean(self.agent_response_times)

    def get_summary(self) -> dict:
        return {
            "total_task_time_seconds": self.get_total_task_time(),
            "average_agent_response_time_seconds": self.get_average_agent_response_time(),
            "num_recorded_agent_calls": len(self.agent_response_times),
        }
