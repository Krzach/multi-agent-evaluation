"""Communication overhead metrics from MAS conversation logs + optional single-agent baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from coding_scenario.base import CodingMASBase


def load_conversation_log_records(log_path: str) -> List[Dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _events(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("record_type") == "event"]


def _sum_duration_ms(events: List[Dict[str, Any]], phase: Optional[str] = None) -> float:
    total = 0.0
    for e in events:
        if phase is not None and e.get("phase") != phase:
            continue
        d = e.get("duration_ms")
        if isinstance(d, (int, float)):
            total += float(d)
    return total


def _sum_tokens(events: List[Dict[str, Any]], phase: Optional[str] = None) -> Dict[str, int]:
    pt = ct = tt = 0
    for e in events:
        if phase is not None and e.get("phase") != phase:
            continue
        u = e.get("token_usage")
        if not isinstance(u, dict):
            continue
        pt += int(u.get("prompt_tokens", 0))
        ct += int(u.get("completion_tokens", 0))
        tt += int(u.get("total_tokens", 0))
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt if tt else pt + ct,
    }


def aggregate_phase_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ev = _events(records)
    return {
        "coordination_duration_ms": _sum_duration_ms(ev, "coordination"),
        "generation_duration_ms": _sum_duration_ms(ev, "generation"),
        "execution_duration_ms": _sum_duration_ms(ev, "execution"),
        "finalization_duration_ms": _sum_duration_ms(ev, "finalization"),
        "coordination_tokens": _sum_tokens(ev, "coordination"),
        "generation_tokens": _sum_tokens(ev, "generation"),
        "execution_tokens": _sum_tokens(ev, "execution"),
        "finalization_tokens": _sum_tokens(ev, "finalization"),
        "total_event_duration_ms": _sum_duration_ms(ev, None),
    }


def compute_communication_overhead(
    *,
    log_path: Optional[str],
    mas_task_seconds: float,
    mas_token_usage: Dict[str, int],
    baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Metric A: coordination-only (phase ``coordination`` time/tokens).
    Metric B: delta vs single-agent baseline (time + tokens).
    """
    records: List[Dict[str, Any]] = []
    if log_path:
        records = load_conversation_log_records(log_path)

    phases = aggregate_phase_metrics(records)
    coord_ms = phases["coordination_duration_ms"]
    coord_sec = coord_ms / 1000.0
    total_wall_ms: Optional[float] = None
    for r in reversed(records):
        if r.get("record_type") == "session_end":
            v = r.get("mas_total_duration_ms")
            if isinstance(v, (int, float)):
                total_wall_ms = float(v)
            break

    denom_sec = mas_task_seconds if mas_task_seconds > 0 else (
        (total_wall_ms / 1000.0) if total_wall_ms else (phases["total_event_duration_ms"] / 1000.0)
    )
    coord_share = (coord_sec / denom_sec) if denom_sec > 0 else 0.0

    metric_a = {
        "coordination_time_seconds": coord_sec,
        "coordination_wall_share_of_task": coord_share,
        "coordination_tokens": phases["coordination_tokens"],
        "phase_durations_ms": {
            "coordination": phases["coordination_duration_ms"],
            "generation": phases["generation_duration_ms"],
            "execution": phases["execution_duration_ms"],
            "finalization": phases["finalization_duration_ms"],
        },
        "session_wall_duration_ms": total_wall_ms,
    }

    metric_b: Dict[str, Any] = {
        "delta_time_seconds": None,
        "delta_tokens": None,
        "baseline": baseline,
    }
    if baseline and isinstance(baseline.get("duration_seconds"), (int, float)):
        base_t = float(baseline["duration_seconds"])
        metric_b["delta_time_seconds"] = mas_task_seconds - base_t
        btok = baseline.get("token_usage") or {}
        mas_tt = int(mas_token_usage.get("total_tokens", 0))
        base_tt = int(btok.get("total_tokens", 0))
        metric_b["delta_tokens"] = mas_tt - base_tt
        metric_b["delta_prompt_tokens"] = int(mas_token_usage.get("prompt_tokens", 0)) - int(
            btok.get("prompt_tokens", 0)
        )
        metric_b["delta_completion_tokens"] = int(
            mas_token_usage.get("completion_tokens", 0)
        ) - int(btok.get("completion_tokens", 0))

    return {
        "log_path": log_path,
        "metric_coordination_overhead": metric_a,
        "metric_delta_vs_single_agent": metric_b,
        "aggregate_phases": phases,
    }


def build_overhead_envelope(
    mas: "CodingMASBase",
    task_input: str,
    mas_output: Dict[str, Any],
    mas_task_seconds: float,
    *,
    include_delta_overhead: bool = False,
) -> Dict[str, Any]:
    """
    Convenience for benchmark runners: metric A always; metric B if baseline run requested.
    """
    from benchmarks.baseline.single_agent import (
        baseline_summary,
        run_single_agent_baseline,
    )

    tu = mas_output.get("token_usage") or {}
    pt = int(tu.get("prompt_tokens", 0))
    ct = int(tu.get("completion_tokens", 0))
    tt = int(tu.get("total_tokens", pt + ct))
    mas_token_usage = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }

    baseline: Optional[Dict[str, Any]] = None
    if include_delta_overhead:
        model_id = getattr(mas, "model_id", None)
        if model_id is not None:
            br = run_single_agent_baseline(task_input, model_id=model_id)
            if br.get("ok"):
                baseline = baseline_summary(br)

    return compute_communication_overhead(
        log_path=mas_output.get("conversation_log_path"),
        mas_task_seconds=mas_task_seconds,
        mas_token_usage=mas_token_usage,
        baseline=baseline,
    )
