#!/usr/bin/env python3
"""Run HumanEval on the first N tasks for LangChain and AutoGen, with repeated trials.

Aggregates correctness, latency, token usage, and collaboration counters; writes GitHub-flavored
Markdown tables to ``results.md`` (configurable via ``--output``), and optionally raw JSON.

Usage::

    python cross_framework_benchmark.py --repeats 3
    python cross_framework_benchmark.py --tasks 5 --repeats 5 --output results.md --save-json runs.json

By default the Markdown report is written to ``results.md`` in the current working directory.

Requires ``OPENAI_API_KEY`` and network for the Hugging Face / OpenAI calls.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

from benchmarks.human_eval.dataset import HumanEvalDataset
from benchmarks.human_eval.runner import HumanEvalRunner
from coding_scenario.autogen.autogen_mas import AutoGenCodingMAS
from coding_scenario.langchain.langchain_mas import LangchainCodingMAS


def _build_mas(framework: str, model_id: str, max_iterations: int) -> Any:
    if framework == "langchain":
        return LangchainCodingMAS(model_id=model_id, max_iterations=max_iterations)
    if framework == "autogen":
        mid = "gpt-5.4-2026-03-05" if model_id == "gpt-5.4" else model_id
        return AutoGenCodingMAS(model_id=mid, max_iterations=max_iterations)
    raise ValueError(f"Unknown framework: {framework}")


def _pick(res: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = res
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    if isinstance(cur, bool):
        return float(cur)
    if isinstance(cur, (int, float)):
        return float(cur)
    return default


def _llm_wall_seconds(res: Dict[str, Any]) -> float:
    agg = res.get("conversation_log_metrics") or {}
    agents = (agg.get("aggregate_agents") or {}).get("by_agent") or {}
    total_ms = 0.0
    for bucket in agents.values():
        if isinstance(bucket, dict):
            total_ms += float(bucket.get("llm_api_duration_ms") or 0.0)
    return total_ms / 1000.0


def _between_nodes_seconds(res: Dict[str, Any]) -> float:
    tw = (res.get("conversation_log_metrics") or {}).get("task_wall_attribution") or {}
    return float(tw.get("between_nodes_time_seconds") or 0.0)


@dataclass
class RunRecord:
    framework: str
    task_id: str
    repeat: int
    result: Dict[str, Any]


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    xs = [float(x) for x in values]
    if not xs:
        return {"n": 0, "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    out = {
        "n": float(len(xs)),
        "mean": statistics.mean(xs),
        "min": min(xs),
        "max": max(xs),
        "median": statistics.median(xs),
    }
    out["stdev"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return out


def _format_stat(s: Dict[str, float], *, unit: str = "", precision: int = 3) -> str:
    if int(s["n"]) == 0:
        return "—"
    m = s["mean"]
    sd = s["stdev"]
    if int(s["n"]) < 2 or sd < 1e-9:
        return f"{m:.{precision}f}{unit}"
    return f"{m:.{precision}f}±{sd:.{precision}f}{unit}"


def _md_escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """GitHub-flavored Markdown pipe table."""
    line_h = "| " + " | ".join(_md_escape_cell(h) for h in headers) + " |"
    line_s = "| " + " | ".join("---" for _ in headers) + " |"
    line_b = ["| " + " | ".join(_md_escape_cell(c) for c in r) + " |" for r in rows]
    return "\n".join([line_h, line_s, *line_b]) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-framework HumanEval benchmark with repeats.")
    p.add_argument("--tasks", type=int, default=5, help="First N HumanEval tasks (default: 5).")
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many times to run each task per framework (default: 3).",
    )
    p.add_argument("--model", default="gpt-5.4", help="Model id (default: gpt-5.4).")
    p.add_argument("--max-iterations", type=int, default=3, help="MAS max_iterations.")
    p.add_argument(
        "--output",
        default="results.md",
        help="Markdown report path (default: results.md).",
    )
    p.add_argument("--save-json", default="", help="Optional path to write all raw run JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    dataset = HumanEvalDataset(split="test")
    tasks = dataset.get_tasks(limit=args.tasks)
    if not tasks:
        raise SystemExit("No tasks loaded.")

    records: List[RunRecord] = []
    frameworks = ("langchain", "autogen")

    print(
        f"HumanEval cross-framework benchmark: {len(tasks)} tasks × {args.repeats} repeats "
        f"× {len(frameworks)} frameworks = {len(tasks) * args.repeats * len(frameworks)} runs."
    )
    print(f"Model={args.model!r}, max_iterations={args.max_iterations}")

    for fw in frameworks:
        mas = _build_mas(fw, args.model, args.max_iterations)
        runner = HumanEvalRunner(mas_instance=mas)
        for rep in range(args.repeats):
            for task in tasks:
                tid = str(task.get("task_id", ""))
                print(f"  [{fw}] repeat {rep + 1}/{args.repeats} task {tid} …", flush=True)
                row = runner.evaluate([task])[0]
                records.append(RunRecord(framework=fw, task_id=tid, repeat=rep, result=row))

    if args.save_json:
        payload = [
            {
                "framework": r.framework,
                "task_id": r.task_id,
                "repeat": r.repeat,
                **r.result,
            }
            for r in records
        ]
        Path(args.save_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote raw runs to {args.save_json}")

    summary_headers = [
        "framework",
        "runs",
        "accuracy %",
        "time (s)",
        "time min-max",
        "tokens",
        "tokens min-max",
        "msgs",
        "LLM wall (s)",
        "between-node (s)",
    ]
    summary_rows: List[List[str]] = []

    for fw in frameworks:
        subset = [r.result for r in records if r.framework == fw]
        correct = [float(_pick(s, "correctness")) for s in subset]
        times = [_pick(s, "time_metrics", "total_task_completion_time_seconds") for s in subset]
        toks = [_pick(s, "cost_metrics", "total_tokens") for s in subset]
        msgs = [_pick(s, "collaboration_metrics", "messages_between_agents") for s in subset]
        llm_w = [_llm_wall_seconds(s) for s in subset]
        btw = [_between_nodes_seconds(s) for s in subset]

        acc_pct = 100.0 * statistics.mean(correct) if correct else 0.0
        ts = _summarize(times)
        tk = _summarize(toks)
        ms = _summarize(msgs)
        lw = _summarize(llm_w)
        bw = _summarize(btw)

        summary_rows.append(
            [
                fw,
                str(len(subset)),
                f"{acc_pct:.1f}",
                _format_stat(ts, unit=""),
                f"{ts['min']:.2f}-{ts['max']:.2f}",
                _format_stat(tk, precision=0, unit=""),
                f"{int(tk['min'])}-{int(tk['max'])}",
                _format_stat(ms, precision=1, unit=""),
                _format_stat(lw, unit=""),
                _format_stat(bw, unit=""),
            ]
        )

    per_task_headers = ["task_id"] + [
        h
        for fw in frameworks
        for h in (f"{fw} pass %", f"{fw} time (s)", f"{fw} tokens")
    ]
    per_task_rows: List[List[str]] = []

    for task in tasks:
        tid = str(task.get("task_id", ""))
        row_cells: List[str] = [tid]
        for fw in frameworks:
            sub = [r.result for r in records if r.framework == fw and r.task_id == tid]
            if not sub:
                row_cells.extend(["—", "—", "—"])
                continue
            cr = [float(_pick(s, "correctness")) for s in sub]
            pass_pct = 100.0 * statistics.mean(cr)
            times = [_pick(s, "time_metrics", "total_task_completion_time_seconds") for s in sub]
            toks = [_pick(s, "cost_metrics", "total_tokens") for s in sub]
            row_cells.append(f"{pass_pct:.0f}")
            row_cells.append(_format_stat(_summarize(times)))
            row_cells.append(_format_stat(_summarize(toks), precision=0))
        per_task_rows.append(row_cells)

    ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    md_lines: List[str] = [
        "# HumanEval cross-framework benchmark",
        "",
        f"- **Generated:** {ts_iso}",
        f"- **Tasks:** {len(tasks)}",
        f"- **Repeats per task (per framework):** {args.repeats}",
        f"- **Model:** `{args.model}`",
        f"- **Max iterations:** {args.max_iterations}",
        "",
        "## Overall (all tasks × repeats)",
        "",
        markdown_table(summary_headers, summary_rows),
        "## Per-task (mean over repeats)",
        "",
        markdown_table(per_task_headers, per_task_rows),
        "### Legend",
        "",
        "- **pass %:** fraction of repeats that passed × 100.",
        "- **time / tokens:** mean ± sample stdev when repeats ≥ 2; otherwise mean only.",
        "",
    ]
    report = "\n".join(md_lines)
    out_path = Path(args.output)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote Markdown report to {out_path.resolve()}")
    print(report)


if __name__ == "__main__":
    main()
