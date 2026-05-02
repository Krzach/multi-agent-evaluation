# Metrics reference

This document describes what each metric means, how it is computed (including differences by MAS implementation), and how to read values from benchmark outputs (JSON arrays of per-task result objects).

The **`metrics/`** package exposes **`conversation_log_metrics.py`**: wall-time splits from JSONL conversation logs (per-agent **time outside the LLM call** vs LLM API time, LangGraph between-node gaps, shares of task wall time). HumanEval and MultiAgentBench runners also attach **`cost_metrics`**, **`time_metrics`**, and **`collaboration_metrics`** from MAS **`answer()`** output.

Coding benchmarks append one object per task:

```json
{
  "task_id": "...",
  "mas_framework": "LangchainCodingMAS",
  "conversation_log_path": "logs/mas_conversations/....jsonl",
  "conversation_log_metrics": { ... },
  "cost_metrics": { ... },
  "time_metrics": { ... },
  "collaboration_metrics": { ... }
}
```

---

## 1. Conversation log metrics (`conversation_log_metrics`)

Computed by **`build_conversation_log_metrics_envelope(mas_output, mas_task_seconds)`**, which calls **`compute_conversation_log_metrics(...)`**. It passes **`between_nodes_duration_ms`** from **`mas_output["langgraph_between_nodes_duration_ms"]`** when that value is numeric; otherwise between-node totals are **0**.

- **`mas_task_seconds`**: wall time for the full `answer()` call, measured by the benchmark runner (`time.perf_counter()` around `mas.answer(...)`).
- **`log_path`**: `mas_output["conversation_log_path"]` when the MAS wrote a JSONL log.

If **`log_path`** is missing or the file does not exist, aggregates are mostly zeros and shares are zero.

### 1.2 Source data: JSONL conversation log

| `record_type`   | Role |
|-----------------|------|
| `session_start` | Session metadata (framework name, model, query, …). |
| `event`         | **`actor`** (Commander / Writer / Safeguard), optional **`phase`**, **`duration_ms`**, optional **`llm_api_duration_ms`**, optional **`token_usage`**, … |
| `session_end`   | May include **`mas_total_duration_ms`**. |

**Per-event timing** (`CodingMASBase.log_conversation_event`):

| Field | Meaning |
|-------|---------|
| **`duration_ms`** | Wall time for the step (ms). |
| **`llm_api_duration_ms`** | Optional. Time inside the provider LLM call (ms). **Time outside the LLM call** for the event (ms) = **`max(0, duration_ms - llm_api_duration_ms)`** — aggregated as **`time_duration_ms`**. |
| **`actor`** | Used to bucket sums: **`commander`**, **`writer`**, **`safeguard`** (case-insensitive); anything else rolls into **`other`**. |

**`phase`** (`coordination`, `generation`, `execution`, `finalization`) is still written for workflow context; **aggregates in `conversation_log_metrics` use `actor` only**, not `phase`.

### 1.3 How metrics are calculated

Implemented in **`metrics/conversation_log_metrics.py`** (`EventMetricsAggregator`, `LogMetricsCalculator`).

#### A. LLM API duration for one event

Same rules as before: explicit **`llm_api_duration_ms`** if ≥ 0; else legacy heuristic from **`token_usage`**; else **0**.

#### B. Time outside the LLM call (per event)

\[
\texttt{time\_ms} = \max(0,\ \texttt{duration\_ms} - \texttt{llm\_api\_ms})
\]

Summed **per `actor` bucket** (`commander`, `writer`, `safeguard`, `other`) as **`time_duration_ms`**.

#### C. Token totals

Summed **`token_usage`** per actor bucket.

#### D. Between-node time (LangGraph)

**Source:** `TimeBetweenNodesCallback` in **`coding_scenario/langchain/callbacks/time_between_nodes.py`**, on **`graph.invoke`** in **`LangchainCodingMAS`** (**`coding_scenario/langchain/langchain_mas.py`**).

**Other MAS builds:** omit **`langgraph_between_nodes_duration_ms`** on **`answer()`** → between-node ms in aggregates are **0**.

#### E. Task wall denominator

\[
\texttt{denom\_seconds} =
\begin{cases}
\texttt{mas\_task\_seconds} & \text{if } > 0 \\
\texttt{session\_end.mas\_total\_duration\_ms} / 1000 & \text{else if present} \\
\texttt{total\_event\_duration\_ms} / 1000 & \text{else}
\end{cases}
\]

#### F. `aggregate_agents`

| Key | Meaning |
|-----|---------|
| **`by_agent`** | For each of **`commander`**, **`writer`**, **`safeguard`**, **`other`**: **`time_duration_ms`**, **`llm_api_duration_ms`**, **`tokens`**. |
| **`total_event_duration_ms`** | Sum of **`duration_ms`** over all events. |
| **`total_time_duration_ms`** | Sum of **`time_duration_ms`** over all events (all actors). |
| **`total_llm_api_duration_ms`** | Sum of LLM ms over all events. |
| **`between_nodes_duration_ms`** | LangGraph callback total (ms); **0** if not supplied. |

#### G. `task_wall_attribution`

Relates logged durations to **overall task wall time** (not “overhead”):

| Key | Meaning |
|-----|---------|
| **`task_wall_denominator_seconds`** | Denominator used for shares (§1.3.E). |
| **`total_llm_api_time_seconds`** | All events, LLM time. |
| **`between_nodes_time_seconds`** | Between-node ms / 1000. |
| **`between_nodes_share_of_task_wall`** | Between-node seconds ÷ denominator. |
| **`between_nodes_measurement_source`** | How between-node time was obtained. |
| **`per_agent`** | For **`commander`**, **`writer`**, **`safeguard`**, **`other`**: **`time_seconds`**, **`llm_api_time_seconds`**, **`time_share_of_task_wall`**, **`llm_api_share_of_task_wall`**, **`tokens`**. |
| **`session_end_wall_duration_ms`** | From `session_end.mas_total_duration_ms` if present, else `null`. |

---

### 1.4 Per-framework: how logs are produced

#### LangChain / LangGraph (`LangchainCodingMAS`)

- **`duration_ms`** / **`llm_api_duration_ms`**: measured around node work and LLM **`invoke`** respectively.
- **Between-node:** callback on **`graph.invoke`** → **`langgraph_between_nodes_duration_ms`** on **`answer()`** → folded into **`aggregate_agents.between_nodes_duration_ms`**.

#### AutoGen / SPADE

Same **`actor`** strings; **`duration_ms`** / **`llm_api_duration_ms`** fidelity differs (see previous caveats in code comments). Legacy logs without **`llm_api_duration_ms`** may attribute most wall time to LLM when tokens exist.

---

## 2. Other benchmark fields

### 2.1 `cost_metrics`

Task-level totals from **`mas_output["token_usage"]`**, not the same as summing per-event tokens in **`aggregate_agents`**.

### 2.2 `time_metrics`

**`total_task_completion_time_seconds`** matches **`mas_task_seconds`** passed into **`build_conversation_log_metrics_envelope`**, so it aligns with **`task_wall_denominator_seconds`** when positive.

### 2.3 `collaboration_metrics`

Runner heuristics (attempt counts, message estimates); not derived from JSONL aggregation.

---

## 3. Reading results in Python / jq

```python
import json
from pathlib import Path

rows = json.loads(Path("path/to/results.json").read_text(encoding="utf-8"))
row = rows[0]
m = row["conversation_log_metrics"]
attr = m["task_wall_attribution"]
cmd = attr["per_agent"]["commander"]
t_s = cmd["time_seconds"]
share = cmd["time_share_of_task_wall"]
```

```bash
jq '.[0].conversation_log_metrics.task_wall_attribution.per_agent.commander' results.json
```

---

## 4. Related files

| File | Role |
|------|------|
| `metrics/conversation_log_metrics.py` | Per-agent aggregates and task-wall attribution. |
| `coding_scenario/langchain/callbacks/time_between_nodes.py` | Between-node callback. |
| `coding_scenario/langchain/langchain_mas.py` | Registers callbacks; logs **`actor`**. |
| `coding_scenario/base.py` | `log_conversation_event`, session helpers. |
| `coding_scenario/conversation_log.schema.json` | JSON Schema for log lines. |
| `benchmarks/human_eval/runner.py`, `benchmarks/multiagentbench/runner.py` | Attach **`conversation_log_metrics`**. |

---

## 5. Caveats

- **Shares** are fractions of benchmark-measured **`answer()`** wall time, not provider latency alone.
- **Between-node** requires LangGraph **`langgraph_node`** metadata on callbacks; re-validate after LangChain/LangGraph upgrades.
- **Legacy logs** without **`llm_api_duration_ms`** can inflate LLM time when tokens exist → **per-agent `time_duration_ms` / `time_seconds` can look artificially low**.
