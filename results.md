# HumanEval cross-framework benchmark

- **Generated:** 2026-05-05 13:16:23 UTC
- **Tasks:** 5
- **Repeats per task (per framework):** 3
- **Model:** `gpt-5.4`
- **Max iterations:** 3
- **Raw results JSON:** `results/cross_framework_results.json`

## Overall (all tasks × repeats)


| framework | runs | accuracy % | time (s)     | time min-max | tokens   | tokens min-max | msgs | LLM wall (s) | between-node (s) |
| --------- | ---- | ---------- | ------------ | ------------ | -------- | -------------- | ---- | ------------ | ---------------- |
| langchain | 15   | 86.7       | 11.984±1.924 | 9.24-15.67   | 3349±655 | 2422-4623      | 5.0  | 11.969±1.924 | 28.541±14.943    |
| autogen   | 15   | 93.3       | 15.421±5.464 | 9.89-29.00   | 3720±565 | 2614-4566      | 5.0  | 15.359±5.466 | 0.021±0.002      |


## Token breakdown (mean ± stdev over runs)


| framework | runs | input tokens | output tokens | total tokens |
| --------- | ---- | ------------ | ------------- | ------------ |
| langchain | 15   | 2717±511     | 631±201       | 3349±655     |
| autogen   | 15   | 2941±478     | 779±164       | 3720±565     |


## Collaboration metrics (mean ± stdev over runs)


| framework | runs | conversation iterations | messages between agents | activated agents | safeguard blocked % |
| --------- | ---- | ----------------------- | ----------------------- | ---------------- | ------------------- |
| langchain | 15   | 0.00                    | 5.00                    | 3.00             | 0.0                 |
| autogen   | 15   | 0.00                    | 5.00                    | 3.00             | 0.0                 |


## Wall-time decomposition (mean ± stdev over runs)

Task wall is runner `answer()` wall time when available; residual = task wall − logged LLM time − logged `execute_code` duration.


| framework | runs | task wall (s) | LLM (s)      | tool exec (s) | residual orch (s) | residual share |
| --------- | ---- | ------------- | ------------ | ------------- | ----------------- | -------------- |
| langchain | 15   | 11.984±1.924  | 11.969±1.924 | 0.000         | 0.015±0.003       | 0.001±0.000    |
| autogen   | 15   | 15.421±5.464  | 15.359±5.466 | 0.000         | 0.063±0.009       | 0.004±0.001    |


## Per-step residual overhead (within-run stats, then mean ± stdev across runs)

Per logged event with `duration_ms`: `max(0, duration − llm_api − tool)`; `tool` is full step time for `execute_code` only.


| framework | runs | events / run | step resid mean (ms) | step resid p95 (ms) | step resid max (ms) |
| --------- | ---- | ------------ | -------------------- | ------------------- | ------------------- |
| langchain | 15   | 9.0          | 0.04±0.04            | 0.15±0.19           | 0.17±0.19           |
| autogen   | 15   | 9.0          | 0.06±0.01            | 0.20±0.06           | 0.24±0.08           |


## Per-task (mean over repeats)


| task_id     | langchain pass % | langchain time (s) | langchain tokens | autogen pass % | autogen time (s) | autogen tokens |
| ----------- | ---------------- | ------------------ | ---------------- | -------------- | ---------------- | -------------- |
| HumanEval/0 | 100              | 13.659±1.933       | 3648±1122        | 100            | 14.728±2.191     | 3880±1098      |
| HumanEval/1 | 100              | 13.524±0.484       | 3519±644         | 100            | 13.637±1.111     | 3735±373       |
| HumanEval/2 | 100              | 10.276±1.228       | 2931±464         | 100            | 12.100±2.415     | 3361±591       |
| HumanEval/3 | 100              | 10.138±0.571       | 3154±484         | 67             | 18.282±9.390     | 3611±354       |
| HumanEval/4 | 33               | 12.322±1.641       | 3492±598         | 100            | 18.360±7.851     | 4011±204       |


### Legend

- **pass %:** fraction of repeats that passed × 100.
- **time / tokens:** mean ± sample stdev when repeats ≥ 2; otherwise mean only.
- **msgs:** `collaboration_metrics.messages_between_agents` (log-derived inter-agent events when available).
- **LLM wall / between-node:** from `conversation_log_metrics` (summed LLM ms; LangGraph/MAS gap callback).
- **Wall-time decomposition / per-step residual:** from `conversation_log_metrics` JSONL-derived fields.

