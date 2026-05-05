# `results.md` metric-by-metric documentation

## 1) Shared JSON schema (per run)

Each run record in `results/cross_framework_results.json` includes these fields used by the tables:

- `framework` (`langchain` or `autogen`)
- `task_id` (`HumanEval/0` ... `HumanEval/4`)
- `repeat` (0..2)
- `correctness`
- `time_metrics.total_task_completion_time_seconds`
- `cost_metrics.input_tokens`
- `cost_metrics.output_tokens`
- `cost_metrics.total_tokens`
- `collaboration_metrics.conversation_iterations`
- `collaboration_metrics.messages_between_agents`
- `collaboration_metrics.activated_agents`
- `collaboration_metrics.safeguard_blocked`
- `conversation_log_metrics.aggregate_agents.by_agent.*.llm_api_duration_ms`
- `conversation_log_metrics.task_wall_attribution.between_nodes_time_seconds`
- `conversation_log_metrics.wall_time_decomposition.*`
- `conversation_log_metrics.per_step_residual_overhead.*`

## 2) Formatting and aggregation rules used in `results.md`

- **Mean/stdev**: sample standard deviation (`statistics.stdev`) when `n >= 2`.
- **Display form**:
  - `mean±stdev` if stdev is non-zero and `n >= 2`.
  - `mean` only otherwise.
- **Accuracy/pass %**:
  - overall: `100 * mean(correctness over framework runs)`
  - per-task row: `100 * mean(correctness over that task's repeats)`
- **Time min-max**: min and max of `time_metrics.total_task_completion_time_seconds`.
- **Tokens min-max**: min and max of `cost_metrics.total_tokens`.
- **Safeguard blocked %**: `100 * mean(collaboration_metrics.safeguard_blocked)`.
- **LLM wall (s)** per run: sum of all `llm_api_duration_ms` under `conversation_log_metrics.aggregate_agents.by_agent`, divided by 1000.
- **between-node (s)** per run: `conversation_log_metrics.task_wall_attribution.between_nodes_time_seconds`.
- **Wall-time decomposition columns**:
  - task wall (s) = `wall_time_decomposition.task_wall_ms / 1000`
  - LLM (s) = `wall_time_decomposition.measured_llm_api_ms / 1000`
  - tool exec (s) = `wall_time_decomposition.measured_tool_execution_ms / 1000`
  - residual orch (s) = `wall_time_decomposition.residual_orchestration_ms / 1000`
  - residual share = `wall_time_decomposition.residual_orchestration_share_of_task_wall`
- **Per-step residual columns** come from:
  - `events / run` = `events_with_duration_count`
  - `step resid mean (ms)` = `mean_step_residual_ms`
  - `step resid p95 (ms)` = `p95_step_residual_ms`
  - `step resid max (ms)` = `max_step_residual_ms`

## 3) What each metric represents (plain-English)

- **accuracy % / pass %**: reliability of producing a correct solution.
- **time (s)**: end-to-end latency to complete one task.
- **tokens (input/output/total)**: model usage cost/throughput proxy.
- **messages between agents**: communication intensity between MAS roles.
- **conversation iterations**: extra back-and-forth rounds triggered by the orchestration.
- **activated agents**: how many distinct agents participated in a run.
- **safeguard blocked %**: safety gate intervention rate.
- **LLM wall (s)**: time spent inside LLM API calls (summed).
- **between-node (s)**: orchestration gap time between nodes/steps (framework-dependent measurement source).
- **task wall (s)**: total measured task duration basis for decomposition.
- **tool exec (s)**: local code execution (`execute_code`) portion of wall time.
- **residual orch (s)**: non-LLM/non-tool orchestration overhead.
- **residual share**: proportion of task wall occupied by residual orchestration overhead.
- **step residual mean/p95/max (ms)**: per-step non-LLM, non-tool overhead distribution (average, tail, worst-case).

