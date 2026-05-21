# HumanEval & AetherCode cross-framework benchmark

- **Generated:** 2026-05-21 11:18:49 UTC
- **HumanEval tasks:** 5
- **AetherCode:** 10 tasks (Easy), 3 repeats per framework
- **Repeats per task (per framework):** 3
- **Model:** `gpt-5.4`
- **Max iterations:** 3
- **Raw results JSON:** `results/cross_framework_results.json`

## Overall (all tasks × repeats)

| framework | runs | accuracy % | time (s) | time min-max | tokens | tokens min-max | msgs | LLM wall (s) | between-node (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aether_langchain | 30 | 36.7 | 58.693±32.607 | 23.31-190.09 | 16132±5486 | 7251-36584 | 5.1±0.7 | 58.687±32.606 | 309.201±177.960 |
| aether_autogen | 30 | 43.3 | 51.476±29.226 | 19.10-143.56 | 14558±4433 | 7496-24094 | 5.0 | 51.423±29.226 | 0.011±0.001 |
| aether_spade | 30 | 23.3 | 37.570±16.899 | 16.32-93.00 | 13360±3438 | 5712-19584 | 5.9±2.0 | 37.286±16.903 | 0.000±0.000 |
| aether_single_agent | 30 | 16.7 | 9.843±10.237 | 2.20-43.20 | 1284±645 | 570-3586 | 0.0 | 9.842±10.237 | 0.000 |
| single_agent | 15 | 100.0 | 1.280±0.289 | 0.87-1.81 | 246±38 | 187-300 | 0.0 | 1.280±0.289 | 0.000 |
| langchain | 15 | 100.0 | 12.892±2.074 | 9.49-17.10 | 3446±529 | 2246-4073 | 5.0 | 12.887±2.074 | 30.345±16.609 |
| autogen | 15 | 100.0 | 12.388±2.454 | 8.70-16.54 | 3383±680 | 2047-4358 | 5.0 | 12.335±2.452 | 0.011±0.002 |
| spade | 15 | 100.0 | 8.307±1.729 | 5.60-11.83 | 2481±441 | 1629-3061 | 5.0 | 8.013±1.735 | 0.000±0.000 |

## Token breakdown (mean ± stdev over runs)

| framework | runs | input tokens | output tokens | total tokens |
| --- | --- | --- | --- | --- |
| aether_langchain | 30 | 12622±3759 | 3510±1893 | 16132±5486 |
| aether_autogen | 30 | 11295±2926 | 3263±1929 | 14558±4433 |
| aether_spade | 30 | 11033±2688 | 2327±1120 | 13360±3438 |
| aether_single_agent | 30 | 719±167 | 565±515 | 1284±645 |
| single_agent | 15 | 206±15 | 40±28 | 246±38 |
| langchain | 15 | 2769±455 | 678±153 | 3446±529 |
| autogen | 15 | 2722±509 | 661±234 | 3383±680 |
| spade | 15 | 2150±364 | 330±128 | 2481±441 |

## Collaboration metrics (mean ± stdev over runs)

| framework | runs | conversation iterations | messages between agents | activated agents | safeguard blocked % |
| --- | --- | --- | --- | --- | --- |
| aether_langchain | 30 | 0.03±0.18 | 5.13±0.73 | 3.00 | 0.0 |
| aether_autogen | 30 | 0.00 | 5.00 | 3.00 | 0.0 |
| aether_spade | 30 | 0.23±0.50 | 5.93±2.02 | 3.00 | 0.0 |
| aether_single_agent | 30 | 1.00 | 0.00 | 3.00 | 0.0 |
| single_agent | 15 | 1.00 | 0.00 | 3.00 | 0.0 |
| langchain | 15 | 0.00 | 5.00 | 3.00 | 0.0 |
| autogen | 15 | 0.00 | 5.00 | 3.00 | 0.0 |
| spade | 15 | 0.00 | 5.00 | 3.00 | 0.0 |

## Wall-time decomposition (mean ± stdev over runs)

Task wall is runner `answer()` wall time when available; residual = task wall − logged LLM time − logged `execute_code` duration.

| framework | runs | task wall (s) | LLM (s) | tool exec (s) | residual orch (s) | residual share |
| --- | --- | --- | --- | --- | --- | --- |
| aether_langchain | 30 | 58.693±32.607 | 58.687±32.606 | 0.000±0.000 | 0.006±0.001 | 0.000±0.000 |
| aether_autogen | 30 | 51.476±29.226 | 51.423±29.226 | 0.000 | 0.053±0.002 | 0.001±0.001 |
| aether_spade | 30 | 37.570±16.899 | 37.286±16.903 | 0.000±0.000 | 0.284±0.039 | 0.009±0.004 |
| aether_single_agent | 30 | 9.843±10.237 | 9.842±10.237 | 0.000 | 0.000±0.000 | 0.000±0.000 |
| single_agent | 15 | 1.280±0.289 | 1.280±0.289 | 0.000 | 0.000±0.000 | 0.000±0.000 |
| langchain | 15 | 12.892±2.074 | 12.887±2.074 | 0.000 | 0.005±0.001 | 0.000±0.000 |
| autogen | 15 | 12.388±2.454 | 12.335±2.452 | 0.000 | 0.053±0.003 | 0.004±0.001 |
| spade | 15 | 8.307±1.729 | 8.013±1.735 | 0.000 | 0.293±0.045 | 0.037±0.010 |

## Per-step residual overhead (within-run stats, then mean ± stdev across runs)

Per logged event with `duration_ms`: `max(0, duration − llm_api − tool)`; `tool` is full step time for `execute_code` only.

| framework | runs | events / run | step resid mean (ms) | step resid p95 (ms) | step resid max (ms) |
| --- | --- | --- | --- | --- | --- |
| aether_langchain | 30 | 9.2±1.1 | 0.03±0.02 | 0.16±0.14 | 0.25±0.21 |
| aether_autogen | 30 | 9.0 | 0.05±0.02 | 0.18±0.13 | 0.27±0.20 |
| aether_spade | 30 | 10.3±2.9 | 0.03±0.01 | 0.14±0.07 | 0.21±0.12 |
| aether_single_agent | 30 | 1.0 | 0.00 | 0.00 | 0.00 |
| single_agent | 15 | 1.0 | 0.00 | 0.00 | 0.00 |
| langchain | 15 | 9.0 | 0.01±0.01 | 0.06±0.05 | 0.08±0.06 |
| autogen | 15 | 9.0 | 0.02±0.00 | 0.07±0.02 | 0.09±0.04 |
| spade | 15 | 9.0 | 0.01±0.00 | 0.04±0.01 | 0.06±0.01 |

## Per-task HumanEval (mean over repeats)

| task_id | single_agent pass % | single_agent time (s) | single_agent tokens | langchain pass % | langchain time (s) | langchain tokens | autogen pass % | autogen time (s) | autogen tokens | spade pass % | spade time (s) | spade tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HumanEval/0 | 100 | 1.262±0.118 | 258 | 100 | 13.970±2.756 | 3376±987 | 100 | 13.710±1.374 | 3332±1150 | 100 | 9.141±0.742 | 2426±695 |
| HumanEval/1 | 100 | 1.747±0.058 | 300 | 100 | 14.169±0.330 | 3725±438 | 100 | 14.913±1.580 | 3763±631 | 100 | 10.612±1.222 | 2799±446 |
| HumanEval/2 | 100 | 1.021±0.166 | 187 | 100 | 11.884±1.792 | 3145±399 | 100 | 9.261±0.548 | 2772±244 | 100 | 5.865±0.281 | 2030±292 |
| HumanEval/3 | 100 | 1.241±0.248 | 245±2 | 100 | 10.872±2.125 | 3222±127 | 100 | 10.330±0.741 | 3183±55 | 100 | 7.907±0.621 | 2474±237 |
| HumanEval/4 | 100 | 1.130±0.115 | 241±1 | 100 | 13.563±1.474 | 3764±357 | 100 | 13.725±1.008 | 3862±517 | 100 | 8.008±0.282 | 2674±133 |

## Per-task AetherCode (mean over repeats)

| task_id | aether_langchain pass % | aether_langchain time (s) | aether_langchain tokens | aether_autogen pass % | aether_autogen time (s) | aether_autogen tokens | aether_spade pass % | aether_spade time (s) | aether_spade tokens | aether_single_agent pass % | aether_single_agent time (s) | aether_single_agent tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60173 | 0 | 59.150±9.071 | 14727±2597 | 0 | 70.082±15.708 | 14944±4526 | 0 | 32.431±2.441 | 11581±3550 | 0 | 6.191±1.317 | 1220±44 |
| 60166 | 0 | 55.700±26.861 | 12445±4907 | 0 | 41.617±17.354 | 10816±3592 | 0 | 38.176±9.549 | 12748±4583 | 0 | 12.132±13.934 | 774±62 |
| 60150 | 100 | 24.847±2.514 | 10917±2209 | 100 | 27.637±5.382 | 10987±1771 | 100 | 21.880±0.725 | 10730±1894 | 67 | 5.976±2.424 | 1162±228 |
| 60185 | 0 | 63.988±8.978 | 15510±1453 | 0 | 47.073±8.933 | 12667±2227 | 0 | 23.603±1.358 | 11069±1450 | 0 | 4.254±0.473 | 974±44 |
| 60165 | 33 | 122.280±63.502 | 26558±8895 | 0 | 84.719±54.238 | 19723±4774 | 0 | 71.740±21.382 | 17846±2548 | 0 | 21.890±18.980 | 2384±1080 |
| 60162 | 0 | 78.788±15.881 | 21406±2976 | 67 | 91.715±33.840 | 21323±2106 | 0 | 47.682±8.957 | 15239±1906 | 0 | 17.161±9.336 | 2095±438 |
| 60174 | 0 | 40.881±9.850 | 12892±920 | 0 | 23.846±7.408 | 9785±828 | 0 | 25.129±11.957 | 10489±4956 | 0 | 2.342±0.177 | 590±32 |
| 60180 | 67 | 39.574±5.410 | 14206±1097 | 67 | 39.895±11.543 | 13798±1013 | 0 | 41.970±15.048 | 16076±732 | 0 | 3.568±0.937 | 900±49 |
| 60149 | 100 | 41.268±11.545 | 14208±2006 | 100 | 35.914±2.742 | 13232±481 | 100 | 28.138±2.941 | 11893±904 | 100 | 6.069±0.178 | 1069±22 |
| 60159 | 67 | 60.452±7.248 | 18450±355 | 100 | 52.262±10.141 | 18310±1078 | 33 | 44.947±6.533 | 15928±705 | 0 | 18.844±13.703 | 1670±122 |

### Legend

- **pass %:** fraction of repeats that passed × 100.
- **time / tokens:** mean ± sample stdev when repeats ≥ 2; otherwise mean only.
- **msgs:** `collaboration_metrics.messages_between_agents` (log-derived inter-agent events when available).
- **LLM wall / between-node:** from `conversation_log_metrics` (summed LLM ms; LangGraph/MAS gap callback).
- **Wall-time decomposition / per-step residual:** from `conversation_log_metrics` JSONL-derived fields.
- **Framework rows** prefixed with `aether_` are AetherCode runs; others are HumanEval.
