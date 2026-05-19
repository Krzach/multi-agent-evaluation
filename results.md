# HumanEval & AetherCode cross-framework benchmark

- **Generated:** 2026-05-19 20:39:51 UTC
- **HumanEval tasks:** 5
- **AetherCode:** 10 tasks (Easy), 3 repeats per framework
- **Repeats per task (per framework):** 3
- **Model:** `gpt-5.4`
- **Max iterations:** 3
- **Raw results JSON:** `results/cross_framework_results.json`

## Overall (all tasks × repeats)

| framework | runs | accuracy % | time (s) | time min-max | tokens | tokens min-max | msgs | LLM wall (s) | between-node (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| langchain | 15 | 86.7 | 12.106±1.761 | 8.65-16.28 | 3537±553 | 2466-4384 | 5.0 | 12.100±1.761 | 29.680±15.873 |
| autogen | 15 | 100.0 | 12.304±1.407 | 10.05-15.22 | 3642±487 | 2805-4256 | 5.0 | 12.246±1.407 | 0.014±0.001 |
| spade | 15 | 93.3 | 8.358±1.651 | 5.52-11.05 | 2474±418 | 1693-3065 | 5.0 | 8.086±1.646 | 0.000±0.000 |
| single_agent | 15 | 0.0 | 1.536±1.374 | 0.76-6.34 | 245±37 | 187-300 | 0.0 | 1.536±1.374 | 0.000 |
| aether_langchain | 30 | 36.7 | 58.693±32.607 | 23.31-190.09 | 16132±5486 | 7251-36584 | 5.1±0.7 | 58.687±32.606 | 309.201±177.960 |
| aether_autogen | 30 | 43.3 | 51.476±29.226 | 19.10-143.56 | 14558±4433 | 7496-24094 | 5.0 | 51.423±29.226 | 0.011±0.001 |
| aether_spade | 30 | 23.3 | 37.570±16.899 | 16.32-93.00 | 13360±3438 | 5712-19584 | 5.9±2.0 | 37.286±16.903 | 0.000±0.000 |
| aether_single_agent | 30 | 16.7 | 9.843±10.237 | 2.20-43.20 | 1284±645 | 570-3586 | 0.0 | 9.842±10.237 | 0.000 |

## Token breakdown (mean ± stdev over runs)

| framework | runs | input tokens | output tokens | total tokens |
| --- | --- | --- | --- | --- |
| langchain | 15 | 2832±451 | 706±197 | 3537±553 |
| autogen | 15 | 2892±432 | 749±130 | 3642±487 |
| spade | 15 | 2150±348 | 325±130 | 2474±418 |
| single_agent | 15 | 206±15 | 39±27 | 245±37 |
| aether_langchain | 30 | 12622±3759 | 3510±1893 | 16132±5486 |
| aether_autogen | 30 | 11295±2926 | 3263±1929 | 14558±4433 |
| aether_spade | 30 | 11033±2688 | 2327±1120 | 13360±3438 |
| aether_single_agent | 30 | 719±167 | 565±515 | 1284±645 |

## Collaboration metrics (mean ± stdev over runs)

| framework | runs | conversation iterations | messages between agents | activated agents | safeguard blocked % |
| --- | --- | --- | --- | --- | --- |
| langchain | 15 | 0.00 | 5.00 | 3.00 | 0.0 |
| autogen | 15 | 0.00 | 5.00 | 3.00 | 0.0 |
| spade | 15 | 0.00 | 5.00 | 3.00 | 0.0 |
| single_agent | 15 | 1.00 | 0.00 | 3.00 | 0.0 |
| aether_langchain | 30 | 0.03±0.18 | 5.13±0.73 | 3.00 | 0.0 |
| aether_autogen | 30 | 0.00 | 5.00 | 3.00 | 0.0 |
| aether_spade | 30 | 0.23±0.50 | 5.93±2.02 | 3.00 | 0.0 |
| aether_single_agent | 30 | 1.00 | 0.00 | 3.00 | 0.0 |

## Wall-time decomposition (mean ± stdev over runs)

Task wall is runner `answer()` wall time when available; residual = task wall − logged LLM time − logged `execute_code` duration.

| framework | runs | task wall (s) | LLM (s) | tool exec (s) | residual orch (s) | residual share |
| --- | --- | --- | --- | --- | --- | --- |
| langchain | 15 | 12.106±1.761 | 12.100±1.761 | 0.000 | 0.007±0.001 | 0.001±0.000 |
| autogen | 15 | 12.304±1.407 | 12.246±1.407 | 0.000 | 0.057±0.002 | 0.005±0.001 |
| spade | 15 | 8.358±1.651 | 8.086±1.646 | 0.000 | 0.272±0.012 | 0.034±0.007 |
| single_agent | 15 | 1.536±1.374 | 1.536±1.374 | 0.000 | 0.000±0.000 | 0.000±0.000 |
| aether_langchain | 30 | 58.693±32.607 | 58.687±32.606 | 0.000±0.000 | 0.006±0.001 | 0.000±0.000 |
| aether_autogen | 30 | 51.476±29.226 | 51.423±29.226 | 0.000 | 0.053±0.002 | 0.001±0.001 |
| aether_spade | 30 | 37.570±16.899 | 37.286±16.903 | 0.000±0.000 | 0.284±0.039 | 0.009±0.004 |
| aether_single_agent | 30 | 9.843±10.237 | 9.842±10.237 | 0.000 | 0.000±0.000 | 0.000±0.000 |

## Per-step residual overhead (within-run stats, then mean ± stdev across runs)

Per logged event with `duration_ms`: `max(0, duration − llm_api − tool)`; `tool` is full step time for `execute_code` only.

| framework | runs | events / run | step resid mean (ms) | step resid p95 (ms) | step resid max (ms) |
| --- | --- | --- | --- | --- | --- |
| langchain | 15 | 9.0 | 0.02±0.01 | 0.08±0.07 | 0.11±0.09 |
| autogen | 15 | 9.0 | 0.03±0.00 | 0.09±0.02 | 0.12±0.03 |
| spade | 15 | 9.0 | 0.01±0.00 | 0.04±0.01 | 0.06±0.02 |
| single_agent | 15 | 1.0 | 0.00 | 0.00 | 0.00 |
| aether_langchain | 30 | 9.2±1.1 | 0.03±0.02 | 0.16±0.14 | 0.25±0.21 |
| aether_autogen | 30 | 9.0 | 0.05±0.02 | 0.18±0.13 | 0.27±0.20 |
| aether_spade | 30 | 10.3±2.9 | 0.03±0.01 | 0.14±0.07 | 0.21±0.12 |
| aether_single_agent | 30 | 1.0 | 0.00 | 0.00 | 0.00 |

## Per-task HumanEval (mean over repeats)

| task_id | langchain pass % | langchain time (s) | langchain tokens | autogen pass % | autogen time (s) | autogen tokens | spade pass % | spade time (s) | spade tokens | single_agent pass % | single_agent time (s) | single_agent tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HumanEval/0 | 33 | 11.997±0.833 | 3374±888 | 100 | 13.565±1.502 | 3645±730 | 67 | 9.068±1.589 | 2391±575 | 0 | 3.138±2.802 | 258 |
| HumanEval/1 | 100 | 14.336±1.854 | 3989±436 | 100 | 12.843±0.492 | 3774±623 | 100 | 9.765±1.119 | 2806±444 | 0 | 1.589±0.128 | 297±3 |
| HumanEval/2 | 100 | 10.596±1.760 | 3047±378 | 100 | 11.159±1.259 | 3358±476 | 100 | 6.054±0.495 | 2019±283 | 0 | 0.831±0.078 | 187 |
| HumanEval/3 | 100 | 11.721±1.732 | 3587±512 | 100 | 10.949±1.011 | 3444±297 | 100 | 9.007±0.987 | 2549±290 | 0 | 1.037±0.208 | 243 |
| HumanEval/4 | 100 | 11.882±0.532 | 3690±91 | 100 | 13.004±0.684 | 3987±205 | 100 | 7.897±1.201 | 2606±115 | 0 | 1.088±0.158 | 240 |

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
