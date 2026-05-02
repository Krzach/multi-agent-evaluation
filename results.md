# HumanEval cross-framework benchmark

- **Generated:** 2026-05-02 (example layout — run `python cross_framework_benchmark.py` to refresh)
- **Tasks:** 5
- **Repeats per task (per framework):** 3
- **Model:** `gpt-5.4`
- **Max iterations:** 3

## Overall (all tasks × repeats)

| framework | runs | accuracy % | time (s) | time min-max | tokens | tokens min-max | msgs | LLM wall (s) | between-node (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| langchain | 15 | 86.7 | 11.603±3.050 | 7.38-16.62 | 3242±568 | 2248-4159 | 6.0 | 11.588±3.049 | 25.096±13.669 |
| autogen | 15 | 86.7 | 10.256±2.064 | 6.90-13.56 | 3169±581 | 2322-4218 | 6.0 | 10.195±2.066 | 0.018±0.003 |

## Per-task (mean over repeats)

| task_id | langchain pass % | langchain time (s) | langchain tokens | autogen pass % | autogen time (s) | autogen tokens |
| --- | --- | --- | --- | --- | --- | --- |
| HumanEval/0 | 67 | 11.672±2.715 | 3170±561 | 67 | 10.547±0.376 | 3097±602 |
| HumanEval/1 | 100 | 15.259±2.046 | 3668±560 | 100 | 12.120±0.880 | 3453±560 |
| HumanEval/2 | 100 | 7.850±0.453 | 2591±300 | 100 | 7.534±0.569 | 2632±280 |
| HumanEval/3 | 100 | 9.670±0.512 | 3028±243 | 100 | 8.951±0.934 | 2958±327 |
| HumanEval/4 | 67 | 13.566±0.733 | 3754±321 | 67 | 12.129±1.852 | 3707±663 |

### Legend

- **pass %:** fraction of repeats that passed × 100.
- **time / tokens:** mean ± sample stdev when repeats ≥ 2; otherwise mean only.
