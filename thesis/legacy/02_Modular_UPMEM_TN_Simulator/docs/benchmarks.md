# Benchmark Workflow

The benchmark pipeline is YAML-first. The YAML file is the experiment contract;
source code supplies reusable pipeline stages and execution routes.

## Single Run

```text
benchmark YAML
  -> load_config
  -> build_tensor_network
  -> plan_task_graph
  -> dispatch route
  -> execute backend
  -> compute reference
  -> validate
  -> write run artifacts
```

The single-run orchestrator is `src/tnsim/runner.py`. It should remain a thin
composition layer. Backend-specific work belongs in `src/tnsim/execution/`.

## Suite Run

```text
suite YAML
  -> run each config
  -> collect result rows
  -> compute per-workload fastest and lowest-energy comparisons
  -> write tables and grouped charts
```

The suite orchestrator is `src/tnsim/suite/orchestrator.py`.

## Metrics

Metrics are extracted from persisted run artifacts, not hidden process state.
This makes old runs analyzable after new metrics are added, as long as the raw
field exists in one of:

- `execution_log.json`;
- `validation_record.json`;
- `task_graph.json`;
- `metrics.jsonl`.

To add a metric, register a function in `src/tnsim/results/summary.py`:

```python
@metric("my_metric")
def _my_metric(a: RunArtifacts):
    return a.execution_log["summary"].get("my_metric")
```

The new metric is then available in `summary.csv`, `summary.json`, and
`summary.md`.

## Energy

Energy is required in every route summary.

- QuEST attempts Linux RAPL package-energy measurement when readable.
- CPU TN uses the YAML `measurement.energy.cpu_watts` estimate.
- UPMEM simulator uses the YAML power estimate and marks the source as simulator
  energy, because it is not real DPU hardware power.

Every summary row records:

- `energy_joules`;
- `energy_source`;
- `estimated_power_watts`;
- `energy_ratio_vs_lowest_same_workload`.

This keeps estimated energy clearly separated from measured energy.

## Charts

Charts are generated without an external plotting dependency. Bars are grouped
by workload and colored by route family:

| Route family | Color |
| --- | --- |
| TN CPU | blue |
| QuEST full-state | orange |
| UPMEM sim | green |

For example, `bell_2q` appears once on the x-axis with adjacent TN CPU, QuEST,
and UPMEM simulator bars.

## Current Suites

```bash
.venv/bin/python scripts/run_suite.py benchmarks/suites/tn_vs_quest_scale.yaml
.venv/bin/python scripts/run_suite.py benchmarks/suites/three_backend_mvp.yaml
```

`tn_vs_quest_scale` covers Bell 2q and GHZ 4/8/12/16. `three_backend_mvp` covers
Bell 2q and GHZ 4q because those are the circuits supported by the frozen MVP
UPMEM DenseGEMM pipeline.
