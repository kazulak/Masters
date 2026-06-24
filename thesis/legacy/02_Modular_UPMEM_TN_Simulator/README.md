# Modular UPMEM Tensor-Network Simulator

This directory is the second-stage thesis runtime. It turns the fast
`../01_MVP_DenseGEMM` validation prototype into a benchmarkable, route-aware
pipeline for comparing tensor-network CPU execution, exact full-state CPU
execution, and the UPMEM DenseGEMM simulator path.

The architecture is deliberately host-orchestrated:

```text
YAML benchmark
  -> circuit builder
  -> tensor-network builder
  -> TaskGraphV2 planner
  -> dispatcher
  -> execution route
  -> validation
  -> result aggregation and charts
```

Current implemented routes:

| Route | Meaning | Status |
| --- | --- | --- |
| `cpu_reference` | Tensor-network contraction on CPU through NumPy/opt_einsum. | Implemented |
| `quest_exact_statevector` | Exact full-state CPU baseline through local QuEST. | Implemented |
| `raw_upmem_dense` | Replays the `01_MVP_DenseGEMM` UPMEM simulator host path. | Implemented for Bell 2q and GHZ 4q |

Future routes such as GPU, SimplePIM, SparseP, custom dense kernels, heuristics,
and PID-Comm collectives must attach to the same YAML, TaskGraph, validation,
and results pipeline.

## Layout

```text
02_Modular_UPMEM_TN_Simulator/
+-- README.md
+-- requirements.txt
+-- scripts/
|   +-- run_benchmark.py
|   +-- run_suite.py
|   +-- analyze_results.py
+-- src/tnsim/
|   +-- core/        # shared dataclasses, IO, utilities
|   +-- config/      # YAML loading and defaults
|   +-- circuits/    # built-in circuits and QASM input
|   +-- network/     # circuit-to-tensor-network conversion
|   +-- task_graph/  # TaskGraphV2 construction
|   +-- dispatch/    # route eligibility and selection
|   +-- execution/   # TN CPU, QuEST, UPMEM simulator routes
|   +-- validation/  # references and correctness metrics
|   +-- records/     # execution-log construction
|   +-- results/     # metric extraction, tables, charts
|   +-- suite/       # multi-run orchestration
|   +-- runner.py    # single-run orchestrator
+-- baselines/
|   +-- quest_exact/
+-- benchmarks/
|   +-- configs/
|   |   +-- tn_cpu/
|   |   +-- quest_exact/
|   |   +-- upmem_sim/
|   +-- suites/
+-- docs/
    +-- architecture.md
    +-- benchmarks.md
    +-- roadmap.md
```

Generated `runs/` and `results/` directories are ignored by git. They are
machine-readable artifacts, not source structure.

## Reading Order

1. `docs/architecture.md` for the committed module boundaries and dependency
   rule.
2. `docs/benchmarks.md` for YAML inputs, result records, metrics, energy, and
   comparison charts.
3. `docs/roadmap.md` for implementation order and what is intentionally not
   integrated yet.

## Run Benchmarks

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make -C baselines/quest_exact
.venv/bin/python scripts/run_suite.py benchmarks/suites/tn_vs_quest_scale.yaml
.venv/bin/python scripts/run_suite.py benchmarks/suites/three_backend_mvp.yaml
```

Run one benchmark:

```bash
.venv/bin/python scripts/run_benchmark.py benchmarks/configs/tn_cpu/bell_2q.yaml
```

Analyze existing runs without rerunning:

```bash
.venv/bin/python scripts/analyze_results.py runs --output-dir results/manual
```

Each run writes:

```text
runs/<experiment_id>/
+-- input_config.yaml
+-- resolved_config.yaml
+-- task_graph.json
+-- execution_log.json
+-- validation_record.json
+-- metrics.jsonl
```

Each suite writes:

```text
results/<suite_id>/
+-- summary.csv
+-- summary.json
+-- summary.md
+-- speedup_vs_fastest.svg
+-- energy_joules.svg
```

The SVG charts group bars by workload, so `bell_2q` displays TN CPU, QuEST
full-state, and UPMEM simulator columns side by side when those runs exist.

## Naming Rule

Experiment ids should describe the algorithm and backend plainly:

```text
<workload>_tn_cpu
<workload>_quest_exact
<workload>_upmem_sim
```

Do not encode temporary implementation details in benchmark names.

## Engineering Rule

Add new behavior inside the pipeline-stage folder that owns it. For example:

- a new circuit generator goes in `circuits/`;
- a new planner goes in `task_graph/`;
- a new execution backend goes in `execution/`;
- a new metric extractor goes in `results/summary.py`;
- a new chart belongs in `results/`.

The orchestrators should stay thin. If `runner.py` or `suite/orchestrator.py`
starts accumulating backend logic, move that logic into the stage module.
