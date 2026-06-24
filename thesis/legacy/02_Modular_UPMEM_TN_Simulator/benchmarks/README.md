# Benchmarks

Benchmarks are YAML files. A config is the canonical input for one reproducible
run: it records the workload, planner, selected route, validation tolerances,
energy settings, and output path.

## Config Groups

```text
configs/
+-- tn_cpu/       # tensor-network contraction on CPU
+-- quest_exact/  # QuEST exact full-state CPU baseline
+-- upmem_sim/    # UPMEM DenseGEMM MVP simulator replay
```

Current local-PC scale coverage:

| Workload | TN CPU | QuEST exact | UPMEM simulator |
| --- | --- | --- | --- |
| `bell_2q` | yes | yes | yes |
| `ghz_4q` | yes | yes | yes |
| `ghz_8q` | yes | yes | no |
| `ghz_12q` | yes | yes | no |
| `ghz_16q` | yes | yes | no |

The UPMEM simulator route is intentionally limited to the circuits implemented
by `../01_MVP_DenseGEMM`: Bell 2q and GHZ 4q.

## Suites

| Suite | Purpose |
| --- | --- |
| `suites/tn_vs_quest_scale.yaml` | Compares TN CPU and QuEST exact on small, medium, and local-large GHZ workloads. |
| `suites/three_backend_mvp.yaml` | Compares TN CPU, QuEST exact, and UPMEM simulator on MVP-sized workloads. |

Run from the project root:

```bash
.venv/bin/python scripts/run_suite.py benchmarks/suites/tn_vs_quest_scale.yaml
.venv/bin/python scripts/run_suite.py benchmarks/suites/three_backend_mvp.yaml
```

Run one config:

```bash
.venv/bin/python scripts/run_benchmark.py benchmarks/configs/tn_cpu/bell_2q.yaml
```

## Outputs

Each run produces:

```text
runs/<experiment_id>/
+-- input_config.yaml
+-- resolved_config.yaml
+-- task_graph.json
+-- execution_log.json
+-- validation_record.json
+-- metrics.jsonl
```

Each suite produces:

```text
results/<suite_id>/
+-- summary.csv
+-- summary.json
+-- summary.md
+-- speedup_vs_fastest.svg
+-- energy_joules.svg
```

The summary extractor is intentionally extensible: add a decorated metric
function in `src/tnsim/results/summary.py`, and it will appear in CSV, JSON, and
Markdown outputs without changing the runner.
