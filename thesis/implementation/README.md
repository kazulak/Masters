# Quantum Bench Thesis Runtime

This is the active implementation for the Master's thesis runtime. It is a
tensor-network contraction runtime for quantum circuit simulation on
UPMEM-style PIM. It is not a standalone GEMM demo: GEMM-lowered dense
contractions are the current backbone, while sparse, quantum-structured,
communication-heavy, and fallback work shares remain part of the planned
runtime design.

The host CPU is the planner, orchestrator, dispatcher, validator, and reporter.
UPMEM work is represented as one future runtime category, `upmem_tn_runtime`.
`L1_WRAM`, `L2_SINGLE_DPU_MRAM`, `L3_MULTI_DPU`, and `L4_OUT_OF_SCOPE` are
internal execution classes inside that runtime, not separate final benchmark
routes.

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the stable structure and
[docs/runtime_architecture_map.md](docs/runtime_architecture_map.md) for the
current status and roadmap.

## Setup

Run commands from this directory:

```bash
cd thesis/implementation
git submodule update --init --recursive external/QuEST external/SimplePIM external/PID-Comm
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml
```

For a fresh clone, prefer:

```bash
git clone --recurse-submodules <repo-url>
```

For RAPL energy measurement, use the helper so `sudo` still uses the thesis
virtual environment:

```bash
scripts/run_energy_suite.sh configs/suites/local_energy.yml
```

## Main Commands

Normal benchmark run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml
```

UPMEM and external-library environment checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-env-check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-external-libs-check
```

Task-level PIM bridge evaluation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --suite configs/suites/pim_bridge_eval_quick.yml --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-bridge-eval --suite configs/suites/pim_bridge_eval_quick.yml --backend upmem_sdk_simulator_dense --execute-external --max-executed-tasks-per-case 2
```

Memory-level and frontier analysis:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench pim-frontier-analysis --suite configs/suites/pim_frontier_pressure_quick.yml
```

Thesis benchmark matrix scaffold:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench benchmark-matrix-report --matrix configs/benchmark_matrix.yml
```

Developer one-task dense bridge checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend mock_numpy_dequantized
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend upmem_sdk_simulator_dense --execute-external
```

Developer one-task generic fallback bridge check:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench generic-task-bridge --case bell_2q --task-index 0 --backend upmem_sdk_simulator_generic_loop --execute-external
```

Artifact-driven result comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench compare-results --inputs runs/<run_dir_a> runs/<run_dir_b> --out runs/manual_compare
```

## Directory Map

```text
implementation/
  configs/                 benchmark suites and thesis matrix inputs
  external/                implementation-local external Git submodules
  native/quest_cpu/        QuEST C runner for CPU full-state baseline
  native/upmem/            UPMEM SDK and future SimplePIM/native bridge code
  scripts/                 helper commands
  src/quantum_bench/       Python runtime package
  tests/                   pytest suite
  runs/                    generated benchmark artifacts, not source
```

`implementation/external` is canonical and populated by Git submodules. The
thesis implementation should be movable without depending on `../legacy/extern`.
Legacy external trees are historical fallback only.

## Current Status

| Area | Status | Evidence boundary |
|---|---|---|
| CPU exact TN | Implemented | Full tensor output, validation-capable |
| QuEST CPU full-state | Implemented baseline | Metrics-only, benchmark-only |
| UPMEM L1 dense | Implemented subset | Task-level UPMEM SDK simulator, split-complex supported when layout is explicit |
| UPMEM L2 dense | Implemented subset | Task-level UPMEM SDK simulator, real-valued only |
| UPMEM generic fallback | Implemented MVP | Task-level UPMEM SDK simulator, real-valued small binary contractions only |
| UPMEM L3 distributed | Model-only | No execution yet |
| SimplePIM GEMM | Candidate | Target UPMEM compute/runtime abstraction for L1/L2 and local tile compute inside L3; not integrated |
| PID-Comm | Candidate | Communication/orchestration substrate across L1/L2/L3, strongest for L3 distributed contraction; not integrated |
| GPU TN/full-state | Planned | Not implemented |
| Sparse/irregular PIM | Planned | SparseP/PRISM/PyGim are references only |

Current UPMEM results are task-level simulator evidence. They are not
full-circuit speedup numbers.

## Evaluation Boundary

Use these rules when interpreting artifacts:

- `cpu_tn_einsum_exact` is the current exact tensor-network reference.
- `quest_cpu_full_state_benchmark` is an external metrics-only baseline.
- `upmem_sdk_simulator_dense` is a developer bridge backend ID, not a final
  benchmark route.
- `upmem_sdk_simulator_generic_loop` is an unoptimized developer bridge backend
  for small binary contractions. It exists for correctness coverage and route
  validation, not performance evidence.
- `upmem_tn_runtime` is the only final UPMEM benchmark category.
- L1/L2/L3 are internal scheduler classes within `upmem_tn_runtime`.
- Shadow CPU fallback runs are diagnostics only and must not be presented as
  UPMEM execution.
- Synthetic pressure workloads are analysis-only and must not enter normal
  benchmark execution.
- Generated artifacts belong under `runs/`; curated thesis results should be
  copied deliberately later, not inferred from every local run.

## External Candidate Libraries

External libraries are tracked as implementation candidates or baselines:

| Candidate | Intended role |
|---|---|
| SimplePIM | Target UPMEM compute/runtime abstraction for L1/L2 and local tile compute inside L3 |
| PID-Comm | Target communication/orchestration substrate across L1/L2/L3, strongest for L3 distributed contraction |
| ATiM | SLR-derived tensor-kernel autotuning candidate; not locally cloned yet |
| SparseP, PRISM, PyGim | Sparse and irregular PIM kernel references |
| PIM-LLM GEMM | Optimized GEMM kernel design reference |
| TransPimLib | Optional special-math support candidate |
| Native UPMEM SDK | Current control/fallback implementation |

Kernel-family usage in artifacts should be interpreted before any timing
comparison. `dense_gemm`, `generic_loop_fallback`, sparse, structured, and
communication/collective work shares are separate parts of the intended runtime;
current generic-loop results are task-level simulator evidence only.

See [external/EXTERNAL_SOURCES.md](external/EXTERNAL_SOURCES.md) for submodule
URLs, pinned commits, roles, and update policy.

## Developer Diagnostics

These commands are useful while building the runtime, but they are not final
benchmark claims:

- `dense-route-coverage`: reports how far each task can progress through dense
  preparation and bridge eligibility.
- `shadow-routed-runtime`: executes the full graph with CPU fallback
  authoritative while recording what-if route evidence.
- `simplepim-microbench`: dry-run SimplePIM dense scaffold; no SimplePIM kernel
  execution.
- `compare-planners`: compares contraction planner outputs and modeled UPMEM
  pressure.
- `probe`: reports registered route/provider metadata.

## Cleanup Policy

- Keep source, configs, tests, and docs in the repo.
- Keep `external/` as implementation-local submodule dependencies.
- Treat `runs/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, and native build
  outputs as generated.
- Do not delete historical runs without explicitly deciding which results are
  still needed for the thesis.
