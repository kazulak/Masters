# Quantum Bench Implementation

Active implementation for the Master's thesis runtime described in
`../CODEX_IMPLEMENTATION_DIRECTION.md` and
`../CODEX_UPMEM_ARCHITECTURE_DIRECTION.md`.

The runtime is route-aware: it builds exact tensor networks from quantum
circuits, plans contractions, asks each provider route whether it can execute,
records skip reasons, validates numerical output, and writes benchmark artifacts
under timestamped `runs/` directories.

The detailed Host-CPU + UPMEM/SimplePIM architecture scaffold is documented in
`docs/runtime_architecture_map.md`. Read it before planning route, target,
planner, or native-code changes.

## Quick Start

```bash
cd thesis/implementation
PYTHONPATH=src ../.venv/bin/python -m pytest -q
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-env-check
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-env-check --run-sample --target simulator
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simplepim-microbench --dry-run --m 8 --k 8 --n 8
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend mock_numpy_dequantized
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 1 --materialization cpu-replay --backend mock_numpy_dequantized
SIMPLEPIM_STUB_BIN=native/upmem/simplepim/simplepim_dense_stub.py PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend simplepim_external_stub --execute-external
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend upmem_sdk_simulator_dense --execute-external
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-route-coverage --case bell_2q
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-route-coverage --suite configs/suites/planner_compare.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench shadow-routed-runtime --case bell_2q
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench shadow-routed-runtime --case bell_2q --shadow-route-policy dense-if-estimate-supported
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench shadow-routed-runtime --case ghz_chain --n-qubits 3 --shadow-route-policy dense-if-no-tiling
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench plot runs/latest
```

For RAPL energy measurement, run the suite with the helper so sudo still uses
the thesis virtual environment:

```bash
scripts/run_energy_suite.sh configs/suites/local_energy.yml
```

## Active Boundaries

- `src/quantum_bench/` contains the Python implementation.
- `configs/suites/` contains the reproducible benchmark suite definitions.
- `native/quest_cpu/` contains only the C QuEST runner used by the
  `quest_cpu_full_state_benchmark` provider.
- `external/QuEST/` contains the local QuEST dependency used by that runner.
- `src/quantum_bench/formats/` contains shared host-side data-format conversion
  records and deterministic fixed-point utilities.
- `src/quantum_bench/targets/upmem/` contains host-side UPMEM WRAM, traffic,
  tile-plan, schedule, SimplePIM probe, dense bridge manifests, and explicit
  SimplePIM dry-run microbenchmark and environment bring-up groundwork shared
  by future UPMEM providers.
- `src/quantum_bench/routing/` contains the task-level route contract,
  analysis-only dynamic router skeleton, shadow route policy records, and
  one-task dense preparation boundary for future UPMEM/SimplePIM execution.
- `src/quantum_bench/tn/materialize.py` contains developer-facing TaskGraph
  CPU replay for materializing one selected task's inputs. It is not the normal
  CPU execution path.
- `src/quantum_bench/bench/dense_task_bridge.py` is a developer-only one-task
  harness that connects a real `ContractionTask` to dense preparation and the
  mock bridge. It can opt into CPU replay for explicit task selection, but it
  does not change normal benchmark routing.
- `src/quantum_bench/bench/dense_route_coverage.py` is a developer-only
  readiness analyzer over every task in selected circuits. It reports dense
  route materialization, preparation, tile-plan, bridge-manifest, and optional
  stub-contract coverage; it is not routed execution.
- `src/quantum_bench/bench/shadow_routed_runtime.py` is a developer-only full
  TaskGraph harness. It executes every task with CPU fallback as the
  authoritative numeric route while recording dense route preparation and
  optional capped bridge/stub evidence as shadow metadata. Its shadow route
  policy fields are what-if decisions only; they never replace CPU fallback.
- `native/upmem/simplepim/` is reserved for future SimplePIM bridge code and
  currently contains the non-executing external contract stub plus the first
  UPMEM SDK simulator dense bridge runner. The runner is in the
  SimplePIM/UPMEM bridge lane but does not use SimplePIM APIs.
- `native/upmem/raw_dense/` is reserved for future raw UPMEM SDK dense kernels.
- `../legacy/` contains old prototypes and generated sudo-owned run folders kept
  out of the active implementation.

## UPMEM/SimplePIM Environment Check

`upmem-env-check` verifies local UPMEM/SimplePIM bring-up before real dense
backend work starts. It records UPMEM SDK tools, SimplePIM source discovery,
safe bounded command probes, and optional simulator sample build/run results
under `runs/<timestamp>_upmem_env_check/`.

SimplePIM discovery checks `--simplepim-home`, then `SIMPLEPIM_HOME`, then the
repo fallback `../legacy/extern/SimplePIM` from this directory. Without
`--run-sample`, simulator and hardware execution are recorded as not verified.
Sample build or simulator-run failure is written as JSON status `failed`, but
the CLI still exits normally after writing the artifact so the failure is
auditable.

## UPMEM SDK Simulator Dense Backend

`upmem_sdk_simulator_dense` is the first real PIM-backed dense bridge backend.
It is explicit and simulator-only:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend upmem_sdk_simulator_dense --execute-external
```

The backend consumes the existing dense bridge manifest, runs a minimal UPMEM
SDK DPU GEMM program through the simulator, writes
`outputs/upmem_sdk_simulator_output.npy`, and validates against
`references/expected_dequantized_output.npy`. It is not a normal suite route,
does not run hardware, does not implement tiling, and does not make dense output
authoritative in the shadow runtime. Any timings it records are bring-up
timings, not performance evidence.
