# UPMEM Tensor-Network Quantum Simulation

This repository is the thesis implementation for benchmarking tensor-network
(TN) quantum-circuit simulation across direct CPU/TN baselines and UPMEM.
It is a research prototype, not a general-purpose quantum simulator.

## Active Flow

```text
SimulationJob
  -> target-neutral TensorNetwork
  -> planner path
  -> ContractionDAG
  -> direct NumPy / Quimb / cotengra / QuEST execution
     or UpmemPlan -> ABI-v4 UPMEM runtime
  -> canonical evidence
  -> report
```

`TensorNetwork` is semantic and non-executable: it describes tensors,
connectivity, inputs, and the requested output. `ContractionDAG` is the sole
logical execution IR: it records the selected contraction order, explicit
dependencies, local slicing branches, and reductions. It contains no DPU
placement, tiles, kernels, scales, binary paths, or machine settings.

`UpmemPlan` is a target-specific mapping of a DAG. It selects the numeric
policy, output/K tiles, topology, stages, and kernel policy without changing
the logical plan. ABI-v4 executes real-valued tiles; complex contractions use
four real products under the selected split-complex policy.

## Commands

All commands run from `thesis/implementation`.

```bash
# Create or update ../.venv, build the CPU QuEST runner, then inspect tools.
make setup
make doctor
make test

# Default software benchmark configuration.
make plan CONFIG=configs/tn_benchmark_reset.yml OUTPUT=runs/reset-plan
make run CONFIG=configs/tn_benchmark_reset.yml OUTPUT=runs/reset-run
make verify INPUT=runs/reset-run
make report INPUT=runs/reset-run REPORT_OUTPUT=runs/reset-report

# Build the active UPMEM ABI-v4 host and DPU binaries.
make build-upmem-runtime UPMEM_TASKLETS=1

# Physical execution is opt-in. Prepare an ignored target-specific copy first.
PYTHONPATH=src ../.venv/bin/python scripts/qualify_m7c_physical.py prepare \
  --template configs/tn_benchmark_physical_smoke.yml \
  --output runs/configs/eth/one-dpu-float32.yml \
  --mode float32-smoke \
  --rank-path /dev/dpu_rank0 \
  --session-root runs/upmem_sessions/eth-one-dpu \
  --expected-cpus 0
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 ../.venv/bin/python -m quantum_bench.cli qualify \
  --config runs/configs/eth/one-dpu-float32.yml \
  --output runs/evidence/eth-one-dpu-float32 \
  --allow-physical
PYTHONPATH=src ../.venv/bin/python scripts/qualify_m7c_physical.py inspect \
  --input runs/evidence/eth-one-dpu-float32 \
  --expected-samples 6 --expected-sessions 6 \
  --numeric-policy split_complex_float32_v1

```

`configs/tn_benchmark_reset.yml` is the default CPU/TN software smoke suite.
`configs/tn_benchmark_physical_smoke.yml` is the one-DPU physical float32
smoke template. Do not edit it for a target machine. The M7C physical
preparation script creates an ignored copy below `runs/configs/eth/`, resolving
the template paths before it writes target-specific binary, session, affinity,
and rank paths. The probe mode emits one float32 measurement; float32 smoke
emits one warmup plus five measurements; int8 smoke is descriptive only.
`configs/m7c_workload_selection.json` preregisters the source-only candidate
selection for the later scaling diagnostic. It selects the deterministic
18-qubit quantization-stress circuit as the primary kernel-scaling workload and
the structurally different 18-qubit GHZ chain as confirmatory evidence; neither
choice used simulator or physical timing. Supply the later scaling configuration
explicitly and check it before an ETH run:

```bash
PYTHONPATH=src ../.venv/bin/python scripts/select_m7c_workload.py --check \
  configs/m7c_workload_selection.json \
  --config configs/tn_benchmark_physical_scaling_diagnostic.yml
```

`plan` writes a deterministic experiment plan without execution. `run` writes
canonical evidence. `verify` checks evidence identities and integrity.
`report` only reads existing evidence and produces tables and plots.

Manifests use `evidence_manifest_v2`, samples use `evidence_sample_v3`,
sessions use `evidence_session_v1`, and reports use `evidence_report_v5`.
Earlier sample evidence is unsupported. Sample `status`
describes whether the complete attempt finished: a validator exception produces
a failed sample, while a policy-reference or accuracy qualification miss
remains a successful sample with its measurement and facts retained.
Policy-reference correctness is reported separately from `accuracy_qualified`.

The collection policy records deterministic warmup and measurement blocks,
their execution order, and fresh-session lifecycle. Reports retain attempted,
successful, failed, and unsupported measurement counts; summarize successful
measurements with median, raw MAD, and a deterministic percentile-bootstrap
interval; and reserve block-paired speedup intervals for admissible physical
comparisons. Topology scaling is emitted separately in `scaling.csv`, never in
CPU-versus-UPMEM `speedups.csv`. Scaling rows retain the semantic/logical,
numeric, kernel, validation, collection, physical-plan, executable, resource,
and dominant-work admission identities; primary comparisons begin from one
DPU or one tasklet, with other increasing-resource pairs marked secondary.
SDK-simulator reports remain diagnostic-only.

## Evidence and Comparison Rules

Each attempted warmup or measurement creates a canonical sample record.
Manifests record problem, logical-plan, physical-plan, executable,
environment, experiment, session, and sample identities. Failed and
unsupported attempts remain visible.

Timing scopes are explicit. `steady_execution_v1` excludes planning and
session lifecycle; `simulation_end_to_end_v1` includes route-specific
preparation through decoded output. Reference calculation, validation,
hashing, and report writing are outside both scopes.

Only evidence with compatible problem, logical-plan relationship, timing
scope, validation, and physical provenance can support a performance claim.
SDK-simulator runs are correctness evidence only.

## Current Capability Boundary

Controlled software tests cover the active CPU/TN and SDK-simulator paths,
including ABI-v4 WRAM-panel float32/int8 simulator checks against CPU replay. Tag
`thesis-m6-software-ready-v1` and its GitHub release bundle exist, establishing
completed M6 software qualification. The reset physical UPMEM route remains
pending and has **not** yet been qualified on ETH hardware. The repository
therefore makes no reset-route claim of speedup, energy efficiency, multi-rank
scaling, broad graph residency, or hardware-calibrated planning.

The active native kernel is `dpu_real_tile_v4_wram_panel_v1`: a bounded dense
real-tile kernel with global shared B panels and tasklet-indexed A/output WRAM
buffers. Tag `thesis-m7a-wram-kernel-software-ready-v1` and its release bundle
record exact-head SDK-simulator qualification for this kernel. That evidence is
software/simulator-qualified only and does not establish a physical timing,
scaling, energy, or kernel-competitiveness result.

The active native runtime uses pinned SimplePIM management types and its
initialization kernel around raw-SDK allocation and dispatch. It is not yet a
qualified high-level SimplePIM scheduler or compute route. The retained
`native/upmem/pidcomm_qualification/` source is standalone future source;
PID-Comm is not an active communication provider or public command. ATiM is
not integrated.

## Repository Layout

```text
src/quantum_bench/
  model.py, circuits.py, lowering.py, planning.py, numerics.py, results.py
  cpu.py, baselines.py, experiment.py, evidence.py, report.py, cli.py
  upmem/plan.py, tiling.py, protocol.py, native_session.py, runtime.py

native/upmem/runtime/       active ABI-v4 host/DPU build
native/upmem/pidcomm_qualification/  standalone compatibility harness
configs/tn_benchmark_reset.yml       software benchmark suite
configs/tn_benchmark_physical_smoke.yml  one-DPU physical smoke template
configs/m7c_workload_selection.json  preregistered M7C workload selection
```

Generated evidence is written below `runs/` and is ignored by Git. Reviewed
historical snapshots remain in `thesis_results/`; reports never edit them.

See [ARCHITECTURE.md](ARCHITECTURE.md) for ownership boundaries and
[STATUS.md](STATUS.md) for implemented versus pending capabilities.
