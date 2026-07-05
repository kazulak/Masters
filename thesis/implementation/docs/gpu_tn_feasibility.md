# GPU Tensor-Network Feasibility Plan

This document defines the next tensor-network parallelization milestone after
the implemented CPU slicing, frontier, and diagnostic hybrid routes. It is a
feasibility plan, not a backend implementation claim.

## Current Boundary

The repository already has a verified GPU route:

- `quest_gpu_full_state_exact`

That route is a QuEST HIP/CUDA full-state simulator route. It is useful for
CPU/GPU full-state comparison, but it is not a GPU tensor-network route and
must not be reported as GPU TN evidence.

The current tensor-network routes are CPU routes:

- `quimb_tn_exact`: serious unsliced CPU TN baseline.
- `quimb_tn_sliced_exact`: executed Quimb/cotengra slicing evidence.
- `cpu_tn_frontier_exact`: diagnostic internal frontier TaskGraph evidence.
- `cpu_tn_hybrid_sliced_frontier_exact`: diagnostic internal hybrid evidence.

## Feasibility Goal

Determine the smallest credible path to one real GPU tensor-network benchmark
route. A future route may be added only after a backend proves all of these:

- it executes tensor-network work on a GPU, not CPU fallback;
- it records GPU device/toolchain metadata;
- it records synchronization/timing scope;
- it validates against an existing exact baseline where output comparison is
  available;
- it emits `parallelism_mode` and `parallelism_evidence_type` fields that do
  not overclaim speedup.

No GPU TN benchmark row should be emitted until those checks pass.

## Candidate Backends

| Candidate | Category | Current fit | Evidence needed before route | Notes |
|---|---|---|---|---|
| NVIDIA cuQuantum cuTensorNet Python | tailored GPU TN library | Strong candidate for NVIDIA cluster. Not usable on the current AMD-only machine without NVIDIA CUDA. | Import `cuquantum`, construct/contract a small TN on GPU, verify no CPU fallback, record device and library versions. | NVIDIA documents cuTensorNet for high-performance tensor-network computations and Python `cuquantum.tensornet` APIs such as `contract`, `contract_path`, and `einsum`. |
| CUDA-Q `tensornet` target | tailored quantum GPU TN simulator | Strong candidate for NVIDIA cluster if CUDA-Q is available. | Run deterministic circuits with `tensornet`, record backend/device metadata, compare against QuEST/Quimb on small cases. | CUDA-Q documents a `tensornet` Tensor Network simulator for shallow-depth/high-width exact circuits, with multi-GPU/multi-node scope. |
| Qiskit Aer `tensor_network` GPU method | tailored simulator using cuTensorNet | Strong candidate for NVIDIA cluster if Aer GPU build is available. | Build/install GPU-enabled Aer, verify `device="GPU"` and `method="tensor_network"`, compare output on shared deterministic circuits. | Qiskit Aer documents `tensor_network` as GPU-only and accelerated by cuTensorNet APIs. |
| Quimb/cotengra with CuPy or another GPU array backend | integration candidate | Useful only if contractions actually execute on GPU and synchronization can be controlled. | Prove tensor operands and contractions stay on GPU, record backend array type, synchronize before timing, validate output. | This keeps the current Quimb route family but requires careful no-CPU-fallback checks. |
| CuPy ROCm generic tensor contraction | AMD feasibility candidate, not preferred SOTA quantum TN baseline | Possible local AMD experiment, but not a tailored quantum TN simulator. | Prove CuPy ROCm install, run small contraction on AMD GPU, label as generic tensor GPU if used. | CuPy ROCm support is experimental and should not be presented as the main SOTA quantum TN path. |

## Evidence Contract For A Future Route

A future GPU TN route should use a new additive route ID chosen during
implementation. Candidate naming should not be finalized here. Required row
semantics:

- `execution_model=tensor_network`
- `contraction_execution_target=gpu`
- `accelerator_kind=amd_gpu` or `nvidia_gpu`
- `backend_family` names the actual library family, not just `gpu`
- `gpu_backend_verified=true`
- `gpu_program_executed=true`
- `cpu_fallback_used=false`
- `parallelism_mode` reflects the executed mode, such as `slicing`,
  `intra_contraction`, or `not_applicable` if the library hides the plan
- `parallelism_evidence_type=executed`
- `validation_method` distinguishes full exact validation from metrics-only
  performance rows
- timing fields distinguish wall time, compute/simulation time, transfer/setup
  time where available, and synchronization status

## Feasibility Probe Output

`simulation-backend-probe` reports GPU TN feasibility separately from existing
full-state GPU verification under:

```text
gpu_probe.gpu_tensor_network_probe
```

This section is intentionally feasibility-only. It records candidate GPU TN
backends and blockers, but it must keep:

- `gpu_tn_backend_route_added=false`
- `gpu_tn_benchmark_records_emitted=false`
- `quest_gpu_full_state_is_gpu_tn_evidence=false`

Candidate rows include:

- `candidate_id`
- `candidate_category`
- `execution_model`
- `backend_family`
- `target_gpu_stack`
- `classification`
- `blocker_reason`
- `tensor_network_gpu_execution_verified`
- `benchmark_route_eligible`
- `benchmark_records_emitted`
- `required_next_evidence`

## Feasibility Checks

The next implementation wave should be an audit/probe, not a full backend:

1. Probe installed packages and tools:
   - `cuquantum`
   - `cudaq`
   - `qiskit_aer`
   - `cupy`
   - CUDA/ROCm device visibility
2. Classify each candidate as:
   - usable now on current AMD ROCm machine;
   - usable later on NVIDIA CUDA cluster;
   - installed but CPU-only in this environment;
   - unavailable;
   - blocked by build/toolchain/device.
3. If a candidate is installed and GPU-visible, run the smallest possible
   tensor-network contraction or circuit example outside the benchmark record
   path first.
4. Write blocker/provenance metadata, but emit zero benchmark rows until a real
   route exists and passes the evidence contract.

## Recommended Next Wave

Wave 2E.62 should be:

**GPU TN Candidate Execution Spike**

Scope:

- use the feasibility probe output to select one candidate environment;
- run one minimal tensor-network GPU execution outside benchmark-row emission;
- do not add a benchmark route yet unless one candidate is installed,
  GPU-visible, and can execute an exact tiny TN path safely with no CPU
  fallback;
- prefer NVIDIA cuTensorNet/CUDA-Q/Aer `tensor_network` for the future NVIDIA
  cluster path;
- treat local AMD ROCm as useful feasibility metadata, not as the primary SOTA
  GPU TN baseline unless a real exact TN backend is proven there.

Non-goals:

- no new GPU TN benchmark rows without verified execution;
- no reuse of `quest_gpu_full_state_exact` as GPU TN evidence;
- no UPMEM work;
- no new dashboard/database/workflow layer;
- no fake speedup fields.

## Wave 2E.62 Implementation Plan

Goal:

Prove whether one GPU tensor-network candidate can execute a minimal exact
tensor-network workload on the available environment, while keeping benchmark
records disabled until the execution contract is satisfied.

Candidate selection order:

1. NVIDIA cluster or CUDA environment:
   - first try `cuquantum_cutensornet`;
   - then `cudaq_tensornet`;
   - then `qiskit_aer_tensor_network`.
2. Current AMD ROCm workstation:
   - treat `cupy_rocm_generic_tensor_contraction` as a local feasibility
     probe only;
   - do not present it as the preferred SOTA quantum TN baseline.
3. Existing Quimb/cotengra CPU routes:
   - only attempt `quimb_cotengra_gpu_array_backend` if tensor storage and
     contraction execution can be proven to stay on GPU.

Required implementation behavior:

- add a small probe function or CLI path only if it reuses
  `simulation-backend-probe`;
- write feasibility/blocker metadata, not benchmark evidence rows;
- run at most one minimal tensor-network contraction or deterministic circuit
  per candidate attempt;
- compare the result against a CPU exact reference when a numeric output is
  available;
- record whether GPU synchronization was performed before timing;
- record package/tool versions and device metadata;
- classify failures as dependency missing, device missing, build/toolchain
  blocked, CPU fallback detected, validation failed, or execution verified.

Required success fields for a candidate marked executable:

- `tensor_network_gpu_execution_verified=true`
- `gpu_program_executed=true`
- `cpu_fallback_used=false`
- `gpu_device_name` present
- `gpu_backend_family` present
- `minimal_tn_validation_status=passed`
- `synchronization_status=synchronized` or an explicit equivalent
- `benchmark_route_eligible=false` until a separate route implementation wave

Hard failure rules:

- if a library silently falls back to CPU, mark `cpu_fallback_detected` and do
  not treat the candidate as verified;
- if only source/API support is detected, keep
  `source_support_is_not_benchmark_evidence=true`;
- if only QuEST full-state GPU works, do not mark GPU TN verified;
- if validation is metrics-only, state that exact output comparability is not
  proven.

Tests for the spike:

- probe output contains the GPU TN section even when no candidate is installed;
- a mocked verified candidate records `tensor_network_gpu_execution_verified`;
- a mocked CPU fallback candidate remains in blocker status;
- no benchmark route is registered by the spike;
- no normalized benchmark records are emitted by the spike;
- existing `quest_gpu_full_state_exact` semantics remain unchanged.

Validation commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simulation-backend-probe
git diff --check -- thesis/implementation
```

Expected output:

A probe artifact or JSON output that makes candidate status explicit. The wave
is successful if it either verifies one minimal GPU TN execution path or records
an exact blocker explaining why no current candidate can be executed. It should
not create thesis benchmark evidence rows yet.

## Source Notes

- NVIDIA cuQuantum documents cuTensorNet as a high-performance tensor-network
  computation library, with examples for serial contraction and distributed
  slicing:
  <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>
- NVIDIA cuQuantum Python documents `cuquantum.tensornet` APIs including
  `contract`, `contract_path`, `einsum`, and `einsum_path`:
  <https://docs.nvidia.com/cuda/cuquantum/latest/python/overview.html>
- CUDA-Q documents a `tensornet` Tensor Network simulator for exact
  shallow-depth/high-width circuits:
  <https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html>
- Qiskit Aer documents a GPU-only `tensor_network` method accelerated by
  cuTensorNet APIs:
  <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html>
- CuPy documents CUDA support and experimental AMD ROCm support:
  <https://docs.cupy.dev/en/stable/install.html>
