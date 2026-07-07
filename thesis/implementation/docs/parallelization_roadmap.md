# Tensor-Network Parallelization Roadmap

This roadmap defines thesis-safe goals for tensor-network contraction
parallelization across CPU, GPU, and UPMEM/PIM. It is a planning artifact, not
an implementation claim. Implementation strategy should be chosen in later
waves after the goals and evidence requirements here are accepted.

The staged implementation strategy is tracked in
[parallelization_implementation_strategy.md](parallelization_implementation_strategy.md).

## Current Baseline

The current implementation already has useful execution evidence, but it does
not yet implement production tensor-network parallel execution:

- `quimb_tn_exact` is the serious unsliced CPU tensor-network baseline.
- `quimb_tn_sliced_exact` provides executed Quimb/cotengra slicing evidence
  with explicit slicing metadata and single-worker slice reconstruction.
- `cpu_tn_einsum_exact` remains a diagnostic sequential internal TaskGraph
  baseline.
- `cpu_tn_frontier_exact` provides executed diagnostic internal TaskGraph
  frontier scheduling evidence.
- `cpu_tn_hybrid_sliced_frontier_exact` provides executed diagnostic internal
  slice-aware TaskGraph plus frontier scheduling evidence on tiny cases.
- UPMEM strict TaskGraph runtime is sequential over tasks.
- UPMEM frontier/wave parallelism is modeled and reported, including the
  `upmem-multi-dpu-assignment` assignment-plan command; it is not executed
  multi-DPU work.
- QuEST GPU is a verified full-state GPU baseline, not a GPU tensor-network
  baseline.
- UPMEM SDK simulator evidence proves strict SDK DPU code-path execution, not
  hardware speedup.

These limitations are acceptable if they are reported explicitly. The purpose
of this roadmap is to define what evidence would be needed before making
stronger parallelism claims.

## Terminology And Evidence Model

| Mode | What is parallelized | Redundant work risk | Evidence needed | Allowed claim | Not allowed |
|---|---|---|---|---|---|
| Slicing-based parallelism | Independent slices of one contraction path. | FLOP ratio/change, repeated boundary contractions, slice reconstruction overhead. | Slice config, slice count, FLOP ratio/change, peak memory, reconstruction validation, wall/compute timing. | "Slicing exposes independent work under this configured path." | Claiming no redundant work, or claiming speedup without matched unsliced baseline. |
| Tree-node/frontier parallelism | Independent ready TaskGraph nodes or subtrees. | Duplicate input materialization, repeated validation, scheduler overhead, poor frontier width from serialized paths. | Frontier widths, executed concurrent tasks, dependency proof, no duplicate contractions, final equivalence. | "The TaskGraph scheduler executed independent contractions concurrently." | Treating modeled frontier width as executed parallel speedup. |
| Intra-contraction parallelism | Tiles or partitions inside one contraction. | Partial-output reduction cost, transfer overhead, synchronization overhead, load imbalance. | Tile/partition plan, worker count, reduction plan, transfer and sync timing, validation. | "This contraction used intra-task workers/DPUs/threads." | Claiming full TN parallelism from a single tiled contraction alone. |
| Hybrid slicing + frontier | Slices and independent TaskGraph nodes run concurrently. | Multiplicative task growth, FLOP ratio/change plus scheduler overhead. | Slice IDs, frontier wave IDs, reconstruction validation, task duplication checks. | "Hybrid mode combines slice-level and graph-level parallelism." | Comparing against the wrong baseline or hiding slicing cost changes. |
| CPU parallelism | CPU threads, BLAS threads, processes, slicing workers, or frontier workers. | Oversubscription, nondeterministic thread settings, BLAS hidden parallelism. | Thread/env settings, process count, BLAS backend, repeat statistics. | "CPU run used controlled host parallelism." | Comparing against uncontrolled thread counts. |
| GPU parallelism | GPU full-state simulation or future GPU TN kernels. | Host/device transfer, asynchronous timing errors, CPU fallback hidden as GPU. | Verified device metadata, synchronization status, no CPU fallback, compute vs wall timing. | "Verified GPU execution for this execution model." | Calling QuEST full-state GPU a GPU TN result. |
| UPMEM/PIM parallelism | DPU task assignment, DPU groups, or intra-DPU kernels. | Host orchestration overhead, transfer cost, quantization/dequantization, DPU sync/reduction. | DPU program execution, DPU count, task assignment, transfer/quantization/sync timing, no CPU contraction fallback. | "Strict UPMEM/PIM path executed these tasks under this mode." | Reporting SDK simulator timing as hardware speedup. |

## Goal Areas

### CPU Tensor-Network Parallelism Goals

- Define explicit Quimb/cotengra slicing configuration in artifacts before
  making slicing claims.
- Record CPU thread and BLAS settings for every CPU TN benchmark row.
- Optionally prototype an internal TaskGraph frontier scheduler to compare
  sequential task execution against graph-level concurrency.
- Compare sequential TaskGraph, frontier TaskGraph, and slicing under matched
  circuits and validation settings.
- Measure FLOP ratio/change, peak/intermediate memory, wall time, compute time,
  scheduler overhead, and output accuracy.

### GPU Full-State And Tensor-Network Goals

- Keep `quest_gpu_full_state_exact` as the verified full-state GPU baseline.
- Treat GPU TN as a separate future capability requiring a real TN-capable GPU
  backend, such as cuTensorNet/cuQuantum/CUDA-Q/Qiskit Aer GPU or another
  verified exact GPU TN path.
- Track candidate selection and evidence gates in
  [gpu_tn_feasibility.md](gpu_tn_feasibility.md).
- Separate GPU full-state speedup claims from GPU TN speedup claims.
- Support an optional NVIDIA cluster path only after device/tool metadata and
  no-CPU-fallback checks are in place.
- Record GPU backend, device, runtime/toolkit, synchronization status, timing
  scope, and validation method for every GPU row.

### UPMEM/PIM Parallelism Goals

- Keep strict sequential UPMEM SDK simulator TaskGraph execution as the current
  baseline.
- Continue using frontier/wave analysis as modeled evidence until execution is
  actually parallelized.
- Use [upmem_multi_dpu_scheduling_design.md](upmem_multi_dpu_scheduling_design.md)
  as the evidence contract before adding multi-DPU execution.
- Eventually implement true multi-DPU task assignment for inter-task and
  intra-task parallelism.
- Distinguish host-level task parallelism from intra-DPU kernel parallelism and
  multi-DPU intra-contraction distribution.
- Measure host transfer cost, quantization cost, dequantization cost, DPU
  program time, synchronization/reduction cost, and host orchestration cost.
- Avoid hardware-speedup claims for SDK simulator runs.

### Hybrid Parallelism Goals

| Hybrid mode | Expected benefit | Main risk | Baseline | Artifact fields needed |
|---|---|---|---|---|
| Slicing only | Lower memory, independent slice work. | FLOP ratio/change and reconstruction overhead. | Unsliced same route. | slice_count, sliced_indices, slicing_flop_ratio, reconstruction_status. |
| Frontier only | Use independent TaskGraph nodes. | Serialized paths may expose little width. | Sequential TaskGraph route. | frontier_wave_id, ready_width, executed_parallel_tasks, scheduler_overhead_s. |
| Slicing + frontier | Expose more independent work. | Task explosion and duplicated work. | Slicing-only and frontier-only. | slice_id, frontier_wave_id, duplicate_contraction_check. |
| Slicing + UPMEM task execution | Bound per-task memory while keeping UPMEM strict path. | Host transfer and quantization may dominate. | Sequential UPMEM same route. | slice_transfer_bytes, quantization_time_s, dpu_invocations. |
| Frontier + multi-DPU execution | Execute ready tasks across DPU groups. | Load imbalance and host scheduling overhead. | Sequential UPMEM same route. | dpu_group_id, task_assignment, dpu_occupancy, sync_time_s. |
| Slicing + frontier + intra-task | Maximum exposed parallel work. | Hardest to attribute and validate. | All simpler modes. | mode_components, per_component_timing, reconstruction_validation. |

## Benchmark And Test Goals

Benchmark tiers should stay separate:

- Correctness tier: small circuits with exact output comparison.
- Performance tier: larger circuits with clearly documented timing scope and
  validation method.
- Scalability tier: sweeps by qubits, depth, task count, slice count, or DPU
  count.
- Boundary tier: first unsupported case and precise blocker reason.
- Simulator-only tier: SDK simulator or model-only evidence, never hardware
  speedup.
- Hardware tier: real device execution with hardware timing and device metadata.

Tests needed before claiming executed parallelism:

- no duplicated or missing contractions;
- deterministic slice reconstruction;
- exact output equivalence where full validation is available;
- metrics-only rows do not pretend to be exact validation;
- CPU fallback detection;
- GPU backend verification and synchronization checks;
- UPMEM strict-path verification with `cpu_fallback_used=false`;
- task assignment and frontier dependency invariants.

## Reporting Goals

Thesis-facing outputs should include:

- parallelism mode comparison table;
- slicing FLOP ratio/change table;
- frontier concurrency table;
- hybrid mode table;
- CPU/GPU/UPMEM route capability matrix;
- supported/unsupported UPMEM boundary table;
- timing breakdown table;
- accuracy and validation table.

Each report must distinguish measured execution, modeled opportunity, simulator
evidence, and hardware evidence.

## Prioritized Goal Ladder

| Goal | Purpose | Route/backend | Required tests | Expected artifacts | Thesis value | Blockers/risks | Status language |
|---|---|---|---|---|---|---|---|
| G0 | Document semantics and current baseline. | Existing CPU/GPU/UPMEM routes. | Doc/link checks; existing tests remain green. | This roadmap; architecture links. | Prevents overclaiming. | Stale docs. | Baseline documentation goal. |
| G1 | Add explicit slicing configuration and evidence goals. | Quimb/cotengra CPU TN route. | Slice reconstruction; exact output equivalence. | Slicing config rows; FLOP ratio/change table. | Enables slicing-specific claims. | Cotengra config complexity; fair baselines. | Implemented as explicit slicing evidence; required for slicing claims. |
| G2 | Define internal sequential-vs-frontier scheduler prototype goals. | Internal TaskGraph route. | DAG dependency invariants; no duplicate tasks. | Frontier execution logs; scheduler timing. | Tests graph-level parallelism independently of slicing. | Scheduler complexity; limited frontier width. | Implemented as diagnostic experimental evidence. |
| G3 | Compare CPU slicing vs frontier experimentally. | CPU TN slicing route and frontier prototype. | Matched validation; controlled CPU thread settings. | Comparison CSV/Markdown/plots. | Separates parallelism sources. | Oversubscription; unfair timing. | Implemented diagnostically; required for hybrid claims. |
| G4 | Define slicing + frontier hybrid experiment goals. | CPU TN hybrid prototype. | Slice/frontier reconstruction; duplicate-work checks. | Hybrid mode table; per-component timing. | Shows composability of parallel modes. | Slicing cost changes and scheduler overhead. | Implemented diagnostically on tiny cases; not a serious baseline. |
| G5 | Define GPU SOTA feasibility or NVIDIA cluster route goals. | Existing QuEST GPU full-state plus future GPU TN backend. | GPU verification; no CPU fallback; synchronization checks. | GPU capability/provenance table; `simulation-backend-probe` GPU TN feasibility section. | Separates full-state GPU and future GPU TN evidence. | Vendor stack availability; CUDA/ROCm mismatch. | Feasibility probe implemented; next step is candidate execution spike, not benchmark rows. |
| G6 | Define UPMEM multi-DPU scheduling design goals. | UPMEM TaskGraph/runtime design. | Task assignment invariants; strict no CPU fallback. | DPU scheduling design; modeled assignment report; modeled/executed comparison fields. | Required before PIM parallelism claims. | Hardware/tooling, PID-Comm integration, reduction costs. | Design and modeled assignment report implemented; executed multi-DPU remains future and optional for current thesis if limitations are documented. |
| G7 | Define UPMEM multi-DPU prototype goals. | Future UPMEM hardware or simulator-supported multi-DPU path. | DPU invocation count; transfer/sync timing; final validation. | Hardware/simulator rows with DPU assignments. | First executed PIM parallelism evidence. | Hardware access and tooling reliability. | Hardware/tooling-dependent. |
| G8 | Define thesis-ready combined comparison goals. | All verified routes only. | Report invariants; validation and timing scope checks. | Final matrix tables/plots. | Connects CPU, GPU, TN, and UPMEM evidence. | Methodology drift; mixed evidence types. | Final comparison goal only after relevant modes exist. |

## Implementation Strategy Deferred

This roadmap intentionally does not choose the implementation sequence, APIs, or
worker model. Later implementation waves should decide those details only after
selecting the specific claim they need to support: slicing, frontier execution,
GPU TN execution, UPMEM multi-DPU execution, or a hybrid combination.
