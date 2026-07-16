# System Architecture

## Research Objective

The implementation evaluates tensor-network (TN) quantum-circuit simulation on
UPMEM processing-in-memory devices. The immediate objective is not a production
simulator. It is a controlled research system in which circuit semantics,
contraction plans, executor configuration, validation, and measurements remain
separable enough to attribute a result to the component being studied.

The central invariant is:

```text
one circuit semantics + one tensor network + one contraction plan
                         -> multiple explicit executors
                         -> normalized, comparable evidence
```

## End-To-End Data Flow

```mermaid
flowchart LR
    C[Quantum circuit<br/>six PIMutation-compatible families<br/>or controlled synthetic case]
    TN[Tensor-network lowering<br/>tensor values, labels, output order]
    P[Contraction planner<br/>opt_einsum / cotengra / custom UPMEM greedy]
    G[TaskGraph<br/>binary contractions + dependencies]
    B[Execution bundle<br/>semantic, TN and plan hashes]

    C --> TN --> P --> G --> B

    B --> CPU[Internal CPU TaskGraph replay]
    C --> QCPU[QuEST CPU full-state]
    C --> QGPU[QuEST HIP/CUDA full-state]
    C --> QUIMB[Quimb/cotengra CPU TN]
    B --> UPMEM[Strict UPMEM TaskGraph runtime]

    CPU --> N[normalized_records.jsonl]
    QCPU --> N
    QGPU --> N
    QUIMB --> N
    UPMEM --> N

    N --> R[Research pack<br/>statistics, source CSVs, plots]
    R --> S[Tracked thesis_results/current]
```

QuEST and Quimb are independent serious baselines. The internal TaskGraph is the
shared representation used for direct CPU-replay versus UPMEM comparisons. It
is not asserted that a Quimb contraction tree has the same plan as the internal
TaskGraph unless a future adapter proves that identity explicitly.

## Execution Identity

`src/quantum_bench/tn/execution_bundle.py` serializes the scientific identity
of an internal execution:

```mermaid
flowchart TD
    O[Ordered gate sequence<br/>wires + parameters] --> CH[circuit_semantics_hash]
    CH --> T[Tensor IDs, labels, shapes,<br/>output labels and einsum form]
    T --> NH[tensor_network_hash]
    NH --> CP[Planner identity + ordered path<br/>+ semantic contraction tasks]
    CP --> PH[contraction_plan_hash]

    PH --> EC1[CPU executor config hash]
    PH --> EC2[UPMEM executor config hash]
    PH --> EC3[Quantized UPMEM config hash]
```

Executor settings are deliberately excluded from `contraction_plan_hash`.
Thus float32 and int8 UPMEM runs can prove that they execute the same plan while
retaining distinct `executor_config_hash` values. Timings, host paths, and
machine metadata are also excluded from semantic hashes.

`contraction_path_structure_hash` additionally captures only the ordered
pairwise path and lowered task structure. It is useful when two planners carry
different identities or objective settings but select the same structural path.

## Module Ownership

| Layer | Active modules | Responsibility | Status |
| --- | --- | --- | --- |
| Circuit semantics | `thesis/implementation/src/quantum_bench/circuits/` | Deterministic circuit definitions and QuEST-compatible semantic mapping | Active |
| TN lowering | `thesis/implementation/src/quantum_bench/tn/network.py` | Convert ordered gates into tensors, labels, and output convention | Active, thesis infrastructure |
| Planning | `thesis/implementation/src/quantum_bench/tn/planners.py`, `upmem_planner.py`, `upmem_path_cost.py`, `upmem_path_cost_v2.py`, `task_graph.py` | Obtain standard-library or versioned custom modeled paths and lower them into dependency tasks | Active; v1 historical and v2 projected-prefix UPMEM objectives are modeled, not hardware-calibrated |
| Execution identity | `thesis/implementation/src/quantum_bench/tn/execution_bundle.py` | Canonical serialization and SHA-256 identities | Active, thesis contribution |
| Serious full-state baseline | `thesis/implementation/src/quantum_bench/providers/full_state/` + `thesis/implementation/external/QuEST/` | QuEST CPU and verified GPU execution | Active |
| Serious CPU TN baseline | `thesis/implementation/src/quantum_bench/providers/exact_tn/quimb_tn.py` | Quimb/cotengra unsliced and sliced exact TN execution | Active |
| Shared-plan CPU reference | `thesis/implementation/src/quantum_bench/providers/exact_tn/cpu_einsum.py`, `cpu_path_replay.py` | Execute the internal TaskGraph on CPU | Active; diagnostic/reference quality |
| Strict UPMEM runtime | `thesis/implementation/src/quantum_bench/targets/upmem/taskgraph_runtime.py`, `numeric_reference.py`, `runtime_evidence.py` | Execute policy/scheduling while keeping CPU references, validation, and evidence construction reviewable | Active, SDK simulator |
| Physical UPMEM bring-up | `targets/upmem/hardware_mvp.py`, `bench/upmem_hardware_mvp.py`, `native/upmem/simplepim/upmem_sdk_dense_hardware_mvp_runner.py` | Guarded one-DPU/one-tasklet dense int8 -> int32 exact functionality evidence | Phase 1A; fixed 2x2/4x4 L1 cases only, no speedup or energy claim |
| Physical generic TaskGraph MVP | `targets/upmem/hardware_generic_mvp.py`, `bench/upmem_hardware_generic_mvp.py`, `targets/upmem/generic_bridge.py`, `native/upmem/simplepim/upmem_sdk_generic_loop_runner.py` | Guarded one-DPU/one-tasklet synthetic real rank-3 x rank-3 -> rank-4 generic contraction with two output tiles | Phase 1B; exact functionality evidence only, no general quantum-TN, speedup, energy, scaling, or scheduler claim |
| Native DPU programs | `thesis/implementation/native/upmem/simplepim/` | Dense and bounded generic host/DPU programs | Active, bounded |
| UPMEM analysis | `thesis/implementation/src/quantum_bench/targets/upmem/tile_plan.py`, `schedule.py`, `tn/upmem_path_cost.py`, planner scoring | Estimate transfer, tiling, frontier, assignment pressure, and objective components | Active; execution coverage remains bounded |
| Evidence writer | `thesis/implementation/src/quantum_bench/bench/simulation_backend_compare.py`, `upmem_mvp_benchmark.py` | Run fixed suites and write canonical normalized evidence | Active |
| Derived analysis | `thesis/implementation/scripts/research_benchmark_pack.py` | Statistics, claim guards, source CSVs, and plots | Active |
| Thesis snapshot | `thesis/implementation/scripts/thesis_snapshot.py`, `thesis_runs.py` | Promote compact tracked evidence and prune stale generated runs | Active |

## Route Roles And Claim Boundaries

| Route | Execution | Research role | Permitted claim |
| --- | --- | --- | --- |
| `quest_cpu_full_state_exact` | QuEST CPU | Serious full-state anchor | CPU full-state correctness/runtime under the recorded timing scope |
| `quest_gpu_full_state_exact` | QuEST HIP/CUDA | Serious optional GPU full-state baseline | GPU full-state runtime only when real GPU execution is verified |
| `quimb_tn_exact` | Quimb/cotengra CPU | Serious CPU TN baseline | External exact TN correctness, planning, contraction, and memory proxy |
| `quimb_tn_sliced_exact` | Quimb/cotengra CPU | Slicing evidence | Executed sliced TN reconstruction; current slices use one worker |
| `cpu_tn_einsum_exact` | Internal NumPy TaskGraph | Shared-plan reference/diagnostic | Correct execution of supported internal plans, not a SOTA TN baseline |
| `cpu_tn_path_replay_*` | Internal NumPy TaskGraph | Quantization diagnostic | CPU cost/error of per-contraction replay; not UPMEM performance |
| `upmem_tn_sdk_simulator_quantized` / strict runtime | UPMEM SDK simulator | Bounded PIM code-path evidence | SDK DPU program execution, support boundary, traffic, and error; no hardware speedup |
| `upmem_tn_hardware_generic_loop_mvp` | UPMEM SDK physical hardware | Synthetic one-task generic TaskGraph functionality MVP | One-DPU/one-tasklet exact int32 validation only; not a quantum-circuit or performance result |
| `planner_candidate_model` | Host planning/model | Path candidate evidence | Standard objectives plus deterministic custom UPMEM-aware greedy selection; modeled only, no execution speedup |

Full-state correctness and performance are separate tiers. `full_dump` rows can
support exact statevector comparison under their cap. `state_output_mode=none`
rows are `metrics_only` and may support compute timing, but not full-output
exactness claims.

## UPMEM Architecture

### Current Executed Path

```mermaid
flowchart TD
    G[Hashed TaskGraph] --> F[Feasibility scan<br/>rank, element, dtype and layout caps]
    F -->|unsupported| X[Explicit boundary record<br/>no CPU contraction fallback]
    F -->|supported| H[Host orchestration]
    H --> Q[Optional per-task quantization<br/>float32 or int8/int32]
    Q --> D[DPU program invocation<br/>dense bridge or generic loop]
    D --> R[Dequantization / tensor reconstruction]
    R --> V[CPU validation against the same TaskGraph]
    V --> E[Normalized evidence<br/>traffic, timing, error, invocation counts]
```

Post-2E.65, the implementation has two explicit bounded tiling states. The
planner exposes dense L1 direct and L2 single-DPU MRAM-resident/WRAM-tiled
plans through `thesis/implementation/src/quantum_bench/targets/upmem/tile_plan.py`.
The strict generic path executes `mram_resident_output_tiled_v1` with output
tiles bounded at 256 elements. The dense bridge has an executable bounded
L2 real-valued tiled path, while complex L2 and shapes outside the recorded
contracts remain explicit boundaries. The SDK simulator proves the native SDK
control path and DPU program invocation, not physical PIM timing.

Current limitation:

> Bounded generic and bounded tiled UPMEM contractions exist; fully general
> UPMEM TN contraction, unrestricted layouts, and hardware timing do not yet
> exist.

### Target Modular Architecture

| Planned module | Purpose | Current state | Candidate source/inspiration |
| --- | --- | --- | --- |
| UPMEM-aware path objective | Score FLOPs, peak intermediate size, host/DPU and MRAM/WRAM movement, tiles, synchronization, and numerical pressure | Deterministic custom greedy planner plus standard baselines; modeled fixed single-DPU policy only | Thesis contribution on top of `opt_einsum`/cotengra |
| Kernel classifier/selector | Choose dense GEMM, generic tiled contraction, permutation/layout, sparse, or collective path | Dense/generic routing scaffolding exists | Thesis architecture |
| Tiled generic contraction | Stream operands/output through MRAM/WRAM under explicit caps | Implemented for bounded output tiling and bounded L2 dense real-valued execution; strict generic coverage remains capped | UPMEM programming model, SimplePIM/ATiM ideas |
| Gate-aware permutation kernels | Replace arithmetic by row/index permutation for gates where mathematically valid | Missing | PIMutation-inspired specialization, thesis adaptation to TN tasks |
| Layout/transpose/slicing kernels | Avoid host materialization and enable bounded subproblems | Missing | Standard TN/PIM techniques; implementation is thesis work |
| Quantization formats | Compare same-plan float32 and integer execution with explicit scale/error | Float32 and int8 generic modes exist | Thesis evaluation; motivated by weak DPU floating point |
| Multi-DPU scheduler | Assign ready contractions/tiles to DPU groups | Assignment/frontier evidence is modeled or simulator-serial | Thesis architecture |
| DPU communication layer | Move intermediate data without unnecessary host round trips | Not integrated | PID-Comm candidate |
| High-level PIM adapter | Reduce host/DPU boilerplate where it does not hide measurement | External code pinned, not active in strict route | SimplePIM candidate |
| Automatic kernel generation | Explore generated DPU kernels for selected contractions | Not integrated | ATiM candidate |
| Sparse kernels | Exploit zero/sparsity structure when measured | Not integrated | SparseP candidate |

The first kernel wave implements a conservative MRAM/WRAM-tiled generic binary
contraction; current evidence can measure its effect using an unchanged
TaskGraph and CPU reference. Specialized permutation, sparse, and
communication kernels should follow only when a recorded workload class makes
their expected benefit testable.

## Provenance And Thesis Contribution

| Item | Provenance | How it is used here |
| --- | --- | --- |
| QuEST | External submodule | Full-state CPU and GPU baseline; no thesis ownership claim |
| Quimb, cotengra, opt_einsum | External Python libraries | Serious CPU TN execution and path planning |
| UPMEM SDK | External platform/toolchain | Simulator today; physical DPU DIMMs planned at ETH |
| SimplePIM | External pinned repository | Candidate abstraction/reference; strict generic evidence currently records `simplepim_api_used=false` |
| PID-Comm | External pinned repository | Candidate future multi-DPU communication layer, not currently executed |
| ATiM, SparseP, related papers/tools | Literature/future dependencies | Design candidates only until a verified adapter exists |
| PIMutation | Prior research and benchmark inspiration | Six circuit families, full-state PIM comparison context, quantization/specialized gate-operation motivation |
| Circuit-to-TN TaskGraph, strict fallback rules, execution identities, UPMEM-aware evaluation | This thesis implementation | Engineering/research contribution to be evaluated; novelty claims remain bounded by experiments |

PIMutation simulates state vectors on PIM. This thesis changes the execution
object to a TN contraction plan. The comparison to PIMutation is therefore
relative and methodological: shared circuit families and a QuEST full-state
anchor are used to explain where TN changes memory and work. It is not presented
as a direct rerun of PIMutation on identical hardware.

## Evidence And Artifact Boundaries

```text
runs/evidence/      generated executions; ignored by Git
runs/comparisons/   generated derived analysis; ignored by Git
thesis_results/     selected compact evidence, source CSVs and plots; tracked
build/              generated native/cache output; ignored by Git
```

An evidence run may contain manifests, environment metadata, normalized rows,
execution bundles, summaries, and bounded case/task artifacts. Derived plots and
comparison tables are forbidden in evidence. `thesis_results/current` includes
checksums and enough normalized evidence to regenerate its report without
rerunning a simulator.

Final performance runs require an explicit physical-core count. The workflow
sets and records OpenMP, OpenBLAS, MKL, and NumExpr thread counts, and captures
the active CPU affinity, frequency governor, and NumPy BLAS implementation.
Tables report repeat dispersion as quartiles/IQR in addition to medians.

Provenance has three explicit stages: benchmark source, report generation, and
snapshot promotion. All selected evidence must share one clean implementation
source commit. Repository-wide dirtiness is retained as context, but files
outside `thesis/implementation` do not invalidate the implementation-scoped
source check.

## Modeled Planner And Report Contract

Planner objective versions are preserved rather than silently rewritten. The
existing `upmem_path_cost_v1` suites are historical v1 evidence. V1 evaluates
the fixed `generic_single_dpu_float32_v1` policy, and complex quantum tensor
networks remain standard-planner baseline evidence with modeled UPMEM
infeasibility under that contract.

The additive v2 suites use `custom_upmem` with
`objective_version: upmem_path_cost_v2` and
`selection_scope: projected_prefix`. V2 is a deterministic greedy planner: at
each step it scores the already selected prefix plus each candidate, so it does
not claim a globally optimal complete path. Its
`PathCostComponentsV2` records modeled FLOPs, largest tensor bytes,
split-complex component invocations and recombination, host payloads, modeled
MRAM/DMA windows, tile iterations, host completion events, and explicit memory
assumptions. Those assumptions distinguish logical task payload, fixed native
MRAM reservation, total configured MRAM capacity, maximum per-buffer payload
ratio, known static WRAM bytes, and configured WRAM budget. They are modeled
capacity fields, not measured DPU occupancy. Infeasible steps carry explicit
rejection reasons.

The v2 numeric contract accepts real inputs and complex-typed inputs with zero
imaginary values as real-valued work. Nonzero complex inputs are represented as
split real/imaginary components and are modeled where the bounded policy
supports them. This is a modeled numeric/execution-policy contract, not a claim
that every arbitrary complex layout or workload is executable.

`configs/suites/diagnostics/planner_objective_motifs.yml` and the v2 manual
planner suites contain controlled real-valued chain, tree, star, cycle, grid,
and FLOP/memory trade-off networks. They exist only to test the modeled
objective. Their records are marked `not_real_quantum_circuit=true` and must
never be used as circuit runtime or UPMEM hardware evidence.

Both v1 and v2 planner rows are evidence of path selection and modeled cost
components only. Neither version supplies a UPMEM hardware performance claim;
SDK-simulator or physical timing requires a separate executor route and an
explicitly recorded execution mode.

Derived figures always remain in the report surface. A figure with no valid
data, no variance, or no implementation is emitted as a visible TODO PNG and
is marked `generated_todo_*` in `plot_manifest.json`; it is listed separately
from valid scientific figures. Plot captions identify measured values,
planner-estimated values, modeled values, software-recorded SDK bytes, SDK
simulator timing, or physical-hardware timing as appropriate.

## Scientific Safety Rules

- A matching case name is insufficient: same-plan claims require matching
  circuit, TN, and contraction-plan hashes.
- Planning and execution timings are reported separately.
- CPU/GPU speedup uses matched repeats and identical timing/output contracts.
- Quimb and internal TaskGraph timings are different implementation families;
  they are not silently treated as same-plan speedup.
- UPMEM SDK simulator timing is never hardware speedup.
- Modeled DPU assignment or pressure is never labeled executed parallelism.
- Unsupported rows remain in tables; they define the architecture boundary.
- Energy remains unavailable unless a real sensor interval is measured and its
  source is recorded.
