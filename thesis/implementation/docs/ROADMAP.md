# TN Benchmarking Roadmap

## Purpose
The end goal is a modular, measurable benchmark of tensor-network (TN)
quantum-circuit simulation on CPU, GPU, and physical UPMEM. The scientific
path is:

```text
SimulationJob -> TensorNetwork -> contraction path -> ContractionDAG
             -> target execution plan -> executor -> validated evidence
```

CPU, GPU, simulator, and UPMEM routes share the circuit, tensor-network,
path, DAG, numeric-policy, validation, and evidence contracts. A route may
replace only its execution modules. QuEST remains a separate full-state
baseline and is compared at the SimulationJob/query boundary, not by sharing
TN internals.

Each stage adds one usable capability and keeps previous routes runnable.
SDK-simulator timings are correctness evidence only. Physical speedup, scaling,
and energy claims require the stated physical gates.

## Current baseline
Implemented and software-tested:

- circuit lowering to `TensorNetwork` and one logical `ContractionDAG`;
- path selection, same-DAG CPU execution, complex128 reference, and replay;
- split-complex float32 and host-packed int8 with shared operand scales;
- local contracted-axis slicing and explicit host reduction;
- ABI-v4 output/K-tile mapping, WRAM-panel dense real-tile execution, and
  four-pass complex simulator execution;
- strict manifests, per-sample/session evidence, and claim guards;
- one-DPU, one-rank SDK-simulator correctness execution.

Not implemented or not qualified in the reset baseline:
- physical reset qualification and tasklet, DPU, rank, DIMM, or slice scaling;
- a high-level SimplePIM scheduler, PID-Comm provider, or ATiM kernel route;
- graph-wide residency, graph-level memory scheduling, or calibrated planning;
- measured energy and final CPU/GPU/UPMEM benchmark evidence.

The native runtime currently uses direct UPMEM SDK allocation and dispatch.
SimplePIM remains pinned as external research source but is not linked or
launched by the active route. PID-Comm and ATiM become active only when a
benchmark trace proves that their provider or kernel executed.

## Invariants for every increment
1. The same `SimulationJob`, TN, path, and DAG can be sent to multiple routes.
2. Planning and mapping are outside steady-state execution timing.
3. Every route returns output, timing scope, backend facts, numeric facts, and
   an explicit success, unsupported, or failed status.
4. Physical routes fail closed: no simulator or CPU fallback is admissible.
5. A comparison states whether it holds the logical plan fixed or compares
   different paths for the same problem.
6. Every accepted result records executable, environment, allocation, release,
   transfer, accuracy, and repetition facts needed by its claim.

## Increment 1: Physical reset qualification
**Goal.** Prove that the active ABI-v4 route executes the current bounded
whole-circuit TN fixture on real UPMEM with no fallback.

**Work.** Build the native runtime from a clean checkout and run one DPU,
one tasklet, both numeric policies, with raw logs and normalized evidence.
**Definition of done.** One ETH run contains at least 2 circuits, 2 numeric
policies, 2 warmup blocks, 30 planned measurement blocks, and complete session
facts. Every sample has
`target_observed=physical_hardware`, exact allocation/release confirmation,
matching physical-plan replay, bounded error against complex128, and no
simulator/CPU marker. A report verifies the capsule without rerunning hardware.

**Claim.** Physical functionality and scoped single-DPU timing only. No
speedup, scaling, energy, or general UPMEM claim.

## Increment 2: Intra-DPU tasklet scaling
**Goal.** Measure whether the same output/K work executes with useful tasklet
parallelism.

**Work.** Keep plan, inputs, binary, and DPU count fixed. Run tasklets at
1, 2, 4, 8, and the supported maximum; record cycles, wall time, bytes, and use.
**Definition of done.** At least 5 repetitions per setting pass correctness and
release checks. The report contains tasklet speedup
`T(1)/T(n)`, median and dispersion, and clearly separates kernel from route
wall time. Unsupported settings are rows, not omissions.

**Claim.** Tasklet behavior on the tested fixture and hardware only.

## Increment 3: Inter-DPU output-tile scaling
**Goal.** Execute independent output tiles concurrently across 1, 2, 4, and
8 physical DPUs where allocated hardware permits.

**Work.** Use one unchanged DAG and plan family. Bulk-submit the DPU set,
verify tile ownership/coverage, and record per-DPU work plus coordinator time.

**Definition of done.** Each DPU count has 5 successful repetitions, exact
output validation, `requested_dpus == allocated_dpus == active_dpus`, and no
fallback. Scaling plots use coordinator wall time; summed DPU work is a
separate metric. The report states the active rank/DIMM topology.

**Claim.** Bounded output-tile scaling, not general multi-DPU acceleration.

## Increment 4: Rank, DIMM, and slice-group scaling
**Goal.** Extend parallel work beyond one rank and expose the existing local
slicing as independent slice groups.

**Work.** Assign existing slice branches deterministically. Run one- and
multi-rank configurations, then multi-DIMM when ETH provides it. Keep host
reduction deterministic and measure it separately.

**Definition of done.** A fixture with at least 4 slice branches runs on 1 and
2 ranks when available, with every branch executed once and reduced once.
Per-rank allocation, work, communication, reduction, and release facts are
retained. Missing hardware produces explicit unsupported rows.

**Claim.** Tested slice/rank behavior only. A rank-aware mapper is not proof of
multi-rank scaling until this gate passes.

## Increment 5: SimplePIM active scheduling
**Goal.** Qualify SimplePIM as an active management and scheduling provider for
one known-good multi-DPU route only when a measured use case justifies it.

**Work.** Integrate the pinned SimplePIM management/initialization interface
behind the executor contract. Keep raw SDK as a differential reference and
reuse the experiment orchestrator.

**Definition of done.** The same compiled work units run through SimplePIM on
1 and 2 DPUs in simulator or physical mode as supported. Evidence identifies
the provider, allocation, dispatch, and release calls. Outputs match CPU
physical-plan replay; provider failure is explicit rather than silently routed
to raw SDK.

**Claim.** SimplePIM provider correctness and measured behavior for the tested
route. It is not a claim that SimplePIM is faster until matched timing scopes
and physical repetitions exist.

## Increment 6: PID-Comm provider
**Goal.** Use PID-Comm for one real communication pattern required by the TN
route, initially slice-result gather or partial-result reduction.

**Work.** Implement one provider adapter with a separately named host-mediated
fallback. Validate uneven payloads and repeated collectives; select it in the
physical plan, not the DAG.

**Definition of done.** A multi-DPU fixture invokes PID-Comm in a traceable
run, produces the same result as deterministic host reduction, and records
payload bytes, collective time, host involvement, synchronization, and
provider identity. A missing/incompatible PID-Comm build is unsupported.

**Claim.** One qualified collective and its cost on tested topology. No claim
of general DPU-to-DPU communication.

## Increment 7: ATiM kernel experiment
**Goal.** Evaluate one ATiM-generated kernel through the same kernel-provider
contract as the handwritten ABI-v4 kernel.

**Work.** Select one existing dense tile operation. Keep the handwritten
kernel as control and record ATiM source/version, binary hash, geometry,
tasklets, WRAM, correctness, and timing.

**Definition of done.** A clean checkout regenerates the selected ATiM kernel,
the kernel is invoked by a real benchmark route, and 5 repetitions per kernel
pass correctness. The comparison holds DAG, numeric policy, topology, and
tile geometry fixed.

**Claim.** The measured kernel experiment only. ATiM is not the default engine
until it passes the same correctness and capability checks.

## Increment 8: Residency and graph-level memory mapping
**Goal.** Move selected intermediate tensors and larger contractions through a
bounded memory plan without changing the logical DAG. The active dense kernel's
local WRAM panel reuse is retained as the lower-level control.

**Work.** Add a lifetime map, MRAM slots, and tiling for one contraction larger
than one WRAM tile. Start with `host_roundtrip`; add `resident_same_group` only
when measurements show it removes transfers.

**Definition of done.** The compiler rejects overlapping/out-of-bounds slots,
WRAM-infeasible tiles, and unsafe int32/int64 accumulation before dispatch.
One fixture shows final-output-only transfer for a resident stage, with
measured H2D/D2H bytes and matching CPU replay. Host-roundtrip remains an
explicit control.

**Claim.** Bounded residency and tiling behavior, not arbitrary graph-wide
residency.

## Increment 9: Hardware-calibrated planner
**Goal.** Replace uncalibrated movement/compute weights with measured
hardware-cost parameters and test whether rankings predict execution.

**Work.** Measure transfer, kernel, reduction, synchronization, and provider
overheads. Keep calibration and holdout circuits separate. Compare opt_einsum,
cotengra, exact small-case search, and the PIM-aware greedy adapter.

**Definition of done.** Each metric has units, origin, model ID, and validity
range. At least 3 candidate paths per holdout case are physically executable.
Report rank correlation, prediction error, selected-vs-best runtime, and
failure cases. “Greedy” is not called “optimal.”

**Claim.** Planner quality within the measured workload and hardware range.

## Increment 10: Energy and final benchmark freeze
**Goal.** Produce the final matched TN benchmark against CPU and GPU, with
energy measured under a declared system boundary.

**Work.** Freeze circuits, queries, paths, numeric policies, topology, warmups,
repetitions, scopes, thresholds, affinity, GPU device, and UPMEM binaries. Add
a sensor interval and report compute, transfer, synchronization, and energy.

**Definition of done.** Every matrix cell is success, unsupported, or failed;
no cell disappears. CPU, GPU, simulator, and physical UPMEM rows share the
same problem identity and state their plan relationship. Physical speedup
requires matched scope, accuracy, clean provenance, and repeated runs.
Energy claims require measured joules and a declared boundary. All tables and
plots regenerate from immutable normalized evidence and a tagged source.

**Final claim.** Only the tested circuit family, TN paths, numeric policies,
hardware configurations, and timing/energy scopes are generalized. The
benchmark must not claim UPMEM acceleration or energy superiority outside that
population.

## Sequencing rule
Implement one increment as a small component change, then run focused tests.
Batch physical work only at qualification gates. Do not add planner
sophistication, broad provider abstractions, or residency before measurements
justify them. Preserve raw SDK as the differential oracle while SimplePIM,
PID-Comm, and ATiM become active providers one bounded experiment at a time.
