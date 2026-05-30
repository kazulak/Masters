# Design Decisions

These decisions define the intended architecture before implementation. They can
change only when experiments, integration constraints, or thesis-scope constraints
justify the change.

## DD-001: The Core Contribution Is Route-Aware TaskGraph Execution

Decision: the thesis contribution is the planner, TaskGraphV2, dispatcher,
cost/profiling records, validation framework, and ablation methodology.

Reasoning: SimplePIM, SparseP, PID-Comm, raw UPMEM kernels, CPU baselines, and
optional GPU baselines are providers. They support the research question but are
not the thesis architecture by themselves.

## DD-002: Preserve The MVP As A Baseline

Decision: keep `../01_MVP_DenseGEMM` as a reproducible baseline and build the
next architecture separately.

Reasoning: the thesis needs a stable before/after comparison. Refactoring the MVP
too early would make it harder to distinguish real architectural improvement from
baseline drift.

## DD-003: Host Owns Pathfinding, Slicing, And Routing

Decision: contraction pathfinding, tensor slicing, route eligibility, memory
planning, and dispatch run on the host CPU.

Reasoning: contraction path search is expensive and irregular. UPMEM DPUs have
tight WRAM limits, integer-native arithmetic, and no dynamic peer communication.

## DD-004: TaskGraphV2 Is The Central Artifact

Decision: every operation must be represented as a TaskGraphV2 `TaskNode` before
it can be executed by any backend.

Reasoning: the task graph is the source of reproducibility, route decisions,
cost estimates, validation, and thesis plots. Route-specific shortcuts must not
bypass it.

## DD-005: Route Dispatch Is Mandatory

Decision: every operation goes through the dispatcher, even when only one route
is enabled.

Reasoning: the dispatcher is the experiment control point. It makes ablations,
forced routes, negative results, and route logs possible.

## DD-006: Route Contract Includes `prepare`

Decision: every route implements `can_execute`, `estimate`, `prepare`, and
`execute`.

Reasoning: preparation includes layout conversion, quantization, CSR conversion,
DPU allocation, tile planning, and buffer setup. If preparation is hidden inside
`execute`, the thesis cannot fairly compare providers.

## DD-007: SimplePIM Is The Default UPMEM Provider

Decision: SimplePIM is the default programming substrate for new UPMEM routes
where it can express the operation and expose enough cost/profile information.

Reasoning: SimplePIM provides a higher-level UPMEM programming model and useful
processing/communication abstractions. The project should benefit from that
productivity, but still measure it against raw/custom paths.

## DD-008: SimplePIM Is A Provider, Not The Architecture

Decision: SimplePIM must live behind `ExecutionRoutePort` and can be replaced or
bypassed by raw/custom providers on a per-task basis.

Reasoning: the thesis architecture is modular dispatch. If SimplePIM is treated
as the whole system, the project loses the ability to test routing, data-format,
and provider tradeoffs.

## DD-009: Raw UPMEM Remains A Baseline And Escape Hatch

Decision: the raw dense UPMEM MVP is wrapped as `RawUPMEMProvider_baseline`.

Reasoning: it is the current reproducible proof of correctness. It also provides
a performance-control path when SimplePIM hides too much or underperforms for a
hot kernel.

## DD-010: Custom Dense Optimization Waits For Measurement

Decision: `CustomDenseProvider` is built only after TaskGraphV2 can compare CPU,
raw UPMEM, and SimplePIM routes under the same profiles.

Reasoning: dense contraction is important, but optimizing it before the logging
and profiling framework exists would create unexplainable speedups or failures.

## DD-011: Heuristics Are A First-Class Route

Decision: diagonal, permutation, scalar, reshape, identity, and trivial
operations should be routed before dense contraction when legal.

Reasoning: avoiding contraction is often better than accelerating contraction.
The effect must be measured with heuristics enabled and disabled.

## DD-012: SparseP Is Conditional

Decision: SparseP is a sparse-linear-algebra provider selected only when density,
conversion cost, and downstream format compatibility justify it.

Reasoning: tensor-network intermediates often become dense. SparseP is strong
evidence for sparse PIM kernels, but it is not a general tensor-network backend.

## DD-013: PID-Comm Belongs In The Collective Layer

Decision: PID-Comm-style functionality is represented as a collective provider,
not as ordinary contraction execution.

Reasoning: collectives handle broadcast, gather, scatter, reduce, and sliced
result aggregation. Pairwise contraction remains a numeric route.

## DD-014: Naive Host Collectives Stay As A Baseline

Decision: implement `NaiveHostCollectiveProvider` before PID-Comm.

Reasoning: PID-Comm needs a comparison point. Small reductions may also be faster
through simple host aggregation.

## DD-015: Data Format Is A First-Class Parameter

Decision: task graph and route logs include selected data format, accumulator,
scale metadata, conversion cost, and validation error.

Reasoning: UPMEM DPUs are integer-native. The thesis should compare data formats
explicitly instead of treating int8 quantization as a hidden implementation
detail.

## DD-016: CPU Reference Is Always Available

Decision: CPU FP64/reference execution remains available for validation, fallback,
small contractions, and final correctness checks.

Reasoning: every non-reference route needs a stable correctness anchor.

## DD-017: GPU Backend Is Optional

Decision: GPU support is postponed unless a baseline can be added cheaply and
reproducibly.

Reasoning: a GPU baseline is useful but not required to prove the UPMEM
architecture. It should not block TaskGraphV2, SimplePIM, dense, sparse, or
collective evaluation.

## DD-018: No Dynamic Inter-DPU Communication

Decision: algorithms cannot require one DPU to request data from another DPU
during task execution.

Reasoning: UPMEM does not support direct dynamic DPU-to-DPU dependency. Any
cross-DPU communication must be represented as explicit collective or host
runtime work.

## DD-019: Every Performance Claim Needs A Correctness Metric

Decision: route and format experiments must report accuracy, not just runtime.

Reasoning: quantization, slicing, reductions, and data-format changes can alter
amplitudes silently. The thesis should report max error, relative error, norm
drift, and fidelity where those metrics apply.

## DD-020: Integration Claims Require Proof

Decision: external systems are treated as design influences or provider
candidates until code, license, SDK version, build path, and benchmark value are
verified.

Reasoning: citing a system and integrating it into this repo are different
claims. The implementation plan should not depend on unverified external
integration.

## DD-021: Modules Are Plugins, Not Services

Decision: provider modules are local adapters behind ports, not network
microservices.

Reasoning: service boundaries would add deployment and communication complexity
without helping the thesis claim. The needed modularity is compile/runtime
replaceability, not distributed deployment.
