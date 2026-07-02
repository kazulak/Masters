# Codex Implementation Direction: Modular Benchmark-Driven PIM/Tensor-Network Thesis Runtime

## Purpose of this document

This document gives high-level direction for the Codex agent working on the Master's thesis repository. It is **not** intended to prescribe the exact file layout or implementation details. The agent should first inspect the existing repository, documentation, tests, current architecture notes, benchmark scripts, and thesis-related files, then prepare a detailed implementation plan consistent with the current codebase.

The goal is to move from a rapid conference/workshop prototype toward a thesis-grade, modular, reproducible implementation that supports the thesis narrative:

> A modular, route-aware runtime for exact tensor-network contraction-based quantum-circuit simulation on conventional and Processing-in-Memory-oriented backends, with reproducible benchmarking and validation against full-state-vector baselines.

The implementation should prioritize correctness, reproducibility, modularity, and clear experiment logging over premature performance optimization.

---

## Big-picture thesis context

The thesis investigates acceleration of **exact tensor-network contraction for quantum-circuit simulation** using **Processing-in-Memory architectures**, especially UPMEM. The work combines:

1. Quantum-circuit simulation.
2. Exact tensor-network contraction, not MPS/PEPS truncation workflows.
3. Processing-in-Memory constraints, especially UPMEM-style digital PIM.
4. Benchmarking against conventional full-state-vector and tensor-network baselines.
5. A modular architecture where specialized kernels can be added without rewriting the system.

The current code has been developed rapidly to produce initial results. The next stage should take a step back and strengthen the implementation architecture so that future kernels, benchmarks, and thesis experiments can be added cleanly.

The desired result is not necessarily a production-grade simulator. The desired result is a thesis-grade research prototype that is:

- reproducible,
- modular,
- benchmarkable,
- validated against trusted baselines,
- honest about limitations,
- extensible for UPMEM kernels and future work.

---

## Primary strategic decision

Build the **benchmark and execution spine first**.

Do not start by fully reconstructing PIMutation, integrating SparseP, or optimizing individual kernels in isolation. Those should become routes/providers inside a common architecture.

The system should make it easy to compare, at minimum:

| Backend family | Full-state-vector simulation | Exact tensor-network contraction |
|---|---|---|
| CPU | QuEST CPU or existing CPU state-vector route | NumPy / opt_einsum / cotengra-style exact contraction |
| GPU | GPU state-vector route if available | GPU exact TN route if available |
| PIM / UPMEM-oriented | PIMutation-style state-vector/permutation route if implemented | DenseGEMM / row-swap / SparseP / other TaskGraph routes |

Not all routes need to be implemented immediately. Missing routes should be represented cleanly as skipped/unavailable with explicit reasons.

---

## Implementation philosophy

### 1. Preserve modularity

Keep the architecture route/provider-oriented. The host should own high-level orchestration:

- circuit loading/generation,
- task-graph construction,
- contraction-path planning,
- route selection,
- data-format conversion,
- validation,
- profiling,
- benchmark output,
- global reductions and aggregation.

Backends/kernels/providers should perform bounded work and report metrics. They should not own global orchestration.

### 2. Avoid spaghetti code

Do not add ad-hoc benchmark scripts that bypass the runtime. Do not create one-off paths for each experiment unless they are explicitly wrapped into the common benchmark interface.

Every new route should be accessible through configuration, not by editing source code.

### 3. Validation is mandatory

Every route that produces numerical output must be validated against a trusted CPU reference where feasible.

At minimum, record:

- max absolute error,
- L2 error,
- state norm or tensor norm,
- fidelity / inner-product error where applicable,
- pass/fail status,
- tolerance used.

For approximate integer/fixed-point/PIM-like routes, validation should be explicit and should not be hidden behind aggregate timing.

### 4. Separate preparation cost from kernel cost

The thesis needs to discuss data movement and orchestration overhead. Therefore benchmark logs should distinguish, where possible:

- circuit generation/loading time,
- contraction-path planning time,
- task-graph lowering time,
- host preparation/conversion/quantization time,
- host-to-device transfer time,
- device/kernel execution time,
- device-to-host transfer time,
- reduction/aggregation time,
- total end-to-end time.

If a backend cannot measure all of these separately, use `null`, `unknown`, or a clearly documented approximation rather than inventing precision.

---

## First milestone: benchmark spine

The first implementation milestone should be a common benchmark pipeline.

The desired user experience should be something like:

```bash
make test
make bench-smoke
python -m <project>.bench run --suite suites/smoke.yml --out runs/smoke
python -m <project>.bench summarize runs/smoke
```

The exact module names and command names may differ depending on the current repository. Use the existing structure where possible.

A benchmark run should produce a self-contained output directory, for example:

```text
runs/<timestamp-or-name>/
  config.yml
  environment.json
  task_graph.json
  execution_log.json
  validation.json
  metrics.csv
  summary.md
```

The exact schema can be adapted, but the output should be machine-readable and thesis-friendly.

---

## Benchmark metadata to capture

For each benchmark case, record as much of the following as the architecture reasonably supports:

```text
run_id
suite_name
case_name
backend_family
route_name
status
skip_reason
n_qubits
depth
circuit_family
gate_set
planner
contraction_path_id
contraction_path_summary
peak_intermediate_size
estimated_flops
estimated_memory_bytes
estimated_host_to_device_bytes
estimated_device_to_host_bytes
actual_host_to_device_bytes
actual_device_to_host_bytes
planning_time_s
host_prepare_time_s
transfer_h2d_time_s
kernel_time_s
transfer_d2h_time_s
reduction_time_s
total_time_s
validation_reference
validation_max_abs_error
validation_l2_error
validation_fidelity_error
validation_passed
notes
```

If some values are unavailable, represent that explicitly.

---

## Initial benchmark suites

The agent should inspect existing tests and examples, then define benchmark suites similar to the following.

### Smoke suite

Purpose: verify that the full pipeline works quickly.

Suggested cases:

- Bell 2q,
- GHZ 4q,
- GHZ 6q or 8q,
- one tiny QAOA p=1 circuit,
- one small random circuit.

Suggested routes:

- CPU full-state vector baseline,
- CPU exact TN contraction,
- any currently working UPMEM/PIM simulation route if available,
- unavailable routes should be skipped with reasons.

### CPU scaling suite

Purpose: provide thesis figures for conventional baselines.

Suggested cases:

- increasing number of qubits,
- increasing circuit depth,
- structured circuits such as GHZ/QAOA/random 1D/circuit families already present in repo.

Suggested comparisons:

- CPU full-state vector,
- CPU exact TN contraction.

### PIM-kernel suite

Purpose: evaluate PIM-oriented primitives in isolation and later inside the TaskGraph runtime.

Suggested cases:

- dense contraction / tiled GEMM microbench,
- permutation / row-swap microbench,
- diagonal gate or sparse-like operation,
- SparseP-style SpMM candidate if available,
- SimplePIM or raw UPMEM route if available.

---

## Route/provider architecture direction

The agent should inspect the existing architecture and adapt this direction to the current code.

A route/provider should conceptually expose operations like:

```text
can_execute(task, context) -> eligibility decision + reason
estimate(task, context) -> cost estimate / expected constraints
prepare(task, context) -> prepared executable object + preparation metrics
execute(prepared, context) -> result + execution metrics
validate(result, reference, context) -> validation record
```

The exact function names and types may differ. The important design requirement is that routing decisions, rejected routes, preparation costs, execution costs, and validation results are visible in the logs.

Each route should be selectable through configuration/YAML/CLI rather than hard-coded.

---

## Important route categories

### CPU full-state-vector route

Purpose: trusted baseline.

Likely implementation source: existing QuEST route or current full-state-vector implementation.

Expectations:

- deterministic benchmark cases,
- correctness reference for small circuits,
- timing and memory metadata where possible,
- clean failure if QuEST is unavailable.

### CPU exact tensor-network contraction route

Purpose: trusted exact TN baseline.

Important: this means **exact tensor-network contraction**, not MPS/PEPS truncation.

Likely implementation source: NumPy, opt_einsum, cotengra, or existing code.

Expectations:

- contraction path recorded,
- peak intermediate estimate recorded where available,
- output validated against CPU full-state-vector route for small cases,
- supports benchmark scaling cases.

### GPU routes

Purpose: desirable comparison, but not required before the benchmark spine exists.

Possible routes:

- GPU full-state-vector,
- GPU exact TN contraction.

Expectations:

- if unavailable, route should be skipped cleanly,
- no hard dependency should break CPU/PIM smoke tests,
- do not make GPU support a blocker for thesis MVP.

### PIM DenseGEMM / tiled contraction route

Purpose: existing or near-existing UPMEM-oriented dense tensor contraction MVP.

Expectations:

- wrap current implementation into the common route interface,
- record quantization/conversion cost separately from kernel execution,
- validate numerical error explicitly,
- keep the old rapid MVP path working if it is useful as a frozen baseline, but do not let it become the main architecture.

### PIMutation-inspired row-swap / permutation route

Purpose: thesis-relevant specialized kernel inspired by PIMutation-style gate handling.

Do not reconstruct the whole PIMutation paper first.

Instead, implement the reusable architectural component:

> A route/provider for permutation-like, row-swap, index-remapping, or thin-structured gates/tasks.

Initial implementation can be host-only or simulated. The important point is to classify eligible tasks, execute them correctly, and validate them against the reference.

Later, this route can be moved to SimplePIM/raw UPMEM if there is time and hardware access.

### SparseP / SpMM route

Purpose: candidate backend for sparse tensor/matrix operations when the operation structure and conversion cost justify it.

Do not integrate SparseP before the benchmark spine is stable.

When implemented, the route should explicitly answer:

- when is the sparse route eligible?
- what conversion is required?
- how much conversion/preparation cost is paid?
- how much data is transferred?
- when is SparseP rejected?
- does it validate against the CPU reference?

---
## Expected Architecture Direction

This project is not intended to become a monolithic quantum simulator or a one-off benchmark script. The implementation should preserve and strengthen the thesis architecture: a modular, benchmark-driven Host-PIM tensor-network execution system.

The central idea is:

* The host CPU performs global work: circuit parsing, tensor-network construction, contraction-path selection, slicing, scheduling, validation, logging, and benchmark orchestration.
* UPMEM/PIM execution providers perform bounded local tasks only: tiled dense kernels, sparse kernels, permutation/row-swap kernels, or other data-local primitives.
* The system should be route/provider based. New kernels should be added as execution routes, not by hard-coding special cases into the main simulator.
* The runtime should make route decisions explicitly and record why a route was selected, rejected, skipped, or failed.
* Benchmarking, validation, and logging are first-class architecture components, not afterthoughts.

### Non-negotiable Design Principles

1. Preserve modularity. Avoid spaghetti code and avoid coupling circuit generation, planning, execution, validation, and plotting into one script.

2. Prefer clear interfaces over clever shortcuts. Each backend or kernel should expose a simple contract such as eligibility check, cost estimate, preparation, execution, validation metadata, and profiling output.

3. Keep CPU reference paths stable. CPU full-state vector and CPU exact tensor-network contraction are correctness baselines. PIM routes should be validated against them.

4. Exact tensor-network contraction means exact contraction of the circuit tensor network, not MPS/PEPS truncation. Approximate tensor-network methods may be discussed in the thesis but should not silently replace the exact benchmark path.

5. UPMEM constraints must shape the architecture:

   * 64 KiB WRAM per DPU means explicit tiling/slicing is mandatory.
   * No native floating point means integer, fixed-point, or block-floating-point routes should be preferred for PIM kernels.
   * No DPU-to-DPU communication means reductions and global synchronization are host-mediated.
   * Host-DPU transfer costs must be measured or estimated separately from kernel execution time.

6. External libraries should be integrated as providers, not as architectural replacements. For example:

   * QuEST may provide CPU/GPU full-state-vector baselines.
   * opt_einsum/cotengra may provide tensor-network pathfinding or contraction planning.
   * SparseP-like logic may provide sparse SpMM-style routes.
   * PIMutation-inspired logic may provide row-swap/permutation routes.
   * SimplePIM/ATiM-like ideas may inform UPMEM execution and tiling.

7. Every benchmark run should produce reproducible artifacts:

   * configuration used,
   * environment information,
   * circuit/workload description,
   * selected route,
   * skipped/rejected route reasons,
   * validation result,
   * runtime metrics,
   * transfer metrics if available,
   * summary table/CSV.

### Intended Route Families

The architecture should support at least the following conceptual route families:

#### CPU full-state-vector route

Baseline using QuEST or equivalent. This is not the main research contribution, but it is necessary for correctness and performance comparison.

#### CPU exact tensor-network route

Baseline using NumPy/opt_einsum/cotengra-style exact contraction. This is the main algorithmic reference for the PIM tensor-network work.

#### GPU full-state-vector route

Optional depending on available dependencies and hardware. It should be represented in the benchmark matrix, even if skipped when unavailable.

#### GPU exact tensor-network route

Optional depending on available dependencies. Possible implementations may use CuPy, cuTensorNet, or another GPU tensor contraction backend. It should not block the core thesis.

#### UPMEM dense route

A tiled GEMM-like route for dense tensor contractions. This route should explicitly handle preparation, quantization/fixed-point conversion if used, WRAM-aware tiling, execution, dequantization, and validation.

#### Sparse route

A route for zero-heavy tensor operations or sparse matrix/tensor representations. SparseP-style ideas should be integrated here if practical. The route must account for sparse-format conversion cost.

#### Heuristic bypass route

A route for operations where full dense contraction is unnecessary. This includes identity operations, diagonal gates, permutation gates, row swaps, reshape-only operations, and separable/simple structures. PIMutation-inspired row swapping belongs here.

### Immediate Implementation Priority

The first priority is not to add more kernels. The first priority is to build the benchmark spine and route-provider architecture so that every future kernel can be added cleanly.

The recommended implementation order is:

1. Create or repair the benchmark runner.
2. Standardize benchmark output files and metrics.
3. Ensure CPU full-state-vector and CPU exact tensor-network baselines work.
4. Add skipped placeholders for GPU and UPMEM routes where dependencies are unavailable.
5. Add route decision logging.
6. Add validation records comparing all routes against CPU references.
7. Add PIMutation-inspired row-swap as a heuristic/bypass route.
8. Add or wrap UPMEM dense route.
9. Add SparseP-style sparse route only after the route interface and benchmarks are stable.

### What to Avoid

Do not rewrite the whole project into a single-purpose simulator.

Do not implement PIMutation as a separate unrelated project.

Do not hard-code benchmark-specific logic inside execution backends.

Do not optimize for speed before correctness, reproducibility, and clean route logging exist.

Do not make GPU or real UPMEM availability mandatory for the core system to run.

Do not let optional dependencies break the benchmark suite; unavailable routes should be skipped with explicit reasons.

### Thesis Alignment

The implementation should support the thesis claim:

This work designs and evaluates a modular Host-PIM architecture for exact tensor-network quantum-circuit simulation under UPMEM constraints. The contribution is not only a kernel, but a route-aware execution system that separates global planning from local PIM execution, validates correctness against CPU baselines, and exposes the data-movement and numerical constraints that determine whether PIM execution is feasible.

---

## PIMutation reconstruction guidance

PIMutation is useful as an architectural reference and a source of specialized kernels such as row swapping/permutation-style operations. However, this thesis is not a PIMutation clone.

The agent should avoid spending large effort on full PIMutation reconstruction before the benchmark and route system exists.

Recommended approach:

1. Identify the operation category from PIMutation that is reusable in this thesis.
2. Represent it as a route/provider inside the modular system.
3. Add unit tests against a CPU reference.
4. Add a microbenchmark case.
5. Record route eligibility, rejection reasons, preparation cost, execution cost, and validation.
6. Only then consider UPMEM/raw implementation.

---

## Testing direction

The implementation should include tests at multiple levels.

### Unit tests

Examples:

- circuit generator produces deterministic circuits for fixed seeds,
- task graph serialization round-trips,
- route eligibility decisions are deterministic,
- row-swap/permutation route matches CPU reference,
- dense contraction route matches NumPy reference,
- validation metrics behave correctly.

### Integration tests

Examples:

- smoke suite runs end-to-end,
- output directory contains required artifacts,
- skipped routes are reported cleanly,
- CPU state-vector and CPU TN agree on small circuits.

### Regression tests

Examples:

- benchmark schema remains stable,
- summary generation works on previous run directories,
- known small circuits produce expected outputs within tolerance.

---

## Documentation direction

The implementation should produce documentation that can be reused directly in the thesis.

Suggested docs:

```text
docs/benchmark_plan.md
docs/runtime_architecture.md
docs/route_provider_contract.md
docs/pimutation_row_swap_route.md
docs/reproducibility.md
```

The exact file names may differ. The goal is to leave a clear trail for Chapter 3 and Chapter 4 of the thesis.

Each new major route should include a short design note explaining:

- what operation it targets,
- when it is eligible,
- what data format it expects,
- what preparation/conversion it requires,
- what metrics it reports,
- what limitations it has.

---

## Definition of Done for the first implementation cycle

The first implementation cycle is complete when:

1. There is a common benchmark runner.
2. A smoke suite can be run from a single command.
3. CPU full-state-vector and CPU exact TN routes run on at least a few small circuits.
4. The CPU TN route is validated against the CPU full-state-vector route.
5. A machine-readable output directory is produced for each benchmark run.
6. Missing GPU/PIM routes are skipped cleanly instead of crashing the whole suite.
7. Existing PIM/DenseGEMM prototype code is either wrapped as an experimental route or explicitly documented as a frozen baseline.
8. There is a clear place for the PIMutation-inspired row-swap route and SparseP route to plug in.
9. Tests cover the benchmark runner, route decisions, validation, and at least one end-to-end smoke benchmark.
10. The documentation explains how to reproduce the smoke benchmark.

---

## Definition of Done for a thesis-grade route

A route is thesis-grade when:

1. It is selectable through configuration.
2. It has deterministic tests.
3. It reports eligibility and rejection reasons.
4. It separates preparation/conversion time from execution time where possible.
5. It reports data movement estimates or actual transfer measurements where possible.
6. It validates against a trusted CPU reference.
7. It writes metrics into the common benchmark schema.
8. It appears in benchmark summaries and can be used in plots.
9. It can be disabled without modifying source code.
10. Its limitations are documented.

Routes that do not meet these criteria should be labeled experimental.

---

## Non-goals for the immediate cycle

Do not prioritize the following before the benchmark spine is working:

- full PIMutation reconstruction,
- full SparseP integration,
- broad GPU support,
- complete reusable library polish,
- aggressive performance tuning,
- large-scale cluster experiments,
- real UPMEM benchmarking,
- major architecture rewrites not required by the thesis MVP.

These can become later milestones after the benchmark and validation system exists.

---

## Suggested agent workflow

The Codex agent should proceed as follows:

1. Inspect the repository structure, existing docs, tests, build system, and benchmark scripts.
2. Identify current working routes and prototypes.
3. Identify current assumptions and fragile points.
4. Prepare a detailed implementation plan before editing code.
5. Keep the first implementation cycle focused on the benchmark spine.
6. Make small, testable changes.
7. Run tests after each significant change.
8. Preserve existing working prototypes unless replacing them is clearly safer.
9. Document any skipped route, unavailable dependency, or architectural compromise.
10. Produce a final summary explaining what was implemented, how to run it, what remains, and how it supports the thesis.

---

## Final instruction to the agent

Optimize for a defensible Master's thesis, not for a perfect simulator.

The implementation should make it easy to say in the thesis:

> The runtime provides a common benchmark-driven execution framework where full-state-vector baselines, exact tensor-network contraction, and PIM-oriented kernels are represented as routes under a shared validation and profiling pipeline. This allows specialized kernels such as DenseGEMM, row-swap/permutation handling, and SparseP-style sparse operations to be developed independently while remaining comparable through the same benchmark schema.

That statement is the architectural target.
