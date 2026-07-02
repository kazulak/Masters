# Codex Direction: UPMEM Implementation Architecture and External Library Integration

This document gives the intended implementation direction for the thesis prototype. The goal is not to build a generic quantum simulator from scratch. The goal is to implement a modular Host-CPU + UPMEM execution architecture for exact tensor-network quantum-circuit simulation, where external libraries are used deliberately as route providers, kernels, planners, or baselines.

Codex should read the existing repository, current architecture documents, benchmark scripts, and thesis material before proposing a detailed implementation plan. This file defines the high-level architectural intent and non-negotiable design direction.

---

## 1. Core Architecture Idea

The intended architecture is the one shown in the Host CPU orchestrator / UPMEM execution layer diagram:

```text
Quantum circuit input
        |
        v
Host CPU orchestrator
  - Circuit analyzer
  - Path optimizer
  - WRAM slicer
  - Data format conversion
  - Dynamic heuristic router
        |
        v
64 KiB DMA tiles
        |
        v
UPMEM execution layer
  - Heuristic route
  - Sparse route
  - Dense GEMM route
  - Optional TransPimLib support
        |
        v
Host aggregation
        |
        v
Amplitude output / validation output
```

The host CPU is responsible for all global, irregular, and algorithmically complex decisions. UPMEM DPUs are responsible only for local, bounded, tile-sized execution tasks.

The architecture should preserve this separation.

---

## 2. Host CPU Orchestrator Responsibilities

The host is the control plane. It should own the complete global view of the circuit, tensor network, contraction plan, available execution routes, validation strategy, and benchmark metadata.

The Host CPU orchestrator should include the following conceptual modules.

### 2.1 Circuit Analyzer

Purpose:

- Read or generate a quantum circuit.
- Convert the circuit into an exact tensor-network representation.
- Detect structural properties that affect routing.

The analyzer should identify at least:

- number of qubits,
- number and types of gates,
- single-qubit vs two-qubit gates,
- diagonal gates,
- permutation-like gates,
- sparse gates,
- dense gates,
- identity / no-op operations,
- separability or low-entanglement opportunities where easy to detect,
- expected output target: full state, selected amplitude, probability, or validation vector.

Important: the thesis focus is exact tensor-network contraction, not approximate MPS/PEPS truncation. Approximate methods can be mentioned in the thesis but should not silently replace the exact TN route.

### 2.2 Path Optimizer

Purpose:

- Choose or import a tensor contraction path.
- Keep the path optimizer modular so that opt_einsum, cotengra, or later custom PIM-aware pathfinding can be swapped in.
- Initially, using existing CPU-oriented path optimizers is acceptable, but the system should expose where PIM-aware cost models would be inserted.

The current CPU/GPU objective usually minimizes FLOPs or peak intermediate memory. For UPMEM, the long-term objective should be transfer-aware:

```text
cost ~= alpha * host_to_dpu_bytes
      + beta  * mram_to_wram_bytes
      + gamma * integer_ops
      + delta * launches
      + epsilon * host_aggregation_cost
```

The implementation does not need to solve the full PIM-aware pathfinding problem immediately, but it should make this future extension obvious.

### 2.3 WRAM Slicer

Purpose:

- Transform contraction tasks into tile-sized tasks that fit the UPMEM 64 KiB WRAM constraint.
- Estimate tile sizes before execution.
- Reject or split tasks that cannot fit WRAM.
- Emit explicit metadata about tile shapes, tile counts, and estimated DMA volume.

UPMEM constraints are central to the thesis. Do not hide them behind generic array operations.

The slicer should track at least:

- input tile shapes,
- output tile shape,
- element type and byte width,
- total tile memory footprint,
- whether double buffering is possible,
- estimated host-to-DPU and DPU-to-host transfer size,
- whether the operation can run on a DPU or must fall back to CPU.

### 2.4 Data Format Conversion

Purpose:

- Convert tensors into formats suitable for selected routes.
- Keep conversion costs visible in the benchmark logs.
- Avoid pretending that quantization or sparse conversion is free.

Possible formats:

- float64 / complex128 for CPU reference,
- float32 / complex64 for GPU or CPU performance baselines,
- int8 / int16 fixed-point for UPMEM kernels,
- block-floating-point where implemented,
- sparse formats for SparseP-style routes,
- permutation descriptors for row-swap / bypass routes.

The data-format module should be route-aware. Dense GEMM, sparse SpMM, row-swap, and transcendental helper routes may require different representations.

### 2.5 Dynamic Heuristic Router

Purpose:

- Inspect each tensor operation or contraction task.
- Decide which execution route should handle it.
- Record why each route was selected, rejected, skipped, or failed.

The router should be explicit and auditable. This is important for the thesis because the research contribution is not only one kernel, but a modular execution architecture.

The route decision should consider:

- operation type,
- tensor density / sparsity,
- gate type,
- tile size and WRAM fit,
- arithmetic requirements,
- expected host-DPU transfer cost,
- conversion cost,
- available hardware / libraries,
- correctness support,
- benchmark configuration.

The router should support at least three conceptual UPMEM execution routes:

1. Heuristic route.
2. Sparse route.
3. Dense GEMM route.

---

## 3. UPMEM Execution Layer

The UPMEM layer is the data plane. It should execute bounded tile tasks produced by the host.

DPUs should not perform global pathfinding, global scheduling, or global reductions. Because UPMEM DPUs lack direct DPU-to-DPU communication, global aggregation must remain host-mediated.

Each UPMEM route should expose a consistent provider interface, for example:

```text
can_execute(task, context) -> RouteDecision
estimate(task, context) -> RouteEstimate
prepare(task, context) -> PreparedTask
execute(prepared_task, context) -> RouteResult
validate(result, reference) -> ValidationResult
```

The exact naming and structure can be decided after reading the repo, but the separation matters.

Each route should report:

- eligibility,
- rejection reason if not eligible,
- preparation time,
- conversion time,
- transfer bytes,
- kernel time,
- total time,
- output format,
- validation error,
- fallback behavior.

---

## 4. Route Families and External Library Use

External libraries should be integrated as route providers, planners, kernels, or baselines. They should not replace the architecture.

### 4.1 CPU Full-State Vector Baseline

Role:

- Correctness and performance baseline.
- QuEST CPU can be used here.

This route is not the thesis novelty. It is necessary to compare full-state vector simulation against exact tensor-network contraction.

Expected use:

```text
Quantum circuit -> QuEST CPU -> reference state / amplitude / probability
```

### 4.2 GPU Full-State Vector Baseline

Role:

- Optional performance baseline.
- QuEST GPU or another GPU state-vector backend can be used if available.

This route should be optional. If dependencies or GPU hardware are missing, the benchmark should skip it with an explicit reason rather than failing.

### 4.3 CPU Exact Tensor-Network Baseline

Role:

- Main exact TN correctness reference.
- Use NumPy, opt_einsum, cotengra, or similar.

This route should represent exact contraction of the circuit tensor network, not MPS/PEPS truncation.

Expected use:

```text
Quantum circuit -> Tensor network -> CPU exact contraction -> reference output
```

### 4.4 GPU Exact Tensor-Network Baseline

Role:

- Optional accelerated exact TN baseline.
- Possible backends: CuPy, cuTensorNet, cotengra with GPU arrays, or another available GPU contraction route.

This route should also be optional and benchmark-skippable.

### 4.5 Dense GEMM Route on UPMEM

Role:

- Execute dense tensor-contraction tiles using GEMM-like kernels.
- This corresponds to the “Dense GEMM route / ATiM tiled GEMM” box in the diagram.

Expected direction:

- Lower selected tensor contractions to matrix multiplication or batched matrix multiplication where possible.
- Slice to WRAM-sized tiles.
- Convert to fixed-point / int8 / int16 / BFP as implemented.
- Send tiles to DPU MRAM/WRAM.
- Run local DPU kernel.
- Return partial results to host.
- Aggregate on host.
- Validate against CPU exact reference.

ATiM-like ideas may be used for tiling and tensor program scheduling. The implementation does not need to fully integrate ATiM immediately, but the code should make it clear where such a route belongs.

### 4.6 Sparse Route on UPMEM

Role:

- Execute sparse or zero-heavy tensor operations using SpMM/SpMV-style kernels.
- This corresponds to the “Sparse route / SparseP SpMV kernels” box in the diagram.

Expected direction:

- Detect zero-heavy gates/tensors or sparse intermediate structures.
- Convert to an appropriate sparse format.
- Account for sparse conversion cost.
- Use SparseP-inspired kernels or external SparseP code if practical.
- Execute only if the estimated benefit exceeds conversion and transfer overhead.
- Fall back to CPU or dense route if sparsity is insufficient.

Important: SparseP should be treated as a route provider/kernel family, not as the entire simulator architecture.

### 4.7 Heuristic Route / Bypass Route

Role:

- Avoid dense GEMM when the operation has special structure.
- This corresponds to the “Heuristic route / gate merge / row swap” box in the diagram.

This route should target operations such as:

- identity gates,
- diagonal gates,
- permutation gates,
- row swaps,
- simple single-qubit gates expressible through index permutation or cheap integer-friendly transformations,
- reshape-only operations,
- gate fusion where it reduces transfer or kernel launches.

PIMutation-inspired row swapping belongs here.

Expected direction:

- Reconstruct only the useful PIMutation idea needed for this thesis: row-swap/permutation-style execution.
- Do not implement a separate PIMutation clone unless it directly helps the modular architecture.
- Provide a route that receives a task, recognizes it as a permutation/row-swap candidate, executes the transformation, and validates it against the CPU reference.

Start with a host-side or simulator-backed prototype if needed. Later it can be moved to raw UPMEM.

### 4.8 TransPimLib Support

Role:

- Optional support for on-DPU transcendental or math-library operations.
- This corresponds to the “TransPimLib support / CORDIC / LUT transcendentals” box in the diagram.

This should not be a first priority. Treat it as an optional support provider for route families that require transcendental or specialized math. The core benchmark and routing architecture should work without it.

---

## 5. Benchmark Architecture

The benchmark system is part of the architecture. It should be implemented early because all routes must plug into it.

The benchmark runner should support a matrix such as:

```text
              Full-state vector       Exact tensor network
CPU           QuEST CPU               NumPy / opt_einsum / cotengra
GPU           QuEST GPU optional      CuPy / cuTensorNet optional
UPMEM/PIM     PIMutation-like route   Dense / Sparse / Heuristic TN routes
```

The UPMEM full-state-vector route is optional and mainly useful for comparison with PIMutation-style ideas. The main thesis focus is UPMEM/PIM support for exact tensor-network contraction routes.

Each benchmark run should write reproducible artifacts:

```text
runs/<timestamp>/
  config.yml
  environment.json
  circuit.json
  task_graph.json
  route_decisions.json
  execution_log.json
  validation.json
  metrics.csv
  summary.md
```

Metrics should include where possible:

- backend,
- route,
- circuit family,
- number of qubits,
- depth,
- gate counts,
- contraction path summary,
- peak intermediate estimate,
- tile count,
- WRAM tile size,
- host preparation time,
- format conversion time,
- host-to-DPU bytes,
- DPU-to-host bytes,
- MRAM-to-WRAM bytes if measurable/estimated,
- kernel time,
- host aggregation time,
- total time,
- memory footprint,
- energy estimate if available,
- numerical error,
- validation status,
- skipped/rejected route reasons.

The benchmark suite must not fail just because optional routes are unavailable. It should record:

```text
status: skipped
reason: GPU unavailable
```

or:

```text
status: skipped
reason: UPMEM SDK not installed
```

---

## 6. Suggested Implementation Order

Codex should prepare a detailed implementation plan after reading the repository. However, the high-level order should be:

### Phase 1: Stabilize the Benchmark Spine

- Create or repair a benchmark CLI.
- Add smoke tests.
- Ensure deterministic outputs.
- Standardize run directories and metrics files.
- Make optional routes skippable.

### Phase 2: Stabilize CPU Reference Routes

- Ensure QuEST CPU full-state vector works.
- Ensure CPU exact tensor-network contraction works.
- Validate both against each other on small circuits.
- Store validation metrics.

### Phase 3: Implement the Route/Provider Interface

- Define route decisions.
- Define route estimates.
- Define prepared tasks.
- Define execution results.
- Ensure all route decisions are logged.
- Add explicit rejection/skipping reasons.

### Phase 4: Implement or Wrap UPMEM Dense Route

- Wrap the existing MVP if available.
- Keep its data conversion, tiling, and execution visible.
- Validate Bell/GHZ and small benchmark circuits.

### Phase 5: Add Heuristic Row-Swap Route

- Reconstruct the PIMutation-inspired row-swap idea as a modular route.
- Add tests for correctness.
- Integrate into route decisions.
- Do not let this become an unrelated PIMutation reimplementation.

### Phase 6: Add Sparse Route

- Add sparse detection and sparse conversion.
- Integrate SparseP-style kernels if practical.
- Benchmark only when conversion cost and sparsity make sense.

### Phase 7: Add Optional GPU Routes

- Add GPU routes only after the CPU and benchmark spine are stable.
- Make them dependency-checked and skippable.

### Phase 8: Thesis-Grade Reporting

- Generate tables and plots from benchmark artifacts.
- Add summary files that can be used directly in the thesis.
- Ensure every result is reproducible.

---

## 7. What to Avoid

Avoid these implementation mistakes:

- Do not build a monolithic simulator.
- Do not make PIMutation reconstruction the main project.
- Do not hard-code Bell/GHZ-specific paths.
- Do not hide transfer or conversion costs.
- Do not make optional dependencies mandatory.
- Do not silently fall back from exact TN to approximate MPS/PEPS.
- Do not optimize kernel speed before validation and benchmark reproducibility exist.
- Do not couple plotting, execution, and circuit construction into one script.
- Do not make UPMEM routes globally mutate benchmark state in hidden ways.
- Do not claim speedup from simulator-only or tiny-circuit experiments without cautious wording.

---

## 8. Definition of Done for a Thesis-Grade Route

A route is thesis-grade only when it satisfies the following:

- It is selectable through configuration.
- It has a clear provider interface.
- It can reject unsupported tasks with explicit reasons.
- It logs preparation, conversion, transfer, kernel, and aggregation timing where applicable.
- It validates against CPU reference output.
- It writes metrics into the common benchmark format.
- It can be skipped without breaking the suite.
- It appears in benchmark summaries.
- It has tests for at least one supported workload.
- It is documented briefly enough that the thesis can explain it.

---

## 9. Desired Thesis Claim Supported by This Architecture

The implementation should support the following claim:

This thesis implements a modular Host-PIM execution architecture for exact tensor-network quantum-circuit simulation under UPMEM constraints. The host CPU performs global circuit analysis, contraction planning, WRAM-aware slicing, format conversion, route selection, and aggregation. The UPMEM execution layer provides specialized local routes for dense, sparse, and heuristic tensor operations, integrating external libraries as providers where appropriate. The system is evaluated through a reproducible benchmark pipeline against CPU/GPU full-state-vector and exact tensor-network baselines.

The focus is architectural feasibility, correctness, reproducibility, and identifying bottlenecks. Large speedups are not required for the thesis to be successful.
