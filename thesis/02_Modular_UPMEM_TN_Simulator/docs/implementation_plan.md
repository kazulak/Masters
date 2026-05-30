# Implementation Plan

This plan starts with structure and observability before optimization. The goal is
a research platform with credible ablations, not a collection of uncontrolled
kernels.

Each stage has three rules:

1. Preserve a working baseline before adding a faster route.
2. Add measurement before making performance claims.
3. Log route choices and failures in a way another developer can reproduce.

## Stage 0: Freeze Current MVP

Goal:

```text
Preserve 01_MVP_DenseGEMM as the reproducible dense int8 UPMEM baseline.
```

Do:

- keep `../01_MVP_DenseGEMM` as a frozen baseline;
- document the exact run command;
- save expected output for Bell and GHZ fixtures;
- save NumPy validation result;
- save current timing report;
- record compiler, UPMEM SDK, CPU, DPU/rank count, and benchmark details;
- keep `../PIMutation_reconstruction/QuEST_implementation` as the CPU
  state-vector baseline.

Do not:

- refactor the MVP into V2;
- change its numerical tolerance;
- silently replace its task graph.

Done criteria:

- Bell and GHZ pass through the MVP.
- Expected outputs and timing reports are documented.
- QuEST/PIMutation benchmark commands are documented for the target machine.

## Stage 1: Domain Core And TaskGraphV2 Replay

Goal:

```text
Reproduce the current dense-GEMM MVP through the new architecture.
```

Implement:

- `TensorMetadata`;
- `TaskNode`;
- `TaskGraphV2`;
- `DataFormat`;
- `CostRecord`;
- `ProfileRecord`;
- `ValidationRecord`;
- minimal `ExecutionRoutePort`;
- `CPUReferenceRoute`;
- `RawUPMEMDenseRoute` wrapper around the MVP logic;
- `execution_log.json`.

Route state:

| Route | Status in this stage |
| --- | --- |
| CPU reference | Implemented. |
| Raw UPMEM dense | Implemented as MVP wrapper. |
| SimplePIM | Candidate only, disabled or unimplemented. |
| Heuristic | Not implemented. |
| SparseP | Not implemented. |
| PID-Comm | Not implemented. |

Done criteria:

- V2 host reference matches NumPy/QuEST for small circuits.
- V2 dense compatibility mode reproduces MVP outputs within current int8
  tolerance.
- Logs explain route selection even when only one non-reference route is enabled.
- TaskGraphV2 can represent K-tiling and collectives even if they are not yet
  implemented.

## Stage 2: SimplePIM Becomes The Default UPMEM Provider

Goal:

```text
Implement equivalent simple or dense UPMEM tasks through SimplePIM and make
SimplePIM the default UPMEM provider where it is legal.
```

Implement:

- `SimplePIMProvider_default`;
- SimplePIM route adapter behind `ExecutionRoutePort`;
- `prepare` step for SimplePIM table/layout setup;
- at least one operation family that can also be run by CPU reference and raw
  UPMEM or host fallback;
- route configuration that can force SimplePIM, force raw UPMEM, or use default
  dispatch.

Compare:

```text
Raw UPMEM MVP route vs SimplePIM provider on equivalent tasks.
```

Decision rule:

```text
if SimplePIM matches or beats raw within the chosen tolerance/margin:
    SimplePIM remains default for that operation family
else:
    SimplePIM remains default for non-hot/simple kernels
    raw/custom remains the dense hot path
```

Done criteria:

- SimplePIM route can be enabled, disabled, and forced.
- Logs separate SimplePIM preparation time from kernel time.
- The thesis has a first SimplePIM-vs-raw comparison table.
- Raw UPMEM fallback remains available.

## Stage 3: Profiling And Empirical Cost Model

Goal:

```text
Make route decisions measurable.
```

Measure per task:

- host packing time;
- format conversion or quantization time;
- route preparation time;
- host-to-DPU DMA time and bytes;
- DPU kernel time;
- DPU-to-host DMA time and bytes;
- host unpack/dequantization time;
- host accumulation/reduction time;
- validation time;
- total wall time;
- numerical error.

Implement:

- stable `ProfileRecord` output;
- aggregate run summary;
- baseline rule-based `CostOracle`;
- mechanism to feed measured records back into later estimates;
- plotting-ready raw result schema.

Done criteria:

- Every executed route emits a complete profile record.
- Dispatcher can print or persist estimated vs measured cost.
- A forced slow route and a default route produce comparable logs.
- No performance claim is made without conversion and validation cost.

## Stage 4: Heuristic Route

Goal:

```text
Avoid unnecessary GEMM.
```

Implement:

- diagonal apply;
- permutation;
- reshape;
- identity/no-op elimination;
- trivial contraction;
- scalar fold;
- conservative gate fusion.

Rules:

- implement host-only first;
- use SimplePIM only if data is already on DPUs or transfer cost is justified;
- cap gate fusion by tensor size, tensor order, and WRAM feasibility;
- always keep dense fallback available.

Ablation:

```text
heuristics on/off
```

Done criteria:

- The execution log states which operations bypassed GEMM and why.
- Heuristic route can be enabled, disabled, and forced where legal.
- At least one benchmark shows measured benefit or a controlled negative result.

## Stage 5: Dense UPMEM V2

Goal:

```text
Make dense contraction thesis-grade.
```

Implement:

- K-tiling;
- multi-DPU tile distribution;
- explicit WRAM declarations and legality checks;
- host-mediated accumulation of K-slices and output slices;
- tasklet/tile parameter sweep;
- double buffering where the SDK path supports it cleanly;
- fixed-point format as the first serious integer-native alternative;
- block-floating-point later if fixed-point is stable.

Compare:

- raw UPMEM baseline;
- SimplePIM default provider;
- custom dense provider;
- CPU reference;
- optional GPU baseline if available.

Done criteria:

- No dense task fails only because `k > TILE_K`; it is either tiled or rejected
  with a precise reason.
- Dense route reports DMA and DPU timing per slice.
- Tile variants can be benchmarked under a fixed task graph.
- The old MVP tile shape remains available as a baseline configuration.
- Accuracy/performance tradeoff is reported for each data format.

## Stage 6: Collectives And PID-Comm

Goal:

```text
Handle multi-DPU sliced outputs explicitly.
```

Implement first:

- `NaiveHostCollectiveProvider`;
- TaskGraphV2 collective tasks;
- rank/DPU allocation metadata;
- reduction timing and memory accounting.

Then implement or integrate:

- `PIDCommCollectiveProvider`, if license, build, SDK, and hardware requirements
  are acceptable.

Compare:

```text
naive gather/reduce vs PID-Comm-style collectives
```

Rules:

- PID-Comm belongs in the collective layer, not ordinary pairwise contraction.
- Small reductions may remain on the naive CPU route.
- Collectives must be selectable and disableable for ablation.

Done criteria:

- Multi-DPU dense route produces the same result as single-DPU dense route within
  selected format tolerance.
- Reduction cost is visible in experiment output.
- PID-Comm comparison is reported if integration is practical; otherwise the
  thesis records why the interface was preserved but provider integration was not
  completed.

## Stage 7: SparseP Provider

Goal:

```text
Test whether sparse PIM kernels help real tensor-network subproblems.
```

Implement:

- sparse detector;
- density estimator;
- CSR/COO or SparseP-required format conversion;
- SparseP adapter behind `ExecutionRoutePort`;
- conversion-cost accounting;
- sparse-output handling;
- densification handling when a later route requires dense input.

Use SparseP when:

- task structure is sparse, diagonal-as-sparse, or graph-local sparse;
- density is below a measured threshold;
- conversion cost is included;
- the next operation can consume sparse output or densification is cheap.

Do not use SparseP when:

- intermediate tensor is dense;
- conversion dominates;
- next task immediately needs dense GEMM;
- heuristic diagonal/permutation handling is simpler.

Main result:

```text
density threshold where SparseP wins or loses after conversion cost
```

Done criteria:

- Sparse route wins on at least one sparse workload or is documented as a
  negative result with measured conversion overhead.
- Dense fallback remains available for all sparse-eligible operations.
- Density threshold plots include preparation and conversion time.

## Stage 8: Route-Aware Planning

Goal:

```text
Move from "plan first, route later" to "plan using route costs."
```

Implement:

- multiple candidate contraction paths;
- `CostOracle` estimates per candidate;
- slicing penalty;
- collective penalty;
- format penalty;
- transfer-aware score;
- route-specific fallback penalty.

Initial score:

```text
score(path) =
  estimated_transfer_cost
+ estimated_compute_cost
+ estimated_prepare_cost
+ estimated_conversion_cost
+ estimated_reduction_cost
+ estimated_error_penalty
+ estimated_fallback_penalty
```

Done criteria:

- The planner chooses a different path than pure `opt_einsum` FLOP minimization
  on at least one controlled workload.
- The log explains why the route-aware path was chosen.
- Runtime and correctness are compared against the original path.

## Stage 9: Thesis Evaluation Lock

Goal:

```text
Freeze experiments for final thesis writing.
```

Implement:

- benchmark definitions with fixed seeds;
- run scripts;
- result schema;
- plotting scripts or notebooks;
- ablation matrix;
- hardware profile capture;
- dependency/version capture.

Done criteria:

- Every figure can be regenerated from checked-in scripts and recorded raw data.
- Every architectural claim has a matching measurement or a clearly labeled
  limitation.
- Negative results are preserved instead of removed.

## Recommended Near-Term Order

1. Freeze and document `01_MVP_DenseGEMM`.
2. Implement TaskGraphV2 in parallel with the MVP graph.
3. Add a CPU reference executor for V2.
4. Wrap the current dense route behind `ExecutionRoutePort`.
5. Add route logging and ablation switches.
6. Add SimplePIM as the default UPMEM provider for the first legal operation
   family.
7. Add full profiling before adding heuristics, custom dense optimization,
   collectives, or SparseP.
