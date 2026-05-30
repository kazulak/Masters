# Open Questions And Ambiguities

This document separates decisions that are now resolved from questions that still
need implementation-time evidence.

## Resolved By The Current Plan

1. Meaning of "UPMEM simulator".
   The project is a quantum-circuit tensor-network runtime accelerated by UPMEM
   hardware. It is not a software simulator of UPMEM hardware behavior.

2. Tensor-network versus state-vector.
   The main runtime is tensor-network centered. QuEST/PIMutation remains a CPU
   state-vector baseline and heuristic inspiration.

3. Architectural center.
   TaskGraphV2 is the central artifact. Providers must not bypass it.

4. SimplePIM role.
   SimplePIM is the default UPMEM provider, not the whole architecture.

5. SparseP role.
   SparseP is a conditional sparse route, not a general tensor-network backend.

6. PID-Comm role.
   PID-Comm belongs in the collective provider layer, not ordinary contraction
   execution.

7. Microservice idea.
   The implementation uses local plugin-style modules behind ports, not network
   microservices.

## Remaining Scope Questions

1. Required output contract.
   The current MVP computes final amplitudes. Larger tensor-network simulations
   may prefer selected amplitudes, expectation values, or full state vectors only
   for small circuits. The output contract affects path planning, validation, and
   memory pressure.

2. CPU tensor-network baseline definition.
   `PIMutation_reconstruction/QuEST_implementation` is a state-vector baseline.
   A fair tensor-network comparison also needs a CPU tensor-network baseline using
   the same contraction path as V2.

3. GPU baseline availability.
   GPU is optional. If hardware and dependencies are easy to use, add a dense GPU
   baseline. Otherwise record it as future work.

## Remaining Integration Questions

4. SimplePIM provider acceptance.
   The repository contains SimplePIM experiments, but the V2 provider still needs
   a clean adapter, build command, SDK compatibility record, license check, and
   profile separation between `prepare` and `execute`.

5. SparseP provider acceptance.
   SparseP should be integrated only if license, build, SDK compatibility, and
   route adapter effort are reasonable. Otherwise implement a controlled local
   sparse prototype and treat SparseP as related work.

6. PID-Comm provider acceptance.
   PID-Comm should be integrated only after naive host collectives exist. If
   integration is blocked, preserve the collective interface and report the
   limitation.

7. External dense/autotuning systems.
   ATiM-style ideas can motivate tile sweeps. Do not add a dependency unless it
   clearly improves the dense route and can be built reproducibly.

## Remaining Numerical Questions

8. Accuracy target.
   The MVP uses an int8 tolerance. V2 needs formal per-workload tolerances for
   amplitude error, relative error, norm drift, fidelity, or observable error.

9. Complex arithmetic route.
   The MVP performs complex GEMM as separate real operations. Dense V2 must decide
   whether to keep split real/imaginary GEMM, pack complex pairs differently, or
   implement a fused complex tile kernel.

10. First non-int8 format.
   The plan prefers fixed-point before block-floating-point, but the exact fixed
   scale policy and accumulator width need a small design note before coding.

11. Error penalty in route-aware planning.
   The mature cost model needs a defensible way to penalize numerical error
   without pretending estimates are exact.

## Remaining Algorithmic Questions

12. Sparse threshold.
   "Sparse workload" needs a measured threshold that includes conversion,
   partitioning, execution, and possible densification cost.

13. Entanglement and routing proxy.
   The implementation needs a cheap host-computable proxy such as tensor order,
   cut size, estimated contraction cost, density, or Schmidt-rank-inspired
   metadata.

14. Gate merging limits.
   Gate merging can reduce dispatch overhead but increase tensor order. The
   planner needs caps based on tensor size, tensor order, WRAM feasibility, and
   downstream route compatibility.

15. Route-aware planner search budget.
   Stage 8 needs a bounded search strategy. Candidate path generation must not
   dominate runtime for the benchmark sizes used in the thesis.

## Remaining Measurement Questions

16. Hardware profile.
   The target machine's rank count, DPU count, SDK version, host CPU, memory, and
   clock behavior must be recorded before final measurements.

17. Energy measurement.
   The CPU baseline mentions RAPL. UPMEM-side energy measurement may require a
   different method or may not be available. Energy claims need a reliable source.

18. Host overhead attribution.
   Planning, dispatch, packing, quantization, DMA, kernel time, dequantization,
   validation, and reduction must be separated. Otherwise the thesis cannot
   explain where time is spent.

19. Fair comparison across precisions.
   CPU FP64, CPU complex128, UPMEM int8, fixed-point, and block-floating-point are
   not numerically equivalent. Comparisons must include both runtime and error.

20. Repetition and variance.
   The experiment harness needs a rule for warmups, repetitions, medians, and
   confidence intervals or interquartile ranges.
