# Open Questions And Ambiguities

These points need clarification before or during implementation. Some are thesis
scope decisions; others are engineering constraints that should be checked on the
target machine.

## Scope Ambiguities

1. Meaning of "UPMEM simulator".
   This plan assumes a quantum circuit simulator accelerated by UPMEM hardware. If
   the goal is instead a software simulator of UPMEM hardware behavior, the
   architecture and metrics need to change.

2. Tensor-network simulator versus state-vector simulator.
   The review guidance mixes tensor-network contraction with PIMutation-style
   state-vector heuristics. The likely direction is a tensor-network simulator with
   heuristic routes for compatible operations, but the thesis should state this
   explicitly.

3. Definition of the CPU SOTA baseline.
   `PIMutation_reconstruction/QuEST_implementation` is a strong state-vector CPU
   baseline. A fair tensor-network comparison may also need a CPU tensor-network
   baseline using the same contraction path.

4. Required output.
   The current MVP computes final amplitudes. Larger tensor-network simulations may
   prefer selected amplitudes or observables. The output contract affects path
   planning, validation, and memory pressure.

## Integration Ambiguities

5. Direct integration versus reimplementation of ideas.
   The literature review names SimplePIM, ATiM, SparseP, TransPimLib, PRISM, and
   PID-Comm. It is not yet clear which should be imported as dependencies, which
   should be reproduced as small local ideas, and which should remain only related
   work.

6. Licensing and build compatibility.
   Any external framework must be checked for license compatibility, SDK version
   compatibility, and reproducible build behavior before the thesis claims
   integration.

7. UPMEM hardware target.
   The number of ranks, DPUs per rank, available MRAM, SDK version, and host CPU
   matter for tile strategy and collectives. The plan needs a recorded hardware
   profile.

## Numerical Ambiguities

8. Accuracy target.
   The MVP uses an int8 tolerance. The next stage needs a formal accuracy target:
   amplitude error, state fidelity, observable error, or a combination.

9. Complex arithmetic route.
   The MVP performs complex GEMM as four real GEMMs. It is not yet decided whether
   dense route v2 should keep this, pack complex pairs differently, or implement a
   fused complex tile kernel.

10. Data-format candidates.
   Block-floating-point, fixed-point, quantization, and library-backed math are all
   possible. The first implementation should choose one additional candidate beyond
   int8, not all at once.

## Algorithmic Ambiguities

11. Sparse threshold.
   "Sparse workload" needs a concrete threshold that includes conversion cost and
   not just nonzero count.

12. Entanglement and routing heuristic.
   The plan mentions highly entangled states, but the implementation needs a cheap
   host-computable proxy such as tensor order, cut size, estimated contraction
   cost, density, or Schmidt-rank-inspired metadata.

13. Gate merging limits.
   Gate merging can reduce dispatch overhead but can increase tensor order. The
   planner needs a cap based on tensor size and WRAM feasibility.

14. Pathfinding library choice.
   The MVP uses `opt_einsum`. A tensor-network-focused pathfinder may improve
   larger circuits, but switching tools should be justified by path quality and
   planning overhead.

## Measurement Ambiguities

15. Energy measurement.
   The CPU baseline mentions RAPL. UPMEM-side energy measurement may require a
   different method or may not be available. Energy claims need a reliable source.

16. Host overhead attribution.
   Planning, packing, quantization, DMA, kernel time, dequantization, and reduction
   must be separated. Otherwise the thesis cannot explain where time is spent.

17. Fair comparison across precisions.
   CPU FP64, CPU complex128, UPMEM int8, and future reduced formats are not
   numerically equivalent. Comparisons must include both runtime and error.
