# Final Architecture

## Purpose

The system is a research implementation for exact tensor-network quantum-circuit
simulation and physical evaluation on UPMEM PIM hardware. It is intentionally a modular
monolith: target-neutral semantics and logical contraction structure are separated from
target mapping, native execution, evidence, and reporting.

## End-to-end structure

```text
SimulationJob
  -> TensorNetwork
  -> complete contraction path
  -> ContractionDAG
  -> execution route

CPU/reference:
  NumPy same-DAG replay
  Quimb/cotengra TN
  QuEST full-state

UPMEM:
  ContractionDAG
  -> UpmemPlan
  -> static DAG-wave scheduling
  -> prepared packed-wave controls
  -> persistent native host
  -> DPU WRAM-panel kernel
  -> deterministic host reconstruction/reduction
  -> canonical evidence

P6 planning:
  complete candidate path
  -> production lowering
  -> metadata-only launch facts
  -> upmem_launch_cost_v1
  -> bounded search/reranking
```

## Semantic and logical layers

`SimulationJob` identifies the requested circuit simulation and output query.

`TensorNetwork` is target-neutral semantic data. It contains tensors, labels,
connectivity, inputs, and requested output ordering. It does not contain a contraction
path or hardware decisions.

A complete contraction path is lowered into `ContractionDAG`. The DAG is the logical
execution IR: binary contractions, dependencies, explicit slicing branches when
declared, and host reductions. It does not contain DPU binary paths or runtime sessions.

## UPMEM mapping and execution

`UpmemPlan` maps the logical DAG onto a declared UPMEM topology and numeric policy. The
final profile is one rank with 1-DPU/T8 and 4-DPU/T8 as the P6 evaluation topologies.

The final scheduler is `static_dag_waves_v1`. Dependency-ready contraction nodes can
execute in one synchronous cohort on disjoint DPU groups. Within a large contraction,
output/K work is distributed across DPUs, and each DPU uses multiple tasklets with
cyclic output-row ownership.

The final transport is `packed_wave_v1` through one persistent native host controller.
The prepared-wave native protocol is ABI-v5. Historical ABI-v4 evidence remains part of
project lineage but is not the final composed executor.

The retained real-product kernel is the WRAM-panel implementation with `KC=64` and
`NC=32`. Four-product complex execution computes RR, II, RI, and IR explicitly.
`fused_when_admitted_v1` places the four products in one launch when the declared memory
layout admits it; otherwise execution remains on the generic UPMEM route.

Intermediates are host-roundtrip in the final production profile. A bounded resident-pair
prototype was evaluated but not retained.

## Numerical policies

`split_complex_float32_v1` is the primary performance policy.

`complex_int8_shared_scale_v1` is a separately characterized approximate numerical
policy. Each complex operand uses a shared scale, DPU products accumulate integer lanes,
and the host reconstructs complex values. Same-policy physical correctness and
full-precision approximation error are different predicates.

No numerical-approximation error is folded into the final float32 P6 execution-cost
score.

## Path-cost model

For one complete lowered plan, P6 uses:

```text
C =
    theta_H * H / s_H
  + theta_P * P / s_P
  + theta_N * N / s_N
  + sum_launch max_dpu(
        theta_M * M[launch,dpu] / s_M
      + theta_W * W[launch,dpu] / s_W
    )
```

`H` is planned host/DPU traffic. `P` is the explicitly inventoried host array-pass
volume proxy. `N` is the declared coordination count. `M` is source-level estimated
MRAM/WRAM movement per DPU/launch. `W` is real arithmetic work per DPU/launch.

Movement and arithmetic are combined on each DPU before the per-launch maximum.
Normalization scales are frozen from development greedy cells before physical fitting.

The score is dimensionless and used for ranking. It is not an elapsed-time predictor.
The final integer weights `[1,2,1,1,5]` are model-ranking parameters; they are not runtime
shares or unique machine constants.

## P6 search architecture

A serial Optuna TPE ask/tell loop proposes `costmod` and `temperature` for exactly one
fresh cotengra `RandomGreedyOptimizer(max_repeats=1)` candidate at a time.

The complete candidate is lowered through the production plan, deterministically admitted
or rejected, scored, and told back to the adaptive sampler before the next ask.

F and U use paired seed schedules:

- F tells conventional complete-tree FLOPs.
- U tells the UPMEM launch cost.

R does not generate a new trace; it reranks exactly the F trace with the frozen UPMEM
model. This makes U versus R the experiment isolating adaptive hardware-aware generation
from hardware-aware selection.

## Evidence and identity

The evidence system keeps problem, TN structure, logical plan, physical plan, executable,
environment, experiment, run, session, and sample identities separate.

Physical execution fails closed: simulator or CPU fallback cannot satisfy a physical
UPMEM request.

Accepted P6 stages require exact round membership, numerical validation, source and
binary identity, resource admission, session release, portable archive checksums, and two
verified durable copies before observations may enter fitting.

Evaluation observations are never allowed into P6 fitting.

## Frozen scientific boundaries

Executor:

```text
459935f586fdd16c82013838e6d27a12604c3093
thesis-upmem-kernel-schedule-system-v1
```

P6 software:

```text
2beea27411c16e90ed76988613ddb00bcc09f942
thesis-upmem-cost-guided-software-v1
```

P6 results package:

```text
8df2ebac61bacd08309ea490309be5a8dcb943b2
thesis-upmem-cost-guided-results-v1
```

## Explicit non-goals of the final thesis implementation

The final system does not claim multi-rank execution, asynchronous host/DPU overlap,
automatic heterogeneous placement, energy efficiency, general DPU-resident tensor
networks, or a globally optimal contraction path.

Those are future research directions, not missing conditions for the completed thesis
artifact.
