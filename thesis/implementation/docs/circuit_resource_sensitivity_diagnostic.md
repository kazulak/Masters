# Circuit-Structure and Resource-Sensitivity Diagnostic

This milestone records a diagnostic-only physical comparison of three
18-qubit tensor-network workloads on one UPMEM rank. It is not a final
performance campaign.

## Frozen experiment

The execution source is `89ecc5527f182f42dc471101f96edf86b0dadefa` and the
canonical experiment is:

- experiment: `892977991db6f3688dfae9ab6adf359629e0780a51c2feb8707abf5ae9271f20`
- run: `20e53f7e-4402-42df-953d-9c7981f9faa4`
- workload cases: Stress18, hidden-shift18, and GHZ18
- routes: one DPU at T1/T4/T8/T12, and one/two/three/four DPUs at T8
- collection: one warmup and five measured blocks per route
- result: 126 successful samples and 126 successful physical sessions
- host/rank: `safari-baguette1`, `/dev/dpu_rank1`, CPU 0
- environment: powersave governor, single-threaded BLAS/OpenMP
- policy: `diagnostic_v1`, `steady_execution_v1`, split-complex float32

Raw evidence is preserved under
`runs/eth/safari-baguette1/89ecc5527f182f42dc471101f96edf86b0dadefa/rerun-20260830T215823Z/`.
The raw and derived checksum manifests are part of that directory.

## Interpretation

Scaling is calculated within each circuit. A circuit is never used as the
denominator for another circuit, and raw runtimes are not pooled across
circuits. Kernel and total-wall speedups are reported separately.

The measured behavior is workload-sensitive: the best total-wall operating
point is T8/D4 for Stress18 and GHZ18, while hidden shift reaches its best
measured point at T12/D2. At the 4-DPU/T8 route, host request overhead and its
payload-record staging component remain substantial across the three cases.
The detailed values are in `analysis/route_statistics.csv`,
`analysis/within_circuit_scaling.csv`, and
`analysis/circuit_resource_sensitivity_summary.json`.

The earlier 125/126 allocation incident is retained as excluded incident
evidence. It is not part of this 126-sample result and does not establish a
scaling claim.

## Supported claims

This result supports physical correctness, exact source and binary
provenance, numerical validation, and descriptive within-circuit tasklet and
one-rank DPU scaling for the listed workload and powersave environment.

It does not support final `physical_performance_v1` estimates, optimized-host
performance, machine-independent acceleration, general circuit-family
scalability, multi-rank scaling, sliced performance, energy claims, or a
universal resource-selection rule. The next optimization must be selected
from a later controlled comparison rather than inferred from this diagnostic.
