# Current Milestone Walkthrough

The authoritative active-route status is maintained in
[README.md](../../README.md#what-is-active).

M4.1--M4.5 are physically accepted bounded milestones. M4.5 is the current
accepted SimplePIM-managed baseline: it provides the bounded descriptor-driven
shared runtime and one- and two-DPU functionality evidence.

M4.6 passed as a development acceptance sweep on one physical DPU for tasklets
`1/2/4/8/16`: 1680 validated rows across 12 small cases, two path variants,
two numeric modes, and seven repeats. The sweep is functionality and diagnostic
tasklet evidence, not a final scaling benchmark.

M5.1 passed a bounded real-float32 output-partition probe on 1/2/4 DPUs.
M5.2 passed the corresponding contracted-axis probe with deterministic
`host_mediated_sum_v1` reduction and maximum absolute error `2.98e-08`. Both
are functionality-only development probes. M5.3 is blocked before allocation:
the pinned PID-Comm source is incompatible with the ETH SDK 2023.1 symbols and
macros. See the [M4/M5 acceptance record](../m4_m5_physical_acceptance.md) for
commands and run IDs.

SimplePIM remains central: it supplies management/allocation for the bounded
M4/M5 routes, while thesis-owned kernels perform the demonstrated contractions.
These milestones do not claim speedup, energy, general distributed execution,
PID-Comm, ATiM, SparseP, multi-rank/DIMM execution, or completed scaling
evidence.
