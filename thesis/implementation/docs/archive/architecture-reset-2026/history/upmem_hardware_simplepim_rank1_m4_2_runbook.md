# M4.2 SimplePIM rank-1 qualification

M4.2 qualifies the pinned SimplePIM array-operator path on two physical DPUs:
two scatter operations, virtual zip, pairwise int64 product, and host-mediated
int64 reduction. It is a capability probe for exactly one rank-1 contraction
task, not a general TaskGraph or performance benchmark.

## Prepare without allocation

```bash
make doctor
make upmem-hw-m4-2-plan
```

The command stages/builds the native route and runs its parser mode. The plan
must report `dpu_allocation_attempted: false`.

## Execute on ETH hardware

```bash
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m4-2
```

The physical run uses two DPUs, twelve tasklets per DPU, one warmup and five
measured repetitions. A failed native run is not retried on a simulator or
CPU.

## Inspect copied evidence

```bash
UPMEM_HW_M4_2_RUN=runs/inbox/eth/m4_2/<timestamp> \
  make upmem-hw-m4-2-report
```

Reports are written under `runs/comparisons/upmem_hardware_simplepim_rank1_m4_2/`.
Raw evidence remains outside Git or in the ignored ETH inbox.

## Claim boundary and next blocker

The route proves SimplePIM allocation and operator invocation for a fixed
deterministic rank-one kernel qualification fixture. It is not a
`ContractionTask` or `TaskGraph` adapter and does not prove generic TaskGraph execution,
physical operand transport from a circuit lowering, PID-Comm communication,
ATiM kernels, persistence, scaling, speedup, or energy efficiency. The native
route currently owns its deterministic vectors; genuine operand transport from
a real `ContractionTask` remains the next adapter milestone. The native
response records SimplePIM-managed allocation and metadata checks. Because the
pinned SimplePIM operator APIs are void, those checks do not independently
validate intermediate table contents; the final int64 result is the numerical
check.
