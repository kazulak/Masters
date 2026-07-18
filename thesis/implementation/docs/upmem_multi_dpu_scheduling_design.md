# Historical UPMEM Multi-DPU Design Notes

The former multi-DPU scheduling design and modeled assignment prototype are
retired from the active Phase B command, Make, pytest, and CI surfaces. The
assignment schema and normalized ownership records remain readable so tracked
historical snapshots can still be verified and reported.

The active UPMEM paths are deliberately narrower:

- `make bench-upmem-sim` runs the strict generic-only SDK simulator boundary;
- `make upmem-hw-taskgraph-resident-plan` prepares the one-DPU resident path
  without allocation; and
- the resident hardware route is manual, one DPU and one tasklet, and provides
  functionality/diagnostic evidence only.

Historical modeled assignments describe ownership and dependency invariants;
they do not prove DPU execution, hardware timing, or speedup. Do not invoke
retired frontier, dense, SimplePIM, or multi-DPU runners. Use the resident
[runbook](upmem_hardware_taskgraph_resident_runbook.md) for physical acceptance
and the snapshot verifier/reader for old evidence.
