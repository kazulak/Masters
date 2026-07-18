# Historical Parallelization Notes

The former CPU frontier/hybrid and UPMEM frontier/multi-DPU prototypes are
retired from the active Phase B command, Make, pytest, and CI surfaces. Their
normalized fields, route labels, assignment records, tables, and plots remain
readable for historical snapshots.

The active evidence surface is:

- exact CPU tensor-network and circuit/path correctness;
- modeled planner v2, tile, slot, and multi-DPU ownership invariants;
- strict generic-only UPMEM SDK simulator evidence; and
- the guarded one-DPU, one-tasklet MRAM-resident TaskGraph route for manual
  physical acceptance.

Use `make bench-upmem-sim` for simulator evidence and the
[resident hardware runbook](upmem_hardware_taskgraph_resident_runbook.md) for
manual hardware acceptance. Historical frontier, hybrid, dense, and SimplePIM
evidence is reportable through the snapshot reader only. It does not establish
current parallel execution, hardware timing, or speedup.
