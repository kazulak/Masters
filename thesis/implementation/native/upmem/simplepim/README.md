# Historical UPMEM Bridge Sources

The former SimplePIM and dense bridge runners in this directory are retired
from the public CLI, Makefile, pytest, and CI surfaces. Their generated
evidence remains readable through normalized records and historical snapshots.

For active use, follow the implementation documentation:

- strict generic-only SDK simulator evidence: `make bench-upmem-sim`;
- manual physical acceptance: `make upmem-hw-taskgraph-resident-plan`, then the
  resident route on the ETH host, as described in
  `docs/upmem_hardware_taskgraph_resident_runbook.md`.

The resident route is one DPU and one tasklet, is correctness-only, and does
not claim hardware speedup. No retired runner in this directory should be
invoked to produce new thesis evidence.
