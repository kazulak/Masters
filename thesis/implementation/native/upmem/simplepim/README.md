# Historical UPMEM Bridge Sources

The former SimplePIM and dense bridge runners in this directory are retired
from the public CLI, Makefile, pytest, and CI surfaces. Their generated
evidence remains readable through normalized records and historical snapshots.

For active use, follow the implementation documentation:

- strict generic-only SDK simulator evidence: `make bench-upmem-sim`;
- manual physical acceptance: `make upmem-hw-taskgraph-resident-plan`, then the
  resident route on the ETH host, as described in
  [the historical taskgraph runbook](../../../docs/history/upmem_hardware_taskgraph_resident_runbook.md).

The base resident route is one DPU and one tasklet, is correctness-only, and
does not claim hardware speedup. M4.6 adds a versioned one-DPU tasklet sweep for
`1/2/4/8/16` tasklets; M5.1 and M5.2 are separate one-contraction distributed
probes using one tasklet per DPU. These are bounded development functionality
routes, not general distributed TN execution or performance evidence. No
retired runner in this directory should be invoked to produce new thesis
evidence.

Physical ETH runs must set `UPMEM_HW_RANK_PATH=/dev/dpu_rankN` together with
`UPMEM_ALLOW_PHYSICAL_HARDWARE=1`. The requested path and effective SDK profile
are recorded; they do not independently prove observed rank identity.
