# Historical Parallelization Roadmap

The earlier parallelization roadmap is retained as historical design context,
not as an active implementation checklist. CPU frontier/hybrid routes and
UPMEM frontier/multi-DPU execution were retired from the Phase B public
command, Make, pytest, and CI surfaces.

The retained contract is intentionally small: planner v2 and multi-DPU
ownership remain modeled invariants, while execution claims are limited to
the active CPU/TN paths, the strict generic-only UPMEM SDK simulator, and the
manual one-DPU, one-tasklet MRAM-resident physical route. Historical route
labels and schema fields remain readable for tracked snapshots.

For current work use `make bench-upmem-sim` or the
[resident hardware runbook](upmem_hardware_taskgraph_resident_runbook.md).
Do not run retired dense, SimplePIM, frontier, hybrid, or multi-DPU runners.
