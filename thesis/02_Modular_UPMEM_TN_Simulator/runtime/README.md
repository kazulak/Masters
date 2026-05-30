# Runtime

Future home for UPMEM host runtime code.

Responsibilities:

- allocate DPUs and ranks;
- load route-specific DPU binaries;
- pack and unpack DMA buffers;
- enforce tile and WRAM budgets;
- launch kernels;
- collect timing and byte counters;
- perform host-mediated reductions and reshuffles.

The runtime should not decide contraction paths. It executes the sized work units
created by the planner and selected by the dispatcher.
