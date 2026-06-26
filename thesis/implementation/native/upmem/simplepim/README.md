# SimplePIM Bridge Placeholder

This directory is reserved for future SimplePIM bridge or wrapper code used by
UPMEM dense GEMM experiments.

No SimplePIM code is implemented here yet. The current runtime only probes for
SimplePIM availability from Python and records metadata in task-route artifacts.

Raw UPMEM SDK experiments should remain separate from this bridge. SimplePIM is
the preferred first dense execution path if it is practical because it should
reduce early SDK and kernel boilerplate while the host-side task, tile-plan, and
validation contracts are still stabilizing.
