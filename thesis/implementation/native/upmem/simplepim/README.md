# SimplePIM Bridge Placeholder

This directory is reserved for future SimplePIM bridge or wrapper code used by
UPMEM dense GEMM experiments.

No SimplePIM code is implemented here yet. The current runtime probes for
SimplePIM availability from Python, records metadata in task-route artifacts,
and provides an explicit host-side dry-run microbenchmark command:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simplepim-microbench --dry-run --m 8 --k 8 --n 8
```

That command writes `simplepim_microbench.json`, performs deterministic
host-side fixed-point conversion and NumPy reference GEMM, and does not execute
SimplePIM or any SimplePIM external command.

Raw UPMEM SDK experiments should remain separate from this bridge. SimplePIM is
the preferred first dense execution path if it is practical because it should
reduce early SDK and kernel boilerplate while the host-side task, tile-plan, and
validation contracts are still stabilizing.
