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

The Python-side dense bridge contract is defined in
`src/quantum_bench/targets/upmem/dense_bridge.py`. It writes an inspectable
Wave 2C.7 file boundary for one prepared dense task:

```text
<bridge_dir>/
  input_manifest.json
  operands/*.npy
  references/*.npy
```

The current mock bridge reads that manifest, writes `output_manifest.json` and
`outputs/mock_dequantized_output.npy`, and performs only NumPy validation of the
file contract. It does not call SimplePIM, native UPMEM, or an external command.

Wave 2C.8 adds the backend adapter interface:

- `mock_numpy_dequantized` executes only the local NumPy mock backend.
- `simplepim_external` is the future external-process backend ID.

For `simplepim_external`, Python records planned command metadata only:
`input_manifest.json`, `output_manifest.json`, `outputs/simplepim_output.npy`,
and SimplePIM environment key names. Even with `execute_external=true`, the
current adapter returns `simplepim_external_execution_not_implemented` and does
not call a subprocess.

Raw UPMEM SDK experiments should remain separate from this bridge. SimplePIM is
the preferred first dense execution path if it is practical because it should
reduce early SDK and kernel boilerplate while the host-side task, tile-plan, and
validation contracts are still stabilizing.
