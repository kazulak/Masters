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
- `simplepim_external_stub` is a non-executing external-process contract stub.

For `simplepim_external`, Python records planned command metadata only:
`input_manifest.json`, `output_manifest.json`, `outputs/simplepim_output.npy`,
and SimplePIM environment key names. Even with `execute_external=true`, the
current adapter returns `simplepim_external_execution_not_implemented` and does
not call a subprocess.

`simplepim_dense_stub.py` is the Wave 2C.11 external contract stub. It is a
Python script that consumes `input_manifest.json`, validates manifest-relative
`.npy` blobs, and writes a `dense_bridge_output` `output_manifest.json` with:

- `backend: simplepim_external_stub`
- `status: stub_executed`
- `output_blob: null`
- `validation_metrics.status: not_applicable`
- `external_command_executed: true`
- `execution_implemented: false`
- `metadata.native_kernel_executed: false`

The stub does not execute SimplePIM, native UPMEM, or any dense kernel. It
writes no output blob by default.

The runtime may call this stub only through the `simplepim_external_stub`
backend, only with `--execute-external`, and only when `SIMPLEPIM_STUB_BIN` is
configured. Relative `SIMPLEPIM_STUB_BIN` values are resolved from the current
working directory; the validation command is intended to be run from
`thesis/implementation`:

```bash
SIMPLEPIM_STUB_BIN=native/upmem/simplepim/simplepim_dense_stub.py PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend simplepim_external_stub --execute-external
```

The adapter invokes the script with `sys.executable`, so the file does not need
executable permissions or a shebang.

Raw UPMEM SDK experiments should remain separate from this bridge. SimplePIM is
the preferred first dense execution path if it is practical because it should
reduce early SDK and kernel boilerplate while the host-side task, tile-plan, and
validation contracts are still stabilizing.

## Environment Verification

Before replacing the stub with real SimplePIM/native execution, run the project
environment check from `thesis/implementation`:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-env-check
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench upmem-env-check --run-sample --target simulator
```

The checker records:

- `UPMEM_HOME`, if configured;
- UPMEM tools found on `PATH`, including `dpu-upmem-dpurte-clang` and
  `dpu-pkg-config`;
- `SIMPLEPIM_HOME`, if configured;
- `SIMPLEPIM_STUB_BIN`, if configured for the non-executing stub;
- the repository fallback `../legacy/extern/SimplePIM` from
  `thesis/implementation`;
- optional SimplePIM sample build and simulator run status.

`SIMPLEPIM_HOME` identifies the source tree but does not prove execution.
Without `--run-sample`, simulator and hardware execution remain not verified.
With `--run-sample --target simulator`, the checker copies the SimplePIM source
tree into the timestamped run directory, builds the `benchmarks/va` sample, and
runs it with `DPU_BACKEND=simulator`. The source copy is intentionally isolated
from the external dependency tree.

Hardware is never exercised by default. `--target hardware` records hardware as
not verified unless a future explicit hardware-safe sample path is added.
Command stdout/stderr are bounded in artifacts; configured homes and tool paths
may be absolute because they describe the local machine, while run artifact
paths are relative.
