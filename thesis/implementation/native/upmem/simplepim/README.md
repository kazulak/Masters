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
- `upmem_sdk_simulator_dense` is the first real simulator-backed dense bridge
  backend. It uses UPMEM SDK C/DPU code, not the SimplePIM API, because this
  SimplePIM tree does not provide a GEMM primitive.

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

## UPMEM SDK Simulator Dense Runner

`upmem_sdk_dense_runner.py` consumes the same `input_manifest.json` as the mock
and stub backends. It copies only the minimal dense source set from
`upmem_sdk_dense/` into the run artifact directory:

```text
bridge/runner_work/
  src/
  build/
  inputs/
  outputs/
```

The runner builds and executes inside `runner_work`; it must not mutate this
repository's native source tree. For `L1_WRAM`, the native buffer contract is
padded row-major int8 inputs and padded row-major little-endian int32 output
with shapes taken from the validated manifest/config. Host code writes
`max_dim x max_dim` buffers, passes explicit `a_stride`, `b_stride`, and
`c_stride` values to the DPU, and the DPU indexes A, B, and C using those
padded strides. This is required for non-square GEMM shapes.

For `L2_SINGLE_DPU_MRAM`, the same runner uses exact row-major int8 MRAM
operand buffers and row-major little-endian int32 C output. The DPU keeps one
output tile accumulator in WRAM, loops over all K tiles, then writes the C tile
to MRAM once. This is an internal UPMEM execution class of the future unified
runtime, not a separate final route.

The supported bring-up command is:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case bell_2q --task-index 0 --backend upmem_sdk_simulator_dense --execute-external
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench dense-task-bridge --case synthetic_l2_square --task-index 0 --backend upmem_sdk_simulator_dense --execute-external
```

This path:

- supports the task-level `L1_WRAM` padded direct subset;
- supports the task-level `L2_SINGLE_DPU_MRAM` real-valued single-DPU
  MRAM/WRAM tiled subset;
- supports int8 operands only in this wave;
- uses a conservative default max dimension of 16, configurable by
  `UPMEM_DENSE_SIM_MAX_DIM`, for L1;
- uses `UPMEM_DENSE_L2_NATIVE_MAX_DIM` and
  `UPMEM_DENSE_L2_MAX_HOST_BLOB_BYTES` for the L2 developer subset;
- supports real GEMM and unambiguous split-complex real/imag layout for L1;
- rejects complex L2 with `complex_l2_not_implemented`;
- runs with `DPU_BACKEND=simulator`;
- writes `outputs/upmem_sdk_simulator_output.npy`;
- validates against `references/expected_dequantized_output.npy`.

Metadata explicitly reports:

- `backend_family: upmem_sdk`
- `simplepim_api_used: false`
- `simplepim_bridge_lane: true`
- `target: simulator`
- `execution_class: L1_WRAM` or `L2_SINGLE_DPU_MRAM`
- `kernel_strategy: l1_padded_direct_v1` or
  `l2_single_dpu_mram_wram_tiled_v1`
- `native_buffer_layout: row_major_padded` for L1 or `row_major` for L2
- `stride_model: explicit_padded_stride_v1` for L1 or
  `row_major_dynamic_stride_v1` for L2
- `upmem_dpu_program_executed: true`
- `simulator_kernel_executed: true`
- `hardware_kernel_executed: false`

The runner CLI has a `--target hardware` value for future compatibility, but
Wave 2E.1 must report `hardware_target_disabled` and must not launch hardware.
Recorded build/run timings are bring-up timings, not final performance
evidence.

Raw UPMEM SDK experiments should remain explicit control/fallback
implementations inside this bridge lane. SimplePIM is the target UPMEM
compute/runtime abstraction for L1/L2 and local tile compute inside L3, but it
must not be claimed as the dense execution path until a SimplePIM kernel is
actually integrated and validated.

## Physical One-DPU Dense MVP

`upmem_sdk_dense_hardware_mvp_runner.py` is deliberately separate from the
simulator runner. It builds an isolated copy of `upmem_sdk_dense/` with
`MAX_DIM=4`, `NR_TASKLETS=1`, and `UPMEM_DENSE_HARDWARE_MVP=1`, uses one
physical DPU, and validates deterministic int8 x int8 -> int32 L1 output
against a CPU reference. It never selects `DPU_BACKEND=simulator` and never
falls back to NumPy or the generic loop.

Use the public runbook rather than invoking the runner directly:

```bash
make PYTHON=.venv/bin/python upmem-hw-mvp-plan
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make PYTHON=.venv/bin/python upmem-hw-mvp
```

This is Phase 1A functionality evidence only. It is not a SimplePIM compute
integration, a generic contraction path, or a hardware performance claim.

## UPMEM SDK Simulator Generic Loop Runner

`upmem_sdk_generic_loop_runner.py` is an intentionally unoptimized fallback
contract for small real-valued binary tensor contractions. It is separate from
the dense GEMM runner and uses:

- `backend_id: upmem_sdk_simulator_generic_loop`
- `kernel_family: generic_loop_fallback`
- `simplepim_api_used: false`
- `native_sdk_control_path: true`

The bridge manifest stores human-readable labels for audit only. The native
contract uses compact integer axis maps, shape arrays, stride arrays, and
relative `.npy` blob paths. The DPU program does not parse labels or JSON-like
structures.

The validation target is quantization-aware:

```text
int8 left x int8 right -> int32 accumulation -> dequantized output
```

`expected_quantized_reference_output.npy` is the CPU int32-accumulation
reference dequantized with the recorded scales. `full_precision_reference_output.npy`
is diagnostic only and is not the native validation target.

The generic kernel exists for coverage and route validation. It is not a
performance kernel and should not be used for speedup claims. It currently
rejects complex tensors with `complex_generic_loop_not_implemented`; native
compile defaults are rank 16, 65536 elements, and a 256-element output tile.
It keeps operands and output in MRAM, reads scalar inputs through aligned
8-byte windows, and uses one DPU/tasklet with no input caching. The surrounding
contract's cap remains unchanged. It guards int32 overflow with:

```text
contracted_combination_count * 127 * 127 <= int32_max
```

Developer command:

```bash
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench generic-task-bridge --case bell_2q --task-index 0 --backend upmem_sdk_simulator_generic_loop --execute-external
```

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
- the implementation-local fallback `external/SimplePIM` from
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
