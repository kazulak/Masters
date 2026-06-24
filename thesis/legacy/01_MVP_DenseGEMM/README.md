# UPMEM Dense-GEMM MVP Baseline

This directory contains the fast dense-GEMM MVP that was built for conference
validation. In the revised thesis architecture it should be treated as:

```text
RawUPMEMProvider_baseline
```

It is not the final simulator architecture. Its job is to preserve the smallest
known-good loop that produces correct quantum-circuit amplitudes by executing
tensor-network contraction tiles through a UPMEM DPU int8 GEMM kernel.

The point of this MVP is correctness, reproducibility, and baseline measurement,
not speed. It proves the full loop exists:

1. Build a tensor network from a small quantum circuit.
2. Ask `opt_einsum` for a pairwise contraction path.
3. Convert each pairwise contraction into a GEMM.
4. Tile the GEMM to respect the UPMEM WRAM budget.
5. Quantize each tile to int8 on the host.
6. Run the int8 GEMM on a DPU.
7. Dequantize and accumulate complex results on the host.
8. Compare the final amplitude tensor against NumPy.

## Architecture Review Outcome

The conference version proved the numerical path but hid too much information for
the TaskGraph-centered V2 design. The MVP needed these changes:

| Gap in conference MVP | Change made here |
| --- | --- |
| Task graph did not identify the route or data format. | `task_graph.json` now marks tasks as `raw_upmem_dense` with `complex_i8_tile_scaled`. |
| Route decisions were implicit in the C host. | Host now writes `execution_log.json` with selected route records. |
| Timing was only aggregate DMA/DPU timing. | Host now emits per-task profile records with prepare, pack/quantize, DMA, DPU, and dequantize/accumulate timing. |
| Validation was console-only and did not affect exit code. | `validate.py` now writes `validation_record.json` and exits nonzero on failure. |
| README described the MVP as the simulator. | README now documents it as a raw UPMEM baseline and V2 replay target. |

The dense numerical path itself is intentionally unchanged. K-tiling, SimplePIM,
SparseP, PID-Comm, multi-DPU scheduling, and route-aware planning belong in
`../02_Modular_UPMEM_TN_Simulator`, not inside this MVP.

## Directory Layout

```text
01_MVP_DenseGEMM/
├── Makefile
├── Param.h
├── README.md
├── python_frontend/
│   ├── generate_plan.py
│   ├── validate.py
│   ├── requirements.txt
│   └── circuits/
│       ├── bell_2q.qasm
│       └── ghz_4q.qasm
├── data_exchange/
│   ├── task_graph.json
│   ├── tensor_data.bin
│   ├── reference_output.npy
│   ├── output_amplitudes.bin
│   ├── execution_log.json
│   └── validation_record.json
├── src_c/
│   ├── main.c
│   ├── quantizer.c
│   ├── quantizer.h
│   └── cjson/
│       ├── cJSON.c
│       └── cJSON.h
└── kernels/
    └── gemm_int8/
        ├── dpu.c
        └── map.h
```

Generated build products are ignored by git:

```text
.venv/
bin/
mvp_host
```

## Hardware Model Used Here

The MVP respects the constraints from the scoping review:

- UPMEM DPU WRAM is treated as a scarce resource.
- DPU floating point is avoided completely.
- DPU work uses native integer multiply: `int8 * int8 -> int32`.
- No inter-DPU communication is assumed.
- Host orchestration controls every contraction step and tile dispatch.

The compile-time tile constants live in `Param.h`:

```c
#define WRAM_SAFE_BYTES (48 * 1024)
#define TILE_ROWS       16
#define TILE_K          256
#define TILE_N          64
#define NR_TASKLETS     8
```

The WRAM data budget check is enforced at compile time:

```text
A_tile: TILE_ROWS x TILE_K int8
B_tile: TILE_K x TILE_N int8
C_tile: TILE_ROWS x TILE_N int32
```

With the current parameters:

```text
16*256 + 256*64 + 16*64*4 = 24576 bytes
```

This stays below the 48 KiB safe data budget.

## Execution Pipeline

### 1. Python Planning

`python_frontend/generate_plan.py` generates three files in `data_exchange/`:

- `task_graph.json`: contraction tasks, tensor shapes, index labels, GEMM dimensions,
  tile counts, `needs_k_tiling`, route metadata, data-format metadata, and static
  cost estimates.
- `tensor_data.bin`: initial tensor amplitudes, stored as real float64 block followed
  by imag float64 block for each tensor.
- `reference_output.npy`: NumPy reference amplitude tensor for validation.

The planner currently uses hardcoded circuits:

- `bell_2q`
- `ghz_4q`

The `.qasm` files are included as documentation/test fixtures, but QASM parsing is not
part of this MVP.

### 2. Host Orchestration

`src_c/main.c` reads `task_graph.json` and `tensor_data.bin`, then runs the
contraction tasks in order. It writes both `output_amplitudes.bin` and
`execution_log.json`.

For each contraction task:

1. Look up input tensors in the host-side registry.
2. Reorder tensor dimensions according to JSON index labels.
3. Reshape the contraction into:

   ```text
   A: m x k
   B: k x n
   C: m x n
   ```

4. Iterate over `(row_block, col_block)` tiles.
5. Dispatch four real int8 GEMMs for complex multiplication:

   ```text
   C_real = A_real @ B_real - A_imag @ B_imag
   C_imag = A_real @ B_imag + A_imag @ B_real
   ```

6. Dequantize `int32` accumulators back to host `double`.
7. Store the result tensor back into the registry.

If any task has `needs_k_tiling: true`, the host aborts clearly. K-tiling is deliberately
out of scope for this MVP.

The host route is fixed:

```text
selected_route = raw_upmem_dense
selected_format = complex_i8_tile_scaled
decision_policy = mvp_fixed_route
```

This is deliberate. V2 should wrap this as the raw dense baseline, not turn this
directory into the full dispatcher.

### 3. DPU Kernel

The DPU code is in:

```text
kernels/gemm_int8/dpu.c
kernels/gemm_int8/map.h
```

`map.h` defines the required DPU entry point:

```c
void map(void)
```

The kernel receives one packed tile through the `DPU_INPUT` host symbol and writes the
result to `DPU_OUTPUT`. Each tasklet owns a contiguous range of output rows.

This project uses direct UPMEM SDK calls in the host:

```c
dpu_alloc
dpu_load
dpu_copy_to
dpu_launch
dpu_copy_from
```

The repository still uses `common.mk` for UPMEM SDK build flags and path conventions,
but this specific MVP kernel is launched directly rather than through SimplePIM's
generic element-wise map wrapper. The reason is practical: the local SimplePIM copy
wraps `map_func(input_element, output_element)`, while this MVP needs one DPU invocation
to consume a whole GEMM tile.

## Setup

Run these commands from this directory:

```bash
cd thesis/01_MVP_DenseGEMM
```

Create an isolated Python environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python_frontend/requirements.txt
```

Build the host binary and DPU binary:

```bash
make
```

This produces:

```text
mvp_host
bin/dpu_gemm_int8
```

## Run Bell State

Generate the Bell contraction plan:

```bash
.venv/bin/python python_frontend/generate_plan.py bell_2q
```

Run the host orchestrator:

```bash
./mvp_host \
  data_exchange/task_graph.json \
  data_exchange/tensor_data.bin \
  data_exchange/output_amplitudes.bin \
  data_exchange/execution_log.json
```

Validate:

```bash
.venv/bin/python python_frontend/validate.py
```

Expected result:

```text
[validate] PASS - amplitude within int8 tolerance
```

The reference Bell state is:

```text
[1/sqrt(2), 0, 0, 1/sqrt(2)]
```

## Run 4-Qubit GHZ State

Generate the GHZ contraction plan:

```bash
.venv/bin/python python_frontend/generate_plan.py ghz_4q
```

Run:

```bash
./mvp_host \
  data_exchange/task_graph.json \
  data_exchange/tensor_data.bin \
  data_exchange/output_amplitudes.bin \
  data_exchange/execution_log.json
```

Validate:

```bash
.venv/bin/python python_frontend/validate.py
```

Expected result:

```text
[validate] PASS - amplitude within int8 tolerance
```

The reference GHZ state has nonzero amplitudes only at `|0000>` and `|1111>`.

## Simulator vs Real Hardware

If no physical UPMEM rank is visible, the SDK prints warnings like:

```text
Fallback to SIMULATOR backend as no hardware device was found
```

That is acceptable for local correctness testing. The same binary still exercises the
UPMEM SDK path:

```text
host -> DPU input symbol -> DPU launch -> DPU output symbol -> host
```

On a machine with real UPMEM hardware visible to the SDK, the same commands should run
on hardware without source changes.

## Output Files

### `task_graph.json`

Contains:

- global tile metadata
- MVP schema version and baseline role
- initial tensor records
- one task per pairwise contraction
- GEMM dimensions: `m`, `k`, `n`
- tile counts: `n_row_blocks`, `n_col_blocks`
- tensor index labels needed by the C host to reshape correctly
- `needs_k_tiling`
- candidate and selected routes
- selected data format
- static byte/op estimates

Quick inspection:

```bash
.venv/bin/python - <<'PY'
import json
with open("data_exchange/task_graph.json") as f:
    doc = json.load(f)
print(doc["meta"])
for task in doc["tasks"]:
    print(task["task_id"], "m/k/n =", task["m"], task["k"], task["n"],
          "needs_k_tiling =", task["needs_k_tiling"])
PY
```

### `tensor_data.bin`

Initial tensors are serialized as:

```text
tensor_0 real float64[]
tensor_0 imag float64[]
tensor_1 real float64[]
tensor_1 imag float64[]
...
```

Offsets are recorded in `task_graph.json`.

### `output_amplitudes.bin`

Written by `mvp_host` as:

```text
int32 n_elements
float64 real[n_elements]
float64 imag[n_elements]
```

`python_frontend/validate.py` reads this file and compares it with
`reference_output.npy`.

### `execution_log.json`

Written by `mvp_host`. Contains:

- baseline role: `RawUPMEMProvider_baseline`
- fixed route decisions
- selected format
- per-task GEMM shape and tile count
- per-task timing:
  - host prepare
  - host pack/quantize
  - host-to-DPU DMA
  - DPU kernel
  - DPU-to-host DMA
  - host dequantize/accumulate
- aggregate timing and byte counts

This is the file V2 should use as evidence when replaying the MVP route.

### `validation_record.json`

Written by `python_frontend/validate.py`. Contains:

- compared route: `raw_upmem_dense`
- reference route: `cpu_reference_numpy`
- selected data format
- tolerance
- max absolute error
- max relative error
- norm drift
- fidelity when available
- pass/fail status

## Performance Report

`mvp_host` prints:

```text
=== Performance Report ===
Total wall time
Host prepare
Host pack/quant
DMA host->DPU
DPU map phase
DMA DPU->host
Host dequant/acc
GEMM tile calls
Bytes host->DPU
Bytes DPU->host
Output path
Execution log path
Output n_elements
```

For simulator runs, the timing is useful only as a software sanity check. For real
hardware runs, the DMA fractions become the baseline measurement this MVP is designed
to expose.

## Clean Build

```bash
make clean
make
```

This removes:

```text
bin/
mvp_host
```

It does not remove:

```text
.venv/
data_exchange/
```

## Important MVP Boundaries

Implemented:

- dense tensor-network contraction
- `opt_einsum` contraction path
- raw UPMEM fixed route metadata
- static route/data-format metadata in `task_graph.json`
- int8 tile quantization
- complex GEMM via four real GEMMs
- DPU-side int8 GEMM
- host-side double accumulation
- per-task execution log
- machine-readable validation record
- Bell and GHZ validation

Not implemented:

- QASM parser
- K-tiling for `k > TILE_K`
- SimplePIM provider
- sparse route
- heuristic route
- multi-DPU row distribution
- collective provider
- gate fusion or contraction-path research heuristics
- route-aware planning
- performance tuning

These omissions are intentional. This MVP is the smallest complete loop that can
produce correct amplitudes through a UPMEM DPU kernel and a baseline artifact that
V2 can replay.

## Troubleshooting

### Python imports fail

Use the local venv:

```bash
.venv/bin/python python_frontend/generate_plan.py bell_2q
```

Do not install into system Python for this project.

### `dpu-upmem-dpurte-clang` not found

Load the UPMEM SDK environment so these commands are available:

```bash
dpu-upmem-dpurte-clang
dpu-pkg-config
```

### `needs_k_tiling` is true

The circuit generated a contraction with `k > TILE_K`. The C host will abort because
K-tiling is out of MVP scope.

### No hardware devices found

The SDK may fall back to the simulator backend. That is fine for correctness checks.
For thesis measurements, rerun on a host where UPMEM ranks are visible to the SDK.
