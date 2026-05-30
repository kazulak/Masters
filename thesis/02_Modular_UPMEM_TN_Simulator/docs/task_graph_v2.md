# Task Graph V2

Task graph v2 is the planned contract between the host planner, dispatcher, and
runtime. It is not implemented yet. The purpose of this document is to avoid
hardcoding architecture decisions into ad hoc JSON fields as the MVP grows.

## Goals

Task graph v2 must make these facts explicit:

- tensor identity, shape, labels, dtype, and storage location;
- operation type and contraction labels;
- legal candidate routes;
- selected route and decision reason;
- slicing plan;
- tile strategy;
- data format and scale metadata;
- host collective requirements;
- expected memory and transfer cost;
- validation/profiling tags.

## Relationship To MVP Task Graph

The MVP graph has:

- `meta`;
- `initial_tensors`;
- `tasks`;
- labels and GEMM dimensions per task;
- `needs_k_tiling`;
- row and column tile counts.

V2 should remain able to represent the MVP dense path, but it must also represent
non-GEMM operations, sparse operations, format variants, multi-DPU slices, and host
reductions.

## Proposed Top-Level Shape

```json
{
  "schema_version": "2.0-draft",
  "meta": {
    "experiment_id": "string",
    "planner": "opt_einsum",
    "planner_seed": 0,
    "hardware_profile": "upmem-local",
    "created_by": "planner"
  },
  "tensors": [],
  "operations": [],
  "routes": [],
  "validation": {},
  "ablation": {}
}
```

## Tensor Record

```json
{
  "id": "tensor_0",
  "shape": [2, 2],
  "labels": [0, 1],
  "logical_dtype": "complex_f64",
  "storage": {
    "kind": "host_binary",
    "path": "data_exchange/tensor_data.bin",
    "offset_real_bytes": 0,
    "offset_imag_bytes": 32
  },
  "lifetime": {
    "produced_by": null,
    "consumed_by": ["op_0"]
  }
}
```

## Operation Record

```json
{
  "id": "op_0",
  "kind": "contraction",
  "inputs": ["tensor_0", "tensor_1"],
  "outputs": ["tensor_2"],
  "labels": {
    "free_A": [2],
    "contracted": [0],
    "free_B": [3],
    "out": [2, 3]
  },
  "gemm_shape": {
    "m": 2,
    "k": 2,
    "n": 2
  },
  "classification": {
    "operation_class": "dense_contraction",
    "density_estimate": 1.0,
    "is_permutation": false,
    "is_diagonal": false,
    "entanglement_hint": "unknown"
  },
  "candidate_routes": ["dense_gemm", "host_reference"],
  "selected_route": "dense_gemm",
  "route_reason": "general dense contraction; heuristic route not applicable"
}
```

## Route-Specific Payload

Dense route example:

```json
{
  "operation_id": "op_0",
  "route": "dense_gemm",
  "format": {
    "name": "complex_i8_tile_scaled",
    "accumulator": "int32",
    "scale_scope": "tile"
  },
  "tiling": {
    "tile_rows": 16,
    "tile_k": 256,
    "tile_cols": 64,
    "k_tiling": false,
    "double_buffering": false,
    "estimated_wram_bytes": 24576
  },
  "slices": [
    {
      "slice_id": "op_0_slice_0",
      "row_range": [0, 2],
      "col_range": [0, 2],
      "k_range": [0, 2],
      "target": {
        "rank": 0,
        "dpu": 0
      }
    }
  ],
  "expected_cost": {
    "host_to_dpu_bytes": 0,
    "dpu_to_host_bytes": 0,
    "int_mac_ops": 8,
    "host_reduction_ops": 0
  }
}
```

Host collective example:

```json
{
  "operation_id": "reduce_0",
  "route": "host_collective",
  "kind": "sum_slices",
  "inputs": ["partial_0", "partial_1"],
  "output": "tensor_final",
  "reason": "UPMEM DPUs cannot reduce peer partials directly"
}
```

## Required Runtime Output

Each executed operation should append or emit a profile record:

```json
{
  "operation_id": "op_0",
  "selected_route": "dense_gemm",
  "status": "ok",
  "timing_seconds": {
    "host_pack": 0.0,
    "dma_out": 0.0,
    "dpu_compute": 0.0,
    "dma_in": 0.0,
    "host_unpack": 0.0,
    "host_reduce": 0.0
  },
  "bytes": {
    "host_to_dpu": 0,
    "dpu_to_host": 0,
    "host_peak_tensor_bytes": 0
  },
  "accuracy": {
    "available": false
  }
}
```

## Compatibility Rule

The first V2 implementation should reproduce the current MVP behavior before
adding new routes. That means:

1. Bell and GHZ fixtures still pass validation.
2. The selected route is always `dense_gemm`.
3. The format is `complex_i8_tile_scaled`.
4. K-tiling remains optional until dense route v2 implements it.
