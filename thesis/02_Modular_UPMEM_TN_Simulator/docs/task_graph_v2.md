# Task Graph V2

TaskGraphV2 is the central artifact of the project. It is the contract between
the host planner, dispatcher, route providers, runtime, validator, and experiment
runner.

Without TaskGraphV2, the project is just a collection of UPMEM kernels. With
TaskGraphV2, every operation has explicit structure, cost, route eligibility,
format choice, validation, and profiling metadata.

## Goals

TaskGraphV2 must make these facts explicit:

- tensor identity, shape, labels, logical dtype, physical format, and location;
- operation kind and index expression;
- tensor structure and density estimates;
- legal candidate routes;
- selected route and rejected routes with reasons;
- selected data format and conversion requirements;
- slicing and tile strategy;
- collective requirements;
- estimated transfer, compute, preparation, conversion, reduction, and error cost;
- measured timing, bytes, status, and correctness metrics.

## Relationship To The MVP Task Graph

The current MVP graph has:

- `meta`;
- `initial_tensors`;
- `tasks`;
- labels and GEMM dimensions per task;
- `needs_k_tiling`;
- row and column tile counts.

V2 must represent this dense path exactly enough to replay it. It must also
represent non-GEMM operations, sparse operations, format variants, multi-DPU
slices, collective reductions, rejected routes, and validation output.

## Top-Level Shape

```json
{
  "schema_version": "2.0-draft",
  "meta": {},
  "hardware": {},
  "ablation": {},
  "tensors": [],
  "tasks": [],
  "route_decisions": [],
  "profiles": [],
  "validation": []
}
```

## Meta Record

```json
{
  "experiment_id": "bell_2q_v2_replay_0001",
  "created_at_utc": "2026-05-30T00:00:00Z",
  "created_by": "planner",
  "planner": {
    "name": "opt_einsum",
    "version": "unknown",
    "seed": 0,
    "mode": "mvp_replay"
  },
  "source": {
    "kind": "qasm",
    "path": "../01_MVP_DenseGEMM/python_frontend/circuits/bell_2q.qasm"
  },
  "run_tags": ["stage_1", "mvp_replay"]
}
```

## Hardware Record

The graph should record the target hardware profile used for legality checks.

```json
{
  "profile_id": "upmem-local",
  "upmem_sdk": "unknown",
  "host_cpu": "unknown",
  "rank_count": 0,
  "dpu_count": 0,
  "dpu_mram_bytes": 67108864,
  "dpu_wram_bytes": 65536,
  "tasklets_per_dpu": 24,
  "notes": []
}
```

Unknown values are allowed in planning fixtures, but experiment runs must fill
them before results are used in the thesis.

## Ablation Record

```json
{
  "enabled_routes": [
    "cpu_reference",
    "raw_upmem_dense",
    "simplepim_default"
  ],
  "disabled_routes": [
    "custom_dense",
    "sparsep",
    "pidcomm_collective"
  ],
  "forced_route": null,
  "format_policy": "route_default",
  "collective_policy": "naive_first",
  "cost_model": "rules_v0"
}
```

## TensorMetadata Record

```json
{
  "id": "tensor_0",
  "shape": [2, 2],
  "labels": ["q0_in", "q0_out"],
  "logical_dtype": "complex_f64",
  "structure": "dense",
  "density_estimate": 1.0,
  "nnz_estimate": 4,
  "format": {
    "name": "complex_f64_host",
    "accumulator": "complex_f64",
    "scale_scope": "none",
    "metadata": {}
  },
  "storage": {
    "location": "host",
    "kind": "host_binary",
    "path": "data_exchange/tensor_data.bin",
    "offset_real_bytes": 0,
    "offset_imag_bytes": 32,
    "byte_length": 64
  },
  "lifetime": {
    "produced_by": null,
    "consumed_by": ["task_0"]
  }
}
```

Allowed `structure` values:

```text
dense
sparse
diagonal
permutation
scalar
identity
unknown
```

## TaskNode Record

Each contraction or transformation becomes a `TaskNode`.

```json
{
  "id": "task_0",
  "op_kind": "contraction",
  "input_tensor_ids": ["tensor_0", "tensor_1"],
  "output_tensor_id": "tensor_2",
  "dependencies": [],
  "index_expression": "ab,bc->ac",
  "input_shapes": [[2, 2], [2, 2]],
  "output_shape": [2, 2],
  "labels": {
    "free_left": ["a"],
    "contracted": ["b"],
    "free_right": ["c"],
    "output": ["a", "c"]
  },
  "gemm_shape": {
    "m": 2,
    "k": 2,
    "n": 2
  },
  "structure": "dense",
  "density_estimate": 1.0,
  "nnz_estimate": 4,
  "candidate_routes": [
    "cpu_reference",
    "simplepim_default",
    "raw_upmem_dense"
  ],
  "selected_route": "raw_upmem_dense",
  "selected_data_format": {
    "name": "complex_i8_tile_scaled",
    "accumulator": "int32",
    "scale_scope": "tile",
    "metadata": {
      "real_imag_layout": "split"
    }
  },
  "rejected_routes_with_reasons": [
    {
      "route": "simplepim_default",
      "reason": "disabled_by_stage_1_mvp_replay"
    }
  ],
  "estimated_cost": {},
  "slicing": {},
  "validation_policy": {
    "reference_route": "cpu_reference",
    "metrics": ["max_abs_error", "max_rel_error", "norm_drift"]
  }
}
```

Allowed `op_kind` values:

```text
contraction
transpose
diagonal_apply
permutation
elementwise
reduction
collective
reshape
format_conversion
host_only
```

## CostRecord

Every selected and rejected route should have an estimate. Rejected routes can
have partial estimates when a legality check fails early.

```json
{
  "task_id": "task_0",
  "route": "raw_upmem_dense",
  "data_format": "complex_i8_tile_scaled",
  "status": "estimated",
  "host_to_dpu_bytes": 4096,
  "dpu_to_host_bytes": 2048,
  "mram_wram_bytes": 8192,
  "integer_ops": 8,
  "host_ops": 0,
  "preparation_cost_seconds": null,
  "conversion_cost_seconds": null,
  "reduction_cost_seconds": null,
  "estimated_total_seconds": null,
  "quantization_error": null,
  "memory": {
    "estimated_wram_bytes": 24576,
    "estimated_mram_bytes": 4096,
    "host_peak_tensor_bytes": 0
  },
  "notes": []
}
```

Cost fields may be unknown during Stage 1. They must still exist so later stages
do not change the schema shape.

## Slicing And Tile Strategy

Dense route example:

```json
{
  "tile_rows": 16,
  "tile_k": 256,
  "tile_cols": 64,
  "k_tiling": false,
  "double_buffering": false,
  "tasklets": 16,
  "estimated_wram_bytes": 24576,
  "slices": [
    {
      "slice_id": "task_0_slice_0",
      "row_range": [0, 2],
      "col_range": [0, 2],
      "k_range": [0, 2],
      "target": {
        "rank": 0,
        "dpu": 0
      },
      "requires_collective": false
    }
  ]
}
```

Sparse route example:

```json
{
  "sparse_format": "csr",
  "density_threshold": 0.05,
  "estimated_conversion_bytes": 1024,
  "estimated_nnz": 128,
  "partitioning": "route_default",
  "requires_densification_after": false
}
```

Collective route example:

```json
{
  "kind": "sum_slices",
  "inputs": ["partial_0", "partial_1"],
  "output": "tensor_final",
  "provider": "naive_host_collective",
  "reason": "multi_dpu_k_slices_require_reduction"
}
```

## RouteDecisionRecord

Route decisions should be easy to inspect without parsing every task field.

```json
{
  "task_id": "task_0",
  "selected_route": "raw_upmem_dense",
  "selected_format": "complex_i8_tile_scaled",
  "decision_policy": "rules_v0",
  "decision_reason": "stage_1_mvp_replay_uses_frozen_raw_dense_baseline",
  "candidate_routes": [
    {
      "route": "cpu_reference",
      "eligible": true,
      "reason": "always_available_for_small_fixture",
      "estimated_cost": {}
    },
    {
      "route": "simplepim_default",
      "eligible": false,
      "reason": "disabled_by_stage_1_mvp_replay",
      "estimated_cost": {}
    },
    {
      "route": "raw_upmem_dense",
      "eligible": true,
      "reason": "dense_contraction_supported_by_mvp_wrapper",
      "estimated_cost": {}
    }
  ]
}
```

## ProfileRecord

Each executed task appends or emits a profile record.

```json
{
  "task_id": "task_0",
  "selected_route": "raw_upmem_dense",
  "selected_format": "complex_i8_tile_scaled",
  "status": "ok",
  "timing_seconds": {
    "host_pack": 0.0,
    "format_conversion": 0.0,
    "route_prepare": 0.0,
    "h2d_dma": 0.0,
    "dpu_kernel": 0.0,
    "d2h_dma": 0.0,
    "host_unpack": 0.0,
    "host_reduce": 0.0,
    "validation": 0.0,
    "total": 0.0
  },
  "bytes": {
    "host_to_dpu": 0,
    "dpu_to_host": 0,
    "mram_wram": 0,
    "host_peak_tensor_bytes": 0
  },
  "hardware": {
    "rank_count": 1,
    "dpu_count": 1,
    "tasklets": 16
  },
  "counters": {},
  "notes": []
}
```

## ValidationRecord

```json
{
  "task_id": "task_0",
  "reference_route": "cpu_reference",
  "compared_route": "raw_upmem_dense",
  "status": "pass",
  "tolerance": {
    "max_abs_error": 0.01,
    "max_rel_error": 0.01,
    "norm_drift": 0.01
  },
  "metrics": {
    "max_abs_error": 0.0,
    "max_rel_error": 0.0,
    "norm_drift": 0.0,
    "fidelity": null
  },
  "notes": []
}
```

## Execution Log

The runtime should write one `execution_log.json` per run. It should contain:

- copied or referenced TaskGraphV2 input;
- route decisions after dispatcher selection;
- profile records after execution;
- validation records;
- environment and dependency versions;
- ablation flags;
- failure records if execution stops.

The log is the primary evidence for thesis claims. It must be machine-readable
and stable enough for plotting scripts.

## Compatibility Rule

The first V2 implementation must reproduce the current MVP behavior before adding
new routes. That means:

1. Bell and GHZ fixtures still pass validation.
2. The selected non-reference route is `raw_upmem_dense`.
3. The selected non-reference format is `complex_i8_tile_scaled`.
4. SimplePIM exists only as a disabled or unimplemented candidate in Stage 1.
5. K-tiling remains represented but optional until dense route v2 implements it.

## Schema Evolution Rule

When a new provider needs extra data, add a nested provider payload rather than
renaming core fields. Core fields should stay stable so old experiment logs remain
readable.

Good:

```json
{
  "provider_payload": {
    "sparsep": {
      "format": "csr",
      "partitioning": "1d"
    }
  }
}
```

Avoid:

```json
{
  "route_specific_magic_field": "..."
}
```
