# Milestone M5 Phase 2 & 3: Technical Implementation Plan

## Overview
This implementation plan covers the refinement and completion of Phase 2 (Python Execution Planner & Descriptors) and Phase 3 (Multi-DPU Test Suite) for the M5 Distributed Single Contraction feature. It follows a strict SMART, production-grade, and KISS methodology, while strictly preserving backward compatibility with M4.6.

## 1. Descriptor Integration (`hardware_taskgraph_resident.py`)

### Problem Statement
While the `PlacementPlan` and `CommunicationPlan` schemas are defined, they are not fully integrated into the `ResidentGraphPackage` and `HardwareTaskGraphResidentProfile`. `build_resident_graph_package` must generate these explicit plans.

### Implementation Details
- **`ResidentGraphPackage`**: Add `placement_plan: Optional[PlacementPlan]` and `communication_plan: Optional[CommunicationPlan]` as properties. Their constructor defaults MUST be `None` to preserve 100% single-DPU M4.6 compatibility.
- **Graceful Serialization**: Update `to_json_dict()` to serialize these plans gracefully (e.g., omitting them or setting to null if `None`), ensuring single-DPU manifests remain clean and unbroken.
- **`HardwareTaskGraphResidentProfile`**: Ensure properties like `requested_dpu_count` are properly mapped to these plans.
- **`build_resident_graph_package`**:
  - Specify that the builder generates default plans or `None` when `profile.requested_dpu_count == 1` as a fallback.
  - When `profile.requested_dpu_count > 1`: Automatically instantiate `PlacementPlan` via `build_m5_placement_plan(dpu_group_count=profile.requested_dpu_count)`.
  - Calculate `logical_bytes` and instantiate `CommunicationPlan` using `build_m5_communication_plan(...)`.
  - Pass these generated plans (or `None`) to the `ResidentGraphPackage` constructor.
  - Make sure memory budgets accurately reflect the replicated operands and resident tensor ownership.

## 2. Multi-DPU Runtime Propagation (`taskgraph_runtime.py` & `runtime_evidence.py`)

### Problem Statement
The runtime evidence and Python executor do not yet propagate the `PlacementPlan` and `CommunicationPlan` through `execute_upmem_taskgraph_runtime`, nor is the `int32` host partial sum reduction helper fully integrated for quantized multi-DPU execution.

### Implementation Details
- **`taskgraph_runtime.py`**:
  - Update `execute_upmem_taskgraph_runtime` to optionally accept or derive `PlacementPlan` and `CommunicationPlan`.
  - Implement a Python helper `reduce_multi_dpu_partial_sums(partial_sums, dtype)` which properly accumulates `int32` partial sums for quantized runs *before* applying requantization logic, and handles standard IEEE 754 float32 addition otherwise.
- **`runtime_evidence.py`**:
  - Update `_summary_payload` and `UpmemTaskGraphRuntimeResult` to accept and serialize `placement_plan: Optional[PlacementPlan]` and `communication_plan: Optional[CommunicationPlan]`. Defaults must be `None`.
  - Serialization must handle `None` values gracefully to keep single-DPU evidence unpolluted.
  - Ensure the JSON evidence explicitly notes `host_mediated_reduction` latency vs. pure kernel execution time where possible.

## 3. Expanded Test Suite (`test_upmem_simplepim_taskgraph_executor.py`)

### Problem Statement
The current tests cover the standalone builders for `PlacementPlan` and `CommunicationPlan`, but lack integration testing with `build_resident_graph_package` and exact validation of `int8`/`int32` quantized reduction.

### Implementation Details
- **`test_m5_resident_graph_package_multi_dpu`**:
  - Verify that `build_resident_graph_package` with `requested_dpu_count > 1` correctly attaches a valid `PlacementPlan` and `CommunicationPlan`.
  - Verify that `build_resident_graph_package` with `requested_dpu_count == 1` defaults to `None` for these plans and doesn't pollute the JSON output.
  - Check the output of `to_json_dict()` contains the `dpu_group_ownership` and `collective_kind` when plans are present.
- **`test_m5_quantized_int32_host_reduction`**:
  - Generate a set of `int32` simulated DPU partial sums.
  - Verify that the `reduce_multi_dpu_partial_sums` helper correctly sums in `int32` space without intermediate overflow, and correctly applies post-reduction requantization to `int8` with the proper absolute maximum scale.
  - Contrast this with incorrect early-requantization to prove numeric contract adherence.

## Acceptance Criteria
- `ResidentGraphPackage.to_json_dict()` explicitly contains `placement_plan` and `communication_plan` when multi-DPU, and handles `None` gracefully for single-DPU backward compatibility.
- Evidence manifests emitted by `execute_upmem_taskgraph_runtime` trace the DPU ownership and reduction strategy.
- Python partial sum reductions for quantized modes accumulate strictly in `int32` before requantization.
- The unit test suite passes with 100% coverage on new multi-DPU plan integration and reduction paths.
