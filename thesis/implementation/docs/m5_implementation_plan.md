# Milestone M5 Implementation Plan: Distributed Single Contraction across Multi-DPU Groups

## 1. Multi-DPU Work Partitioning

### 1.1 Output-Tile Decomposition
**Design**:
Instead of computing the entire output tensor on one DPU, we divide the independent output elements across a group of DPUs (e.g., 2, 4, or 8). 
Each DPU is responsible for a non-overlapping contiguous slice of the output. 
Both DPUs receive the complete contracted operands (or the necessary subset) and operate independently.

**C Changes (`common.h` / `host.c` / `dpu.c`)**:
- Remove hardcoded `dpu_alloc(1, ...)` in `host.c` and replace it with `dpu_alloc(request.requested_dpus, ...)` where `requested_dpus` is passed in the manifest.
- Update `resident_operation_t` in `common.h` to include `dpu_slice_offset` and `dpu_slice_elements` or use `DPU_FOREACH` and DPU-specific buffer transfers (`dpu_push_xfer`) to send distinct descriptor parameters to each DPU.
- Each DPU kernel (`dpu.c`) computes its assigned range of `output_elements` based on its specific `dpu_slice_offset`.

### 1.2 Contracted-Axis Partial-Sum Decomposition
**Design**:
When the contracted dimension is large, partitioning along the output dimension is insufficient due to MRAM limits. Instead, all DPUs calculate the full output shape, but only loop over a slice of the contracted dimensions. 
This yields a partial sum per DPU, necessitating a reduction step to yield the final output tensor.

**C Changes**:
- Update `upmem_generic_args_t` in `common.h` to specify `contracted_offset` and `contracted_elements_slice`.
- Each DPU computes a partial result in its output MRAM slot.

## 2. Communication & Reduction

### 2.1 Host-Mediated vs. PID-Comm Reduction with Fallback Logic
**Design**:
For partial-sum decomposition, the results from multiple DPUs must be reduced. The execution flow dictates a strict hierarchy for resolving the reduction strategy:
- **PID-Comm collective reduction**: Leverage the external PID-Comm module to perform `reduce-scatter` or `all-reduce` directly across the DPU group, minimizing host-side bandwidth bottlenecks. This is the primary targeted path.
- **Host-mediated reduction (Fallback Path)**: If PID-Comm is uninitialized, physically unavailable on the specific DIMM/rank topology, or explicitly disabled in the configuration, the runtime will automatically fall back to host-mediated reduction. In this fallback path, the host issues a synchronized `dpu_copy_from` across all participating DPUs, allocates host CPU buffers for the partial sums, computes the accumulation using CPU native floating-point or integer arithmetic, and either commits the result to the host filesystem or scatters it back to the DPUs for subsequent tasks.

### 2.2 CommunicationPlan Schema
In Python, introduce an explicit `CommunicationPlan` class to capture transfers:
```python
@dataclass(frozen=True)
class CommunicationPlan:
    logical_bytes: int
    application_visible_transfer_bytes: int
    collective_kind: str  # "host_mediated_reduction", "pid_comm_all_reduce", "pid_comm_reduce_scatter", etc.
    participants: tuple[str, ...]  # e.g. DPU IDs or group identifiers
    source_ownership: str
    destination_ownership: str
    sync_points: int

    def to_json_dict(self) -> JsonDict: ...
```

## 3. PlacementPlan & Memory Budgets

### 3.1 PlacementPlan Schema
We formalize memory budgets and execution locality in the Python planner:
```python
@dataclass(frozen=True)
class PlacementPlan:
    dpu_group_ownership: tuple[int, ...]
    tasklets_per_dpu: int
    resident_tensor_ownership: Mapping[str, tuple[int, ...]]
    replicated_operands: tuple[str, ...]
    memory_budgets: Mapping[str, int]  # e.g. "mram_pool_bytes": 524288
    
    def to_json_dict(self) -> JsonDict: ...
```
This descriptor enables the host to decide whether to slice an output or replicate an operand based on the target DPU memory capacity.

### 3.2 Numeric Contract for Quantized Accumulators
**Design**:
When distributing partial sums across DPUs, the accumulation mechanism must maintain strict error bounds:
- **Float32 Accumulators**: Standard IEEE 754 float32 precision is used. Due to the non-associativity of floating-point addition, the multi-DPU reduction order may yield a slightly different exact bitwise result than a sequential single-DPU run. The validation phase will bound this difference using exact tolerances calibrated for multi-DPU tree-reduction vs sequential reduction.
- **Quantized (int8/int32) Accumulators**: DPUs computing quantized inner products multiply int8 operands and accumulate into **int32** partial sums. To prevent overflow and truncation error during multi-DPU reduction, the `int32` partial sums are transferred and reduced *before* any requantization to `int8` occurs. The host (or PID-Comm) strictly accumulates these `int32` values. Only after the global reduction is complete is the final absolute maximum scale computed and the combined result requantized back to int8 if required by the subsequent task.

## 4. Python Runtime & Evidence Integration

### 4.1 Surface Multi-DPU Descriptors
In `src/quantum_bench/targets/upmem/hardware_taskgraph_resident.py`:
- `HardwareTaskGraphResidentProfile` will be updated to handle `requested_dpu_count > 1` effectively (currently assumes bounded or 1).
- `ResidentAllocation` and `ResidentOperationDescriptor` generation will be adjusted to produce multiple per-DPU variants or a single broadcast descriptor with DPU-specific execution intervals.

In `src/quantum_bench/targets/upmem/taskgraph_runtime.py`:
- Consume `PlacementPlan` and `CommunicationPlan` and attach them to the execution bundle.
- Support scaling `dpu_group_count` in `upmem_taskgraph_executor_config` and `execute_upmem_taskgraph_runtime`.

### 4.2 Multi-DPU Evidence Metrics
- Record per-DPU occupancy, multi-DPU active cycles, and imbalance in the summary artifact.
- Track scaling efficiency (strong/weak scaling) in `schedule_metadata` and final JSON payload.
- Separate host-reduction latency from pure kernel latency.

## 5. Deadlock & Race Safety across DPU Group
**Design**:
Multi-DPU execution introduces risks of barrier deadlocks or read-after-write hazards. To ensure zero race conditions across multi-DPU launches:
- **Strict Host-Coordinated Synchronization**: The host executes a global barrier (`dpu_sync()` or synchronous `dpu_launch`) across the entire assigned DPU group before any cross-DPU communication (whether via PID-Comm or host-mediated fallback) can occur.
- **Explicit Ownership Transfer**: No DPU is allowed to read a resident MRAM tensor that is actively being written to by another DPU or the host. `CommunicationPlan` explicit `sync_points` enforce that the producer DPU group has fully reached the completed execution state before the consumer DPU group (or host reduction phase) begins reading.
- **PID-Comm Deadlock Avoidance**: PID-Comm collectives are always invoked uniformly across the identical DPU rank group without diverging control flow among tasklets. The host will not launch overlapping asynchronous requests to the same DPU group.

## 6. Unit & Integration Tests

### 6.1 Python Test Suite Updates
In `tests/test_upmem_simplepim_taskgraph_executor.py`:
- Add test: `test_m5_output_tile_multi_dpu_plan` verifying that given a 2-DPU configuration, the output tile is evenly split and `PlacementPlan` reflects distinct output ownership.
- Add test: `test_m5_partial_sum_host_reduction_plan` verifying `CommunicationPlan` generation with `collective_kind="host_mediated_reduction"` and correct application-visible transfer bytes.
- Add test: `test_m5_reference_validation` running a simulated or bounded physical workload verifying the multi-DPU partial sums reconstruct the exact CPU baseline tensor within numeric tolerance.

### Acceptance Criteria (Gate)
- Tile and partial sum ownership are exact (no missing elements, no overlapping compute).
- Reduction error is strictly bounded by the declared numeric contract for both float32 and int8/int32 (validated against exact CPU references).
- The fallback logic gracefully reroutes to host-mediated reduction without crashing when PID-Comm is disabled.
- Synchronization points are explicit, completely avoiding race conditions and deadlocks in multi-DPU execution.
- Observed collectives and transfers perfectly match the `CommunicationPlan` predictions in bytes.
- Multi-DPU execution operates correctly on ETH hardware and is clearly distinguished from theoretical modeled assignment.
