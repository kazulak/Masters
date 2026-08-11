# M7: PIM-Aware Path Cost Model & Contraction Path Optimizer Engine Implementation Plan

## Overview
This document outlines the SMART, production-grade, KISS technical implementation plan for Milestone M7. It introduces a custom UPMEM-aware contraction path optimizer that integrates into the existing `opt_einsum` interface, allowing the contraction planning process to natively evaluate the physical costs (flops, transfers, WRAM/MRAM constraints) of UPMEM architecture using the `PathCostComponentsV2` framework.

## 1. Custom `opt_einsum` Path Optimizer
**Target File**: `src/quantum_bench/tn/upmem_path_optimizer.py`

### `PIMCostWeights` Dataclass
- **Purpose**: Parameterize the cost function for path evaluation.
- **Implementation**: Define a `@dataclass` holding scalar weights: `w_flops`, `w_h2d`, `w_d2h`, `w_mram_dma`, `w_wram`, `w_sync`, `w_complex_penalty`.
- **Default Values**: Match the `balanced_literature_informed` profile values from `upmem_path_cost_v2.py`.

### `PIMPathCostOptimizer` API Contract
- **Purpose**: Implement the custom path optimizer compliant with `opt_einsum`'s custom cost function interface.
- **Explicit Signature**:
  If using `opt_einsum.paths.greedy` with a custom `cost_fn`, the exact signature expected by `opt_einsum` is:
  `cost_fn(size12: int, size1: int, size2: int, k12: int, k1: int, k2: int) -> float`
  However, this signature lacks tensor structural awareness (like shape indices).
  Alternatively, providing a full custom path optimizer signature natively supported by `opt_einsum.contract_path(..., optimize=pim_path_finder)` requires:
  ```python
  def pim_path_finder(
      inputs: list[set[str]],
      output: set[str],
      size_dict: dict[str, int],
      memory_limit: int | None = None,
      **kwargs
  ) -> list[tuple[int, int]]:
      ...
  ```
  We will implement `pim_path_finder` using this exact signature, internally executing a greedy or randomized-greedy algorithm that queries `model_upmem_task_cost_v2` for each candidate pairwise contraction.
- **Implementation Details**:
  - Accepts a `PIMCostWeights` instance upon initialization (via a closure or functor object wrapping the `pim_path_finder` signature).
  - Evaluates pairwise contraction candidates during the path search by mapping candidates to `ContractionTask` mockups.
  - Calls `model_upmem_task_cost_v2` to retrieve `PathCostComponentsV2`.
  - Computes a scalar cost using the weighted sum of `PathCostComponentsV2` metrics.
  - Falls back to `math.inf` for candidates flagged as `feasibility=False`.

## 2. Planner Integration
**Target File**: `src/quantum_bench/tn/planners.py`

### Register New Planners & Integration Mechanism
- **Purpose**: Expose the custom optimizer through existing graph planning tools.
- **Integration Mechanism**:
  `opt_einsum` accepts a callable as the `optimize` argument in `oe.contract_path(..., optimize=callable)`. 
  `upmem_pim_cost_greedy` will instantiate our `PIMPathCostOptimizer` callable and pass it directly to `oe.contract_path`.
- **Implementation**:
  - Add `upmem_pim_cost_greedy` and `upmem_pim_cost_random` to the planner registry / factory methods (`planner_from_config`).
  - `upmem_pim_cost_greedy` invokes `oe.contract_path(..., optimize=pim_path_finder_instance)`.
  - `upmem_pim_cost_random` wraps `pim_path_finder` in a randomized search logic compliant with the same callable interface.

### Metadata Propagation
- **Implementation**:
  - Update `PlannerIdentity` and `PlannerResult` metadata gathering to include the instantiated `PIMCostWeights` and the normalized cost components for the selected path.
  - Ensure compatibility with `plan_task_graph_with_config` and `plan_task_graph`.

## 3. Unit Test Suite
**Target File**: `tests/test_upmem_path_optimizer.py`

### Custom Path Optimization Verification
- **Implementation**:
  - Provide standard quantum circuit tensor networks (Bell, QRNG, GHZ).
  - Verify that `upmem_pim_cost_greedy` successfully completes a path search without errors and yields a valid contraction tree.

### Weight Sensitivity
- **Implementation**:
  - Define extreme configurations of `PIMCostWeights` (e.g., heavily penalize memory transfers `w_h2d = 1000` vs heavily penalize flops `w_flops = 1000`).
  - Assert that varying weights strictly changes the resulting contraction tree ordering (the generated path tuple differs between configurations).

### Compatibility Tests
- **Implementation**:
  - Instantiate graphs using `plan_task_graph_with_config` specifying `engine: custom_upmem` and `optimize: upmem_pim_cost_greedy`.
  - Ensure graph attributes, task dependencies, and metadata dictionaries are populated correctly and conform to existing M0-M6 contracts.
