# Milestone M7: Functional Technical Implementation Plan

## Overview

This document outlines the exhaustive technical implementation plan for Milestone M7 (PIM-Aware Path Cost Model & Contraction Path Optimizer Engine), adhering strictly to the required Functional Programming Architectural Directives.

## 1. Centralized Immutable Parameter Registry

We replace scattered cost constants with a centralized, inspectable, and frozen dataclass (`PIMCostParameters`). This registry holds all immutable weights, scale multipliers, capacity constraints, and policy defaults, ensuring easy auditing by the user.

```python
from dataclasses import dataclass, field, replace
from typing import Mapping, Any

@dataclass(frozen=True)
class PIMCostParameters:
    # Immutable weights
    w_flops: float = 1.0
    w_h2d: float = 1.0
    w_d2h: float = 1.0
    w_mram_dma: float = 1.0
    w_wram: float = 1.0
    w_sync: float = 1.0
    w_complex_penalty: float = 1.0
    
    # Scale multipliers
    scale_h2d: float = 1.0
    scale_d2h: float = 1.0
    
    # Policy constraints
    memory_limit: int | None = None
    
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PIMCostParameters":
        """Instantiate parameters from an abstract dictionary mapping."""
        valid_keys = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)
```

## 2. Pure Functions & Explicit State Transitions

The plan strictly prohibits hidden side-effects or mutable OOP class states. Path optimization uses explicit state transition objects passed through pure functions. 

```python
@dataclass(frozen=True)
class PathSearchState:
    active_tensors: tuple[tuple[int, ...], ...]
    size_dict: Mapping[int, int]
    history: tuple[tuple[int, int], ...]
    total_cost: float
    parameters: PIMCostParameters

def calculate_pim_step_cost(
    t1_labels: tuple[int, ...], 
    t2_labels: tuple[int, ...], 
    size_dict: Mapping[int, int], 
    params: PIMCostParameters
) -> float:
    """Pure cost calculation utilizing the centralized PIMCostParameters."""
    # (Implementation of specific cost calculation logic based on w_flops, w_h2d, etc.)
    return 0.0

def eval_pair_step(state: PathSearchState, pair: tuple[int, int]) -> tuple[PathSearchState, float]:
    """Pure state transformation evaluating the cost of contracting a pair."""
    i, j = pair
    t1 = state.active_tensors[i]
    t2 = state.active_tensors[j]
    
    step_cost = calculate_pim_step_cost(t1, t2, state.size_dict, state.parameters)
    
    # Pure generation of next active sequence
    next_active = _derive_next_active(state.active_tensors, pair)
    
    next_state = PathSearchState(
        active_tensors=next_active,
        size_dict=state.size_dict,
        history=state.history + (pair,),
        total_cost=state.total_cost + step_cost,
        parameters=state.parameters
    )
    return next_state, step_cost
```

## 3. Abstract & Flexible Function Signatures (KISS)

In adherence to KISS principles, all top-level functions and class instantiations accept dictionary/mapping configuration objects (`config: Mapping[str, Any]`), avoiding rigid explicit positional arguments.

## 4. OptEinsum & TaskGraph Integration

We supply a pure functional path finder compliant with the custom path optimizer interface expected by `opt_einsum`.

```python
def pim_path_finder_functional(
    inputs: list[set[int]], 
    output: set[int], 
    size_dict: dict[int, int], 
    memory_limit: int | None = None, 
    **kwargs
) -> list[tuple[int, int]]:
    """Pure functional path optimizer for opt_einsum."""
    config = kwargs.get("config", {})
    params = PIMCostParameters.from_config(config)
    
    if memory_limit is not None:
        params = replace(params, memory_limit=memory_limit)
        
    initial_state = PathSearchState(
        active_tensors=tuple(tuple(s) for s in inputs),
        size_dict=size_dict,
        history=(),
        total_cost=0.0,
        parameters=params
    )
    
    # Executes pure search logic (e.g., greedy algorithm recursively transforming state)
    final_state = _greedy_search_pure(initial_state)
    return list(final_state.history)
```

Integration within `src/quantum_bench/tn/planners.py`:

```python
class UpmemPIMCostGreedyPlanner:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options = dict(options) if options else {}
        self.identity = PlannerIdentity(
            planner_engine="custom_upmem",
            planner_id="upmem_pim_cost_greedy",
            planner_kind="external_path_optimizer",
            optimize_mode="upmem_pim_cost_greedy",
            objective="pim_cost",
            cost_basis="upmem_pim_model",
            target_estimate_key=None,
            options=self.options,
            planner_config=self.options,
        )

    def plan(self, network: TensorNetworkValue) -> PlannerResult:
        import opt_einsum as oe
        from quantum_bench.tn.network import interleaved_einsum_args
        
        start = time.perf_counter()
        path, path_info = oe.contract_path(
            *interleaved_einsum_args(network), 
            optimize=pim_path_finder_functional,
            config=self.options
        )
        planning_time_s = time.perf_counter() - start
        
        return PlannerResult(
            identity=self.identity,
            path=tuple(tuple(int(item) for item in step) for step in path),
            path_info_text=str(path_info),
            largest_intermediate=None,
            naive_flops=None,
            optimized_flops=None,
            planning_time_s=planning_time_s,
            metadata={"planner_config_hash": self.identity.planner_config_hash}
        )

def planner_from_config(config: dict[str, Any] | None) -> PathPlanner:
    # Existing configurations...
    config = config or {}
    engine = str(config.get("engine", "opt_einsum"))
    optimize_mode = str(config.get("optimize", "greedy"))
    
    if engine == "custom_upmem" and optimize_mode == "upmem_pim_cost_greedy":
        return UpmemPIMCostGreedyPlanner(options=config)
    # ...
```

## 5. Comprehensive Unit Test Suite

The test suite ensures robustness and alignment with architectural constraints:

- `test_pim_cost_parameters_inspection()`: Validates `from_config` instantiations, ensuring missing values fallback to defaults and unused mappings are ignored.
- `test_eval_pair_step_pure_state_transition()`: Validates that `PathSearchState` and `PIMCostParameters` instances remain unmutated (verified by ID checks and assertions) post `eval_pair_step` invocation.
- `test_weight_sensitivity()`: Injects varying values (e.g., highly weighted `w_h2d`) to verify accurate cost penalty shifts within `calculate_pim_step_cost`.
- `test_task_graph_integration()`: Runs a benchmark `TensorNetworkValue` through `plan_task_graph_with_config` utilizing `engine="custom_upmem"` and `optimize="upmem_pim_cost_greedy"` to guarantee integration without execution failure.
