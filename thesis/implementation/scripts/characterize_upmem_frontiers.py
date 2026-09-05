#!/usr/bin/env python3
"""Bounded, source-only DAG frontier and UPMEM headroom census."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from math import prod
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from quantum_bench.lowering import (
    build_contraction_dag,
    contraction_dag_hash,
    lower_tensor_network,
)
from quantum_bench.model import ContractNode, ContractionDAG, GraphNode, ReduceNode, make_simulation_job
from quantum_bench.upmem.plan import UpmemPlan, UpmemTopology, UpmemWorkUnit, plan_upmem, physical_plan_id
from quantum_bench.upmem.protocol import (
    COMPLETION_BYTES,
    CONTROL_BYTES,
    MRAM_POOL_BYTES,
    WRAM_PANEL_KC,
)
from quantum_bench.upmem.tiling import canonical_label_geometry
from quantum_bench.circuits import builtin_circuit


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SCRIPT = Path(__file__).with_name("characterize_upmem_execution.py")
SCHEMA = "upmem_frontier_census_v1"
TIMING_REASON = "source_only_no_execution"
FOUR_PRODUCT_COUNT = 4
EXPECTED_CELL_COUNT = 40
EXPECTED_EXCLUSION_COUNT = 4
RUNTIME_OUTPUT_DTYPE = np.dtype("complex64")
RESIDENT_PAIR_NUMERIC_POLICY = "split_complex_float32_v1"
MRAM_RESERVATION_COMPONENTS = ("2A", "2B", "4C")
RESIDENT_PAIR_UNVERIFIED_REASONS = (
    "same_dpu_locality_unverified",
    "full_intermediate_layout_unverified",
    "intermediate_reconstruction_unverified",
    "no_split_k_unverified",
    "scale_handling_unverified",
)
LIVENESS_EXCLUDED_CATEGORIES = (
    "encoded_operands",
    "raw_lane_values",
    "transport_copies",
    "runtime_object_overheads",
)
LIVENESS_MEMORY_SCOPE = (
    "partial logical tensor payload accounting, not whole-host admission/RSS"
)
LIVENESS_ALIAS_CAVEAT = (
    "logical nbytes can double-count shared NumPy storage aliases"
)


def _load_execution_census() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_upmem_execution_census_v1", EXECUTION_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load existing census script: {EXECUTION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_execution_census = _load_execution_census()
BASE_SHA = _execution_census.BASE_SHA
LOWERING_TIMEOUT = getattr(_execution_census, "LOWERING_TIMEOUT", 60.0)
POLICIES = _execution_census.POLICIES
POOL_FILES = _execution_census.POOL_FILES
CIRCUITS = _execution_census.CIRCUITS
frozen_cells = _execution_census.frozen_cells
characterize_cell = _execution_census.characterize_cell


class DependencyIndex:
    """Validated deterministic dependency metadata for one semantic DAG."""

    __slots__ = (
        "nodes",
        "predecessors",
        "successors",
        "topological_order",
        "cohorts",
    )

    def __init__(
        self,
        *,
        nodes: Mapping[str, GraphNode],
        predecessors: Mapping[str, tuple[str, ...]],
        successors: Mapping[str, tuple[str, ...]],
        topological_order: tuple[str, ...],
        cohorts: tuple[tuple[str, ...], ...],
    ) -> None:
        self.nodes = nodes
        self.predecessors = predecessors
        self.successors = successors
        self.topological_order = topological_order
        self.cohorts = cohorts


def _node_views(node: GraphNode) -> tuple[Any, ...]:
    if isinstance(node, ContractNode):
        return (node.left, node.right)
    if isinstance(node, ReduceNode):
        return node.inputs
    raise TypeError(f"unsupported DAG node type: {type(node).__name__}")


def _validate_dependency_index(dag: ContractionDAG) -> DependencyIndex:
    nodes: dict[str, GraphNode] = {}
    for node in dag.nodes:
        if not isinstance(node, (ContractNode, ReduceNode)):
            raise TypeError(f"unsupported DAG node type: {type(node).__name__}")
        if not isinstance(node.node_id, str) or not node.node_id:
            raise ValueError("DAG node IDs must be nonempty strings")
        if node.node_id in nodes:
            raise ValueError(f"duplicate DAG node ID: {node.node_id!r}")
        nodes[node.node_id] = node

    predecessors: dict[str, tuple[str, ...]] = {}
    successors_mutable: dict[str, list[str]] = {
        node_id: [] for node_id in nodes
    }
    for node_id, node in nodes.items():
        dependencies = tuple(node.dependencies)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"duplicate dependency in node {node_id!r}")
        missing = sorted(set(dependencies) - nodes.keys())
        if missing:
            raise ValueError(
                f"missing dependency for node {node_id!r}: {missing[0]!r}"
            )
        predecessors[node_id] = tuple(sorted(dependencies))
        for dependency in dependencies:
            successors_mutable[dependency].append(node_id)

    successors = {
        node_id: tuple(sorted(values))
        for node_id, values in successors_mutable.items()
    }
    remaining = {node_id: len(predecessors[node_id]) for node_id in nodes}
    ready = sorted(node_id for node_id, count in remaining.items() if count == 0)
    topological: list[str] = []
    while ready:
        node_id = ready.pop(0)
        topological.append(node_id)
        for successor in successors[node_id]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.append(successor)
        ready.sort()
    if len(topological) != len(nodes):
        raise ValueError("cyclic DAG dependencies")

    cohort_remaining = {node_id: len(predecessors[node_id]) for node_id in nodes}
    cohort_ready = sorted(
        node_id for node_id, count in cohort_remaining.items() if count == 0
    )
    cohorts: list[tuple[str, ...]] = []
    while cohort_ready:
        cohort = tuple(cohort_ready)
        cohorts.append(cohort)
        next_ready: list[str] = []
        for node_id in cohort:
            for successor in successors[node_id]:
                cohort_remaining[successor] -= 1
                if cohort_remaining[successor] == 0:
                    next_ready.append(successor)
        cohort_ready = sorted(next_ready)

    return DependencyIndex(
        nodes=nodes,
        predecessors=predecessors,
        successors=successors,
        topological_order=tuple(topological),
        cohorts=tuple(cohorts),
    )


def validate_dependency_graph(dag: ContractionDAG) -> DependencyIndex:
    """Validate explicit node dependencies and return deterministic graph facts."""

    return _validate_dependency_index(dag)


validate_dag_dependencies = validate_dependency_graph


def dependency_ready_cohorts(dag: ContractionDAG) -> tuple[tuple[str, ...], ...]:
    """Return sorted dependency-ready cohorts without changing the DAG."""

    return validate_dependency_graph(dag).cohorts


def frontier_widths(dag: ContractionDAG) -> tuple[int, ...]:
    """Return the width of every deterministic dependency-ready cohort."""

    return tuple(len(cohort) for cohort in dependency_ready_cohorts(dag))


def _dtype_bytes(dtype: str) -> int:
    try:
        return int(np.dtype(dtype).itemsize)
    except TypeError as exc:
        raise ValueError(f"unsupported tensor dtype {dtype!r}") from exc


def _tensor_descriptors(dag: ContractionDAG) -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    for tensor in dag.tensors:
        if tensor.id in descriptors:
            raise ValueError(f"duplicate tensor ID: {tensor.id!r}")
        descriptors[tensor.id] = tensor
    for node in dag.nodes:
        output_id = node.output.id
        if output_id in descriptors:
            raise ValueError(f"duplicate produced tensor ID: {output_id!r}")
        descriptors[output_id] = node.output
    return descriptors


def _geometry(node: ContractNode) -> dict[str, int | str | tuple[int, ...]]:
    batch, m, k, n = canonical_label_geometry(
        node.left.labels,
        node.left.shape,
        node.right.labels,
        node.right.shape,
        node.output.labels,
    )
    category = geometry_category(batch, m, n, k)
    return {
        "b": batch,
        "m": m,
        "n": n,
        "k": k,
        "geometry": (batch, m, n, k),
        "geometry_category": category,
        "one_product_real_mac_work": batch * m * n * k,
        "four_product_real_mac_work": FOUR_PRODUCT_COUNT * batch * m * n * k,
    }


def geometry_category(batch: int, m: int, n: int, k: int) -> str:
    """Classify canonical B/M/N/K geometry into one deterministic bucket."""

    if batch > 1:
        return "batched_gemm"
    if m == 1 and n == 1:
        return "dot"
    if k == 1:
        return "outer_product"
    if m == 1 or n == 1:
        return "gemv"
    if m == n:
        return "square_gemm"
    if min(m, n) <= 8 or max(m, n) >= 4 * min(m, n):
        return "skinny_gemm"
    return "rectangular_gemm"


def _plan_units(plan: UpmemPlan | None) -> dict[str, tuple[UpmemWorkUnit, ...]]:
    units: dict[str, list[UpmemWorkUnit]] = defaultdict(list)
    if plan is not None:
        for stage in plan.stages:
            for unit in stage.work_units:
                units[unit.node_id].append(unit)
    return {node_id: tuple(values) for node_id, values in units.items()}


def _plan_order(dag: ContractionDAG, plan: UpmemPlan | None, index: DependencyIndex) -> tuple[str, ...]:
    if plan is None:
        return index.topological_order
    order = tuple(
        node_id for stage in plan.stages for node_id in stage.node_ids
    )
    if len(order) != len(index.nodes) or set(order) != set(index.nodes):
        raise ValueError("UPMEM plan does not cover the complete DAG")
    if len(set(order)) != len(order):
        raise ValueError("UPMEM plan repeats a DAG node")
    positions = {node_id: position for position, node_id in enumerate(order)}
    for node_id, predecessors in index.predecessors.items():
        if any(positions[dependency] >= positions[node_id] for dependency in predecessors):
            raise ValueError(f"UPMEM plan violates dependency order for {node_id!r}")
    return order


def _runtime_tensor_bytes(
    tensor_id: str,
    descriptors: Mapping[str, Any],
    *,
    inputs: Mapping[str, np.ndarray] | None,
    produced_ids: set[str],
) -> int:
    if inputs is not None and tensor_id in inputs:
        return int(np.asarray(inputs[tensor_id]).nbytes)
    descriptor = descriptors[tensor_id]
    elements = prod(descriptor.shape) if descriptor.shape else 1
    if tensor_id in produced_ids:
        return elements * int(RUNTIME_OUTPUT_DTYPE.itemsize)
    return elements * _dtype_bytes(descriptor.dtype)


def _liveness_estimates(
    dag: ContractionDAG,
    index: DependencyIndex,
    order: tuple[str, ...],
    *,
    inputs: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    descriptors = _tensor_descriptors(dag)
    produced_ids = {node.output.id for node in dag.nodes}
    uses: dict[str, list[int]] = defaultdict(list)
    for position, node_id in enumerate(order):
        for view in _node_views(index.nodes[node_id]):
            if view.tensor_id not in descriptors:
                raise ValueError(
                    f"node {node_id!r} references unknown tensor {view.tensor_id!r}"
                )
            uses[view.tensor_id].append(position)
    last_use = {tensor_id: max(positions) for tensor_id, positions in uses.items()}
    final_tensor_id = dag.output.tensor_id
    if final_tensor_id not in descriptors:
        raise ValueError(f"DAG output references unknown tensor {final_tensor_id!r}")

    logical_bytes = {
        tensor_id: int(
            (prod(descriptor.shape) if descriptor.shape else 1)
            * _dtype_bytes(descriptor.dtype)
        )
        for tensor_id, descriptor in descriptors.items()
    }
    runtime_bytes = {
        tensor_id: _runtime_tensor_bytes(
            tensor_id,
            descriptors,
            inputs=inputs,
            produced_ids=produced_ids,
        )
        for tensor_id in descriptors
    }

    theoretical_live: set[str] = set()
    theoretical_peak_bytes = 0
    theoretical_peak_ids: tuple[str, ...] = ()
    theoretical_snapshots: list[dict[str, Any]] = []
    for position, node_id in enumerate(order):
        node = index.nodes[node_id]
        required = {view.tensor_id for view in _node_views(node)}
        theoretical_live.update(required)
        theoretical_live.add(node.output.id)
        current_bytes = sum(runtime_bytes[tensor_id] for tensor_id in theoretical_live)
        if current_bytes > theoretical_peak_bytes:
            theoretical_peak_bytes = current_bytes
            theoretical_peak_ids = tuple(sorted(theoretical_live))
        theoretical_snapshots.append(
            {
                "node_id": node_id,
                "live_tensor_ids": sorted(theoretical_live),
                "live_tensor_bytes": current_bytes,
            }
        )
        for tensor_id in required:
            if (
                last_use.get(tensor_id) == position
                and tensor_id != final_tensor_id
            ):
                theoretical_live.discard(tensor_id)
        if node.output.id not in uses and node.output.id != final_tensor_id:
            theoretical_live.discard(node.output.id)

    retained_live = set(dag_tensor.id for dag_tensor in dag.tensors)
    retained_peak_bytes = sum(runtime_bytes[tensor_id] for tensor_id in retained_live)
    retained_peak_ids = tuple(sorted(retained_live))
    retained_snapshots: list[dict[str, Any]] = []
    for node_id in order:
        node = index.nodes[node_id]
        retained_live.add(node.output.id)
        current_bytes = sum(runtime_bytes[tensor_id] for tensor_id in retained_live)
        if current_bytes > retained_peak_bytes:
            retained_peak_bytes = current_bytes
            retained_peak_ids = tuple(sorted(retained_live))
        retained_snapshots.append(
            {
                "node_id": node_id,
                "live_tensor_ids": sorted(retained_live),
                "live_tensor_bytes": current_bytes,
            }
        )

    return {
        "tensor_logical_bytes": dict(sorted(logical_bytes.items())),
        "tensor_runtime_bytes": dict(sorted(runtime_bytes.items())),
        "minimal_theoretical": {
            "policy": "just_in_time_tensor_values_with_last_use_release_v1",
            "memory_scope": LIVENESS_MEMORY_SCOPE,
            "estimate_kind": "minimal_logical_tensor_payload_accounting",
            "process_memory_lower_bound_claimed": False,
            "peak_live_tensor_bytes": theoretical_peak_bytes,
            "peak_live_tensor_count": len(theoretical_peak_ids),
            "peak_live_tensor_ids": list(theoretical_peak_ids),
            "snapshots": theoretical_snapshots,
        },
        "actual_retained_runtime": {
            "policy": "runtime_working_mapping_retains_inputs_and_all_outputs_v1",
            "memory_scope": LIVENESS_MEMORY_SCOPE,
            "estimate_kind": "partial_logical_tensor_payload_accounting",
            "process_memory_lower_bound_claimed": False,
            "numpy_storage_aliases_deduplicated": False,
            "logical_nbytes_alias_double_count_possible": True,
            "whole_host_admission": False,
            "excluded_retained_categories": list(LIVENESS_EXCLUDED_CATEGORIES),
            "excluded_retained_categories_reason": (
                "source census does not reconstruct these retained runtime buffers"
            ),
            "accounting_caveats": [
                LIVENESS_ALIAS_CAVEAT,
                "does not measure process RSS or whole-host memory admission",
            ],
            "peak_live_tensor_bytes": retained_peak_bytes,
            "peak_live_tensor_count": len(retained_peak_ids),
            "peak_live_tensor_ids": list(retained_peak_ids),
            "snapshots": retained_snapshots,
        },
        "memory_scope": LIVENESS_MEMORY_SCOPE,
        "process_memory_lower_bound_claimed": False,
        "numpy_storage_aliases_deduplicated": False,
        "logical_nbytes_alias_double_count_possible": True,
        "whole_host_admission": False,
        "excluded_retained_categories": list(LIVENESS_EXCLUDED_CATEGORIES),
        "accounting_caveats": [
            LIVENESS_ALIAS_CAVEAT,
            "does not measure process RSS or whole-host memory admission",
        ],
        "measured_timing": None,
        "measured_timing_reason": TIMING_REASON,
    }


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _mram_extents(
    m: int, n: int, k: int, numeric_policy: str
) -> dict[str, int | bool | str | None]:
    if numeric_policy not in POLICIES:
        raise ValueError(f"unsupported UPMEM numeric policy: {numeric_policy!r}")
    input_element_bytes = 1 if numeric_policy == "complex_int8_shared_scale_v1" else 4
    left_plane_bytes = _align8(m * k * input_element_bytes)
    right_plane_bytes = _align8(k * n * input_element_bytes)
    output_plane_bytes = _align8(m * n * 4)
    single = left_plane_bytes + right_plane_bytes + output_plane_bytes
    fused_operands = 2 * left_plane_bytes + 2 * right_plane_bytes
    fused_outputs = FOUR_PRODUCT_COUNT * output_plane_bytes
    fused = fused_operands + fused_outputs
    admitted = fused <= MRAM_POOL_BYTES
    return {
        "numeric_policy": numeric_policy,
        "left_plane_bytes_aligned": left_plane_bytes,
        "right_plane_bytes_aligned": right_plane_bytes,
        "output_plane_bytes_aligned": output_plane_bytes,
        "single_product_live_mram_bytes": single,
        "fused_four_product_operand_bytes": fused_operands,
        "fused_four_product_output_bytes": fused_outputs,
        "fused_four_product_live_mram_bytes": fused,
        "mram_reservation_bytes": fused,
        "mram_reservation_components": list(MRAM_RESERVATION_COMPONENTS),
        "mram_reservation_scope": "A_B_C_data_only_v1",
        "control_bytes_outside_mram_reservation": CONTROL_BYTES,
        "completion_bytes_outside_mram_reservation": COMPLETION_BYTES,
        "control_completion_mram_reserved_bytes": 0,
        "control_completion_memory_scope": "WRAM_outside_MRAM_reservation_v1",
        "mram_limit_bytes": MRAM_POOL_BYTES,
        "fused_four_product_admitted": admitted,
        "fused_four_product_headroom_bytes": MRAM_POOL_BYTES - fused,
        "admission_reason": None
        if admitted
        else "fused_four_product_live_mram_exceeds_512KiB",
        "fusion_route": (
            "fused_four_product_generic_upmem"
            if admitted
            else "generic_upmem_no_retile"
        ),
        "nonfit_fallback": None if admitted else "generic_upmem_no_retile",
        "retile_requested": False,
    }


def fused_four_product_mram(
    node: ContractNode | UpmemWorkUnit,
    numeric_policy: str,
) -> dict[str, int | bool | str | None]:
    """Estimate one fused complex tile using the existing v4 MRAM limit."""

    if isinstance(node, UpmemWorkUnit):
        return _mram_extents(node.m_size, node.n_size, node.k_size, numeric_policy)
    if not isinstance(node, ContractNode):
        raise TypeError("fused_four_product_mram expects ContractNode or UpmemWorkUnit")
    _, m, k, n = canonical_label_geometry(
        node.left.labels,
        node.left.shape,
        node.right.labels,
        node.right.shape,
        node.output.labels,
    )
    return _mram_extents(m, n, k, numeric_policy)


def _operation_mram(
    node: ContractNode,
    units: tuple[UpmemWorkUnit, ...],
    numeric_policy: str,
) -> dict[str, Any]:
    if not units:
        return {
            "plan_backed": False,
            "fused_four_product_live_mram_bytes": None,
            "fused_four_product_admitted": None,
            "mram_reservation_bytes": None,
            "mram_reservation_components": list(MRAM_RESERVATION_COMPONENTS),
            "mram_reservation_scope": "A_B_C_data_only_v1",
            "control_bytes_outside_mram_reservation": CONTROL_BYTES,
            "completion_bytes_outside_mram_reservation": COMPLETION_BYTES,
            "control_completion_mram_reserved_bytes": 0,
            "control_completion_memory_scope": "WRAM_outside_MRAM_reservation_v1",
            "mram_limit_bytes": MRAM_POOL_BYTES,
            "admission_reason": "retained_physical_plan_unavailable",
            "fusion_route": "not_planned",
            "nonfit_tile_count": None,
            "nonfit_tile_fraction": None,
            "nonfit_fallback": None,
            "retile_requested": False,
        }
    unit_facts = [fused_four_product_mram(unit, numeric_policy) for unit in units]
    peak = max(int(fact["fused_four_product_live_mram_bytes"]) for fact in unit_facts)
    single_peak = max(int(fact["single_product_live_mram_bytes"]) for fact in unit_facts)
    nonfit_tile_count = sum(
        not bool(fact["fused_four_product_admitted"]) for fact in unit_facts
    )
    admitted = peak <= MRAM_POOL_BYTES
    return {
        "plan_backed": True,
        "single_product_live_mram_bytes": single_peak,
        "fused_four_product_live_mram_bytes": peak,
        "mram_reservation_bytes": peak,
        "mram_reservation_components": list(MRAM_RESERVATION_COMPONENTS),
        "mram_reservation_scope": "A_B_C_data_only_v1",
        "control_bytes_outside_mram_reservation": CONTROL_BYTES,
        "completion_bytes_outside_mram_reservation": COMPLETION_BYTES,
        "control_completion_mram_reserved_bytes": 0,
        "control_completion_memory_scope": "WRAM_outside_MRAM_reservation_v1",
        "fused_four_product_admitted": admitted,
        "fused_four_product_headroom_bytes": MRAM_POOL_BYTES - peak,
        "mram_limit_bytes": MRAM_POOL_BYTES,
        "admission_reason": None
        if admitted
        else "fused_four_product_live_mram_exceeds_512KiB",
        "tile_count": len(units),
        "nonfit_tile_count": nonfit_tile_count,
        "nonfit_tile_fraction": nonfit_tile_count / len(units),
        "fusion_route": (
            "fused_four_product_generic_upmem"
            if admitted
            else "generic_upmem_no_retile"
        ),
        "nonfit_fallback": None if admitted else "generic_upmem_no_retile",
        "retile_requested": False,
    }


def _k_accumulation_facts(
    units: tuple[UpmemWorkUnit, ...], *, plan_backed: bool
) -> dict[str, Any]:
    """Separate request-local KC panels from host reduction across work units."""

    if not plan_backed:
        return {
            "within_request_kc_panel_size": WRAM_PANEL_KC,
            "within_request_kc_panel_count": None,
            "within_request_kc_panel_accumulation_count": None,
            "within_request_kc_panel_accumulation_scope": (
                "one_work_unit_request_only"
            ),
            "host_reduced_inter_workunit_k_chunk_count": None,
            "host_reduced_inter_workunit_k_output_tile_count": None,
            "host_reduced_inter_workunit_k_chunk_scope": (
                "host_reduced_across_distinct_work_units"
            ),
            "implicit_resident_k": False,
            "implicit_resident_k_reason": (
                "K is not reserved across work units; only request-local KC panels are modeled"
            ),
        }

    panels = [
        (unit.k_size + WRAM_PANEL_KC - 1) // WRAM_PANEL_KC
        for unit in units
    ]
    output_tile_counts = Counter(
        (
            unit.batch_start,
            unit.batch_size,
            unit.m_start,
            unit.m_size,
            unit.n_start,
            unit.n_size,
        )
        for unit in units
    )
    host_reduced_chunks = sum(max(0, count - 1) for count in output_tile_counts.values())
    return {
        "within_request_kc_panel_size": WRAM_PANEL_KC,
        "within_request_kc_panel_count": sum(panels),
        "within_request_kc_panel_accumulation_count": sum(
            max(0, panel_count - 1) for panel_count in panels
        ),
        "within_request_kc_panel_accumulation_scope": (
            "one_work_unit_request_only"
        ),
        "host_reduced_inter_workunit_k_chunk_count": host_reduced_chunks,
        "host_reduced_inter_workunit_k_output_tile_count": sum(
            count > 1 for count in output_tile_counts.values()
        ),
        "host_reduced_inter_workunit_k_chunk_scope": (
            "host_reduced_across_distinct_work_units"
        ),
        "implicit_resident_k": False,
        "implicit_resident_k_reason": (
            "K is not reserved across work units; only request-local KC panels are modeled"
        ),
    }


def _critical_path(
    index: DependencyIndex,
    weights: Mapping[str, int],
) -> dict[str, Any]:
    distance: dict[str, int] = {}
    paths: dict[str, tuple[str, ...]] = {}
    for node_id in index.topological_order:
        candidates = [
            (distance[predecessor], paths[predecessor])
            for predecessor in index.predecessors[node_id]
        ]
        if candidates:
            best_distance = max(item[0] for item in candidates)
            # Equal-work paths choose the lowest node-ID tuple without relying
            # on object or hash iteration order.
            best_path = min(
                item[1] for item in candidates if item[0] == best_distance
            )
        else:
            best_distance, best_path = 0, ()
        distance[node_id] = best_distance + int(weights[node_id])
        paths[node_id] = best_path + (node_id,)

    if not distance:
        return {
            "real_mac_work": 0,
            "node_ids": [],
            "end_node_id": None,
            "measured_timing": None,
            "measured_timing_reason": TIMING_REASON,
        }
    maximum = max(distance.values())
    end_node_id = min(node_id for node_id, value in distance.items() if value == maximum)
    return {
        "real_mac_work": maximum,
        "node_ids": list(paths[end_node_id]),
        "end_node_id": end_node_id,
        "per_node_path_work": {
            node_id: distance[node_id] for node_id in index.topological_order
        },
        "measured_timing": None,
        "measured_timing_reason": TIMING_REASON,
    }


def critical_path_real_mac_work(
    dag: ContractionDAG,
    *,
    plan: UpmemPlan | None = None,
) -> int:
    """Return deterministic four-real-product critical-path MAC work."""

    return int(characterize_dag(dag, plan=plan)["critical_path"]["real_mac_work"])


def _cohort_facts(
    index: DependencyIndex,
    operations: Mapping[str, Mapping[str, Any]],
    *,
    plan: UpmemPlan | None,
) -> list[dict[str, Any]]:
    units_by_node = _plan_units(plan)
    rows: list[dict[str, Any]] = []
    for cohort_index, cohort in enumerate(index.cohorts):
        units = [unit for node_id in cohort for unit in units_by_node.get(node_id, ())]
        internal_wave_keys = {
            (unit.node_id, unit.wave) for unit in units
        }
        occupied_slots = {
            (unit.node_id, unit.wave, unit.logical_rank, unit.logical_dpu)
            for unit in units
            if unit.estimated_arithmetic_work > 0
        }
        capacity = len(internal_wave_keys) * (plan.topology.dpu_count if plan else 0)
        rows.append(
            {
                "cohort_index": cohort_index,
                "node_ids": list(cohort),
                "frontier_width": len(cohort),
                "real_mac_work": sum(
                    int(operations[node_id]["real_mac_work"]) for node_id in cohort
                ),
                "contract_node_count": sum(
                    operations[node_id]["node_kind"] == "contract" for node_id in cohort
                ),
                "reduce_node_count": sum(
                    operations[node_id]["node_kind"] == "reduce" for node_id in cohort
                ),
                "planned_work_unit_count": len(units) if plan else None,
                "planned_internal_wave_count": len(internal_wave_keys) if plan else None,
                "occupied_dpu_slots": len(occupied_slots) if plan else None,
                "dpu_slot_capacity": capacity if plan else None,
                "dpu_slot_fill_ratio": (
                    len(occupied_slots) / capacity if capacity else None
                ),
            }
        )
    return rows


def _resident_pairs(
    index: DependencyIndex,
    operations: Mapping[str, Mapping[str, Any]],
    liveness: Mapping[str, Any],
    *,
    plan: UpmemPlan | None,
    numeric_policy: str,
) -> list[dict[str, Any]]:
    """Report unqualified local-memory candidates, never resident admission."""

    runtime_bytes = liveness["tensor_runtime_bytes"]
    descriptors = {
        node_id: index.nodes[node_id] for node_id in index.nodes
    }
    tensor_consumers: dict[str, set[str]] = defaultdict(set)
    for consumer_id, node in descriptors.items():
        for view in _node_views(node):
            tensor_consumers[view.tensor_id].add(consumer_id)

    rows: list[dict[str, Any]] = []
    for successor_id in index.topological_order:
        successor = index.nodes[successor_id]
        successor_inputs = {view.tensor_id for view in _node_views(successor)}
        for predecessor_id in index.predecessors[successor_id]:
            predecessor = index.nodes[predecessor_id]
            pair_id = f"{predecessor_id}->{successor_id}"
            intermediate_ids = sorted(
                {predecessor.output.id} & successor_inputs
            )
            hard_reasons: list[str] = []
            if plan is None:
                hard_reasons.append("retained_physical_plan_unavailable")
            if not isinstance(predecessor, ContractNode) or not isinstance(
                successor, ContractNode
            ):
                hard_reasons.append("resident_pair_requires_two_contract_nodes")
            if not intermediate_ids:
                hard_reasons.append("dependency_is_not_tensor_edge")
            if any(
                tensor_consumers[tensor_id] - {successor_id}
                for tensor_id in intermediate_ids
            ):
                hard_reasons.append("producer_output_has_other_consumers")
            if numeric_policy != RESIDENT_PAIR_NUMERIC_POLICY:
                hard_reasons.append("numeric_policy_not_float32_initial_probe")
            first_mram = operations[predecessor_id].get(
                "fused_four_product_live_mram_bytes"
            )
            second_mram = operations[successor_id].get(
                "fused_four_product_live_mram_bytes"
            )
            if first_mram is None or second_mram is None:
                hard_reasons.append("operation_mram_headroom_unavailable")
            if any(
                not operations[node_id].get("fused_four_product_admitted", False)
                for node_id in (predecessor_id, successor_id)
            ):
                hard_reasons.append("individual_fused_operation_exceeds_mram")
            intermediate_bytes = sum(
                int(runtime_bytes[tensor_id]) for tensor_id in intermediate_ids
            )
            pair_peak = None
            if first_mram is not None and second_mram is not None:
                pair_peak = max(
                    int(first_mram), int(second_mram) + intermediate_bytes
                )
                if pair_peak > MRAM_POOL_BYTES:
                    hard_reasons.append("resident_pair_live_mram_exceeds_512KiB")
            fallback_routes = sorted(
                {
                    operations[node_id].get("nonfit_fallback")
                    for node_id in (predecessor_id, successor_id)
                    if operations[node_id].get("nonfit_fallback") is not None
                }
            )
            memory_candidate_reasons = sorted(set(hard_reasons))
            admission_reasons = sorted(
                set(hard_reasons) | set(RESIDENT_PAIR_UNVERIFIED_REASONS)
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "predecessor_node_id": predecessor_id,
                    "successor_node_id": successor_id,
                    "predecessor_kind": operations[predecessor_id]["node_kind"],
                    "successor_kind": operations[successor_id]["node_kind"],
                    "intermediate_tensor_ids": intermediate_ids,
                    "intermediate_runtime_bytes": intermediate_bytes,
                    "intermediate_runtime_memory_scope": LIVENESS_MEMORY_SCOPE,
                    "intermediate_runtime_aliases_deduplicated": False,
                    "pair_peak_live_mram_bytes": pair_peak,
                    "pair_peak_is_full_layout_proof": False,
                    "pair_peak_process_memory_lower_bound_claimed": False,
                    "mram_limit_bytes": MRAM_POOL_BYTES,
                    "numeric_policy": numeric_policy,
                    "initial_probe_numeric_policy": RESIDENT_PAIR_NUMERIC_POLICY,
                    "resident_pair_scope": "bounded_memory_candidate_only_v1",
                    "memory_candidate": not memory_candidate_reasons,
                    "memory_candidate_reasons": memory_candidate_reasons,
                    "admitted": False,
                    "admission_status": "unqualified_memory_candidate",
                    "admission_reasons": admission_reasons,
                    "same_dpu_locality_verified": False,
                    "full_intermediate_layout_verified": False,
                    "intermediate_reconstruction_verified": False,
                    "no_split_k_verified": False,
                    "scale_handling_verified": False,
                    "generic_upmem_nonfit_fallback": fallback_routes or None,
                    "retile_requested": False,
                    "measured_timing": None,
                    "measured_timing_reason": TIMING_REASON,
                }
            )
    return rows


def characterize_dag(
    dag: ContractionDAG,
    *,
    plan: UpmemPlan | None = None,
    inputs: Mapping[str, np.ndarray] | None = None,
    numeric_policy: str = POLICIES[0],
) -> dict[str, Any]:
    """Extract bounded graph/headroom facts from actual model DAG records."""

    index = validate_dependency_graph(dag)
    if plan is not None:
        if not isinstance(plan, UpmemPlan):
            raise TypeError("plan must be an UpmemPlan or None")
        if plan.logical_plan_id != contraction_dag_hash(dag):
            raise ValueError("UPMEM plan logical identity does not match DAG")
        if plan.numeric_policy != numeric_policy:
            raise ValueError("UPMEM plan numeric policy does not match census policy")
    order = _plan_order(dag, plan, index)
    units_by_node = _plan_units(plan)
    operations: dict[str, dict[str, Any]] = {}
    for node_id in order:
        node = index.nodes[node_id]
        if isinstance(node, ContractNode):
            geometry = _geometry(node)
            planned_one_product = sum(
                int(unit.estimated_arithmetic_work)
                for unit in units_by_node.get(node_id, ())
            )
            semantic_one_product = int(geometry["one_product_real_mac_work"])
            one_product = planned_one_product if plan is not None else semantic_one_product
            mram = _operation_mram(node, units_by_node.get(node_id, ()), numeric_policy)
            operation = {
                "operation_id": node_id,
                "node_id": node_id,
                "node_kind": "contract",
                "predecessors": list(index.predecessors[node_id]),
                "successors": list(index.successors[node_id]),
                "input_tensor_ids": [node.left.tensor_id, node.right.tensor_id],
                "output_tensor_id": node.output.id,
                "geometry": list(geometry["geometry"]),
                "geometry_category": geometry["geometry_category"],
                "b": geometry["b"],
                "m": geometry["m"],
                "n": geometry["n"],
                "k": geometry["k"],
                "semantic_one_product_real_mac_work": semantic_one_product,
                "one_product_real_mac_work": one_product,
                "four_product_real_mac_work": FOUR_PRODUCT_COUNT * one_product,
                "real_mac_work": FOUR_PRODUCT_COUNT * one_product,
                "planned_work_unit_count": len(units_by_node.get(node_id, ())) if plan else None,
                "planned_internal_wave_count": (
                    len({unit.wave for unit in units_by_node.get(node_id, ())})
                    if plan
                    else None
                ),
                **mram,
                **_k_accumulation_facts(
                    units_by_node.get(node_id, ()), plan_backed=plan is not None
                ),
            }
        elif isinstance(node, ReduceNode):
            operation = {
                "operation_id": node_id,
                "node_id": node_id,
                "node_kind": "reduce",
                "predecessors": list(index.predecessors[node_id]),
                "successors": list(index.successors[node_id]),
                "input_tensor_ids": [view.tensor_id for view in node.inputs],
                "output_tensor_id": node.output.id,
                "geometry": None,
                "geometry_category": "reduce",
                "b": None,
                "m": None,
                "n": None,
                "k": None,
                "semantic_one_product_real_mac_work": 0,
                "one_product_real_mac_work": 0,
                "four_product_real_mac_work": 0,
                "real_mac_work": 0,
                "planned_work_unit_count": 0 if plan else None,
                "planned_internal_wave_count": 0 if plan else None,
                "plan_backed": bool(plan),
                "fused_four_product_live_mram_bytes": None,
                "fused_four_product_admitted": None,
                "mram_reservation_bytes": None,
                "mram_reservation_components": list(MRAM_RESERVATION_COMPONENTS),
                "mram_reservation_scope": "A_B_C_data_only_v1",
                "control_bytes_outside_mram_reservation": CONTROL_BYTES,
                "completion_bytes_outside_mram_reservation": COMPLETION_BYTES,
                "control_completion_mram_reserved_bytes": 0,
                "control_completion_memory_scope": "WRAM_outside_MRAM_reservation_v1",
                "mram_limit_bytes": MRAM_POOL_BYTES,
                "admission_reason": "reduce_node_has_no_real_mac_tile",
                "fusion_route": "not_applicable_reduce",
                **_k_accumulation_facts((), plan_backed=plan is not None),
            }
        else:  # pragma: no cover - validated by _validate_dependency_index.
            raise TypeError(type(node).__name__)
        operation.update(
            measured_timing=None,
            measured_timing_reason=TIMING_REASON,
        )
        operations[node_id] = operation

    liveness = _liveness_estimates(dag, index, order, inputs=inputs)
    critical_path = _critical_path(
        index,
        {node_id: int(operation["real_mac_work"]) for node_id, operation in operations.items()},
    )
    cohort_rows = _cohort_facts(index, operations, plan=plan)
    resident_pairs = _resident_pairs(
        index,
        operations,
        liveness,
        plan=plan,
        numeric_policy=numeric_policy,
    )
    geometry_counts = Counter(
        operation["geometry_category"]
        for operation in operations.values()
        if operation["node_kind"] == "contract"
    )
    mram_values = [
        int(operation["fused_four_product_live_mram_bytes"])
        for operation in operations.values()
        if operation.get("fused_four_product_live_mram_bytes") is not None
    ]
    mram_admitted = [
        bool(operation["fused_four_product_admitted"])
        for operation in operations.values()
        if operation.get("fused_four_product_admitted") is not None
    ]
    contract_operations = [
        operation
        for operation in operations.values()
        if operation["node_kind"] == "contract"
    ]
    k_facts_available = plan is not None
    k_accumulation = {
        "within_request_kc_panel_size": WRAM_PANEL_KC,
        "within_request_kc_panel_count": (
            sum(
                int(operation["within_request_kc_panel_count"] or 0)
                for operation in contract_operations
            )
            if k_facts_available
            else None
        ),
        "within_request_kc_panel_accumulation_count": (
            sum(
                int(operation["within_request_kc_panel_accumulation_count"] or 0)
                for operation in contract_operations
            )
            if k_facts_available
            else None
        ),
        "host_reduced_inter_workunit_k_chunk_count": (
            sum(
                int(operation["host_reduced_inter_workunit_k_chunk_count"] or 0)
                for operation in contract_operations
            )
            if k_facts_available
            else None
        ),
        "host_reduced_inter_workunit_k_output_tile_count": (
            sum(
                int(operation["host_reduced_inter_workunit_k_output_tile_count"] or 0)
                for operation in contract_operations
            )
            if k_facts_available
            else None
        ),
        "implicit_resident_k": False,
        "scope": (
            "request_local_kc_panels_plus_explicit_host_reduction_across_work_units"
        ),
    }
    nonfit_tile_count = sum(
        int(operation.get("nonfit_tile_count") or 0)
        for operation in contract_operations
    )
    nonfit_operation_count = sum(
        operation.get("nonfit_tile_count", 0) not in (None, 0)
        for operation in contract_operations
    )
    memory_candidates = sum(
        pair["memory_candidate"] for pair in resident_pairs
    )
    return {
        "node_count": len(index.nodes),
        "contract_node_count": sum(
            isinstance(node, ContractNode) for node in index.nodes.values()
        ),
        "reduce_node_count": sum(
            isinstance(node, ReduceNode) for node in index.nodes.values()
        ),
        "topological_order": list(index.topological_order),
        "dependency_ready_cohorts": [list(cohort) for cohort in index.cohorts],
        "frontier_widths": [len(cohort) for cohort in index.cohorts],
        "maximum_frontier_width": max((len(cohort) for cohort in index.cohorts), default=0),
        "cohort_facts": cohort_rows,
        "critical_path": critical_path,
        "total_real_mac_work": sum(
            int(operation["real_mac_work"]) for operation in operations.values()
        ),
        "operations": [operations[node_id] for node_id in order],
        "predecessors": {
            node_id: list(index.predecessors[node_id]) for node_id in order
        },
        "successors": {
            node_id: list(index.successors[node_id]) for node_id in order
        },
        "liveness": liveness,
        "k_accumulation": k_accumulation,
        "fused_four_product_mram": {
            "mram_limit_bytes": MRAM_POOL_BYTES,
            "reservation_components": list(MRAM_RESERVATION_COMPONENTS),
            "reservation_scope": "A_B_C_data_only_v1",
            "control_completion_mram_reserved_bytes": 0,
            "control_completion_memory_scope": "WRAM_outside_MRAM_reservation_v1",
            "peak_live_mram_bytes": max(mram_values, default=None),
            "minimum_headroom_bytes": (
                MRAM_POOL_BYTES - max(mram_values) if mram_values else None
            ),
            "all_contract_operations_admitted": (
                all(mram_admitted) if mram_admitted else None
            ),
            "plan_backed": plan is not None,
            "nonfit_tile_count": nonfit_tile_count if plan is not None else None,
            "nonfit_operation_count": (
                nonfit_operation_count if plan is not None else None
            ),
            "generic_upmem_no_retile_operation_count": (
                nonfit_operation_count if plan is not None else None
            ),
            "retile_requested": False,
        },
        "resident_pairs": resident_pairs,
        "resident_pair_summary": {
            "memory_candidate_count": memory_candidates,
            "admitted_count": 0,
            "admission_policy": "never_admit_without_layout_locality_scale_proof_v1",
            "numeric_probe_policy": RESIDENT_PAIR_NUMERIC_POLICY,
        },
        "geometry_category_counts": dict(sorted(geometry_counts.items())),
        "measured_timing": None,
        "measured_timing_reason": TIMING_REASON,
    }


def _reconstruct_cell(
    cell: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    ContractionDAG | None,
    UpmemPlan | None,
    Mapping[str, np.ndarray],
]:
    if cell.get("logical_plan_id") is None:
        return (
            {
                **cell,
                "status": "rejected",
                "rejection_reasons": [
                    "retained_candidate_has_no_logical_identity"
                ],
                "operations": [],
                "reconstruction_performed": False,
                "reconstruction_reason": "no_logical_plan_identity",
            },
            None,
            None,
            {},
        )

    base = characterize_cell(dict(cell))
    circuit = builtin_circuit(cell["circuit"]["name"], cell["circuit"]["parameters"])
    network, inputs = lower_tensor_network(make_simulation_job(circuit))
    dag = build_contraction_dag(network, tuple(tuple(pair) for pair in cell["path"]))
    observed_identity = contraction_dag_hash(dag)
    expected_identity = cell.get("logical_plan_id")
    if expected_identity is not None and observed_identity != expected_identity:
        if base.get("observed_logical_plan_id") != observed_identity:
            raise ValueError("existing characterization and frontier reconstruction disagree")
        return base, dag, None, inputs
    if base["status"] != "eligible":
        return base, dag, None, inputs

    topology = UpmemTopology(**cell["topology"])
    plan = plan_upmem(
        dag,
        numeric_policy=cell["numeric_policy"],
        topology=topology,
    )
    observed_physical_id = physical_plan_id(plan)
    if observed_physical_id != base.get("physical_plan_id"):
        raise ValueError("existing characterization and frontier plan identity disagree")
    expected_units = sum(
        len(stage.work_units) for stage in plan.stages
    )
    if expected_units != base.get("planned_work_unit_count"):
        raise ValueError("existing characterization and frontier plan work disagree")
    return base, dag, plan, inputs


def _attach_frontier_census(
    base: Mapping[str, Any], graph: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **base,
        "frontier_census": graph,
        "dependency_ready_cohorts": graph["dependency_ready_cohorts"],
        "frontier_widths": graph["frontier_widths"],
        "critical_path_real_mac_work": graph["critical_path"]["real_mac_work"],
        "per_operation": graph["operations"],
        "liveness": graph["liveness"],
        "fused_four_product_mram": graph["fused_four_product_mram"],
        "resident_pairs": graph["resident_pairs"],
        "geometry_category_counts": graph["geometry_category_counts"],
        "measured_timing": None,
        "measured_timing_reason": TIMING_REASON,
        "cpu_fallback_used": False,
        "hardware_execution": False,
    }


def _without_frontier_census(
    base: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    return {
        **base,
        "frontier_census": None,
        "frontier_census_reason": reason,
        "measured_timing": None,
        "measured_timing_reason": TIMING_REASON,
        "cpu_fallback_used": False,
        "hardware_execution": False,
    }


def characterize_frontier_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve one existing cell and attach source-only frontier facts."""

    base, dag, plan, inputs = _reconstruct_cell(cell)
    if dag is None:
        return _without_frontier_census(
            base, base.get("reconstruction_reason", "frontier_not_reconstructed")
        )
    graph = characterize_dag(
        dag,
        plan=plan,
        inputs=inputs,
        numeric_policy=cell["numeric_policy"],
    )
    return _attach_frontier_census(base, graph)


def _rejected_frontier_record(
    cell: Mapping[str, Any], reason: str, *, diagnostic: str | None = None
) -> dict[str, Any]:
    base: dict[str, Any] = {
        **cell,
        "status": "rejected",
        "rejection_reasons": [reason],
        "operations": [],
        "reconstruction_performed": False,
        "reconstruction_reason": reason,
    }
    if diagnostic:
        base["diagnostic"] = diagnostic
    return _without_frontier_census(base, reason)


def isolated_frontier_cell(
    cell: Mapping[str, Any], *, timeout: float = LOWERING_TIMEOUT
) -> dict[str, Any]:
    """Run eligible reconstruction in the existing bounded subprocess pattern."""

    if cell.get("logical_plan_id") is None:
        return characterize_frontier_cell(cell)
    try:
        run = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(cell),
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
    except subprocess.TimeoutExpired:
        return _rejected_frontier_record(cell, "lowering_timeout")
    if run.returncode:
        return _rejected_frontier_record(
            cell,
            "frontier_reconstruction_error",
            diagnostic=run.stderr,
        )
    try:
        return json.loads(run.stdout)
    except json.JSONDecodeError:
        return _rejected_frontier_record(
            cell,
            "frontier_reconstruction_error",
            diagnostic="worker did not return valid JSON",
        )


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_identity() -> tuple[str, bool]:
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    return source, dirty


def _selection_manifest(cells: list[dict[str, Any]], pools: dict[str, Any]) -> dict[str, Any]:
    source, dirty = _source_identity()
    return {
        "runtime_base_sha": BASE_SHA,
        "source_sha": source,
        "source_dirty": dirty,
        "candidate_pools": pools,
        "cells": cells,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _csv_rows(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in results:
        graph = cell.get("frontier_census")
        cell_common = {
            "cell_id": cell["cell_id"],
            "circuit_id": cell["circuit_id"],
            "candidate_path_id": cell["candidate_path_id"],
            "numeric_policy": cell["numeric_policy"],
            "dpu_count": cell["topology"]["dpu_count"],
            "cell_status": cell["status"],
            "rejection_reasons": ";".join(cell.get("rejection_reasons", ())),
            "measured_timing": None,
            "measured_timing_reason": TIMING_REASON,
        }
        if graph is None:
            rows.append(
                {
                    **cell_common,
                    "row_kind": "cell",
                    "frontier_census_reason": cell.get(
                        "frontier_census_reason",
                        cell.get("reconstruction_reason", "frontier_not_reconstructed"),
                    ),
                }
            )
            continue
        common = {
            **cell_common,
            "frontier_widths": graph["frontier_widths"],
            "maximum_frontier_width": graph["maximum_frontier_width"],
            "critical_path_real_mac_work": cell["critical_path_real_mac_work"],
            "minimal_theoretical_peak_live_tensor_bytes": graph["liveness"][
                "minimal_theoretical"
            ]["peak_live_tensor_bytes"],
            "actual_retained_runtime_peak_live_tensor_bytes": graph["liveness"][
                "actual_retained_runtime"
            ]["peak_live_tensor_bytes"],
            "liveness_memory_scope": graph["liveness"].get(
                "memory_scope", LIVENESS_MEMORY_SCOPE
            ),
            "actual_retained_runtime_is_partial_logical_tensor_payload_accounting": graph["liveness"][
                "actual_retained_runtime"
            ].get("estimate_kind")
            == "partial_logical_tensor_payload_accounting",
            "whole_host_admission": graph["liveness"].get(
                "whole_host_admission", False
            ),
            "excluded_retained_categories": graph["liveness"].get(
                "excluded_retained_categories", []
            ),
            "fused_four_product_peak_live_mram_bytes": graph[
                "fused_four_product_mram"
            ]["peak_live_mram_bytes"],
            "fused_four_product_admitted": graph["fused_four_product_mram"][
                "all_contract_operations_admitted"
            ],
            "mram_reservation_components": graph[
                "fused_four_product_mram"
            ].get("reservation_components", list(MRAM_RESERVATION_COMPONENTS)),
            "control_completion_mram_reserved_bytes": graph[
                "fused_four_product_mram"
            ].get("control_completion_mram_reserved_bytes", 0),
            "nonfit_tile_count": graph["fused_four_product_mram"].get(
                "nonfit_tile_count"
            ),
            "generic_upmem_no_retile_operation_count": graph[
                "fused_four_product_mram"
            ].get("generic_upmem_no_retile_operation_count"),
            "within_request_kc_panel_accumulation_count": graph.get(
                "k_accumulation", {}
            ).get("within_request_kc_panel_accumulation_count"),
            "host_reduced_inter_workunit_k_chunk_count": graph.get(
                "k_accumulation", {}
            ).get("host_reduced_inter_workunit_k_chunk_count"),
            "implicit_resident_k": graph.get("k_accumulation", {}).get(
                "implicit_resident_k", False
            ),
            "resident_pair_memory_candidate_count": graph.get(
                "resident_pair_summary", {}
            ).get("memory_candidate_count", 0),
            "resident_pair_admitted_count": graph.get(
                "resident_pair_summary", {}
            ).get("admitted_count", 0),
        }
        operations = graph["operations"]
        if not operations:
            rows.append({**common, "row_kind": "cell"})
            continue
        for operation in operations:
            rows.append({
                **common,
                "row_kind": "operation",
                "node_id": operation["node_id"],
                "node_kind": operation["node_kind"],
                "predecessors": operation["predecessors"],
                "successors": operation["successors"],
                "geometry_category": operation["geometry_category"],
                "b": operation["b"],
                "m": operation["m"],
                "n": operation["n"],
                "k": operation["k"],
                "real_mac_work": operation["real_mac_work"],
                "planned_work_unit_count": operation["planned_work_unit_count"],
                "fused_four_product_live_mram_bytes": operation.get(
                    "fused_four_product_live_mram_bytes"
                ),
                "fused_four_product_admitted": operation.get(
                    "fused_four_product_admitted"
                ),
                "fusion_route": operation.get("fusion_route"),
                "nonfit_tile_count": operation.get("nonfit_tile_count"),
                "retile_requested": operation.get("retile_requested", False),
                "mram_reservation_components": operation.get(
                    "mram_reservation_components", list(MRAM_RESERVATION_COMPONENTS)
                ),
                "control_completion_mram_reserved_bytes": operation.get(
                    "control_completion_mram_reserved_bytes", 0
                ),
                "within_request_kc_panel_accumulation_count": operation.get(
                    "within_request_kc_panel_accumulation_count"
                ),
                "host_reduced_inter_workunit_k_chunk_count": operation.get(
                    "host_reduced_inter_workunit_k_chunk_count"
                ),
                "implicit_resident_k": operation.get("implicit_resident_k", False),
                "operation_measured_timing": None,
                "operation_measured_timing_reason": TIMING_REASON,
            })
    return [
        {key: _csv_value(value) for key, value in row.items()}
        for row in rows
    ]


def write_census(output: Path) -> dict[str, Any]:
    """Write the bounded JSON/CSV census into an ignored run directory."""

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    cells, pools = frozen_cells()
    exclusion_count = sum(cell.get("logical_plan_id") is None for cell in cells)
    if len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError(f"frozen cell count changed: expected {EXPECTED_CELL_COUNT}")
    if exclusion_count != EXPECTED_EXCLUSION_COUNT:
        raise ValueError(
            f"frozen exclusion count changed: expected {EXPECTED_EXCLUSION_COUNT}"
        )
    output.mkdir(parents=True, exist_ok=True)
    selection = _selection_manifest(cells, pools)
    selected_path = output / "selected_paths.json"
    selected_path.write_bytes(_canonical_bytes(selection))

    results: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        record = (
            characterize_frontier_cell(cell)
            if cell.get("logical_plan_id") is None
            else isolated_frontier_cell(cell)
        )
        results.append(record)
        print(
            f"{index + 1}/{len(cells)} {cell['circuit_id']} "
            f"{cell['numeric_policy']} D{cell['topology']['dpu_count']} "
            f"{record['status']}",
            file=sys.stderr,
            flush=True,
        )

    geometry_counts: Counter[str] = Counter()
    for record in results:
        geometry_counts.update(record.get("geometry_category_counts", {}))
    graph_records = [
        record["frontier_census"]
        for record in results
        if record.get("frontier_census") is not None
    ]
    resident_pairs = [
        pair
        for graph in graph_records
        for pair in graph.get("resident_pairs", [])
    ]
    nonfit_tile_count = sum(
        int(graph.get("fused_four_product_mram", {}).get("nonfit_tile_count") or 0)
        for graph in graph_records
    )
    nonfit_operation_count = sum(
        int(
            graph.get("fused_four_product_mram", {}).get(
                "nonfit_operation_count"
            )
            or 0
        )
        for graph in graph_records
    )
    memory_candidate_count = sum(
        int(graph.get("resident_pair_summary", {}).get("memory_candidate_count", 0))
        for graph in graph_records
    )
    k_accumulation = {
        "within_request_kc_panel_size": WRAM_PANEL_KC,
        "within_request_kc_panel_accumulation_count": sum(
            int(
                graph.get("k_accumulation", {}).get(
                    "within_request_kc_panel_accumulation_count"
                )
                or 0
            )
            for graph in graph_records
        ),
        "host_reduced_inter_workunit_k_chunk_count": sum(
            int(
                graph.get("k_accumulation", {}).get(
                    "host_reduced_inter_workunit_k_chunk_count"
                )
                or 0
            )
            for graph in graph_records
        ),
        "implicit_resident_k": False,
    }
    source, dirty = _source_identity()
    report = {
        "schema_version": SCHEMA,
        "source_sha": source,
        "source_dirty": dirty,
        "runtime_base_sha": BASE_SHA,
        "candidate_pools": pools,
        "selection_sha256": _digest(selected_path.read_bytes()),
        "cell_count": len(results),
        "exclusion_count": sum(record["status"] != "eligible" for record in results),
        "eligible_count": sum(record["status"] == "eligible" for record in results),
        "preserved_cell_ids": [record["cell_id"] for record in results],
        "geometry_category_counts": dict(sorted(geometry_counts.items())),
        "limits": {
            "mram_pool_bytes": MRAM_POOL_BYTES,
            "wram_panel_kc": WRAM_PANEL_KC,
            "control_bytes_outside_mram_reservation": CONTROL_BYTES,
            "completion_bytes_outside_mram_reservation": COMPLETION_BYTES,
            "control_completion_mram_reserved_bytes": 0,
            "mram_reservation_components": list(MRAM_RESERVATION_COMPONENTS),
        },
        "reconstruction": {
            "eligible_cell_timeout_s": LOWERING_TIMEOUT,
            "excluded_cells_reconstructed": False,
            "excluded_cell_reason": "no_logical_plan_identity",
        },
        "resident_pair_policy": {
            "scope": "bounded_memory_candidate_only_v1",
            "numeric_probe_policy": RESIDENT_PAIR_NUMERIC_POLICY,
            "pair_edge_count": len(resident_pairs),
            "admitted_count": 0,
            "memory_candidate_count": memory_candidate_count,
            "same_dpu_locality_verified": False,
            "full_intermediate_layout_verified": False,
            "intermediate_reconstruction_verified": False,
            "no_split_k_verified": False,
            "scale_handling_verified": False,
            "never_whole_host_admission": True,
        },
        "k_accumulation": k_accumulation,
        "fused_headroom_summary": {
            "nonfit_tile_count": nonfit_tile_count,
            "nonfit_operation_count": nonfit_operation_count,
            "nonfit_route": "generic_upmem_no_retile",
            "retile_requested": False,
        },
        "execution": {
            "physical": False,
            "sdk": False,
            "timing_measured": False,
            "cpu_fallback_used": False,
            "candidate_generation": False,
        },
        "timing": {"measured": None, "reason": TIMING_REASON},
        "cells": results,
        "counts": dict(Counter(record["status"] for record in results)),
    }
    json_path = output / "upmem_frontier_census.json"
    json_path.write_bytes(_canonical_bytes(report))

    rows = _csv_rows(results)
    columns = sorted({key for row in rows for key in row})
    csv_path = output / "upmem_frontier_census.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    files = [selected_path, json_path, csv_path]
    (output / "SHA256SUMS").write_text(
        "".join(f"{_digest(path.read_bytes())}  {path.name}\n" for path in sorted(files)),
        encoding="ascii",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "upmem_frontier_census",
    )
    args = parser.parse_args(argv)
    if args.worker:
        print(
            json.dumps(
                characterize_frontier_cell(json.load(sys.stdin)),
                allow_nan=False,
            )
        )
        return
    report = write_census(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "cell_count": report["cell_count"],
                "exclusion_count": report["exclusion_count"],
                "eligible_count": report["eligible_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
