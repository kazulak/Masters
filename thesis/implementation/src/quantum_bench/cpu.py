"""Single-run NumPy execution of the canonical contraction DAG."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import time
from typing import NoReturn

import numpy as np

from quantum_bench.lowering import validate_contraction_dag, validate_dag_inputs
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode, TensorView
from quantum_bench.numerics import (
    EncodedComplexTensor,
    NumericPolicy,
    contract_complex_products,
    decode_complex_products,
    encode_complex_tensor,
)
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    Measurement,
    UnsupportedExecution,
)
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemStage,
    UpmemWorkUnit,
    physical_plan_id,
    validate_upmem_plan,
)
from quantum_bench.upmem.tiling import (
    M5TileLowering,
    M5TileLimits,
    lower_binary_contraction,
)


_SUPPORTED_POLICIES = (
    "split_complex_float32_v1",
    "split_complex_int8_shared_scale_v1",
)
_BACKEND_FACTS = {"backend_id": "numpy_cpu_v1", "execution_class": "cpu_host"}
_REPLAY_BACKEND_ID = "cpu_upmem_plan_replay_v1"
_REPLAY_EXECUTION_CLASS = "cpu_physical_plan_reference"
_REPLAY_KERNEL_POLICY = "real_tile_four_product_v1"
_REPLAY_INTERMEDIATE_POLICY = "host_roundtrip_v1"


def run_cpu_once(
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    numeric_policy: NumericPolicy,
    *,
    scope_id: str = "steady_execution_v1",
) -> ExecutionSample:
    """Execute one validated DAG run and return its output and measurements."""

    validate_contraction_dag(dag)
    validate_dag_inputs(dag, inputs)
    if not isinstance(numeric_policy, str) or numeric_policy not in _SUPPORTED_POLICIES:
        raise UnsupportedExecution(
            stage="preflight",
            reason=f"CPU route does not implement numeric policy {numeric_policy!r}",
            capability="numeric_policy",
        )
    order = _topological_order(dag)
    if (
        numeric_policy == "split_complex_int8_shared_scale_v1"
        and not _int8_output_is_derived(dag, order)
    ):
        raise UnsupportedExecution(
            stage="preflight",
            reason="requested output is not fully derived from int8-policy contractions",
            capability="numeric_policy_applicability",
        )
    if scope_id != "steady_execution_v1":
        raise UnsupportedExecution(
            stage="preflight",
            reason=f"CPU route does not implement timing scope {scope_id!r}",
            capability="timing_scope",
        )

    producer_node_ids = {node.output.id: node.node_id for node in order}
    int8_reduce_inputs = (
        _int8_reduce_input_order(order, producer_node_ids)
        if numeric_policy == "split_complex_int8_shared_scale_v1"
        else {}
    )
    working = {tensor_id: np.asarray(array) for tensor_id, array in inputs.items()}
    encode_s = 0.0
    kernel_s = 0.0
    host_reduce_s = 0.0
    decode_s = 0.0
    encode_executed = False
    kernel_executed = False
    host_reduce_executed = False
    decode_executed = False
    saturation_real = 0
    saturation_imag = 0
    started = time.perf_counter()

    for node in order:
        if isinstance(node, ContractNode):
            encode_executed = True
            try:
                encode_started = time.perf_counter()
                left = _materialize_view(node.left, working)
                right = _materialize_view(node.right, working)
                left_encoded = encode_complex_tensor(left, numeric_policy)
                right_encoded = encode_complex_tensor(right, numeric_policy)
                encode_s += time.perf_counter() - encode_started
                saturation_real += (
                    left_encoded.saturation_real + right_encoded.saturation_real
                )
                saturation_imag += (
                    left_encoded.saturation_imag + right_encoded.saturation_imag
                )
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_execution_failed("encode", exc)

            kernel_executed = True
            try:
                kernel_started = time.perf_counter()
                products = contract_complex_products(
                    node, left_encoded, right_encoded, numeric_policy
                )
                result = products[0]
                if tuple(result.shape) != node.output.shape:
                    raise ValueError(
                        f"Node {node.node_id} produced shape {result.shape}; expected {node.output.shape}"
                    )
                kernel_s += time.perf_counter() - kernel_started
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_execution_failed("kernel", exc)

            decode_executed = True
            try:
                decode_started = time.perf_counter()
                result = decode_complex_products(
                    products,
                    left_encoded.scale,
                    right_encoded.scale,
                    numeric_policy,
                )
                if tuple(result.shape) != node.output.shape:
                    raise ValueError(
                        f"Node {node.node_id} decoded shape {result.shape}; expected {node.output.shape}"
                    )
                result = _to_complex64(result)
                decode_s += time.perf_counter() - decode_started
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_execution_failed("decode", exc)
        elif isinstance(node, ReduceNode):
            host_reduce_executed = True
            try:
                reduce_started = time.perf_counter()
                reduce_inputs = int8_reduce_inputs.get(node.node_id, node.inputs)
                values = [
                    _to_complex64(_materialize_view(view, working))
                    for view in reduce_inputs
                ]
                with np.errstate(over="ignore", invalid="ignore"):
                    result = np.add.reduce(tuple(values), axis=0, dtype=np.complex64)
                if tuple(result.shape) != node.output.shape:
                    raise ValueError(
                        f"Node {node.node_id} reduced shape {result.shape}; expected {node.output.shape}"
                    )
                result = _to_complex64(result)
                host_reduce_s += time.perf_counter() - reduce_started
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_execution_failed("host_reduce", exc)
        else:  # pragma: no cover - GraphNode is closed by the model contract.
            raise TypeError(f"unsupported DAG node: {type(node).__name__}")
        working[node.output.id] = np.asarray(result)

    try:
        output = _to_complex64(_materialize_view(dag.output, working))
    except ExecutionFailed:
        raise
    except Exception as exc:
        _raise_execution_failed("finalize", exc)
    total_wall_s = time.perf_counter() - started
    measurement = Measurement(
        scope_id=scope_id,
        total_wall_s=total_wall_s,
        encode_s=encode_s if encode_executed else None,
        kernel_s=kernel_s if kernel_executed else None,
        host_reduce_s=host_reduce_s if host_reduce_executed else None,
        decode_s=decode_s if decode_executed else None,
    )
    return ExecutionSample(
        output=output,
        measurement=measurement,
        backend_facts=_BACKEND_FACTS,
        numeric_facts={
            "numeric_policy": numeric_policy,
            "saturation_real": saturation_real,
            "saturation_imag": saturation_imag,
        },
    )


def run_complex128_reference(
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Replay the DAG with direct complex128 NumPy contractions."""

    validate_contraction_dag(dag)
    validate_dag_inputs(dag, inputs)
    working = {tensor_id: np.asarray(array) for tensor_id, array in inputs.items()}
    for node in _topological_order(dag):
        if isinstance(node, ContractNode):
            left = _materialize_view(node.left, working)
            right = _materialize_view(node.right, working)
            left_indices, right_indices, output_indices = _einsum_indices(node)
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.einsum(
                    left,
                    left_indices,
                    right,
                    right_indices,
                    output_indices,
                    dtype=np.complex128,
                    optimize=False,
                )
            _require_finite_complex128(result, "contraction")
        elif isinstance(node, ReduceNode):
            values = [_materialize_view(view, working) for view in node.inputs]
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.add.reduce(
                    tuple(np.asarray(value, dtype=np.complex128) for value in values),
                    axis=0,
                    dtype=np.complex128,
                )
            _require_finite_complex128(result, "reduction")
        else:  # pragma: no cover - GraphNode is closed by the model contract.
            raise TypeError(f"unsupported DAG node: {type(node).__name__}")
        if tuple(result.shape) != node.output.shape:
            raise ValueError(
                f"Node {node.node_id} produced shape {result.shape}; expected {node.output.shape}"
            )
        working[node.output.id] = result
    output = np.array(
        _materialize_view(dag.output, working),
        dtype=np.complex128,
        copy=True,
        order="C",
    )
    _require_finite_complex128(output, "final output")
    output.setflags(write=False)
    return output


def replay_upmem_plan_once(
    dag: ContractionDAG,
    plan: UpmemPlan,
    inputs: Mapping[str, np.ndarray],
    *,
    scope_id: str = "steady_execution_v1",
) -> ExecutionSample:
    """Execute the final UPMEM physical plan using a deterministic CPU oracle.

    This deliberately follows the real-tile ABI policy rather than the direct
    NumPy contraction path: canonicalization is performed by the same tiler,
    each K chunk is accumulated in the policy dtype, and the four real product
    lanes are reconstructed before any host reduction.
    """

    validate_contraction_dag(dag)
    validate_dag_inputs(dag, inputs)
    if scope_id != "steady_execution_v1":
        raise UnsupportedExecution(
            stage="preflight",
            reason=f"UPMEM plan replay does not implement timing scope {scope_id!r}",
            capability="timing_scope",
        )
    if plan.intermediate_policy != _REPLAY_INTERMEDIATE_POLICY:
        raise UnsupportedExecution(
            stage="preflight",
            reason=f"unsupported intermediate policy {plan.intermediate_policy!r}",
            capability="intermediate_policy",
        )
    if plan.kernel_policy != _REPLAY_KERNEL_POLICY:
        raise UnsupportedExecution(
            stage="preflight",
            reason=f"unsupported kernel policy {plan.kernel_policy!r}",
            capability="kernel_policy",
        )
    _validate_replay_stage_shape(dag, plan)
    validate_upmem_plan(dag, plan)
    order = _topological_order(dag)
    if plan.numeric_policy == "split_complex_int8_shared_scale_v1":
        producer_node_ids = {node.output.id: node.node_id for node in order}
        if not _int8_output_is_derived(dag, order):
            raise UnsupportedExecution(
                stage="preflight",
                reason="requested output is not fully derived from int8-policy contractions",
                capability="numeric_policy_applicability",
            )
        _int8_reduce_input_order(order, producer_node_ids)
    elif plan.numeric_policy != "split_complex_float32_v1":
        raise UnsupportedExecution(
            stage="preflight",
            reason=f"unsupported numeric policy {plan.numeric_policy!r}",
            capability="numeric_policy",
        )

    plan_id = physical_plan_id(plan)
    backend_facts = {
        "backend_id": _REPLAY_BACKEND_ID,
        "execution_class": _REPLAY_EXECUTION_CLASS,
        "physical_plan_id": plan_id,
        "physical_plan_consumed": True,
        "requested_dpus": plan.topology.dpu_count,
        "rank_count": plan.topology.rank_count,
        "tasklets_per_dpu": plan.topology.tasklets_per_dpu,
        "hardware_execution": False,
    }
    limits = _replay_limits(plan.numeric_policy)
    working = {tensor_id: np.asarray(array) for tensor_id, array in inputs.items()}
    raw_lanes: list[tuple[str, str, str, np.ndarray]] = []
    encoded_operands: list[tuple[str, str, EncodedComplexTensor]] = []
    preparation_s = 0.0
    encode_s = 0.0
    kernel_s = 0.0
    decode_s = 0.0
    host_reduce_s = 0.0
    preparation_executed = False
    encode_executed = False
    kernel_executed = False
    decode_executed = False
    host_reduce_executed = False
    started = time.perf_counter()

    for stage, node in _replay_stage_nodes(dag, plan):
        if isinstance(node, ContractNode):
            preparation_executed = True
            try:
                preparation_started = time.perf_counter()
                left = _materialize_view(node.left, working)
                right = _materialize_view(node.right, working)
                left_real = lower_binary_contraction(
                    node,
                    np.asarray(left.real, dtype=np.float32),
                    np.asarray(right.real, dtype=np.float32),
                    limits=limits,
                )
                left_imag = lower_binary_contraction(
                    node,
                    np.asarray(left.imag, dtype=np.float32),
                    np.asarray(right.imag, dtype=np.float32),
                    limits=limits,
                )
                _validate_replay_lowerings(left_real, left_imag, stage, node)
                preparation_s += time.perf_counter() - preparation_started
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_replay_failed("preparation", exc, backend_facts)

            try:
                encode_started = time.perf_counter()
                left_canonical = _combine_canonical_planes(
                    left_real, left_imag, left=True
                )
                right_canonical = _combine_canonical_planes(
                    left_real, left_imag, left=False
                )
                left_encoded = encode_complex_tensor(
                    left_canonical, plan.numeric_policy
                )
                right_encoded = encode_complex_tensor(
                    right_canonical, plan.numeric_policy
                )
                encode_s += time.perf_counter() - encode_started
                encode_executed = True
                encoded_operands.extend(
                    (
                        (node.node_id, "left", left_encoded),
                        (node.node_id, "right", right_encoded),
                    )
                )
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_replay_failed("encode", exc, backend_facts)

            try:
                kernel_started = time.perf_counter()
                partials: dict[
                    str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                ] = {}
                for unit in stage.work_units:
                    lanes = _replay_tile_lanes(
                        left_encoded, right_encoded, unit, plan.numeric_policy
                    )
                    partials[_lowering_tile_id(unit, node)] = lanes
                    for lane, value in zip(
                        ("rr", "ii", "ri", "ir"), lanes, strict=True
                    ):
                        raw_lanes.append(
                            (node.node_id, unit.stable_tile_id, lane, value)
                        )
                kernel_s += time.perf_counter() - kernel_started
                kernel_executed = True
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_replay_failed("kernel", exc, backend_facts)

            try:
                decode_started = time.perf_counter()
                lane_outputs = tuple(
                    left_real.assemble(
                        {tile_id: lanes[index] for tile_id, lanes in partials.items()},
                        dtype=np.float32
                        if plan.numeric_policy == "split_complex_float32_v1"
                        else np.int64,
                    )
                    for index in range(4)
                )
                result = decode_complex_products(
                    lane_outputs,
                    left_encoded.scale,
                    right_encoded.scale,
                    plan.numeric_policy,
                )
                if tuple(result.shape) != node.output.shape:
                    raise ValueError(
                        f"Node {node.node_id} decoded shape {result.shape}; expected {node.output.shape}"
                    )
                result = _to_complex64(result)
                decode_s += time.perf_counter() - decode_started
                decode_executed = True
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_replay_failed("decode", exc, backend_facts)
        else:
            host_reduce_executed = True
            try:
                reduce_started = time.perf_counter()
                producer_node_ids = {
                    candidate.output.id: candidate.node_id for candidate in order
                }
                ordered_inputs = sorted(
                    node.inputs,
                    key=lambda view: (
                        producer_node_ids.get(view.tensor_id, ""),
                        view.tensor_id,
                        view.slice_spec,
                    ),
                )
                values = [
                    _to_complex64(_materialize_view(view, working))
                    for view in ordered_inputs
                ]
                result = np.array(values[0], dtype=np.complex64, copy=True)
                for value in values[1:]:
                    with np.errstate(over="ignore", invalid="ignore"):
                        result = np.add(result, value, dtype=np.complex64)
                if tuple(result.shape) != node.output.shape:
                    raise ValueError(
                        f"Node {node.node_id} reduced shape {result.shape}; expected {node.output.shape}"
                    )
                result = _to_complex64(result)
                host_reduce_s += time.perf_counter() - reduce_started
            except ExecutionFailed:
                raise
            except Exception as exc:
                _raise_replay_failed("host_reduce", exc, backend_facts)
        working[node.output.id] = np.asarray(result)

    try:
        output = _to_complex64(_materialize_view(dag.output, working))
    except ExecutionFailed:
        raise
    except Exception as exc:
        _raise_replay_failed("finalize", exc, backend_facts)
    total_wall_s = time.perf_counter() - started
    operand_records = tuple(
        _operand_record(node_id, side, encoded)
        for node_id, side, encoded in sorted(
            encoded_operands, key=lambda item: (item[0], item[1])
        )
    )
    numeric_facts = {
        "numeric_policy": plan.numeric_policy,
        "saturation_real": sum(
            encoded.saturation_real for _, _, encoded in encoded_operands
        ),
        "saturation_imag": sum(
            encoded.saturation_imag for _, _, encoded in encoded_operands
        ),
        "operand_records": operand_records,
        "raw_lane_records": tuple(
            _raw_lane_record(node_id, tile_id, lane, value)
            for node_id, tile_id, lane, value in sorted(
                raw_lanes, key=_raw_lane_sort_key
            )
        ),
    }
    return ExecutionSample(
        output=output,
        measurement=Measurement(
            scope_id=scope_id,
            total_wall_s=total_wall_s,
            preparation_s=preparation_s if preparation_executed else None,
            encode_s=encode_s if encode_executed else None,
            kernel_s=kernel_s if kernel_executed else None,
            decode_s=decode_s if decode_executed else None,
            host_reduce_s=host_reduce_s if host_reduce_executed else None,
        ),
        backend_facts=backend_facts,
        numeric_facts=numeric_facts,
    )


def _validate_replay_stage_shape(dag: ContractionDAG, plan: UpmemPlan) -> None:
    nodes = {node.node_id: node for node in dag.nodes}
    for stage in plan.stages:
        stage_nodes = [nodes.get(node_id) for node_id in stage.node_ids]
        if any(node is None for node in stage_nodes):
            unknown = next(
                node_id
                for node_id, node in zip(stage.node_ids, stage_nodes)
                if node is None
            )
            raise ValueError(f"UPMEM stage references unknown node {unknown!r}")
        if stage.kind == "contract_batch" and any(
            not isinstance(node, ContractNode) for node in stage_nodes
        ):
            raise UnsupportedExecution(
                stage="preflight",
                reason="contract_batch stage does not contain only ContractNodes",
                capability="upmem_replay_stage_shape",
            )
        if stage.kind == "host_reduce" and (
            len(stage_nodes) != 1 or not isinstance(stage_nodes[0], ReduceNode)
        ):
            raise UnsupportedExecution(
                stage="preflight",
                reason="host_reduce stage does not contain one ReduceNode",
                capability="upmem_replay_stage_shape",
            )


def _replay_stage_nodes(
    dag: ContractionDAG, plan: UpmemPlan
) -> tuple[tuple[UpmemStage, ContractNode | ReduceNode], ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    entries: list[tuple[UpmemStage, ContractNode | ReduceNode]] = []
    for stage in plan.stages:
        for node_id in stage.node_ids:
            node = nodes[node_id]
            entries.append(
                (
                    UpmemStage(
                        stage_id=stage.stage_id,
                        kind=stage.kind,
                        node_ids=(node_id,),
                        work_units=tuple(
                            unit for unit in stage.work_units if unit.node_id == node_id
                        ),
                    ),
                    node,
                )
            )
    return tuple(entries)


def _node_for_stage(
    dag: ContractionDAG, stage: UpmemStage
) -> ContractNode | ReduceNode:
    nodes = {node.node_id: node for node in dag.nodes}
    try:
        return nodes[stage.node_ids[0]]
    except KeyError as exc:
        raise ValueError(
            f"UPMEM stage references unknown node {stage.node_ids[0]!r}"
        ) from exc


def _replay_limits(policy: str) -> M5TileLimits:
    if policy == "split_complex_float32_v1":
        return M5TileLimits.float32()
    if policy == "split_complex_int8_shared_scale_v1":
        return M5TileLimits.host_packed_int8()
    raise UnsupportedExecution(
        stage="preflight",
        reason=f"unsupported numeric policy {policy!r}",
        capability="numeric_policy",
    )


def _combine_canonical_planes(
    real_lowering: M5TileLowering,
    imag_lowering: M5TileLowering,
    *,
    left: bool,
) -> np.ndarray:
    real = real_lowering.canonical.left if left else real_lowering.canonical.right
    imag = imag_lowering.canonical.left if left else imag_lowering.canonical.right
    if real.shape != imag.shape:
        raise ValueError("real and imaginary canonical planes have different shapes")
    result = np.empty(real.shape, dtype=np.complex64)
    result.real = np.asarray(real, dtype=np.float32)
    result.imag = np.asarray(imag, dtype=np.float32)
    return result


def _validate_replay_lowerings(
    real_lowering: M5TileLowering,
    imag_lowering: M5TileLowering,
    stage: UpmemStage,
    node: ContractNode,
) -> None:
    real = real_lowering.canonical
    imag = imag_lowering.canonical
    if (
        (real.b, real.m, real.k, real.n) != (imag.b, imag.m, imag.k, imag.n)
        or real.batch_labels != imag.batch_labels
        or real.free_left_labels != imag.free_left_labels
        or real.contracted_labels != imag.contracted_labels
        or real.free_right_labels != imag.free_right_labels
        or real.canonical_output_labels != imag.canonical_output_labels
        or real.label_dimensions != imag.label_dimensions
    ):
        raise ValueError("real and imaginary canonical metadata differ")
    if real_lowering.output_tiles != imag_lowering.output_tiles:
        raise ValueError("real and imaginary output tiles differ")
    if real_lowering.k_chunks != imag_lowering.k_chunks:
        raise ValueError("real and imaginary K chunks differ")
    if real_lowering.tiles != imag_lowering.tiles:
        raise ValueError("real and imaginary tiles differ")
    expected = {f"{node.node_id}:{tile.id}": tile for tile in real_lowering.tiles}
    units = stage.work_units
    if len({unit.stable_tile_id for unit in units}) != len(units):
        raise ValueError("UPMEM stage contains duplicate tile IDs")
    unit_ids = {unit.stable_tile_id for unit in units}
    if unit_ids != set(expected):
        raise ValueError("UPMEM stage tile IDs do not match lowered tiles")
    for unit in units:
        tile = expected[unit.stable_tile_id]
        if (
            unit.node_id != node.node_id
            or unit.batch_start != tile.batch_index
            or unit.batch_size != 1
            or unit.m_start != tile.m_start
            or unit.m_size != tile.m_size
            or unit.n_start != tile.n_start
            or unit.n_size != tile.n_size
            or unit.k_start != tile.k_start
            or unit.k_size != tile.k_size
            or unit.estimated_input_bytes != tile.left_bytes + tile.right_bytes
            or unit.estimated_output_bytes != tile.output_bytes
            or unit.aligned_mram_bytes != tile.aligned_mram_bytes
            or unit.estimated_arithmetic_work != tile.m_size * tile.n_size * tile.k_size
        ):
            raise ValueError(
                f"UPMEM work unit {unit.stable_tile_id!r} differs from lowered tile"
            )


def _replay_tile_lanes(
    left: EncodedComplexTensor,
    right: EncodedComplexTensor,
    unit: UpmemWorkUnit,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_real = left.real[
        unit.batch_start,
        unit.m_start : unit.m_start + unit.m_size,
        unit.k_start : unit.k_start + unit.k_size,
    ]
    left_imag = left.imag[
        unit.batch_start,
        unit.m_start : unit.m_start + unit.m_size,
        unit.k_start : unit.k_start + unit.k_size,
    ]
    right_real = right.real[
        unit.batch_start,
        unit.k_start : unit.k_start + unit.k_size,
        unit.n_start : unit.n_start + unit.n_size,
    ]
    right_imag = right.imag[
        unit.batch_start,
        unit.k_start : unit.k_start + unit.k_size,
        unit.n_start : unit.n_start + unit.n_size,
    ]
    if policy == "split_complex_float32_v1":
        dtype = np.dtype(np.float32)
        lanes = []
        for left_plane, right_plane in (
            (left_real, right_real),
            (left_imag, right_imag),
            (left_real, right_imag),
            (left_imag, right_real),
        ):
            accumulator = np.zeros((unit.m_size, unit.n_size), dtype=dtype)
            for k_index in range(unit.k_size):
                product = np.multiply(
                    left_plane[:, k_index, None],
                    right_plane[k_index, None, :],
                    dtype=dtype,
                )
                accumulator = np.add(accumulator, product, dtype=dtype)
            lanes.append(accumulator)
        return tuple(lanes)  # type: ignore[return-value]

    lanes = []
    for left_plane, right_plane in (
        (left_real, right_real),
        (left_imag, right_imag),
        (left_real, right_imag),
        (left_imag, right_real),
    ):
        left_int = np.asarray(left_plane, dtype=np.int32)
        right_int = np.asarray(right_plane, dtype=np.int32)
        accumulator = np.zeros((unit.m_size, unit.n_size), dtype=np.int32)
        for k_index in range(unit.k_size):
            product = np.multiply(
                left_int[:, k_index, None],
                right_int[k_index, None, :],
                dtype=np.int32,
            )
            accumulator = np.add(accumulator, product, dtype=np.int32)
        lanes.append(accumulator)
    return tuple(lanes)  # type: ignore[return-value]


def _lowering_tile_id(unit: UpmemWorkUnit, node: ContractNode) -> str:
    prefix = f"{node.node_id}:"
    if not unit.stable_tile_id.startswith(prefix):
        raise ValueError("UPMEM work unit tile ID omits node identity")
    return unit.stable_tile_id[len(prefix) :]


def _operand_record(
    node_id: str, side: str, encoded: EncodedComplexTensor
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "side": side,
        "scale": float(encoded.scale),
        "saturation_real": int(encoded.saturation_real),
        "saturation_imag": int(encoded.saturation_imag),
        "real_dtype": encoded.real.dtype.str,
        "imag_dtype": encoded.imag.dtype.str,
        "shape": tuple(int(value) for value in encoded.real.shape),
        "real_sha256": _payload_sha256(encoded.real),
        "imag_sha256": _payload_sha256(encoded.imag),
    }


def _raw_lane_record(
    node_id: str,
    tile_id: str,
    lane: str,
    value: np.ndarray,
) -> dict[str, object]:
    integer = np.issubdtype(value.dtype, np.integer)
    dtype = np.dtype("<i4" if integer else "<f4")
    return {
        "node_id": node_id,
        "stable_tile_id": tile_id,
        "lane": lane,
        "dtype": dtype.str,
        "shape": tuple(int(item) for item in value.shape),
        "sha256": _payload_sha256(value, dtype=dtype),
        "exact": integer,
    }


def _raw_lane_sort_key(
    value: tuple[str, str, str, np.ndarray],
) -> tuple[str, str, int]:
    lane_order = {"rr": 0, "ii": 1, "ri": 2, "ir": 3}
    return value[0], value[1], lane_order.get(value[2], len(lane_order))


def _payload_sha256(value: np.ndarray, *, dtype: np.dtype | None = None) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _raise_replay_failed(
    stage: str,
    error: Exception,
    backend_facts: Mapping[str, object],
) -> NoReturn:
    reason = str(error).strip() or type(error).__name__
    raise ExecutionFailed(
        stage=stage, reason=reason, backend_facts=backend_facts
    ) from error


def _topological_order(dag: ContractionDAG) -> tuple[ContractNode | ReduceNode, ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    remaining = {node.node_id: set(node.dependencies) for node in dag.nodes}
    ordered: list[ContractNode | ReduceNode] = []
    while remaining:
        ready = sorted(node_id for node_id, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError("ContractionDAG contains a dependency cycle")
        for node_id in ready:
            ordered.append(nodes[node_id])
            del remaining[node_id]
        for deps in remaining.values():
            deps.difference_update(ready)
    return tuple(ordered)


def _int8_output_is_derived(
    dag: ContractionDAG,
    order: tuple[ContractNode | ReduceNode, ...],
) -> bool:
    derived: dict[str, bool] = {}
    for node in order:
        if isinstance(node, ContractNode):
            derived[node.output.id] = True
        else:
            derived[node.output.id] = all(
                derived.get(view.tensor_id, False) for view in node.inputs
            )
    return derived.get(dag.output.tensor_id, False)


def _int8_reduce_input_order(
    order: tuple[ContractNode | ReduceNode, ...],
    producer_node_ids: Mapping[str, str],
) -> dict[str, tuple[TensorView, ...]]:
    reduced_inputs: dict[str, tuple[TensorView, ...]] = {}
    for node in order:
        if not isinstance(node, ReduceNode):
            continue
        missing = [
            view.tensor_id
            for view in node.inputs
            if view.tensor_id not in producer_node_ids
        ]
        if missing:
            raise UnsupportedExecution(
                stage="preflight",
                reason=f"int8 reduction {node.node_id!r} consumes non-derived tensors {sorted(missing)}",
                capability="numeric_policy_applicability",
            )
        reduced_inputs[node.node_id] = tuple(
            sorted(
                node.inputs,
                key=lambda view: (
                    producer_node_ids[view.tensor_id],
                    view.tensor_id,
                    view.slice_spec,
                ),
            )
        )
    return reduced_inputs


def _materialize_view(
    view: TensorView, tensors: Mapping[str, np.ndarray]
) -> np.ndarray:
    try:
        array = tensors[view.tensor_id]
    except KeyError as exc:
        raise ValueError(
            f"Tensor {view.tensor_id} is not available for execution"
        ) from exc
    if not view.slice_spec:
        result = np.asarray(array)
    else:
        indices: list[slice | int] = [slice(None)] * array.ndim
        fixed_axes: set[int] = set()
        for axis, value in view.slice_spec:
            if axis in fixed_axes or axis < 0 or axis >= array.ndim:
                raise ValueError(
                    f"Tensor view {view.tensor_id} has invalid fixed axis {axis}"
                )
            if value < 0 or value >= array.shape[axis]:
                raise ValueError(
                    f"Tensor view {view.tensor_id} has invalid fixed value {value}"
                )
            fixed_axes.add(axis)
            indices[axis] = int(value)
        result = np.asarray(array[tuple(indices)])
    if tuple(result.shape) != view.shape:
        raise ValueError(
            f"Tensor view {view.tensor_id} produced shape {result.shape}; expected {view.shape}"
        )
    return result


def _to_complex64(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("execution result must be numeric")
    with np.errstate(over="ignore", invalid="ignore"):
        real = np.asarray(array.real, dtype=np.float64)
        imag = np.asarray(array.imag, dtype=np.float64)
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ValueError("execution result is nonfinite")
    limit = np.finfo(np.float32).max
    if np.any(np.abs(real) > limit) or np.any(np.abs(imag) > limit):
        raise ValueError("execution result cannot be represented as complex64")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.array(array, dtype=np.complex64, copy=True, order="C")
    if not np.all(np.isfinite(converted)):
        raise ValueError("execution result cannot be represented as finite complex64")
    converted.setflags(write=False)
    return converted


def _require_finite_complex128(value: np.ndarray, stage: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"complex128 reference produced a nonfinite {stage}")


def _raise_execution_failed(stage: str, error: Exception) -> NoReturn:
    reason = str(error).strip() or type(error).__name__
    raise ExecutionFailed(
        stage=stage,
        reason=reason,
        backend_facts=_BACKEND_FACTS,
    ) from error


def _einsum_indices(node: ContractNode) -> tuple[list[int], list[int], list[int]]:
    labels = list(
        dict.fromkeys((*node.left.labels, *node.right.labels, *node.output_labels))
    )
    if len(labels) > 52:
        raise ValueError("contraction uses too many distinct labels for NumPy einsum")
    mapping = {label: index for index, label in enumerate(labels)}
    return (
        [mapping[label] for label in node.left.labels],
        [mapping[label] for label in node.right.labels],
        [mapping[label] for label in node.output_labels],
    )


__all__ = [
    "replay_upmem_plan_once",
    "run_cpu_once",
    "run_complex128_reference",
]
