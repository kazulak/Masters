"""Single-run NumPy execution of the canonical contraction DAG."""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import NoReturn

import numpy as np

from quantum_bench.lowering import validate_contraction_dag, validate_dag_inputs
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode, TensorView
from quantum_bench.numerics import (
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


_SUPPORTED_POLICIES = (
    "split_complex_float32_v1",
    "split_complex_int8_shared_scale_v1",
)
_BACKEND_FACTS = {"backend_id": "numpy_cpu_v1", "execution_class": "cpu_host"}


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
                saturation_real += left_encoded.saturation_real + right_encoded.saturation_real
                saturation_imag += left_encoded.saturation_imag + right_encoded.saturation_imag
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
                values = [_to_complex64(_materialize_view(view, working)) for view in reduce_inputs]
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
    output = np.array(_materialize_view(dag.output, working), dtype=np.complex128, copy=True, order="C")
    _require_finite_complex128(output, "final output")
    output.setflags(write=False)
    return output


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
        missing = [view.tensor_id for view in node.inputs if view.tensor_id not in producer_node_ids]
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


def _materialize_view(view: TensorView, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
    try:
        array = tensors[view.tensor_id]
    except KeyError as exc:
        raise ValueError(f"Tensor {view.tensor_id} is not available for execution") from exc
    if not view.slice_spec:
        result = np.asarray(array)
    else:
        indices: list[slice | int] = [slice(None)] * array.ndim
        fixed_axes: set[int] = set()
        for axis, value in view.slice_spec:
            if axis in fixed_axes or axis < 0 or axis >= array.ndim:
                raise ValueError(f"Tensor view {view.tensor_id} has invalid fixed axis {axis}")
            if value < 0 or value >= array.shape[axis]:
                raise ValueError(f"Tensor view {view.tensor_id} has invalid fixed value {value}")
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
    labels = list(dict.fromkeys((*node.left.labels, *node.right.labels, *node.output_labels)))
    if len(labels) > 52:
        raise ValueError("contraction uses too many distinct labels for NumPy einsum")
    mapping = {label: index for index, label in enumerate(labels)}
    return (
        [mapping[label] for label in node.left.labels],
        [mapping[label] for label in node.right.labels],
        [mapping[label] for label in node.output_labels],
    )


__all__ = ["run_cpu_once", "run_complex128_reference"]
