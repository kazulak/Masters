"""Canonical circuit, tensor-network, and contraction-DAG lowering.

This module deliberately contains no tensor arrays, planner metadata, costs, or
target execution details.  The path uses opt_einsum's dynamic active-list
convention: every pair refers to positions in the active tensor list at that
step, and the produced tensor is appended to that list.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import product
from typing import Iterable, Mapping, Sequence

from quantum_bench.circuits import gate_structure, gate_tensor
from quantum_bench.model import (
    ContractionDAG,
    ContractNode,
    GraphNode,
    ReduceNode,
    SimulationJob,
    SliceSpec,
    TensorNetwork,
    TensorSpec,
    TensorView,
)

import numpy as np


_EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_LIST_EINSUM_SENTINEL = "__label_list_einsum_required__"


def _index_symbols(
    label_sets: list[tuple[int, ...]], output_labels: tuple[int, ...]
) -> dict[int, str]:
    labels = sorted(
        {label for labels in label_sets for label in labels} | set(output_labels)
    )
    if len(labels) > len(_EINSUM_SYMBOLS):
        raise ValueError("Too many tensor indices for NumPy einsum symbol set")
    return {label: _EINSUM_SYMBOLS[index] for index, label in enumerate(labels)}


def _label_count(
    label_sets: list[tuple[int, ...]], output_labels: tuple[int, ...]
) -> int:
    return len({label for labels in label_sets for label in labels} | set(output_labels))


def _supports_string_einsum(
    label_sets: list[tuple[int, ...]], output_labels: tuple[int, ...]
) -> bool:
    return _label_count(label_sets, output_labels) <= len(_EINSUM_SYMBOLS)


def lower_tensor_network(
    job: SimulationJob,
) -> tuple[TensorNetwork, dict[str, np.ndarray]]:
    """Lower a supported simulation job into metadata and numerical inputs."""

    if not isinstance(job, SimulationJob):
        raise TypeError("lower_tensor_network requires a SimulationJob")
    if job.query != "pre_measurement_statevector":
        raise ValueError(f"Unsupported simulation query: {job.query!r}")
    if job.parameters:
        raise ValueError(
            "pre_measurement_statevector lowering does not support simulation parameters"
        )
    if job.seed is not None:
        raise ValueError(
            "pre_measurement_statevector lowering supports only deterministic jobs; seed must be None"
        )

    circuit = job.circuit
    tensors: list[tuple[TensorSpec, np.ndarray]] = []
    counter = 0
    wire_label: dict[int, int] = {}
    zero = np.array([1.0, 0.0], dtype=np.complex128)

    for wire in range(circuit.n_qubits):
        label = counter
        counter += 1
        wire_label[wire] = label
        spec = TensorSpec(f"tensor_{len(tensors)}", (label,), zero.shape, "dense")
        tensors.append((spec, zero.copy()))

    for op_index, op in enumerate(circuit.operations):
        _validate_wires(op.wires, circuit.n_qubits)
        input_labels = tuple(wire_label[wire] for wire in op.wires)
        output_labels = tuple(range(counter, counter + len(op.wires)))
        counter += len(op.wires)
        for wire, label in zip(op.wires, output_labels):
            wire_label[wire] = label
        array = gate_tensor(op)
        spec = TensorSpec(
            id=f"tensor_{len(tensors)}",
            labels=input_labels + output_labels,
            shape=array.shape,
            structure=gate_structure(op.gate),
            produced_by=f"circuit_op_{op_index}",
        )
        tensors.append((spec, array))

    output_labels = tuple(wire_label[wire] for wire in range(circuit.n_qubits))
    expression = build_full_einsum_expression(
        [tensor for tensor, _ in tensors], output_labels
    )
    network = TensorNetwork(
        circuit=circuit,
        tensors=tuple(tensor for tensor, _ in tensors),
        output_labels=output_labels,
        einsum_expression=expression,
    )
    inputs = {tensor.id: np.asarray(array) for tensor, array in tensors}
    validate_tensor_inputs(network, inputs)
    return network, inputs


def validate_tensor_inputs(
    network: TensorNetwork,
    inputs: Mapping[str, np.ndarray],
) -> None:
    """Validate that execution inputs match network descriptors exactly."""

    specs = {tensor.id: tensor for tensor in network.tensors}
    if len(specs) != len(network.tensors):
        raise ValueError("Tensor network contains duplicate tensor ids")
    if set(inputs) != set(specs):
        missing = sorted(set(specs) - set(inputs))
        extra = sorted(set(inputs) - set(specs))
        raise ValueError(
            f"Tensor input ids do not match network: missing={missing} extra={extra}"
        )
    for tensor_id, value in inputs.items():
        array = np.asarray(value)
        if tuple(array.shape) != specs[tensor_id].shape:
            raise ValueError(
                f"Tensor input {tensor_id} shape {array.shape} "
                f"does not match descriptor {specs[tensor_id].shape}"
            )
        if array.dtype != np.dtype(specs[tensor_id].dtype):
            raise ValueError(
                f"Tensor input {tensor_id} dtype {array.dtype} "
                f"does not match descriptor {specs[tensor_id].dtype}"
            )


def validate_dag_inputs(
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
) -> None:
    """Validate input arrays against the tensor descriptors in a DAG."""

    specs = {tensor.id: tensor for tensor in dag.tensors}
    if len(specs) != len(dag.tensors):
        raise ValueError("ContractionDAG contains duplicate input tensor IDs")
    actual_ids = set(inputs)
    expected_ids = set(specs)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Tensor input ids do not match DAG: missing={missing} extra={extra}"
        )
    for tensor_id, value in inputs.items():
        array = np.asarray(value)
        descriptor = specs[tensor_id]
        if tuple(array.shape) != descriptor.shape:
            raise ValueError(
                f"Tensor input {tensor_id} shape {array.shape} "
                f"does not match descriptor {descriptor.shape}"
            )
        if array.dtype != np.dtype(descriptor.dtype):
            raise ValueError(
                f"Tensor input {tensor_id} dtype {array.dtype} "
                f"does not match descriptor {descriptor.dtype}"
            )


def build_full_einsum_expression(
    tensors: list[TensorSpec], output_labels: tuple[int, ...]
) -> str:
    label_sets = [tensor.labels for tensor in tensors]
    if not _supports_string_einsum(label_sets, output_labels):
        return f"{LABEL_LIST_EINSUM_SENTINEL}:labels={_label_count(label_sets, output_labels)}"
    symbols = _index_symbols([tensor.labels for tensor in tensors], output_labels)
    operands = [
        "".join(symbols[label] for label in tensor.labels) for tensor in tensors
    ]
    output = "".join(symbols[label] for label in output_labels)
    return ",".join(operands) + "->" + output


def choose_slice_labels(
    node: ContractNode, *, minimum_slice_count: int
) -> tuple[int, ...]:
    """Choose contracted labels whose Cartesian slices reach a minimum count."""

    if (
        isinstance(minimum_slice_count, bool)
        or not isinstance(minimum_slice_count, int)
        or minimum_slice_count < 2
    ):
        raise ValueError("minimum_slice_count must be at least 2")

    candidates = [
        (label, _label_dimension_from_views(label, node.left, node.right))
        for label in node.contracted_labels
    ]
    candidates = [candidate for candidate in candidates if candidate[1] > 1]
    candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))

    selected: list[int] = []
    slice_count = 1
    for label, dimension in candidates:
        selected.append(label)
        slice_count *= dimension
        if slice_count >= minimum_slice_count:
            return tuple(selected)
    raise ValueError(
        f"Cannot reach minimum slice count {minimum_slice_count} from contracted dimensions"
    )


def slice_contraction(
    dag: ContractionDAG, *, node_id: str, labels: tuple[int, ...]
) -> ContractionDAG:
    """Rewrite one contraction into Cartesian fixed-index partials and a reduction."""

    validate_contraction_dag(dag)
    target_index = next(
        (index for index, node in enumerate(dag.nodes) if node.node_id == node_id),
        None,
    )
    if target_index is None:
        raise ValueError(f"Unknown slice node {node_id!r}")
    target = dag.nodes[target_index]
    if not isinstance(target, ContractNode):
        raise ValueError(f"Slice node {node_id!r} is not a contract node")
    if target.left.slice_spec or target.right.slice_spec:
        raise ValueError(
            f"Slice node {node_id!r} already has fixed input indices; "
            "nested slicing is unsupported"
        )

    if not labels:
        raise ValueError("Slice labels must be nonempty")
    if any(isinstance(label, bool) or not isinstance(label, int) for label in labels):
        raise ValueError("Slice labels must be integers")
    if len(set(labels)) != len(labels):
        raise ValueError("Slice labels must be unique")
    if any(label not in target.contracted_labels for label in labels):
        raise ValueError(
            f"Slice labels {labels} are not contracted by node {node_id!r}"
        )

    canonical_labels = tuple(sorted(labels))
    dimensions = tuple(
        _label_dimension_from_views(label, target.left, target.right)
        for label in canonical_labels
    )
    if any(dimension <= 1 for dimension in dimensions):
        invalid = next(
            (label, dimension)
            for label, dimension in zip(canonical_labels, dimensions)
            if dimension <= 1
        )
        raise ValueError(
            f"Cannot slice contracted label {invalid[0]} with dimension {invalid[1]}"
        )

    partial_nodes: list[ContractNode] = []
    partial_views: list[TensorView] = []
    occupied_node_ids = {node.node_id for node in dag.nodes if node is not target}
    occupied_tensor_ids = {tensor.id for tensor in dag.tensors}
    occupied_tensor_ids.update(
        node.output.id for node in dag.nodes if node is not target
    )
    occupied_tensor_ids.add(target.output.id)
    remaining_labels = tuple(
        label for label in target.contracted_labels if label not in canonical_labels
    )
    for values in product(*(range(dimension) for dimension in dimensions)):
        assignments = tuple(zip(canonical_labels, values))
        if len(assignments) == 1:
            assignment_id = f"{assignments[0][0]}_{assignments[0][1]}"
            slice_separator = "_"
        else:
            assignment_id = "__".join(
                f"label_{label}_value_{value}" for label, value in assignments
            )
            slice_separator = "__"
        left = _fix_view(target.left, assignments)
        right = _fix_view(target.right, assignments)
        partial_node_id = _unique_generated_id(
            f"{target.node_id}__slice{slice_separator}{assignment_id}",
            occupied_node_ids,
        )
        partial_output_id = _unique_generated_id(
            f"{target.output.id}__slice{slice_separator}{assignment_id}",
            occupied_tensor_ids,
        )
        partial_output = replace(
            target.output, id=partial_output_id, produced_by=partial_node_id
        )
        partial_node = replace(
            target,
            node_id=partial_node_id,
            left=left,
            right=right,
            output=partial_output,
            contracted_labels=remaining_labels,
        )
        partial_nodes.append(partial_node)
        partial_views.append(_view(partial_output))

    if len(canonical_labels) == 1:
        reduction_base = f"{target.node_id}__reduce_{canonical_labels[0]}"
    else:
        reduction_id = "__".join(f"label_{label}" for label in canonical_labels)
        reduction_base = f"{target.node_id}__reduce__{reduction_id}"
    reduce_node_id = _unique_generated_id(reduction_base, occupied_node_ids)
    reduced_output = replace(target.output, produced_by=reduce_node_id)
    reduce_node = ReduceNode(
        node_id=reduce_node_id,
        inputs=tuple(partial_views),
        output=reduced_output,
        reduced_labels=canonical_labels,
        dependencies=tuple(node.node_id for node in partial_nodes),
    )

    rewritten_nodes: list[GraphNode] = []
    for index, node in enumerate(dag.nodes):
        if index == target_index:
            rewritten_nodes.extend(partial_nodes)
            rewritten_nodes.append(reduce_node)
            continue
        if target.node_id in node.dependencies:
            node = replace(
                node,
                dependencies=tuple(
                    reduce_node_id if dependency == target.node_id else dependency
                    for dependency in node.dependencies
                ),
            )
        rewritten_nodes.append(node)

    rewritten = replace(dag, nodes=tuple(rewritten_nodes))
    validate_contraction_dag(rewritten)
    return rewritten


def apply_slicing(dag: ContractionDAG, spec: SliceSpec) -> ContractionDAG:
    """Compatibility wrapper for one-label contraction slicing."""

    return slice_contraction(dag, node_id=spec.node_id, labels=(spec.label,))


def build_contraction_dag(
    network: TensorNetwork,
    path: Iterable[Iterable[int]],
) -> ContractionDAG:
    """Materialize a semantic DAG from a tensor network and dynamic path.

    ``path`` follows opt_einsum's dynamic active-list convention.  Planner
    identity and timing are intentionally not accepted or retained here.
    """

    spec = network
    active = list(spec.tensors)
    nodes: list[GraphNode] = []
    produced_by: dict[str, str] = {}

    for step_index, raw_pair in enumerate(path):
        node_id = f"contract_{step_index}"
        node, next_active = build_contract_node(
            active,
            tuple(raw_pair),
            spec.output_labels,
            produced_by=produced_by,
            node_id=node_id,
            output_id=f"result_{step_index}",
        )
        nodes.append(node)
        produced_by[node.output.id] = node_id
        active = list(next_active)

    if len(active) != 1:
        raise ValueError(
            f"Contraction path ended with {len(active)} active tensors; expected one"
        )

    output = _view(active[0])
    if output.labels != tuple(spec.output_labels):
        raise ValueError(
            f"Final tensor labels {output.labels} do not match network output labels {spec.output_labels}"
        )
    dag = ContractionDAG(tensors=tuple(spec.tensors), nodes=tuple(nodes), output=output)
    validate_contraction_dag(dag)
    return dag


def build_contract_node(
    active: Sequence[TensorSpec],
    pair: Sequence[int],
    final_output_labels: tuple[int, ...],
    *,
    produced_by: Mapping[str, str | None],
    node_id: str,
    output_id: str,
) -> tuple[ContractNode, tuple[TensorSpec, ...]]:
    """Lower one dynamic pair into target-neutral semantic DAG metadata."""

    normalized_pair = tuple(int(item) for item in pair)
    if len(normalized_pair) != 2:
        raise ValueError(
            f"Only binary contraction paths are supported; got {normalized_pair}"
        )
    i, j = sorted(normalized_pair)
    if i < 0 or i == j or j >= len(active):
        raise ValueError(
            f"Invalid contraction pair {normalized_pair} for {len(active)} active tensors"
        )

    left = active[i]
    right = active[j]
    _validate_shared_dimensions(left, right)
    remaining = [tensor for index, tensor in enumerate(active) if index not in (i, j)]
    remaining_labels = {label for tensor in remaining for label in tensor.labels}
    output_labels = _output_labels(left, right, remaining_labels, final_output_labels)
    contracted_labels = tuple(
        dict.fromkeys(
            label for label in left.labels + right.labels if label not in output_labels
        )
    )
    output_shape = tuple(
        _label_dimension(label, left, right) for label in output_labels
    )
    output = TensorSpec(
        id=output_id,
        labels=output_labels,
        shape=output_shape,
        structure="dense",
        dtype=_common_dtype(left, right),
        produced_by=node_id,
    )
    dependencies = tuple(
        dependency
        for tensor in (left, right)
        if (dependency := produced_by.get(tensor.id)) is not None
    )
    node = ContractNode(
        node_id=node_id,
        left=_view(left),
        right=_view(right),
        output=output,
        contracted_labels=contracted_labels,
        output_labels=output_labels,
        dependencies=dependencies,
    )
    return node, tuple((*remaining, output))


def validate_contraction_dag(dag: ContractionDAG) -> None:
    """Raise ``ValueError`` when a semantic DAG is malformed."""

    tensor_specs = {tensor.id: tensor for tensor in dag.tensors}
    if len(tensor_specs) != len(dag.tensors):
        raise ValueError("ContractionDAG contains duplicate input tensor IDs")
    node_ids = [node.node_id for node in dag.nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("ContractionDAG contains duplicate node IDs")
    nodes = {node.node_id: node for node in dag.nodes}
    produced: dict[str, str] = {}
    descriptors = dict(tensor_specs)
    descriptors.update({node.output.id: node.output for node in dag.nodes})
    for node in dag.nodes:
        output_id = node.output.id
        if output_id in tensor_specs or output_id in produced:
            raise ValueError(f"Tensor ID {output_id} has multiple producers")
        produced[output_id] = node.node_id
    for node in dag.nodes:
        for dependency in node.dependencies:
            if dependency not in nodes:
                raise ValueError(
                    f"Node {node.node_id} has unknown dependency {dependency}"
                )
    _validate_dependencies(nodes)

    for node in dag.nodes:
        if node.node_id in node.dependencies:
            raise ValueError(f"Node {node.node_id} depends on itself")
        if len(set(node.dependencies)) != len(node.dependencies):
            raise ValueError(f"Node {node.node_id} has duplicate dependencies")
        if isinstance(node, ContractNode):
            views = (node.left, node.right)
        else:
            if not node.inputs:
                raise ValueError(f"Reduce node {node.node_id} has no inputs")
            views = node.inputs
        for view in views:
            _validate_view(view, descriptors, node.node_id)
        if isinstance(node, ContractNode):
            _validate_contract_algebra(node, descriptors)
        else:
            _validate_reduce_algebra(node, descriptors)
        input_producers = {
            produced[view.tensor_id] for view in views if view.tensor_id in produced
        }
        if input_producers != set(node.dependencies):
            raise ValueError(
                f"Node {node.node_id} dependencies do not match producers of consumed tensors"
            )
        if (
            isinstance(node, ContractNode)
            and node.left.tensor_id == node.right.tensor_id
        ):
            raise ValueError(
                f"Contract node {node.node_id} contracts a tensor with itself"
            )

    output_ids = set(tensor_specs) | set(produced)
    if dag.output.tensor_id not in output_ids:
        raise ValueError(f"DAG output references unknown tensor {dag.output.tensor_id}")
    _validate_view(dag.output, descriptors, "DAG output")


def contraction_dag_hash(dag: ContractionDAG) -> str:
    """Return a canonical hash of semantic graph content only."""

    validate_contraction_dag(dag)
    payload = {
        "tensors": sorted(
            (_tensor_payload(tensor) for tensor in dag.tensors),
            key=_canonical_json,
        ),
        "nodes": sorted(
            (_semantic_node_payload(dag, node) for node in dag.nodes),
            key=_canonical_json,
        ),
        "output": _semantic_view_payload(dag, dag.output),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _view(spec: TensorSpec) -> TensorView:
    return TensorView(tensor_id=spec.id, labels=spec.labels, shape=spec.shape)


def _label_dimension_from_views(label: int, left: TensorView, right: TensorView) -> int:
    for view in (left, right):
        if label in view.labels:
            return view.shape[view.labels.index(label)]
    raise ValueError(f"Contracted label {label} is absent from both source views")


def _fix_view(view: TensorView, assignments: tuple[tuple[int, int], ...]) -> TensorView:
    values = dict(assignments)
    fixed = tuple(
        (axis, values[label])
        for axis, label in enumerate(view.labels)
        if label in values
    )
    fixed_axes = {axis for axis, _ in fixed}
    return TensorView(
        tensor_id=view.tensor_id,
        labels=tuple(
            label for axis, label in enumerate(view.labels) if axis not in fixed_axes
        ),
        shape=tuple(
            size for axis, size in enumerate(view.shape) if axis not in fixed_axes
        ),
        slice_spec=view.slice_spec + fixed,
    )


def _unique_generated_id(base: str, occupied: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in occupied:
        candidate = f"{base}__generated_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _validate_view(
    view: TensorView,
    descriptors: dict[str, TensorSpec],
    owner: str,
) -> None:
    descriptor = descriptors.get(view.tensor_id)
    if descriptor is None:
        raise ValueError(f"Node {owner} references unknown tensor {view.tensor_id}")
    if len(view.labels) != len(view.shape):
        raise ValueError(
            f"Tensor view {view.tensor_id} has mismatched labels and shape"
        )
    fixed_axes: set[int] = set()
    for item in view.slice_spec:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                f"Tensor view {view.tensor_id} has invalid fixed index {item!r}"
            )
        axis, value = item
        if not isinstance(axis, int) or not isinstance(value, int):
            raise ValueError(
                f"Tensor view {view.tensor_id} has invalid fixed index {item!r}"
            )
        if axis in fixed_axes:
            raise ValueError(f"Tensor view {view.tensor_id} has duplicate fixed axes")
        fixed_axes.add(axis)
        if axis < 0 or axis >= len(descriptor.labels):
            raise ValueError(
                f"Tensor view {view.tensor_id} has invalid fixed axis {axis}"
            )
        if value < 0 or value >= descriptor.shape[axis]:
            raise ValueError(
                f"Tensor view {view.tensor_id} has invalid fixed value {value}"
            )

    remaining_labels = tuple(
        label for axis, label in enumerate(descriptor.labels) if axis not in fixed_axes
    )
    remaining_shape = tuple(
        size for axis, size in enumerate(descriptor.shape) if axis not in fixed_axes
    )
    if (view.labels, view.shape) != (remaining_labels, remaining_shape):
        raise ValueError(
            f"Tensor view {view.tensor_id} does not match its descriptor after fixed axes"
        )


def _validate_contract_algebra(
    node: ContractNode,
    descriptors: dict[str, TensorSpec],
) -> None:
    left = descriptors[node.left.tensor_id]
    right = descriptors[node.right.tensor_id]
    if len(set(node.left.labels)) != len(node.left.labels):
        raise ValueError(f"Contract node {node.node_id} has duplicate left labels")
    if len(set(node.right.labels)) != len(node.right.labels):
        raise ValueError(f"Contract node {node.node_id} has duplicate right labels")
    if len(set(node.output_labels)) != len(node.output_labels):
        raise ValueError(f"Contract node {node.node_id} has duplicate output labels")
    if len(set(node.contracted_labels)) != len(node.contracted_labels):
        raise ValueError(
            f"Contract node {node.node_id} has duplicate contracted labels"
        )
    input_labels = set(node.left.labels) | set(node.right.labels)
    output_labels = set(node.output_labels)
    contracted_labels = set(node.contracted_labels)
    if output_labels & contracted_labels:
        raise ValueError(f"Contract node {node.node_id} outputs a contracted label")
    if output_labels | contracted_labels != input_labels:
        raise ValueError(
            f"Contract node {node.node_id} does not account for every input label"
        )
    for label in set(node.left.labels) & set(node.right.labels):
        left_dim = node.left.shape[node.left.labels.index(label)]
        right_dim = node.right.shape[node.right.labels.index(label)]
        if left_dim != right_dim:
            raise ValueError(
                f"Contract node {node.node_id} has mismatched label {label} dimensions"
            )
    expected_shape = tuple(
        _view_label_dimension(label, node.left, node.right)
        for label in node.output_labels
    )
    if node.output.labels != node.output_labels or node.output.shape != expected_shape:
        raise ValueError(
            f"Contract node {node.node_id} output does not match its inputs"
        )
    if node.output.dtype != left.dtype or node.output.dtype != right.dtype:
        raise ValueError(
            f"Contract node {node.node_id} output dtype does not match its inputs"
        )


def _validate_reduce_algebra(
    node: ReduceNode, descriptors: dict[str, TensorSpec]
) -> None:
    if len(set(node.reduced_labels)) != len(node.reduced_labels):
        raise ValueError(f"Reduce node {node.node_id} has duplicate reduced labels")
    output = node.output
    for view in node.inputs:
        if (view.labels, view.shape) != (output.labels, output.shape):
            raise ValueError(
                f"Reduce node {node.node_id} inputs do not match its output"
            )


def _view_label_dimension(label: int, left: TensorView, right: TensorView) -> int:
    if label in left.labels:
        return left.shape[left.labels.index(label)]
    return right.shape[right.labels.index(label)]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _semantic_view_payload(dag: ContractionDAG, view: TensorView) -> dict[str, object]:
    producers = {node.output.id: node for node in dag.nodes}
    producer = producers.get(view.tensor_id)
    tensor_ref: object
    if producer is None:
        tensor_ref = {"input": view.tensor_id}
    else:
        tensor_ref = {"producer": _semantic_node_key(dag, producer)}
    return {
        "tensor": tensor_ref,
        "labels": view.labels,
        "shape": view.shape,
        "slice_spec": tuple(sorted(view.slice_spec)),
    }


def _semantic_node_key(dag: ContractionDAG, node: GraphNode) -> str:
    return _canonical_json(_semantic_node_payload(dag, node))


def _semantic_node_payload(dag: ContractionDAG, node: GraphNode) -> dict[str, object]:
    if isinstance(node, ContractNode):
        operands = sorted(
            (
                _semantic_view_payload(dag, node.left),
                _semantic_view_payload(dag, node.right),
            ),
            key=_canonical_json,
        )
        return {
            "kind": "contract",
            "operands": operands,
            "output": _descriptor_payload(node.output),
            "contracted_labels": node.contracted_labels,
            "output_labels": node.output_labels,
        }
    return {
        "kind": "reduce",
        "inputs": [_semantic_view_payload(dag, view) for view in node.inputs],
        "output": _descriptor_payload(node.output),
        "reduced_labels": node.reduced_labels,
    }


def _descriptor_payload(tensor: TensorSpec) -> dict[str, object]:
    return {
        "labels": tensor.labels,
        "shape": tensor.shape,
        "structure": tensor.structure,
        "dtype": tensor.dtype,
    }


def _common_dtype(left: TensorSpec, right: TensorSpec) -> str:
    if left.dtype != right.dtype:
        raise ValueError(
            f"Cannot contract tensors with different dtypes: {left.dtype!r}, {right.dtype!r}"
        )
    return left.dtype


def _validate_shared_dimensions(left: TensorSpec, right: TensorSpec) -> None:
    for label in set(left.labels) & set(right.labels):
        left_dim = left.shape[left.labels.index(label)]
        right_dim = right.shape[right.labels.index(label)]
        if left_dim != right_dim:
            raise ValueError(f"Label {label} has dimensions {left_dim} and {right_dim}")


def _label_dimension(label: int, left: TensorSpec, right: TensorSpec) -> int:
    if label in left.labels:
        return left.shape[left.labels.index(label)]
    return right.shape[right.labels.index(label)]


def _output_labels(
    left: TensorSpec,
    right: TensorSpec,
    remaining_labels: set[int],
    final_output_labels: tuple[int, ...],
) -> tuple[int, ...]:
    keep = remaining_labels | set(final_output_labels)
    present = set(left.labels) | set(right.labels)
    if not remaining_labels and set(final_output_labels).issubset(present):
        return tuple(final_output_labels)
    labels: list[int] = []
    for label in left.labels + right.labels:
        if label in keep and label not in labels:
            labels.append(label)
    return tuple(labels)


def _validate_dependencies(nodes: dict[str, GraphNode]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("ContractionDAG dependencies contain a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in nodes[node_id].dependencies:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)


def _tensor_payload(tensor: TensorSpec) -> dict[str, object]:
    return {
        "id": tensor.id,
        "labels": tensor.labels,
        "shape": tensor.shape,
        "structure": tensor.structure,
        "dtype": tensor.dtype,
    }


def _validate_wires(wires: tuple[int, ...], n_qubits: int) -> None:
    if not wires:
        raise ValueError("Gate operation has no wires")
    seen: set[int] = set()
    for wire in wires:
        if wire < 0 or wire >= n_qubits:
            raise ValueError(f"Wire {wire} outside qreg size {n_qubits}")
        if wire in seen:
            raise ValueError(f"Duplicate wire {wire} in gate operation")
        seen.add(wire)


def _view_payload(view: TensorView) -> dict[str, object]:
    return {
        "tensor_id": view.tensor_id,
        "labels": view.labels,
        "shape": view.shape,
        "slice_spec": view.slice_spec,
    }


def _node_payload(node: GraphNode) -> dict[str, object]:
    if isinstance(node, ContractNode):
        return {
            "kind": "contract",
            "node_id": node.node_id,
            "left": _view_payload(node.left),
            "right": _view_payload(node.right),
            "output": _tensor_payload(node.output),
            "contracted_labels": node.contracted_labels,
            "output_labels": node.output_labels,
            "dependencies": node.dependencies,
        }
    return {
        "kind": "reduce",
        "node_id": node.node_id,
        "inputs": [_view_payload(view) for view in node.inputs],
        "output": _tensor_payload(node.output),
        "reduced_labels": node.reduced_labels,
        "dependencies": node.dependencies,
    }


__all__ = [
    "lower_tensor_network",
    "validate_tensor_inputs",
    "validate_dag_inputs",
    "build_full_einsum_expression",
    "build_contract_node",
    "choose_slice_labels",
    "slice_contraction",
    "apply_slicing",
    "build_contraction_dag",
    "contraction_dag_hash",
    "validate_contraction_dag",
]
