"""Small, target-neutral contraction DAG values and pure construction helpers.

This module deliberately contains no tensor arrays, planner metadata, costs, or
target execution details.  The path uses opt_einsum's dynamic active-list
convention: every pair refers to positions in the active tensor list at that
step, and the produced tensor is appended to that list.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence, TypeAlias

from quantum_bench.core.records import TensorNetworkSpec, TensorSpec
from quantum_bench.tn.network import TensorNetworkValue


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorView:
    """A semantic tensor reference with optional fixed indices."""

    tensor_id: str
    labels: tuple[int, ...]
    shape: tuple[int, ...]
    slice_spec: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class SliceSpec:
    """One bounded semantic slice of one contraction label."""

    node_id: str
    label: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ContractNode:
    """One binary tensor contraction in the semantic graph."""

    node_id: str
    left: TensorView
    right: TensorView
    output: TensorSpec
    contracted_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ReduceNode:
    """Explicit sum reconstruction for sliced partial results."""

    node_id: str
    inputs: tuple[TensorView, ...]
    output: TensorSpec
    reduced_labels: tuple[int, ...] = ()
    dependencies: tuple[str, ...] = ()


GraphNode: TypeAlias = ContractNode | ReduceNode


@dataclass(frozen=True, slots=True, kw_only=True)
class ContractionDAG:
    """Planner-independent semantic contraction graph."""

    tensors: tuple[TensorSpec, ...]
    nodes: tuple[GraphNode, ...]
    output: TensorView


def apply_slicing(dag: ContractionDAG, spec: SliceSpec) -> ContractionDAG:
    """Rewrite one contraction into fixed-index partials and one reduction.

    This intentionally supports exactly one contracted label.  Target-local
    tiling remains outside the semantic graph and is not represented here.
    """

    validate_contraction_dag(dag)
    target_index = next(
        (index for index, node in enumerate(dag.nodes) if node.node_id == spec.node_id),
        None,
    )
    if target_index is None:
        raise ValueError(f"Unknown slice node {spec.node_id!r}")
    target = dag.nodes[target_index]
    if not isinstance(target, ContractNode):
        raise ValueError(f"Slice node {spec.node_id!r} is not a contract node")
    if target.left.slice_spec or target.right.slice_spec:
        raise ValueError(
            f"Slice node {spec.node_id!r} already has fixed input indices; "
            "nested slicing is unsupported"
        )
    if spec.label not in target.contracted_labels:
        raise ValueError(
            f"Label {spec.label} is not contracted by node {spec.node_id!r}"
        )

    dimension = _label_dimension_from_views(spec.label, target.left, target.right)
    if dimension <= 1:
        raise ValueError(f"Cannot slice contracted label {spec.label} with dimension {dimension}")

    partial_nodes: list[ContractNode] = []
    partial_views: list[TensorView] = []
    for value in range(dimension):
        left = _fix_view(target.left, spec.label, value)
        right = _fix_view(target.right, spec.label, value)
        partial_node_id = f"{target.node_id}__slice_{spec.label}_{value}"
        partial_output_id = f"{target.output.id}__slice_{spec.label}_{value}"
        partial_output = replace(target.output, id=partial_output_id, produced_by=partial_node_id)
        partial_node = replace(
            target,
            node_id=partial_node_id,
            left=left,
            right=right,
            output=partial_output,
            contracted_labels=tuple(
                label for label in target.contracted_labels if label != spec.label
            ),
        )
        partial_nodes.append(partial_node)
        partial_views.append(_view(partial_output))

    reduce_node_id = f"{target.node_id}__reduce_{spec.label}"
    reduced_output = replace(target.output, produced_by=reduce_node_id)
    reduce_node = ReduceNode(
        node_id=reduce_node_id,
        inputs=tuple(partial_views),
        output=reduced_output,
        reduced_labels=(spec.label,),
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


def build_contraction_dag(
    network: TensorNetworkSpec | TensorNetworkValue,
    path: Iterable[Iterable[int]],
) -> ContractionDAG:
    """Materialize a semantic DAG from a tensor network and dynamic path.

    ``path`` follows opt_einsum's dynamic active-list convention.  Planner
    identity and timing are intentionally not accepted or retained here.
    """

    spec = network.spec if isinstance(network, TensorNetworkValue) else network
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
        raise ValueError(f"Contraction path ended with {len(active)} active tensors; expected one")

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
        raise ValueError(f"Only binary contraction paths are supported; got {normalized_pair}")
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
    output_shape = tuple(_label_dimension(label, left, right) for label in output_labels)
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
                raise ValueError(f"Node {node.node_id} has unknown dependency {dependency}")
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
            produced[view.tensor_id]
            for view in views
            if view.tensor_id in produced
        }
        if input_producers != set(node.dependencies):
            raise ValueError(
                f"Node {node.node_id} dependencies do not match producers of consumed tensors"
            )
        if isinstance(node, ContractNode) and node.left.tensor_id == node.right.tensor_id:
            raise ValueError(f"Contract node {node.node_id} contracts a tensor with itself")

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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _view(spec: TensorSpec) -> TensorView:
    return TensorView(tensor_id=spec.id, labels=spec.labels, shape=spec.shape)


def _label_dimension_from_views(label: int, left: TensorView, right: TensorView) -> int:
    for view in (left, right):
        if label in view.labels:
            return view.shape[view.labels.index(label)]
    raise ValueError(f"Contracted label {label} is absent from both source views")


def _fix_view(view: TensorView, label: int, value: int) -> TensorView:
    if label not in view.labels:
        return view
    axis = view.labels.index(label)
    return TensorView(
        tensor_id=view.tensor_id,
        labels=view.labels[:axis] + view.labels[axis + 1 :],
        shape=view.shape[:axis] + view.shape[axis + 1 :],
        slice_spec=view.slice_spec + ((axis, value),),
    )


def _validate_view(
    view: TensorView,
    descriptors: dict[str, TensorSpec],
    owner: str,
) -> None:
    descriptor = descriptors.get(view.tensor_id)
    if descriptor is None:
        raise ValueError(f"Node {owner} references unknown tensor {view.tensor_id}")
    if len(view.labels) != len(view.shape):
        raise ValueError(f"Tensor view {view.tensor_id} has mismatched labels and shape")
    fixed_axes: set[int] = set()
    for item in view.slice_spec:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Tensor view {view.tensor_id} has invalid fixed index {item!r}")
        axis, value = item
        if not isinstance(axis, int) or not isinstance(value, int):
            raise ValueError(f"Tensor view {view.tensor_id} has invalid fixed index {item!r}")
        if axis in fixed_axes:
            raise ValueError(f"Tensor view {view.tensor_id} has duplicate fixed axes")
        fixed_axes.add(axis)
        if axis < 0 or axis >= len(descriptor.labels):
            raise ValueError(f"Tensor view {view.tensor_id} has invalid fixed axis {axis}")
        if value < 0 or value >= descriptor.shape[axis]:
            raise ValueError(f"Tensor view {view.tensor_id} has invalid fixed value {value}")

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
        raise ValueError(f"Contract node {node.node_id} has duplicate contracted labels")
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
            raise ValueError(f"Contract node {node.node_id} has mismatched label {label} dimensions")
    expected_shape = tuple(
        _view_label_dimension(label, node.left, node.right) for label in node.output_labels
    )
    if node.output.labels != node.output_labels or node.output.shape != expected_shape:
        raise ValueError(f"Contract node {node.node_id} output does not match its inputs")
    if node.output.dtype != left.dtype or node.output.dtype != right.dtype:
        raise ValueError(f"Contract node {node.node_id} output dtype does not match its inputs")


def _validate_reduce_algebra(node: ReduceNode, descriptors: dict[str, TensorSpec]) -> None:
    if len(set(node.reduced_labels)) != len(node.reduced_labels):
        raise ValueError(f"Reduce node {node.node_id} has duplicate reduced labels")
    output = node.output
    for view in node.inputs:
        if (view.labels, view.shape) != (output.labels, output.shape):
            raise ValueError(f"Reduce node {node.node_id} inputs do not match its output")


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
            (_semantic_view_payload(dag, node.left), _semantic_view_payload(dag, node.right)),
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
        raise ValueError(f"Cannot contract tensors with different dtypes: {left.dtype!r}, {right.dtype!r}")
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
    "ContractNode",
    "ContractionDAG",
    "GraphNode",
    "ReduceNode",
    "SliceSpec",
    "TensorView",
    "apply_slicing",
    "build_contraction_dag",
    "contraction_dag_hash",
    "validate_contraction_dag",
]
