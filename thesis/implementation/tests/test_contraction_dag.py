from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

from quantum_bench.core.records import TensorNetworkSpec, TensorSpec, TensorValue
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode, SliceSpec, TensorView
from quantum_bench.lowering import (
    apply_slicing,
    build_contraction_dag,
    contraction_dag_hash,
    validate_contraction_dag,
)
from quantum_bench.tn.network import TensorNetworkValue


def _contains_array(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if is_dataclass(value):
        return any(_contains_array(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, tuple):
        return any(_contains_array(item) for item in value)
    return False


def _network(*, provenance: str | None = None) -> TensorNetworkValue:
    left = TensorSpec("a", (0, 1), (2, 3), "dense", produced_by=provenance)
    right = TensorSpec("b", (1, 2), (3, 4), "dense", produced_by=provenance)
    spec = TensorNetworkSpec(
        circuit=None,  # type: ignore[arg-type]
        tensors=(left, right),
        output_labels=(0, 2),
        einsum_expression="ab,bc->ac",
    )
    return TensorNetworkValue(
        spec=spec,
        tensors=[TensorValue(left, np.ones(left.shape)), TensorValue(right, np.ones(right.shape))],
    )


def test_hash_is_deterministic_and_excludes_array_payloads() -> None:
    first = build_contraction_dag(_network().spec, ((0, 1),))
    second = build_contraction_dag(_network().spec, ((0, 1),))

    assert contraction_dag_hash(first) == contraction_dag_hash(second)
    assert not hasattr(first, "network")
    assert all(isinstance(tensor, TensorSpec) for tensor in first.tensors)
    assert not _contains_array(first)


def test_same_path_has_same_identity_without_provenance() -> None:
    first = build_contraction_dag(_network(provenance="planner_a").spec, ((0, 1),))
    second = build_contraction_dag(_network(provenance="planner_b").spec, ((0, 1),))

    assert contraction_dag_hash(first) == contraction_dag_hash(second)


def test_different_paths_have_different_identity() -> None:
    network = TensorNetworkValue(
        spec=TensorNetworkSpec(
            circuit=None,  # type: ignore[arg-type]
            tensors=(
                TensorSpec("a", (0, 1), (2, 2), "dense"),
                TensorSpec("b", (1, 2), (2, 2), "dense"),
                TensorSpec("c", (2, 0), (2, 2), "dense"),
            ),
            output_labels=(),
            einsum_expression="ab,bc,ca->",
        ),
        tensors=[],
    )

    left_associative = build_contraction_dag(network.spec, ((0, 1), (0, 1)))
    right_associative = build_contraction_dag(network.spec, ((1, 2), (0, 1)))

    assert contraction_dag_hash(left_associative) != contraction_dag_hash(right_associative)


def test_independent_contraction_order_has_same_identity() -> None:
    tensors = (
        TensorSpec("a", (0, 1), (2, 2), "dense"),
        TensorSpec("b", (1, 2), (2, 2), "dense"),
        TensorSpec("c", (3, 4), (2, 2), "dense"),
        TensorSpec("d", (4, 5), (2, 2), "dense"),
    )
    spec = TensorNetworkSpec(None, tensors, (0, 2, 3, 5), "ab,bc,de,ef->acdf")  # type: ignore[arg-type]

    first = build_contraction_dag(spec, ((0, 1), (0, 1), (0, 1)))
    second = build_contraction_dag(spec, ((2, 3), (0, 1), (0, 1)))

    assert contraction_dag_hash(first) == contraction_dag_hash(second)


def test_single_label_slicing_rewrites_and_sums_partial_contractions() -> None:
    dag = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec("a", (0, 1), (2, 3), "dense", dtype="float64"),
                TensorSpec("b", (1, 2), (3, 4), "dense", dtype="float64"),
            ),
            (0, 2),
            "ab,bc->ac",
        ),
        ((0, 1),),
    )  # type: ignore[arg-type]
    sliced = apply_slicing(dag, SliceSpec(node_id=dag.nodes[0].node_id, label=1))

    partials = [node for node in sliced.nodes if isinstance(node, ContractNode)]
    reduce = sliced.nodes[-1]
    assert len(partials) == 3
    assert isinstance(reduce, ReduceNode)
    assert all(node.contracted_labels == () for node in partials)
    assert all(node.output.labels == (0, 2) and node.output.shape == (2, 4) for node in partials)
    assert [node.left.slice_spec for node in partials] == [((1, value),) for value in range(3)]
    assert [node.right.slice_spec for node in partials] == [((0, value),) for value in range(3)]
    assert reduce.output.id == dag.output.tensor_id
    assert sliced.output == dag.output

    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    partial_values = [np.outer(left[:, value], right[value, :]) for value in range(3)]
    assert np.array_equal(sum(partial_values), np.einsum("ab,bc->ac", left, right))
    assert contraction_dag_hash(sliced) != contraction_dag_hash(dag)
    assert contraction_dag_hash(sliced) == contraction_dag_hash(
        apply_slicing(dag, SliceSpec(node_id=dag.nodes[0].node_id, label=1))
    )


def test_slicing_rewrites_downstream_dependency_but_keeps_tensor_id() -> None:
    tensors = (
        TensorSpec("a", (0, 1), (2, 3), "dense"),
        TensorSpec("b", (1, 2), (3, 4), "dense"),
        TensorSpec("c", (2, 3), (4, 5), "dense"),
    )
    dag = build_contraction_dag(
        TensorNetworkSpec(None, tensors, (0, 3), "ab,bc,cd->ad"),
        ((0, 1), (0, 1)),
    )  # type: ignore[arg-type]
    target = dag.nodes[0]
    downstream = dag.nodes[1]
    sliced = apply_slicing(dag, SliceSpec(node_id=target.node_id, label=1))
    rewritten = next(node for node in sliced.nodes if node.node_id == downstream.node_id)
    reduce = next(node for node in sliced.nodes if isinstance(node, ReduceNode))

    assert isinstance(rewritten, ContractNode)
    assert target.output.id in (rewritten.left.tensor_id, rewritten.right.tensor_id)
    assert rewritten.dependencies == (reduce.node_id,)
    assert sliced.output.tensor_id == dag.output.tensor_id


def test_slicing_rejects_unknown_noncontract_and_invalid_labels() -> None:
    dag = build_contraction_dag(_network().spec, ((0, 1),))
    with pytest.raises(ValueError, match="Unknown slice node"):
        apply_slicing(dag, SliceSpec(node_id="missing", label=1))

    sliced = apply_slicing(dag, SliceSpec(node_id=dag.nodes[0].node_id, label=1))
    partial = next(node for node in sliced.nodes if isinstance(node, ContractNode))
    with pytest.raises(ValueError, match="nested slicing is unsupported"):
        apply_slicing(sliced, SliceSpec(node_id=partial.node_id, label=1))
    with pytest.raises(ValueError, match="not a contract node"):
        apply_slicing(sliced, SliceSpec(node_id=sliced.nodes[-1].node_id, label=1))
    with pytest.raises(ValueError, match="not contracted"):
        apply_slicing(dag, SliceSpec(node_id=dag.nodes[0].node_id, label=0))

    dimension_one = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (TensorSpec("a", (0, 1), (2, 1), "dense"), TensorSpec("b", (1, 2), (1, 2), "dense")),
            (0, 2),
            "ab,bc->ac",
        ),
        ((0, 1),),
    )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimension 1"):
        apply_slicing(dimension_one, SliceSpec(node_id=dimension_one.nodes[0].node_id, label=1))


def test_fixed_view_shape_and_value_are_validated_against_descriptor() -> None:
    dag = build_contraction_dag(_network().spec, ((0, 1),))
    node = dag.nodes[0]
    assert isinstance(node, ContractNode)
    invalid = replace(
        node,
        left=replace(node.left, labels=(0,), shape=(3,), slice_spec=((1, 3),)),
    )
    malformed = replace(dag, nodes=(invalid,))

    with pytest.raises(ValueError, match="invalid fixed value"):
        validate_contraction_dag(malformed)


def test_dependencies_must_match_consumed_generated_tensors_exactly() -> None:
    tensors = (
        TensorSpec("a", (0, 1), (2, 3), "dense"),
        TensorSpec("b", (1, 2), (3, 4), "dense"),
        TensorSpec("c", (2, 3), (4, 5), "dense"),
    )
    dag = build_contraction_dag(
        TensorNetworkSpec(None, tensors, (0, 3), "ab,bc,cd->ad"),
        ((0, 1), (0, 1)),
    )  # type: ignore[arg-type]
    downstream = dag.nodes[1]
    assert isinstance(downstream, ContractNode)

    omitted = replace(dag, nodes=(dag.nodes[0], replace(downstream, dependencies=())))
    with pytest.raises(ValueError, match="dependencies do not match"):
        validate_contraction_dag(omitted)

    duplicate = replace(
        dag,
        nodes=(dag.nodes[0], replace(downstream, dependencies=(downstream.dependencies[0],) * 2)),
    )
    with pytest.raises(ValueError, match="duplicate dependencies"):
        validate_contraction_dag(duplicate)


def test_invalid_dependency_is_rejected() -> None:
    node = ContractNode(
        node_id="contract_0",
        left=TensorView(tensor_id="a", labels=(0,), shape=(2,)),
        right=TensorView(tensor_id="b", labels=(0,), shape=(2,)),
        output=TensorSpec("out", (), (), "dense"),
        contracted_labels=(0,),
        output_labels=(),
        dependencies=("missing",),
    )
    dag = ContractionDAG(
        tensors=(TensorSpec("a", (0,), (2,), "dense"), TensorSpec("b", (0,), (2,), "dense")),
        nodes=(node,),
        output=TensorView(tensor_id="out", labels=(), shape=()),
    )

    with pytest.raises(ValueError, match="unknown dependency"):
        validate_contraction_dag(dag)


def test_dependency_cycle_is_rejected() -> None:
    a = TensorSpec("a", (0,), (2,), "dense")
    b = TensorSpec("b", (0,), (2,), "dense")
    first = ContractNode(
        node_id="first",
        left=TensorView(tensor_id="a", labels=(0,), shape=(2,)),
        right=TensorView(tensor_id="b", labels=(0,), shape=(2,)),
        output=TensorSpec("out_a", (), (), "dense"),
        contracted_labels=(0,),
        output_labels=(),
        dependencies=("second",),
    )
    second = replace(first, node_id="second", output=TensorSpec("out_b", (), (), "dense"), dependencies=("first",))
    dag = ContractionDAG(
        tensors=(a, b),
        nodes=(first, second),
        output=TensorView(tensor_id="out_b", labels=(), shape=()),
    )

    with pytest.raises(ValueError, match="cycle"):
        validate_contraction_dag(dag)
