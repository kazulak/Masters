from dataclasses import replace

import numpy as np
import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.lowering import (
    build_full_einsum_expression,
    build_contraction_dag,
    choose_slice_labels,
    contraction_dag_hash,
    lower_tensor_network,
    slice_contraction,
    validate_contraction_dag,
    validate_tensor_inputs,
)
from quantum_bench.model import (
    ContractionDAG,
    ContractNode,
    TensorNetwork,
    TensorSpec,
    TensorView,
    make_simulation_job,
)


def _network(
    tensors: tuple[TensorSpec, ...],
    output_labels: tuple[int, ...],
    expression: str,
) -> TensorNetwork:
    return TensorNetwork(
        circuit=builtin_circuit("bell_2q"),
        tensors=tensors,
        output_labels=output_labels,
        einsum_expression=expression,
    )


def _two_label_dag() -> ContractionDAG:
    return build_contraction_dag(
        _network(
            (
                TensorSpec("a", (0, 1, 2), (3, 2, 2), "dense", dtype="float64"),
                TensorSpec("b", (2, 3, 1), (2, 4, 2), "dense", dtype="float64"),
            ),
            (0, 3),
            "abc,cdb->ad",
        ),
        ((0, 1),),
    )


def test_network_structure_and_values_are_separate():
    network, inputs = lower_tensor_network(make_simulation_job(builtin_circuit("bell_2q")))

    assert tuple(inputs) == tuple(tensor.id for tensor in network.tensors)
    assert all(not hasattr(tensor, "array") for tensor in network.tensors)
    assert all(isinstance(array, np.ndarray) for array in inputs.values())


def test_lowering_rejects_nonempty_simulation_parameters():
    job = make_simulation_job(
        builtin_circuit("bell_2q"),
        parameters=(("mode", "sample"),),
    )
    with pytest.raises(ValueError, match="does not support simulation parameters"):
        lower_tensor_network(job)


def test_lowering_rejects_seeded_simulation_jobs():
    job = make_simulation_job(builtin_circuit("bell_2q"), seed=7)
    with pytest.raises(ValueError, match="seed must be None"):
        lower_tensor_network(job)


def test_tensor_input_validation_rejects_missing_and_wrong_shape():
    network, inputs = lower_tensor_network(make_simulation_job(builtin_circuit("bell_2q")))

    missing = dict(inputs)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing"):
        validate_tensor_inputs(network, missing)

    first_id = next(iter(inputs))
    wrong = dict(inputs)
    wrong[first_id] = np.zeros((3,), dtype=np.complex128)
    with pytest.raises(ValueError, match="shape"):
        validate_tensor_inputs(network, wrong)


def test_tensor_input_validation_rejects_extra_ids():
    network, inputs = lower_tensor_network(make_simulation_job(builtin_circuit("bell_2q")))
    extra = dict(inputs)
    extra["unexpected"] = np.zeros((1,), dtype=np.complex128)

    with pytest.raises(ValueError, match="extra"):
        validate_tensor_inputs(network, extra)


def test_tensor_input_validation_rejects_dtype_mismatch():
    network, inputs = lower_tensor_network(make_simulation_job(builtin_circuit("bell_2q")))
    first_id = next(iter(inputs))
    wrong = dict(inputs)
    wrong[first_id] = np.zeros(inputs[first_id].shape, dtype=np.float32)

    with pytest.raises(ValueError, match="dtype"):
        validate_tensor_inputs(network, wrong)


def test_full_einsum_expression_uses_numpy_letters_at_exact_limit():
    tensors = [TensorSpec(f"tensor_{label}", (label,), (2,), "dense") for label in range(52)]

    expression = build_full_einsum_expression(tensors, (0,))

    assert not expression.startswith("__label_list_einsum_required__")


def test_full_einsum_expression_uses_label_list_sentinel_above_numpy_limit():
    tensors = [TensorSpec(f"tensor_{label}", (label,), (2,), "dense") for label in range(53)]

    expression = build_full_einsum_expression(tensors, (0,))

    assert expression == "__label_list_einsum_required__:labels=53"


def test_contraction_dag_identity_is_deterministic_and_path_specific():
    tensors = (
        TensorSpec("a", (0, 1), (2, 2), "dense"),
        TensorSpec("b", (1, 2), (2, 2), "dense"),
        TensorSpec("c", (2, 0), (2, 2), "dense"),
    )
    network = _network(tensors, (), "ab,bc,ca->")

    first = build_contraction_dag(network, ((0, 1), (0, 1)))
    repeated = build_contraction_dag(network, ((0, 1), (0, 1)))
    alternate = build_contraction_dag(network, ((1, 2), (0, 1)))

    assert contraction_dag_hash(first) == contraction_dag_hash(repeated)
    assert contraction_dag_hash(first) != contraction_dag_hash(alternate)


def test_choose_slice_labels_is_deterministic_by_dimension_then_label():
    node = ContractNode(
        node_id="selector",
        left=TensorView(
            tensor_id="left",
            labels=(10, 8, 5, 7),
            shape=(2, 4, 4, 1),
        ),
        right=TensorView(
            tensor_id="right",
            labels=(10, 8, 5, 7),
            shape=(2, 4, 4, 1),
        ),
        output=TensorSpec("out", (), (), "dense"),
        contracted_labels=(10, 8, 5, 7),
        output_labels=(),
    )

    assert choose_slice_labels(node, minimum_slice_count=2) == (5,)
    assert choose_slice_labels(node, minimum_slice_count=16) == (5, 8)
    with pytest.raises(ValueError, match="minimum slice count"):
        choose_slice_labels(node, minimum_slice_count=33)


def test_multi_label_slicing_creates_cartesian_partials_and_reduce_node():
    dag = _two_label_dag()

    sliced = slice_contraction(dag, node_id="contract_0", labels=(2, 1))

    partials = [node for node in sliced.nodes if isinstance(node, ContractNode)]
    reduction = sliced.nodes[-1]
    assert len(partials) == 4
    assert reduction.reduced_labels == (1, 2)
    assert reduction.output.id == dag.output.tensor_id
    assert all(node.contracted_labels == () for node in partials)
    assert [node.left.slice_spec for node in partials] == [
        ((1, 0), (2, 0)),
        ((1, 0), (2, 1)),
        ((1, 1), (2, 0)),
        ((1, 1), (2, 1)),
    ]
    assert [node.right.slice_spec for node in partials] == [
        ((0, 0), (2, 0)),
        ((0, 1), (2, 0)),
        ((0, 0), (2, 1)),
        ((0, 1), (2, 1)),
    ]
    assert contraction_dag_hash(sliced) == contraction_dag_hash(
        slice_contraction(dag, node_id="contract_0", labels=(1, 2))
    )


def test_slicing_rewrites_the_direct_downstream_dependency():
    dag = build_contraction_dag(
        _network(
            (
                TensorSpec("a", (0, 1, 2), (2, 2, 2), "dense"),
                TensorSpec("b", (2, 3, 1), (2, 2, 2), "dense"),
                TensorSpec("c", (3, 4), (2, 3), "dense"),
            ),
            (0, 4),
            "abc,cdb,de->ae",
        ),
        ((0, 1), (0, 1)),
    )

    sliced = slice_contraction(dag, node_id="contract_0", labels=(1, 2))
    reduction = next(node for node in sliced.nodes if node.node_id.startswith("contract_0__reduce"))
    downstream = next(node for node in sliced.nodes if node.node_id == "contract_1")

    assert downstream.dependencies == (reduction.node_id,)
    assert sliced.output.tensor_id == dag.output.tensor_id


def test_dag_validation_rejects_invalid_fixed_view_missing_dependency_and_cycle():
    dag = _two_label_dag()
    node = dag.nodes[0]
    assert isinstance(node, ContractNode)
    invalid_view = replace(
        node,
        left=replace(node.left, labels=(0,), shape=(3,), slice_spec=((1, 3),)),
    )
    with pytest.raises(ValueError, match="invalid fixed value"):
        validate_contraction_dag(replace(dag, nodes=(invalid_view,)))

    missing_dependency = replace(node, dependencies=("missing",))
    with pytest.raises(ValueError, match="unknown dependency"):
        validate_contraction_dag(replace(dag, nodes=(missing_dependency,)))

    first = ContractNode(
        node_id="first",
        left=TensorView(tensor_id="a", labels=(0,), shape=(2,)),
        right=TensorView(tensor_id="b", labels=(0,), shape=(2,)),
        output=TensorSpec("out_a", (), (), "dense"),
        contracted_labels=(0,),
        output_labels=(),
        dependencies=("second",),
    )
    second = replace(
        first,
        node_id="second",
        output=TensorSpec("out_b", (), (), "dense"),
        dependencies=("first",),
    )
    cyclic = ContractionDAG(
        tensors=(
            TensorSpec("a", (0,), (2,), "dense"),
            TensorSpec("b", (0,), (2,), "dense"),
        ),
        nodes=(first, second),
        output=TensorView(tensor_id="out_b", labels=(), shape=()),
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_contraction_dag(cyclic)
