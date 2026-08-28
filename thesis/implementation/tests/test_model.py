from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from collections.abc import Mapping

import pytest

from quantum_bench.model import (
    CircuitOperation,
    CircuitSpec,
    ContractionDAG,
    ContractNode,
    ReduceNode,
    SimulationJob,
    SliceSpec,
    TensorNetwork,
    TensorSpec,
    TensorView,
    make_simulation_job,
    validate_circuit_spec,
)


def _circuit() -> CircuitSpec:
    return CircuitSpec(
        name="test",
        n_qubits=1,
        operations=(CircuitOperation(gate="x", wires=(0,)),),
        source={"gate": "x", "wires": [0]},
    )


def test_factory_normalizes_parameters_and_direct_jobs_require_normalization() -> None:
    job = make_simulation_job(
        _circuit(),
        parameters=(("z", 2), ("a", "value"), ("m", None)),
    )

    assert job.parameters == (("a", "value"), ("m", None), ("z", 2))
    assert job.query == "pre_measurement_statevector"
    assert job.seed is None
    with pytest.raises(ValueError, match="strictly key-sorted"):
        SimulationJob(_circuit(), parameters=(("z", 1), ("a", 2)))


@pytest.mark.parametrize(
    "parameters, message",
    [
        ((("", 1),), "nonempty"),
        ((("a", 1), ("a", 2)), "Duplicate"),
        ((("a", []),), "non-scalar"),
    ],
)
def test_factory_rejects_invalid_parameters(parameters, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_simulation_job(_circuit(), parameters=parameters)


def test_direct_job_rejects_invalid_query_bool_seed_and_parameter_shape() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        SimulationJob(_circuit(), query="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        SimulationJob(_circuit(), seed=True)
    with pytest.raises(ValueError, match="Duplicate"):
        SimulationJob(_circuit(), parameters=(("a", 1), ("a", 2)))
    with pytest.raises(ValueError, match="tuple"):
        SimulationJob(_circuit(), parameters=[("a", 1)])  # type: ignore[arg-type]


def test_circuit_source_is_deeply_immutable_and_detached_from_input() -> None:
    original = {"nested": {"values": [1, {"flag": True}]}}
    circuit = CircuitSpec(name="test", n_qubits=1, operations=(), source=original)

    original["nested"]["values"].append(2)
    assert isinstance(circuit.source, Mapping)
    assert not isinstance(circuit.source, dict)
    assert circuit.source["nested"]["values"] == (1, {"flag": True})
    with pytest.raises(TypeError):
        circuit.source["nested"]["values"][1]["flag"] = False
    with pytest.raises(TypeError):
        circuit.source["new"] = "value"
    with pytest.raises(AttributeError):
        circuit.source.update({"new": "value"})  # type: ignore[attr-defined]

    detached = deepcopy(circuit.source)
    assert isinstance(detached, dict)
    assert detached == dict(circuit.source)
    assert detached is not circuit.source


def test_circuit_source_rejects_unordered_and_unsupported_values() -> None:
    with pytest.raises(TypeError, match="unordered sets"):
        CircuitSpec(name="test", n_qubits=1, operations=(), source={"items": {1, 2}})
    with pytest.raises(TypeError, match="does not support values"):
        CircuitSpec(name="test", n_qubits=1, operations=(), source={"value": object()})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(value: float) -> None:
    with pytest.raises(TypeError, match="non-finite"):
        CircuitSpec(name="test", n_qubits=1, operations=(), source={"value": value})
    with pytest.raises(ValueError, match="finite"):
        make_simulation_job(_circuit(), parameters=(("value", value),))


@pytest.mark.parametrize("n_qubits", [True, False, 0, -1, 1.0, "1"])
def test_circuit_validator_rejects_invalid_qubit_counts(n_qubits: object) -> None:
    circuit = CircuitSpec("invalid", n_qubits, (), {})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive non-bool integer"):
        make_simulation_job(circuit)


@pytest.mark.parametrize(
    ("gate", "wires", "params"),
    [
        ("i", (0,), ()),
        ("h", (0,), ()),
        ("x", (0,), ()),
        ("y", (0,), ()),
        ("z", (0,), ()),
        ("s", (0,), ()),
        ("t", (0,), ()),
        ("ry", (0,), (0.5,)),
        ("rz", (0,), (-0.5,)),
        ("cx", (0, 1), ()),
        ("cnot", (0, 1), ()),
        ("cz", (0, 1), ()),
        ("swap", (0, 1), ()),
    ],
)
def test_circuit_validator_accepts_the_exact_active_gate_set(
    gate: str, wires: tuple[int, ...], params: tuple[float, ...]
) -> None:
    circuit = CircuitSpec("valid", 2, (CircuitOperation(gate, wires, params),), {})

    assert validate_circuit_spec(circuit) is None


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (CircuitOperation("H", (0,)), "unsupported gate"),
        (CircuitOperation("measure", (0,)), "unsupported gate"),
        (CircuitOperation("h", (0, 1)), "exactly 1 wire"),
        (CircuitOperation("cx", (0,)), "exactly 2 wire"),
        (CircuitOperation("h", (0,), (0.5,)), "exactly 0 parameter"),
        (CircuitOperation("ry", (0,)), "exactly 1 parameter"),
        (CircuitOperation("cx", (0, 0)), "unique"),
        (CircuitOperation("x", (True,)), "non-bool integers"),
        (CircuitOperation("x", (2,)), "outside"),
        (CircuitOperation("x", (-1,)), "outside"),
        (CircuitOperation("ry", (0,), (True,)), "numeric"),
        (CircuitOperation("ry", (0,), (float("nan"),)), "finite"),
        (CircuitOperation("rz", (0,), (float("inf"),)), "finite"),
    ],
)
def test_make_simulation_job_rejects_malformed_circuits(
    operation: CircuitOperation, message: str
) -> None:
    circuit = CircuitSpec("invalid", 2, (operation,), {})

    with pytest.raises(ValueError, match=message):
        make_simulation_job(circuit)


def test_circuit_validator_requires_circuit_operations() -> None:
    circuit = CircuitSpec("invalid", 1, ("x",), {})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="CircuitOperation"):
        make_simulation_job(circuit)


def test_model_records_are_frozen() -> None:
    job = make_simulation_job(_circuit())
    tensor = TensorSpec(id="a", labels=(0,), shape=(2,), structure="input")
    view = TensorView(tensor_id="a", labels=(0,), shape=(2,))
    network = TensorNetwork(
        circuit=_circuit(),
        tensors=(tensor,),
        output_labels=(0,),
        einsum_expression="a->a",
    )

    with pytest.raises(FrozenInstanceError):
        job.seed = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        tensor.shape = (1,)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        view.labels = (1,)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        network.einsum_expression = "a->"  # type: ignore[misc]


def test_tensor_network_contains_only_semantic_fields() -> None:
    assert {field.name for field in fields(TensorNetwork)} == {
        "circuit",
        "tensors",
        "output_labels",
        "einsum_expression",
    }
    assert hasattr(TensorNetwork, "__slots__")
    assert not hasattr(TensorNetwork, "array")
    assert not hasattr(TensorNetwork, "path")
    assert not hasattr(TensorNetwork, "dependencies")


def test_contraction_records_are_model_only_and_explicit() -> None:
    left = TensorView(tensor_id="a", labels=(0, 1), shape=(2, 2))
    right = TensorView(tensor_id="b", labels=(1, 2), shape=(2, 2))
    output = TensorSpec(id="c", labels=(0, 2), shape=(2, 2), structure="contraction")
    contract = ContractNode(
        node_id="contract-0",
        left=left,
        right=right,
        output=output,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    reduced = ReduceNode(
        node_id="reduce-0",
        inputs=(TensorView(tensor_id="c", labels=(0, 2), shape=(2, 2)),),
        output=output,
        reduced_labels=(1,),
    )
    dag = ContractionDAG(
        tensors=(output,),
        nodes=(contract, reduced),
        output=TensorView(tensor_id="c", labels=(0, 2), shape=(2, 2)),
    )

    assert contract.contracted_labels == (1,)
    assert reduced.reduced_labels == (1,)
    assert dag.nodes == (contract, reduced)
    assert SliceSpec(node_id="contract-0", label=1).label == 1
