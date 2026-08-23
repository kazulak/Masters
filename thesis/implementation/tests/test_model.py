from dataclasses import FrozenInstanceError, asdict, fields
from copy import deepcopy
import json
from collections.abc import Mapping

import pytest

from quantum_bench.core.records import CircuitSpec as LegacyCircuitSpec
from quantum_bench.core.records import (
    TensorNetworkSpec,
    TensorSpec as LegacyTensorSpec,
    to_jsonable,
)
from quantum_bench.model import (
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
)
from quantum_bench.tn.graph import (
    ContractionDAG as GraphContractionDAG,
    ContractNode as GraphContractNode,
    ReduceNode as GraphReduceNode,
    SliceSpec as GraphSliceSpec,
    TensorView as GraphTensorView,
)


def _circuit() -> CircuitSpec:
    return CircuitSpec(name="test", n_qubits=1, operations=(), source={})


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


def test_direct_job_rejects_invalid_query_and_bool_seed() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        SimulationJob(_circuit(), query="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        SimulationJob(_circuit(), seed=True)


def test_direct_job_rejects_duplicate_and_non_tuple_parameters() -> None:
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
        circuit.source["nested"]["values"] += (3,)
    with pytest.raises(TypeError):
        circuit.source["new"] = "value"
    with pytest.raises(AttributeError):
        circuit.source.update({"new": "value"})  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        dict.__setitem__(circuit.source, "new", "value")

    assert dict(circuit.source) == {
        "nested": {"values": (1, {"flag": True})},
    }
    assert circuit.source == {"nested": {"values": (1, {"flag": True})}}
    detached = deepcopy(circuit.source)
    assert isinstance(detached, dict)
    assert detached == dict(circuit.source)
    assert detached is not circuit.source
    assert circuit.source.get("missing") is None
    assert list(circuit.source) == ["nested"]


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


def test_frozen_model_records_reject_assignment() -> None:
    job = make_simulation_job(_circuit())
    network = TensorNetwork(circuit=_circuit(), tensors=(), output_labels=(), einsum_expression="")
    with pytest.raises(FrozenInstanceError):
        job.seed = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        network.einsum_expression = "x"  # type: ignore[misc]


def test_legacy_serialization_preserves_source_content() -> None:
    source = {"nested": {"values": [1, 2]}, "enabled": True}
    circuit = CircuitSpec(name="test", n_qubits=1, operations=(), source=source)
    expected_asdict = {
        "name": "test",
        "n_qubits": 1,
        "operations": (),
        "source": {"nested": {"values": (1, 2)}, "enabled": True},
    }
    assert asdict(circuit) == expected_asdict
    assert json.loads(json.dumps(asdict(circuit))) == {
        "name": "test",
        "n_qubits": 1,
        "operations": [],
        "source": source,
    }
    serialized = to_jsonable(circuit)
    expected_json = {
        "name": "test",
        "n_qubits": 1,
        "operations": [],
        "source": source,
    }
    assert serialized == expected_json
    assert json.dumps(serialized)


def test_model_records_are_frozen_and_network_has_only_semantic_fields() -> None:
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


def test_compatibility_reexports_are_object_identical() -> None:
    assert LegacyCircuitSpec is CircuitSpec
    assert LegacyTensorSpec is TensorSpec
    assert TensorNetworkSpec is TensorNetwork
    assert GraphTensorView is TensorView
    assert GraphSliceSpec is SliceSpec
    assert GraphContractNode is ContractNode
    assert GraphReduceNode is ReduceNode
    assert GraphContractionDAG is ContractionDAG
