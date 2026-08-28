from pathlib import Path

import pytest

from quantum_bench.circuits import (
    builtin_circuit,
    load_circuit,
    quest_compatible_circuit,
)


@pytest.mark.parametrize(
    ("alias", "canonical", "params"),
    [
        ("quantization-stress", "quantization_stress", {"n_qubits": 3}),
        ("quant_stress", "quantization_stress", {"n_qubits": 3}),
        ("bernstein_vazirani", "bv", {"n_qubits": 3}),
        ("parity", "xor", {"n_qubits": 3}),
        ("bb_n", "bb84", {"n_qubits": 3}),
        ("dense_coding", "edc", {"n_qubits": 3}),
        ("hidden_shift", "hs", {"allocated_qubits": 4}),
    ],
)
def test_builtin_aliases_preserve_generated_operations(
    alias: str, canonical: str, params: dict
) -> None:
    aliased = builtin_circuit(alias, params)
    expected = builtin_circuit(canonical, params)

    assert aliased.n_qubits == expected.n_qubits
    assert aliased.operations == expected.operations


@pytest.mark.parametrize(
    ("alias", "canonical", "params"),
    [
        ("bernstein_vazirani", "bv", {"n_qubits": 3}),
        ("parity", "xor", {"n_qubits": 3}),
        ("bb_n", "bb84", {"n_qubits": 3}),
        ("dense_coding", "edc", {"n_qubits": 3}),
        ("hidden_shift", "hs", {"allocated_qubits": 4}),
    ],
)
def test_quest_aliases_preserve_generated_operations(
    alias: str, canonical: str, params: dict
) -> None:
    aliased = quest_compatible_circuit(alias, params)
    expected = quest_compatible_circuit(canonical, params)

    assert aliased.n_qubits == expected.n_qubits
    assert aliased.operations == expected.operations


@pytest.mark.parametrize("generator", [builtin_circuit, quest_compatible_circuit])
def test_qubit_count_alias_preserves_generated_operations(generator) -> None:
    canonical = generator("qrng", {"n_qubits": 3})
    aliased = generator("qrng", {"qubits": 3})

    assert aliased.n_qubits == canonical.n_qubits
    assert aliased.operations == canonical.operations


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("bell_2q", {}),
        ("ghz_4q", {}),
        ("ghz_chain", {"n_qubits": 3}),
        ("qrng", {"n_qubits": 3}),
        ("quantization_stress", {"n_qubits": 3, "repeat_layers": 2}),
        ("bv", {"n_qubits": 3}),
        ("xor", {"n_qubits": 3}),
        ("bb84", {"n_qubits": 3}),
        ("edc", {"n_qubits": 3}),
        ("hs", {"logical_qubits": 2, "allocated_qubits": 4}),
    ],
)
def test_builtin_source_metadata_round_trips_explicitly(
    name: str, params: dict
) -> None:
    circuit = builtin_circuit(name, params)
    regenerated = builtin_circuit(name, dict(circuit.source))

    assert regenerated.n_qubits == circuit.n_qubits
    assert regenerated.operations == circuit.operations


@pytest.mark.parametrize("name", ["qrng", "bb84", "bv", "edc", "xor"])
def test_quest_source_metadata_round_trips_explicitly(name: str) -> None:
    circuit = quest_compatible_circuit(
        name, {"n_qubits": 3, "repeat_layers": 2}
    )
    regenerated = quest_compatible_circuit(name, dict(circuit.source))

    assert regenerated.n_qubits == circuit.n_qubits
    assert regenerated.operations == circuit.operations


def test_quest_hs_source_metadata_round_trips_explicitly() -> None:
    circuit = quest_compatible_circuit(
        "hs", {"logical_qubits": 2, "allocated_qubits": 4, "repeat_layers": 2}
    )

    regenerated = quest_compatible_circuit("hs", dict(circuit.source))

    assert regenerated.n_qubits == circuit.n_qubits
    assert regenerated.operations == circuit.operations


@pytest.mark.parametrize(
    ("generator", "name", "params"),
    [
        (builtin_circuit, "qrng", {"unknown": 1}),
        (builtin_circuit, "bell_2q", {"n_qubits": 2}),
        (builtin_circuit, "qrng", {"depth": 2}),
        (quest_compatible_circuit, "qrng", {"depth": 2}),
        (quest_compatible_circuit, "hs", {"unknown": 1}),
    ],
)
def test_generator_schemas_reject_unknown_or_irrelevant_keys(
    generator, name: str, params: dict
) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        generator(name, params)


@pytest.mark.parametrize("value", [True, False, 0, -1, 3.0, "3", None])
@pytest.mark.parametrize("generator", [builtin_circuit, quest_compatible_circuit])
def test_generators_reject_invalid_or_lossily_coerced_qubit_counts(
    generator, value: object
) -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        generator("qrng", {"n_qubits": value})


@pytest.mark.parametrize("value", [True, 0, -1, 2.0, "2", None])
def test_generators_reject_invalid_layer_counts(value: object) -> None:
    with pytest.raises(ValueError, match="repeat_layers"):
        quest_compatible_circuit("qrng", {"repeat_layers": value})
    with pytest.raises(ValueError, match="repeat_layers"):
        builtin_circuit("quantization_stress", {"repeat_layers": value})


def test_generators_reject_ambiguous_or_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="aliases"):
        builtin_circuit("qrng", {"n_qubits": 3, "qubits": 3})
    with pytest.raises(ValueError, match="must match"):
        quest_compatible_circuit(
            "hs", {"n_qubits": 4, "allocated_qubits": 6}
        )
    with pytest.raises(ValueError, match="half"):
        builtin_circuit("hs", {"logical_qubits": 3, "allocated_qubits": 4})


def test_load_circuit_does_not_coerce_names() -> None:
    with pytest.raises(ValueError, match="nonempty string"):
        load_circuit(
            {"circuit": {"kind": "builtin", "name": 1}}, Path.cwd()
        )
