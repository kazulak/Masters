"""Deterministic family-aligned OpenQASM definitions for the P6 workload.

The builders emit only the pre-measurement unitary circuit.  They are deliberately
small and have no file, candidate, planner, or execution side effects.  Parameters
are fixed by the returned instance metadata; ``build_family_qasm`` is useful for
small reference checks with the same definitions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib


_QASM_HEADER = (
    "OPENQASM 2.0;\n"
    'include "qelib1.inc";\n'
)
_FAMILIES = {"bb84", "bv", "edc", "hs", "qrng", "xor"}
_HS_REFERENCE = (
    "QASMBench:small/hs4_n4/hs4_n4.qasm@"
    "357b942396d5c2b7cbc1c229c585a6ef5ccaebac"
)


def _positive_int(parameters: Mapping[str, object], key: str) -> int:
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive non-bool integer")
    return value


def _exact_parameters(
    parameters: Mapping[str, object], expected: set[str]
) -> None:
    if not isinstance(parameters, Mapping):
        raise TypeError("family parameters must be a mapping")
    missing = expected - set(parameters)
    extra = set(parameters) - expected
    if missing or extra:
        raise ValueError(
            f"family parameters must contain exactly {sorted(expected)!r}; "
            f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )


def _qasm_bytes(n_qubits: int, operations: list[str]) -> bytes:
    if any(
        token in operation.lower()
        for operation in operations
        for token in ("measure", "reset", "creg", "if(")
    ):
        raise ValueError("family workload must remain a pre-measurement unitary")
    return (
        f"{_QASM_HEADER}qreg q[{n_qubits}];\n"
        + "".join(f"{operation}\n" for operation in operations)
    ).encode("ascii")


def _definition(
    family: str, parameters: Mapping[str, object]
) -> tuple[int, list[str], dict[str, object]]:
    if not isinstance(family, str) or family.lower() not in _FAMILIES:
        raise ValueError(f"unsupported family {family!r}")
    family = family.lower()

    if family == "qrng":
        _exact_parameters(parameters, {"n_qubits"})
        n_qubits = _positive_int(parameters, "n_qubits")
        return n_qubits, [f"h q[{wire}];" for wire in range(n_qubits)], {
            "operation_identity": "single_h_layer",
            "preparation": "one_h_per_wire",
            "semantic_expectation": "uniform_computational_basis_amplitudes",
        }

    if family == "bv":
        _exact_parameters(parameters, {"data_qubits"})
        data_qubits = _positive_int(parameters, "data_qubits")
        ancilla = data_qubits
        secret = tuple(1 if wire % 2 == 0 else 0 for wire in range(data_qubits))
        operations = [f"x q[{ancilla}];", f"h q[{ancilla}];"]
        operations.extend(f"h q[{wire}];" for wire in range(data_qubits))
        operations.extend(
            f"cx q[{wire}],q[{ancilla}];"
            for wire, bit in enumerate(secret)
            if bit
        )
        operations.extend(f"h q[{wire}];" for wire in range(data_qubits))
        return data_qubits + 1, operations, {
            "operation_identity": "alternating_nonzero_secret_phasekickback",
            "preparation": "target_x_then_h_and_data_h_oracle_h",
            "secret_bits_q0_first": "".join(str(bit) for bit in secret),
            "semantic_expectation": "data_secret_and_target_minus_state",
        }

    if family == "bb84":
        _exact_parameters(parameters, {"n_qubits"})
        n_qubits = _positive_int(parameters, "n_qubits")
        bits = tuple(wire % 2 for wire in range(n_qubits))
        bases = tuple((wire // 2) % 2 for wire in range(n_qubits))
        operations: list[str] = []
        for wire, bit, basis in zip(range(n_qubits), bits, bases):
            if bit:
                operations.append(f"x q[{wire}];")
            if basis:
                operations.append(f"h q[{wire}];")
        return n_qubits, operations, {
            "operation_identity": "alternating_bits_two_wire_basis_cycle",
            "preparation": "bit_x_then_basis_h",
            "bits_q0_first": "".join(str(bit) for bit in bits),
            "bases_q0_first": "".join(str(basis) for basis in bases),
            "semantic_expectation": "declared_bb84_product_states",
        }

    if family == "edc":
        _exact_parameters(parameters, {"data_qubits"})
        data_qubits = _positive_int(parameters, "data_qubits")
        if data_qubits < 2:
            raise ValueError("EDC data_qubits must be at least 2")
        syndrome_qubits = data_qubits - 1
        error_wire = data_qubits // 2
        operations = ["ry(pi/3) q[0];"]
        operations.extend(
            f"cx q[0],q[{wire}];" for wire in range(1, data_qubits)
        )
        operations.append(f"x q[{error_wire}];")
        operations.extend(
            operation
            for check in range(syndrome_qubits)
            for operation in (
                f"cx q[{check}],q[{data_qubits + check}];",
                f"cx q[{check + 1}],q[{data_qubits + check}];",
            )
        )
        return data_qubits + syndrome_qubits, operations, {
            "operation_identity": "ry_encoder_single_x_adjacent_coherent_syndrome",
            "preparation": "ry_pi_over_3_then_repetition_encoder",
            "error_wire": error_wire,
            "syndrome_wires_q0_first": list(
                range(data_qubits, data_qubits + syndrome_qubits)
            ),
            "semantic_expectation": "uncorrected_adjacent_parity_syndrome",
        }

    if family == "hs":
        _exact_parameters(parameters, {"allocated_qubits"})
        allocated_qubits = _positive_int(parameters, "allocated_qubits")
        if allocated_qubits < 2 or allocated_qubits % 2:
            raise ValueError("HS allocated_qubits must be a positive even integer")
        operations = []
        for control in range(0, allocated_qubits, 2):
            target = control + 1
            operations.extend(
                [
                    f"h q[{control}];",
                    f"h q[{target}];",
                    f"x q[{control}];",
                    f"h q[{target}];",
                    f"cx q[{control}],q[{target}];",
                    f"h q[{target}];",
                    f"x q[{control}];",
                    f"h q[{control}];",
                    f"h q[{target}];",
                    f"h q[{target}];",
                    f"cx q[{control}],q[{target}];",
                    f"h q[{target}];",
                    f"h q[{control}];",
                    f"h q[{target}];",
                ]
            )
        return allocated_qubits, operations, {
            "operation_identity": "pinned_hs_pair_block_v1",
            "preparation": "one_pinned_pair_block_per_disjoint_pair",
            "pair_block_gate_sequence": (
                "Hc,Ht,Xc,Ht,CX,Ht,Xc,Hc,Ht,Ht,CX,Ht,Hc,Ht"
            ),
            "source_reference": _HS_REFERENCE,
            "semantic_expectation": "each_zero_pair_maps_to_control_one_target_zero",
        }

    _exact_parameters(parameters, {"data_qubits"})
    data_qubits = _positive_int(parameters, "data_qubits")
    parity = data_qubits
    operations = [f"h q[{wire}];" for wire in range(data_qubits)]
    operations.extend(
        f"cx q[{wire}],q[{parity}];" for wire in range(data_qubits)
    )
    return data_qubits + 1, operations, {
        "operation_identity": "uniform_data_h_then_cnot_parity",
        "preparation": "uniform_data_h_layer_before_parity_cnot",
        "parity_ancilla": parity,
        "semantic_expectation": "uniform_data_with_computed_parity_ancilla",
    }


def build_family_qasm(
    family: str, parameters: Mapping[str, object]
) -> bytes:
    """Return deterministic ASCII OpenQASM2 bytes for one family definition."""

    n_qubits, operations, _ = _definition(family, parameters)
    return _qasm_bytes(n_qubits, operations)


_INSTANCE_SPECS: tuple[tuple[str, str, str, dict[str, object]], ...] = (
    ("qrng_train_total16_r1", "qrng", "training", {"n_qubits": 16}),
    ("qrng_test_total17_r1", "qrng", "test", {"n_qubits": 17}),
    ("bv_train_total16_r1", "bv", "training", {"data_qubits": 15}),
    ("bv_test_total17_r1", "bv", "test", {"data_qubits": 16}),
    ("bb84_train_total16_r1", "bb84", "training", {"n_qubits": 16}),
    ("bb84_test_total17_r1", "bb84", "test", {"n_qubits": 17}),
    ("edc_train_data8_syndrome7_total15_r1", "edc", "training", {"data_qubits": 8}),
    ("edc_test_data9_syndrome8_total17_r1", "edc", "test", {"data_qubits": 9}),
    ("hs_train_total16_r1", "hs", "training", {"allocated_qubits": 16}),
    ("hs_test_total18_r1", "hs", "test", {"allocated_qubits": 18}),
    ("xor_train_total16_r1", "xor", "training", {"data_qubits": 15}),
    ("xor_test_total17_r1", "xor", "test", {"data_qubits": 16}),
)


def _gate_counts(operations: list[str]) -> dict[str, int]:
    counts = Counter(operation.split(" ", 1)[0].split("(", 1)[0] for operation in operations)
    one_qubit = sum(
        count
        for gate, count in counts.items()
        if gate in {"h", "x", "ry"}
    )
    two_qubit = sum(count for gate, count in counts.items() if gate == "cx")
    return {
        "1q": one_qubit,
        "2q": two_qubit,
        "total": len(operations),
        **dict(sorted(counts.items())),
    }


def _metadata(
    instance_id: str,
    family: str,
    split: str,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    n_qubits, operations, definition_metadata = _definition(family, parameters)
    qasm = _qasm_bytes(n_qubits, operations)
    metadata: dict[str, object] = {
        "instance_id": instance_id,
        "family": family,
        "split": split,
        "repeat_layers": 1,
        "query": "pre_measurement_statevector",
        "source_kind": "family_aligned_qasm_v1",
        "source_id": f"family_aligned_{family}_v1",
        "parameters": dict(parameters),
        "total_qubits": n_qubits,
        "qasm_sha256": hashlib.sha256(qasm).hexdigest(),
        "gate_counts": _gate_counts(operations),
        "operation_count": len(operations),
    }
    metadata.update(definition_metadata)
    return metadata


def _instance_spec(instance_id: str) -> tuple[str, str, str, dict[str, object]]:
    for spec in _INSTANCE_SPECS:
        if spec[0] == instance_id:
            return spec[0], spec[1], spec[2], dict(spec[3])
    raise ValueError(f"unknown family workload instance {instance_id!r}")


def build_instance(instance_id: str) -> tuple[bytes, dict[str, object]]:
    """Return QASM bytes and independent metadata for one declared instance."""

    instance_id, family, split, parameters = _instance_spec(instance_id)
    qasm = build_family_qasm(family, parameters)
    return qasm, _metadata(instance_id, family, split, parameters)


def all_instance_metadata() -> tuple[dict[str, object], ...]:
    """Return metadata for exactly the twelve declared train/test instances."""

    return tuple(
        _metadata(instance_id, family, split, parameters)
        for instance_id, family, split, parameters in _INSTANCE_SPECS
    )


__all__ = ["all_instance_metadata", "build_family_qasm", "build_instance"]
