from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from quantum_bench.circuits import parse_openqasm2
from quantum_bench.cpu import run_complex128_reference
from quantum_bench.lowering import build_contraction_dag, lower_tensor_network
from quantum_bench.model import make_simulation_job, validate_circuit_spec
from quantum_bench.planning import plan_opt_einsum


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upmem_family_workload.py"
SPEC = importlib.util.spec_from_file_location("upmem_family_workload", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
workload = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workload
SPEC.loader.exec_module(workload)


def _reference_state(qasm: bytes, tmp_path: Path, name: str) -> np.ndarray:
    path = tmp_path / f"{name}.qasm"
    path.write_bytes(qasm)
    circuit = parse_openqasm2(path)
    validate_circuit_spec(circuit)
    network, inputs = lower_tensor_network(make_simulation_job(circuit))
    contraction_path, _ = plan_opt_einsum(network, optimize="greedy")
    dag = build_contraction_dag(network, contraction_path)
    return np.asarray(run_complex128_reference(dag, inputs)).reshape(-1)


def test_declared_instances_cover_exact_family_split_and_size_matrix() -> None:
    expected = {
        ("qrng", "training"): (16, {"n_qubits": 16}),
        ("qrng", "test"): (17, {"n_qubits": 17}),
        ("bv", "training"): (16, {"data_qubits": 15}),
        ("bv", "test"): (17, {"data_qubits": 16}),
        ("bb84", "training"): (16, {"n_qubits": 16}),
        ("bb84", "test"): (17, {"n_qubits": 17}),
        ("edc", "training"): (15, {"data_qubits": 8}),
        ("edc", "test"): (17, {"data_qubits": 9}),
        ("hs", "training"): (16, {"allocated_qubits": 16}),
        ("hs", "test"): (18, {"allocated_qubits": 18}),
        ("xor", "training"): (16, {"data_qubits": 15}),
        ("xor", "test"): (17, {"data_qubits": 16}),
    }

    rows = workload.all_instance_metadata()
    assert len(rows) == 12
    assert len({row["instance_id"] for row in rows}) == 12
    assert {(row["family"], row["split"]) for row in rows} == set(expected)
    for row in rows:
        key = (row["family"], row["split"])
        total_qubits, parameters = expected[key]
        assert row["total_qubits"] == total_qubits
        assert row["parameters"] == parameters
        assert row["repeat_layers"] == 1
        assert row["query"] == "pre_measurement_statevector"
        assert row["source_kind"] == "family_aligned_qasm_v1"


def test_declared_qasm_is_parser_valid_and_hash_bound(tmp_path: Path) -> None:
    for row in workload.all_instance_metadata():
        qasm = workload.build_family_qasm(row["family"], row["parameters"])
        assert isinstance(qasm, bytes)
        assert hashlib.sha256(qasm).hexdigest() == row["qasm_sha256"]
        assert b"measure" not in qasm.lower()
        assert b"reset" not in qasm.lower()
        assert b"creg" not in qasm.lower()

        path = tmp_path / f"{row['instance_id']}.qasm"
        path.write_bytes(qasm)
        circuit = parse_openqasm2(path)
        validate_circuit_spec(circuit)
        assert circuit.n_qubits == row["total_qubits"]
        assert row["operation_count"] == len(circuit.operations)
        assert row["gate_counts"]["total"] == len(circuit.operations)
        assert row["gate_counts"]["1q"] == sum(
            len(operation.wires) == 1 for operation in circuit.operations
        )
        assert row["gate_counts"]["2q"] == sum(
            len(operation.wires) == 2 for operation in circuit.operations
        )


def test_bytes_and_metadata_are_deterministic_and_returned_metadata_is_isolated() -> None:
    first_qasm, first_metadata = workload.build_instance("xor_train_total16_r1")
    second_qasm, second_metadata = workload.build_instance("xor_train_total16_r1")
    assert first_qasm == second_qasm
    assert first_metadata == second_metadata
    assert first_metadata["qasm_sha256"] == hashlib.sha256(first_qasm).hexdigest()

    first_metadata["parameters"]["data_qubits"] = 2
    third_qasm, third_metadata = workload.build_instance("xor_train_total16_r1")
    assert third_qasm == first_qasm
    assert third_metadata["parameters"] == {"data_qubits": 15}


def test_qrng_bv_and_bb84_have_declared_small_state_semantics(tmp_path: Path) -> None:
    qrng = _reference_state(
        workload.build_family_qasm("qrng", {"n_qubits": 2}), tmp_path, "qrng"
    )
    np.testing.assert_allclose(qrng, np.full(4, 0.5), atol=1e-12)

    bv = _reference_state(
        workload.build_family_qasm("bv", {"data_qubits": 2}), tmp_path, "bv"
    )
    expected_bv = np.zeros(8, dtype=np.complex128)
    expected_bv[4] = 1 / np.sqrt(2)
    expected_bv[5] = -1 / np.sqrt(2)
    np.testing.assert_allclose(bv, expected_bv, atol=1e-12)

    bb84 = _reference_state(
        workload.build_family_qasm("bb84", {"n_qubits": 4}), tmp_path, "bb84"
    )
    expected_bb84 = np.zeros(16, dtype=np.complex128)
    expected_bb84[4:8] = [0.5, -0.5, 0.5, -0.5]
    np.testing.assert_allclose(bb84, expected_bb84, atol=1e-12)


def test_xor_prepares_uniform_data_before_computing_parity(tmp_path: Path) -> None:
    qasm = workload.build_family_qasm("xor", {"data_qubits": 2})
    path = tmp_path / "xor_small.qasm"
    path.write_bytes(qasm)
    circuit = parse_openqasm2(path)
    assert [(op.gate, op.wires) for op in circuit.operations] == [
        ("h", (0,)),
        ("h", (1,)),
        ("cx", (0, 2)),
        ("cx", (1, 2)),
    ]
    state = _reference_state(qasm, tmp_path, "xor_small_reference")
    expected = np.zeros(8, dtype=np.complex128)
    for index in (0, 3, 5, 6):
        expected[index] = 0.5
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_edc_has_encoder_error_and_coherent_adjacent_syndrome(tmp_path: Path) -> None:
    qasm = workload.build_family_qasm("edc", {"data_qubits": 3})
    path = tmp_path / "edc_small.qasm"
    path.write_bytes(qasm)
    circuit = parse_openqasm2(path)
    assert [(op.gate, op.wires) for op in circuit.operations] == [
        ("ry", (0,)),
        ("cx", (0, 1)),
        ("cx", (0, 2)),
        ("x", (1,)),
        ("cx", (0, 3)),
        ("cx", (1, 3)),
        ("cx", (1, 4)),
        ("cx", (2, 4)),
    ]
    assert circuit.operations[0].params == pytest.approx((np.pi / 3,))

    state = _reference_state(qasm, tmp_path, "edc_small_reference")
    expected = np.zeros(32, dtype=np.complex128)
    expected[0b01011] = np.sqrt(3) / 2
    expected[0b10111] = 0.5
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_hs_uses_pinned_pair_block_and_maps_zero_pair_to_ten(tmp_path: Path) -> None:
    qasm = workload.build_family_qasm("hs", {"allocated_qubits": 2})
    path = tmp_path / "hs_small.qasm"
    path.write_bytes(qasm)
    circuit = parse_openqasm2(path)
    assert [(op.gate, op.wires) for op in circuit.operations] == [
        ("h", (0,)),
        ("h", (1,)),
        ("x", (0,)),
        ("h", (1,)),
        ("cx", (0, 1)),
        ("h", (1,)),
        ("x", (0,)),
        ("h", (0,)),
        ("h", (1,)),
        ("h", (1,)),
        ("cx", (0, 1)),
        ("h", (1,)),
        ("h", (0,)),
        ("h", (1,)),
    ]
    state = _reference_state(qasm, tmp_path, "hs_small_reference")
    np.testing.assert_allclose(state, np.array([0, 0, 1, 0], dtype=np.complex128), atol=1e-12)


def test_builder_rejects_unknown_or_noncanonical_parameters() -> None:
    with pytest.raises(ValueError, match="unsupported family"):
        workload.build_family_qasm("not_a_family", {"n_qubits": 2})
    with pytest.raises(ValueError, match="exactly"):
        workload.build_family_qasm("qrng", {"n_qubits": 2, "repeat_layers": 1})
    with pytest.raises(ValueError, match="even"):
        workload.build_family_qasm("hs", {"allocated_qubits": 3})


def test_hs_disjoint_pairs_preserve_all_output_wires(tmp_path: Path) -> None:
    state = _reference_state(
        workload.build_family_qasm("hs", {"allocated_qubits": 4}), tmp_path, "hs_pairs"
    )
    expected = np.zeros(16, dtype=np.complex128)
    expected[0b1010] = 1
    np.testing.assert_allclose(state, expected, atol=1e-12)
