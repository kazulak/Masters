#!/usr/bin/env python3
"""Run deterministic software-only sequential statevector conformance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from quantum_bench.baselines import run_quest_cpu
from quantum_bench.circuits import builtin_circuit, quest_compatible_circuit
from quantum_bench.cpu import run_complex128_reference, run_cpu_once
from quantum_bench.lowering import (
    build_contraction_dag,
    choose_slice_labels,
    lower_tensor_network,
    slice_contraction,
)
from quantum_bench.model import CircuitOperation, CircuitSpec, make_simulation_job
from quantum_bench.planning import plan_opt_einsum


ROOT = Path(__file__).resolve().parents[1]
QUEST_RUNNER = ROOT / "native" / "quest_cpu" / "bin" / "quest_runner"
COMPLEX128_LIMIT = 1.0e-12
FLOAT32_ATOL = 1.0e-5
FLOAT32_RTOL = 1.0e-5
FLOAT32_RELATIVE_L2_LIMIT = 1.0e-5
FLOAT32_NORM_DRIFT_LIMIT = 2.0e-5
SLICED_NODE_ID = "contract_24"
SLICED_MINIMUM_COUNT = 4


def _complex_orientation_3q() -> CircuitSpec:
    return CircuitSpec(
        "complex_orientation_3q",
        3,
        (
            CircuitOperation("y", (0,)),
            CircuitOperation("s", (1,)),
            CircuitOperation("t", (2,)),
            CircuitOperation("ry", (1,), (-0.37,)),
            CircuitOperation("rz", (0,), (0.61,)),
            CircuitOperation("h", (2,)),
            CircuitOperation("cx", (2, 0)),
            CircuitOperation("cz", (1, 2)),
            CircuitOperation("swap", (0, 1)),
        ),
        {"kind": "sequential_conformance", "name": "complex_orientation_3q"},
    )


def _basis_order_2q() -> CircuitSpec:
    return CircuitSpec(
        "basis_order_2q",
        2,
        (CircuitOperation("x", (0,)),),
        {"kind": "sequential_conformance", "name": "basis_order_2q"},
    )


def _fixture_specs() -> tuple[dict[str, Any], ...]:
    return (
        {"fixture_id": "basis_order_2q", "circuit": _basis_order_2q()},
        {"fixture_id": "Bell2", "circuit": builtin_circuit("bell_2q")},
        {
            "fixture_id": "complex_orientation_3q",
            "circuit": _complex_orientation_3q(),
        },
        {
            "fixture_id": "GHZ5",
            "circuit": builtin_circuit("ghz_chain", {"n_qubits": 5}),
        },
        {
            "fixture_id": "QuEST-compatible QRNG3",
            "circuit": quest_compatible_circuit("qrng", {"n_qubits": 3}),
            "quest": True,
        },
        {
            "fixture_id": "QuEST-compatible BV5",
            "circuit": quest_compatible_circuit("bv", {"n_qubits": 5}),
            "quest": True,
        },
        {
            "fixture_id": "Stress18",
            "circuit": builtin_circuit(
                "quantization_stress", {"n_qubits": 18, "repeat_layers": 2}
            ),
            "float32": True,
        },
        {
            "fixture_id": "sliced Stress4",
            "circuit": builtin_circuit(
                "quantization_stress", {"n_qubits": 4, "repeat_layers": 2}
            ),
            "float32": True,
            "sliced": True,
        },
    )


def _analytic_state(fixture_id: str) -> np.ndarray | None:
    if fixture_id == "basis_order_2q":
        state = np.zeros(4, dtype=np.complex128)
        state[1] = 1.0
        return state
    if fixture_id == "Bell2":
        state = np.zeros(4, dtype=np.complex128)
        state[[0, 3]] = 1.0 / math.sqrt(2.0)
        return state
    if fixture_id == "GHZ5":
        state = np.zeros(32, dtype=np.complex128)
        state[[0, 31]] = 1.0 / math.sqrt(2.0)
        return state
    if fixture_id == "QuEST-compatible QRNG3":
        return np.full(8, 1.0 / math.sqrt(8.0), dtype=np.complex128)
    if fixture_id == "QuEST-compatible BV5":
        state = np.zeros(32, dtype=np.complex128)
        state[[0, 16]] = 1.0 / math.sqrt(2.0)
        return state
    return None


def _direct_quimb_state(circuit_spec: CircuitSpec) -> np.ndarray:
    import quimb.tensor as qtn

    circuit = qtn.Circuit(circuit_spec.n_qubits)
    for operation in circuit_spec.operations:
        gate = "CX" if operation.gate.lower() == "cnot" else operation.gate.upper()
        circuit.apply_gate(gate, *operation.params, *operation.wires)
    return np.asarray(
        circuit.to_dense(reverse=True, optimize="greedy"), dtype=np.complex128
    ).reshape(-1)


def _dag(circuit_spec: CircuitSpec, *, sliced: bool) -> tuple[Any, dict[str, np.ndarray]]:
    network, inputs = lower_tensor_network(make_simulation_job(circuit_spec))
    path, _ = plan_opt_einsum(network, optimize="greedy")
    dag = build_contraction_dag(network, path)
    if sliced:
        node = next(node for node in dag.nodes if node.node_id == SLICED_NODE_ID)
        labels = choose_slice_labels(node, minimum_slice_count=SLICED_MINIMUM_COUNT)
        dag = slice_contraction(dag, node_id=SLICED_NODE_ID, labels=labels)
    return dag, inputs


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.complex128).reshape(-1)
    expected = np.asarray(expected, dtype=np.complex128).reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError(f"statevector shape mismatch: {actual.shape} != {expected.shape}")
    difference = actual - expected
    expected_norm = float(np.linalg.norm(expected))
    overlap = np.vdot(expected, actual)
    phase = np.exp(-1j * np.angle(overlap)) if overlap != 0.0 else 1.0
    return {
        "max_abs_error": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2_error": (
            float(np.linalg.norm(difference) / expected_norm)
            if expected_norm
            else float(np.linalg.norm(difference))
        ),
        "norm_drift": abs(float(np.linalg.norm(actual)) - expected_norm),
        "phase_aligned_max_abs_error": float(
            np.max(np.abs(actual * phase - expected), initial=0.0)
        ),
    }


def _comparison(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    policy: str,
) -> dict[str, Any]:
    metrics = _metrics(actual, expected)
    if policy == "complex128_1e-12":
        raw_allclose = bool(
            np.allclose(actual, expected, atol=COMPLEX128_LIMIT, rtol=COMPLEX128_LIMIT)
        )
        passed = raw_allclose and all(
            metrics[field] <= COMPLEX128_LIMIT
            for field in ("max_abs_error", "relative_l2_error", "norm_drift")
        )
    elif policy == "float32_1e-5":
        raw_allclose = bool(
            np.allclose(actual, expected, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL)
        )
        passed = (
            raw_allclose
            and metrics["relative_l2_error"] <= FLOAT32_RELATIVE_L2_LIMIT
            and metrics["norm_drift"] <= FLOAT32_NORM_DRIFT_LIMIT
        )
    else:  # pragma: no cover - closed by this script.
        raise ValueError(f"unknown conformance policy: {policy}")
    return {
        "comparison": name,
        "policy": policy,
        "raw_phase_sensitive_allclose": raw_allclose,
        **metrics,
        "passed": passed,
    }


def _circuit_record(circuit: CircuitSpec) -> dict[str, Any]:
    operations = [
        {
            "gate": operation.gate,
            "wires": list(operation.wires),
            "params": list(operation.params),
        }
        for operation in circuit.operations
    ]
    encoded = json.dumps(operations, sort_keys=True, separators=(",", ":")).encode()
    return {
        "name": circuit.name,
        "n_qubits": circuit.n_qubits,
        "operation_count": len(operations),
        "operations_sha256": hashlib.sha256(encoded).hexdigest(),
        "source": dict(circuit.source),
    }


def run_conformance() -> dict[str, Any]:
    fixture_records: list[dict[str, Any]] = []
    for fixture in _fixture_specs():
        fixture_id = fixture["fixture_id"]
        circuit = fixture["circuit"]
        job = make_simulation_job(circuit)
        quimb_state = _direct_quimb_state(circuit)
        dag, inputs = _dag(circuit, sliced=bool(fixture.get("sliced")))
        dag_state = np.asarray(run_complex128_reference(dag, inputs)).reshape(-1, order="F")
        comparisons = [
            _comparison(
                "thesis_dag_complex128_vs_direct_quimb",
                dag_state,
                quimb_state,
                policy="complex128_1e-12",
            )
        ]
        analytic = _analytic_state(fixture_id)
        if analytic is not None:
            comparisons.extend(
                (
                    _comparison(
                        "direct_quimb_vs_analytic",
                        quimb_state,
                        analytic,
                        policy="complex128_1e-12",
                    ),
                    _comparison(
                        "thesis_dag_complex128_vs_analytic",
                        dag_state,
                        analytic,
                        policy="complex128_1e-12",
                    ),
                )
            )
        if fixture.get("quest"):
            quest_state = run_quest_cpu(job, runner=QUEST_RUNNER).output
            comparisons.extend(
                (
                    _comparison(
                        "quest_cpu_vs_direct_quimb",
                        quest_state,
                        quimb_state,
                        policy="complex128_1e-12",
                    ),
                    _comparison(
                        "quest_cpu_vs_thesis_dag_complex128",
                        quest_state,
                        dag_state,
                        policy="complex128_1e-12",
                    ),
                )
            )
        if fixture.get("float32"):
            float32_state = np.asarray(
                run_cpu_once(dag, inputs, "split_complex_float32_v1").output
            ).reshape(-1, order="F")
            comparisons.append(
                _comparison(
                    "thesis_dag_float32_vs_direct_quimb",
                    float32_state,
                    quimb_state,
                    policy="float32_1e-5",
                )
            )
        record = {
            "fixture_id": fixture_id,
            "circuit": _circuit_record(circuit),
            "oracles": ["direct_quimb_circuit", "thesis_dag_complex128"],
            "comparisons": comparisons,
            "passed": all(comparison["passed"] for comparison in comparisons),
        }
        if analytic is not None:
            record["oracles"].append("analytic")
        if fixture.get("quest"):
            record["oracles"].append("quest_cpu")
        if fixture.get("float32"):
            record["oracles"].append("thesis_dag_float32")
        if fixture.get("sliced"):
            record["slicing"] = {
                "node_id": SLICED_NODE_ID,
                "minimum_slice_count": SLICED_MINIMUM_COUNT,
                "sdk_simulator_coverage": (
                    "tests/test_cli_report.py::"
                    "test_sliced_conformance_retains_strict_sdk_simulator_coverage"
                ),
            }
        fixture_records.append(record)
    return {
        "schema_version": "sequential_statevector_conformance_v1",
        "execution_class": "software_only",
        "phase_aligned_metric_is_diagnostic_only": True,
        "policies": {
            "complex128_1e-12": {
                "raw_allclose_atol": COMPLEX128_LIMIT,
                "raw_allclose_rtol": COMPLEX128_LIMIT,
                "max_abs_error_max": COMPLEX128_LIMIT,
                "relative_l2_error_max": COMPLEX128_LIMIT,
                "norm_drift_max": COMPLEX128_LIMIT,
            },
            "float32_1e-5": {
                "raw_allclose_atol": FLOAT32_ATOL,
                "raw_allclose_rtol": FLOAT32_RTOL,
                "relative_l2_error_max": FLOAT32_RELATIVE_L2_LIMIT,
                "norm_drift_max": FLOAT32_NORM_DRIFT_LIMIT,
            },
        },
        "fixtures": fixture_records,
        "passed": all(fixture["passed"] for fixture in fixture_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact = run_conformance()
    payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="ascii")
    print(payload, end="")
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
