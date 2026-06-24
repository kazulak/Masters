from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.bench.config import load_suite
from quantum_bench.circuits import builtin_circuit, manifest
from quantum_bench.tn import build_tensor_network, plan_task_graph
from quantum_bench.providers import route_registry
from quantum_bench.validation import compute_reference, validate


ROOT = Path(__file__).resolve().parents[1]


def test_builtin_manifest_gate_counts() -> None:
    circuit = builtin_circuit("ghz_chain", {"n_qubits": 4})
    record = manifest(circuit)
    assert record["gate_counts"] == {"1q": 1, "2q": 3, "total": 4}
    assert record["gate_set"] == ["cx", "h"]


def test_task_graph_serializes_and_validates_reference() -> None:
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    reference, _ = compute_reference(network)
    result = validate(reference, reference)
    json.dumps(graph, default=lambda obj: getattr(obj, "__dict__", str(obj)))
    assert graph.tasks
    assert result.passed
    assert result.l2_error == 0.0


def test_suite_config_loads() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "smoke.yml")
    assert suite["suite_id"] == "smoke"
    assert suite["_schema_version"] == 2
    assert suite["route_policy"]["routes"][0] == "cpu_tn_einsum_exact"
    assert suite["route_policy"]["routes"][1] == "upmem_dense_int8_placeholder"


def test_suite_v2_config_loads_with_separate_workloads_and_routes() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "smoke_v2.yml")
    assert suite["_schema_version"] == 2
    assert [case["case_id"] for case in suite["cases"]] == ["bell_2q", "ghz_4q"]
    assert suite["route_policy"]["routes"] == ["cpu_tn_einsum_exact", "upmem_dense_int8_placeholder"]
    assert suite["_route_configs"][0]["required"] is True


def test_route_probe_and_upmem_skip_reason() -> None:
    routes = route_registry(ROOT)
    assert routes["cpu_tn_einsum_exact"].probe().available
    assert routes["cpu_tn_einsum_exact"].identity.output_contract == "final_tensor"
    assert routes["cpu_tn_einsum_exact"].identity.validation_mode == "compare_output"
    assert routes["quest_cpu_full_state_benchmark"].identity.output_contract == "metrics_only"
    assert routes["quest_cpu_full_state_benchmark"].identity.validation_mode == "benchmark_only"
    assert routes["upmem_dense_int8_placeholder"].identity.hardware_target == "upmem_dpu"
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    suite = load_suite(ROOT / "configs" / "suites" / "smoke.yml")
    from quantum_bench.core.records import BenchmarkContext

    context = BenchmarkContext(ROOT, ROOT / "runs" / "test", suite, suite["cases"][0], suite["_route_configs"][1], 0, suite["tolerances"], 30, 2)
    can_execute, reason = routes["upmem_dense_int8_placeholder"].can_execute(graph, context)
    assert not can_execute
    assert reason
