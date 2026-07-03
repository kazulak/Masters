from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantum_bench.bench import __main__ as bench_main
from quantum_bench.bench.config import suite_path
from quantum_bench.bench.shadow_routed_runtime import (
    run_shadow_routed_runtime,
    validate_cli_options,
)
import quantum_bench.bench.shadow_routed_runtime as shadow_module
import quantum_bench.targets.upmem.dense_bridge as dense_bridge_module
from quantum_bench.circuits import builtin_circuit
from quantum_bench.providers.exact_tn.cpu_einsum import _execute_task_sequence
from quantum_bench.targets.upmem import annotate_task_graph_with_upmem_estimates
from quantum_bench.tn import build_tensor_network, plan_task_graph, with_path_cost_summary


ROOT = Path(__file__).resolve().parents[1]


def _stub_path() -> Path:
    return ROOT / "native" / "upmem" / "simplepim" / "simplepim_dense_stub.py"


def _load_payload(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "shadow_routed_runtime.json").read_text(encoding="utf-8"))


def test_shadow_routed_runtime_bell_uses_cpu_fallback_authoritatively(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(tmp_path, case="bell_2q", env={})
    payload = _load_payload(run_dir)
    rows = payload["rows"]
    case_summary = payload["case_summaries"][0]
    encoded = json.dumps(payload)

    assert payload["schema_version"] == "shadow_routed_runtime_v1"
    assert payload["status"] == "completed"
    assert case_summary["status"] == "passed"
    assert case_summary["final_validation"]["passed"] is True
    assert rows
    assert all(row["authoritative_route"] == "cpu_fallback" for row in rows)
    assert all(row["selected_authoritative_route"] == "cpu_fallback" for row in rows)
    assert all(row["final_route_used_for_tensor"] == "cpu_fallback" for row in rows)
    assert all(row["shadow_policy_id"] == "cpu-only" for row in rows)
    assert all(row["shadow_policy_status"] == "selected_cpu" for row in rows)
    assert all(row["shadow_policy_reason"] == "policy_cpu_only" for row in rows)
    assert all(row["router_selected_route"] == "cpu_fallback" for row in rows)
    assert all(row["cpu_execution_status"] == "passed" for row in rows)
    assert all(any(route["route_id"] == "dense_gemm" for route in row["candidate_routes"]) for row in rows)
    assert all(any(route["route_id"] == "cpu_fallback" and route["is_selected"] for route in row["candidate_routes"]) for row in rows)
    assert all(row["dense_prepare_status"] in {"prepared", "simplepim_unavailable"} for row in rows)
    assert all(row["external_command_executed"] is False for row in rows)
    assert all(row["execution_implemented"] is False for row in rows)
    assert all(row["native_kernel_executed"] is False for row in rows)
    assert "prepared_operands" not in encoded
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    assert str(tmp_path) not in encoded
    assert (run_dir / "cases" / "bell_2q" / "shadow_routed_runtime.jsonl").exists()
    policy_summary = payload["summary"]["shadow_policy_summary"]
    assert policy_summary["shadow_policy_id"] == "cpu-only"
    assert policy_summary["cpu_fallback_count"] == len(rows)
    assert policy_summary["dense_gemm_candidate_count"] == len(rows)


def test_shadow_cpu_sequence_matches_cpu_exact_route_for_bell() -> None:
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    graph = with_path_cost_summary(graph)
    expected, expected_metadata = _execute_task_sequence(graph, network)

    tensors = {tensor.spec.id: np.asarray(tensor.array, dtype=np.complex128) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    live_ids = set(tensors)
    remaining = shadow_module._remaining_input_uses(graph)
    final_id = graph.tasks[-1].output_tensor_id
    for task in graph.tasks:
        result = shadow_module._execute_cpu_fallback_task(task, tensors)
        assert result["status"] == "passed"
        output = np.asarray(result["output"], dtype=np.complex128)
        tensors[task.output_tensor_id] = output
        labels[task.output_tensor_id] = task.output_labels
        live_ids.add(task.output_tensor_id)
        shadow_module._release_dead_inputs(task, tensors, labels, live_ids, remaining)

    actual, transposed = shadow_module._order_final_tensor(tensors[final_id], labels[final_id], graph.network.output_labels)

    np.testing.assert_allclose(actual, expected)
    assert transposed == expected_metadata["final_transpose_applied"]
    assert labels[final_id] == expected_metadata["final_tensor_labels"]


def test_shadow_routed_runtime_default_does_not_call_subprocess(tmp_path: Path, monkeypatch) -> None:
    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("default shadow-routed-runtime must not call subprocess")

    monkeypatch.setattr(shadow_module, "capture_environment", lambda root_dir: {})
    monkeypatch.setattr(dense_bridge_module.subprocess, "run", forbidden_subprocess)
    run_dir = run_shadow_routed_runtime(tmp_path, case="bell_2q", env={})
    payload = _load_payload(run_dir)

    assert payload["summary"]["external_command_executed_count"] == 0
    assert all(row["external_command_executed"] is False for row in payload["rows"])


def test_shadow_bridge_none_with_zero_cap_is_eligibility_only(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        case="bell_2q",
        dense_shadow="bridge",
        bridge_backend="none",
        max_bridge_artifacts=0,
        env={},
    )
    payload = _load_payload(run_dir)

    assert payload["status"] == "completed"
    assert payload["summary"]["bridge_manifest_eligible_task_count"] > 0
    assert payload["summary"]["bridge_artifact_written_count"] == 0
    assert not list((run_dir / "cases" / "bell_2q").glob("dense_bridge/**/input_manifest.json"))


def test_shadow_bridge_none_writes_capped_input_manifests_only(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        case="bell_2q",
        dense_shadow="bridge",
        bridge_backend="none",
        max_bridge_artifacts=1,
        env={},
    )
    payload = _load_payload(run_dir)
    written = [row for row in payload["rows"] if row["bridge_artifact_written"]]

    assert len(written) == 1
    assert written[0]["bridge_status"] is None
    assert (run_dir / written[0]["bridge_artifact_path"]).exists()
    assert not (run_dir / Path(written[0]["bridge_artifact_path"]).parent / "output_manifest.json").exists()


def test_shadow_bridge_mock_is_capped_and_not_authoritative(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        case="bell_2q",
        dense_shadow="bridge",
        bridge_backend="mock_numpy_dequantized",
        max_bridge_artifacts=1,
        env={},
    )
    payload = _load_payload(run_dir)
    written = [row for row in payload["rows"] if row["bridge_artifact_written"]]

    assert payload["summary"]["bridge_artifact_written_count"] == 1
    assert payload["case_summaries"][0]["final_validation"]["passed"] is True
    assert len(written) == 1
    assert written[0]["bridge_status"] == "mock_executed"
    assert written[0]["bridge_validation_metrics"]["max_abs_error"] < 1.0e-12
    assert written[0]["final_route_used_for_tensor"] == "cpu_fallback"
    assert all(row["selected_authoritative_route"] == "cpu_fallback" for row in payload["rows"])


def test_shadow_bridge_artifact_cap_is_per_run_across_suite_cases(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        suite_path=ROOT / "configs" / "suites" / "diagnostics" / "planner_compare.yml",
        dense_shadow="bridge",
        bridge_backend="mock_numpy_dequantized",
        max_bridge_artifacts=1,
        env={},
    )
    payload = _load_payload(run_dir)

    assert {row["case_id"] for row in payload["rows"]} == {"bell_2q", "ghz_3q"}
    assert payload["summary"]["case_count"] == 2
    assert payload["summary"]["bridge_artifact_written_count"] == 1
    assert sum(1 for row in payload["rows"] if row["bridge_artifact_written"]) == 1


def test_shadow_policy_dense_if_estimate_supported_selects_dense_without_changing_authority(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        case="bell_2q",
        shadow_route_policy="dense-if-estimate-supported",
        env={},
    )
    payload = _load_payload(run_dir)
    rows = payload["rows"]
    policy_summary = payload["summary"]["shadow_policy_summary"]

    assert payload["status"] == "completed"
    assert payload["case_summaries"][0]["final_validation"]["passed"] is True
    assert rows
    assert all(row["shadow_policy_status"] == "selected_dense" for row in rows)
    assert all(row["shadow_policy_selected_route"] == "dense_gemm" for row in rows)
    assert all(row["authoritative_route"] == "cpu_fallback" for row in rows)
    assert all(row["selected_authoritative_route"] == "cpu_fallback" for row in rows)
    assert all(row["final_route_used_for_tensor"] == "cpu_fallback" for row in rows)
    assert policy_summary["selected_route_counts"]["dense_gemm"] == len(rows)
    assert policy_summary["cpu_fallback_count"] == 0
    assert policy_summary["total_host_to_dpu_bytes_for_policy_dense"] > 0
    assert policy_summary["total_dpu_to_host_bytes_for_policy_dense"] > 0
    assert policy_summary["total_mram_to_wram_bytes_for_policy_dense"] > 0


def test_shadow_policy_bridge_ready_works_with_prepare_without_artifacts(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        case="bell_2q",
        dense_shadow="prepare",
        shadow_route_policy="dense-if-bridge-ready",
        env={},
    )
    payload = _load_payload(run_dir)
    rows = payload["rows"]

    assert payload["summary"]["bridge_artifact_written_count"] == 0
    assert all(row["bridge_manifest_eligible"] is True for row in rows)
    assert all(row["shadow_policy_selected_route"] == "dense_gemm" for row in rows)
    assert all(row["final_route_used_for_tensor"] == "cpu_fallback" for row in rows)


def test_shadow_policy_summary_counts_match_rows(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        case="bell_2q",
        shadow_route_policy="dense-if-no-tiling",
        env={},
    )
    payload = _load_payload(run_dir)
    rows = payload["rows"]
    policy_summary = payload["summary"]["shadow_policy_summary"]

    assert policy_summary["dense_gemm_candidate_count"] == sum(
        1
        for row in rows
        for route in row["candidate_routes"]
        if route["route_id"] == "dense_gemm" and route["estimate"]["supported"]
    )
    assert policy_summary["cpu_fallback_count"] == sum(
        1 for row in rows if row["shadow_policy_selected_route"] == "cpu_fallback"
    )
    assert policy_summary["selected_route_counts"]["dense_gemm"] == sum(
        1 for row in rows if row["shadow_policy_selected_route"] == "dense_gemm"
    )


def test_shadow_stub_explicit_check_is_capped_and_non_numeric(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        case="bell_2q",
        dense_shadow="stub",
        bridge_backend="simplepim_external_stub",
        execute_external=True,
        max_bridge_artifacts=1,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )
    payload = _load_payload(run_dir)
    checked = [row for row in payload["rows"] if row["external_command_executed"]]

    assert payload["status"] == "completed"
    assert payload["summary"]["external_command_executed_count"] == 1
    assert payload["summary"]["native_kernel_executed_count"] == 0
    assert len(checked) == 1
    row = checked[0]
    assert row["bridge_status"] == "stub_executed"
    assert row["bridge_validation_metrics"] == {"status": "not_applicable", "reason": "external_stub_no_output_blob"}
    assert row["final_route_used_for_tensor"] == "cpu_fallback"
    output_manifest = run_dir / Path(row["bridge_artifact_path"]).parent / "output_manifest.json"
    manifest_payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest_payload["status"] == "stub_executed"
    assert manifest_payload["output_blob"] is None
    assert manifest_payload["execution_implemented"] is False


def test_shadow_stub_failure_is_warning_when_cpu_fallback_validates(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=7, stdout="", stderr="stub failed")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", fake_run)
    run_dir = run_shadow_routed_runtime(
        tmp_path,
        case="bell_2q",
        dense_shadow="stub",
        bridge_backend="simplepim_external_stub",
        execute_external=True,
        max_bridge_artifacts=1,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )
    payload = _load_payload(run_dir)
    failed_bridge_rows = [row for row in payload["rows"] if row["bridge_status"] == "failed"]

    assert payload["status"] == "completed"
    assert payload["case_summaries"][0]["final_validation"]["passed"] is True
    assert payload["summary"]["warning_count"] == 1
    assert len(failed_bridge_rows) == 1
    assert failed_bridge_rows[0]["final_route_used_for_tensor"] == "cpu_fallback"


def test_shadow_csv_uses_json_strings_for_nested_fields(tmp_path: Path) -> None:
    run_dir = run_shadow_routed_runtime(tmp_path, case="bell_2q", env={})
    with (run_dir / "shadow_routed_runtime.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    candidate_routes = json.loads(rows[0]["candidate_routes"])
    output_shape = json.loads(rows[0]["output_shape"])
    policy_blockers = json.loads(rows[0]["shadow_policy_blockers"])
    assert isinstance(candidate_routes, list)
    assert isinstance(output_shape, list)
    assert isinstance(policy_blockers, list)


def test_shadow_cli_rejects_invalid_external_combinations() -> None:
    invalid_cases = [
        {
            "suite_path": None,
            "case": "bell_2q",
            "dense_shadow": "prepare",
            "bridge_backend": "mock_numpy_dequantized",
            "execute_external": False,
            "max_bridge_artifacts": 0,
            "shadow_route_policy": "cpu-only",
        },
        {
            "suite_path": None,
            "case": "bell_2q",
            "dense_shadow": "bridge",
            "bridge_backend": "mock_numpy_dequantized",
            "execute_external": False,
            "max_bridge_artifacts": 0,
            "shadow_route_policy": "cpu-only",
        },
        {
            "suite_path": None,
            "case": "bell_2q",
            "dense_shadow": "bridge",
            "bridge_backend": "simplepim_external_stub",
            "execute_external": False,
            "max_bridge_artifacts": 1,
            "shadow_route_policy": "cpu-only",
        },
        {
            "suite_path": None,
            "case": "bell_2q",
            "dense_shadow": "stub",
            "bridge_backend": "simplepim_external_stub",
            "execute_external": True,
            "max_bridge_artifacts": 0,
            "shadow_route_policy": "cpu-only",
        },
        {
            "suite_path": None,
            "case": "bell_2q",
            "dense_shadow": "none",
            "bridge_backend": "none",
            "execute_external": False,
            "max_bridge_artifacts": 0,
            "shadow_route_policy": "dense-if-bridge-ready",
        },
    ]
    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            validate_cli_options(**kwargs, env={})


def test_shadow_cli_dispatch(monkeypatch, capsys, tmp_path: Path) -> None:
    called: dict[str, object] = {}

    def fake_run_shadow_routed_runtime(root_dir: Path, **kwargs: object) -> Path:
        called["root_dir"] = root_dir
        called.update(kwargs)
        run_dir = tmp_path / "runs" / "fake_shadow_runtime"
        run_dir.mkdir(parents=True)
        return run_dir

    monkeypatch.setattr(shadow_module, "run_shadow_routed_runtime", fake_run_shadow_routed_runtime)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m quantum_bench.bench",
            "shadow-routed-runtime",
            "--case",
            "bell_2q",
            "--dense-shadow",
            "bridge",
            "--bridge-backend",
            "none",
            "--max-bridge-artifacts",
            "1",
            "--shadow-route-policy",
            "dense-if-estimate-supported",
        ],
    )

    assert bench_main.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert called["case"] == "bell_2q"
    assert called["dense_shadow"] == "bridge"
    assert called["bridge_backend"] == "none"
    assert called["max_bridge_artifacts"] == 1
    assert called["shadow_route_policy"] == "dense-if-estimate-supported"
    assert payload["status"] == "completed"
    assert payload["run_dir"].endswith("fake_shadow_runtime")


def test_shadow_cli_rejects_execute_external_with_bad_combination(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m quantum_bench.bench",
            "shadow-routed-runtime",
            "--case",
            "bell_2q",
            "--dense-shadow",
            "bridge",
            "--execute-external",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        bench_main.main()
    assert exc_info.value.code == 2
