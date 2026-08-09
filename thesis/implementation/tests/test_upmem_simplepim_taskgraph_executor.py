from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from quantum_bench.bench import upmem_simplepim_taskgraph as route
from quantum_bench.circuits import load_circuit
from quantum_bench.targets.upmem import simplepim_taskgraph_executor as executor
from quantum_bench.targets.upmem.execution_plan_v1 import (
    PLACEMENT_FRONTIER,
    PLACEMENT_SINGLE,
    compile_plan,
    serialize_schedule,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    build_resident_graph_package,
)
from quantum_bench.tn import (
    build_tensor_network,
    plan_task_graph_with_config,
    with_execution_identity,
)


ROOT = Path(__file__).resolve().parents[1]
QASM = "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm"


def _native_sdk_available() -> bool:
    return all(
        shutil.which(name) is not None
        for name in ("make", "dpu-pkg-config", "dpu-upmem-dpurte-clang")
    )


def _build_fixture(tmp_path: Path) -> dict[str, object]:
    if not _native_sdk_available():
        pytest.skip("UPMEM SDK compiler tools are unavailable")
    result = executor.build(tmp_path / "native_build", prepare_only=True)
    assert result["allocation_attempted"] is False
    assert result["launch_attempted"] is False
    return result


def _native_validation_inputs(
    tmp_path: Path,
    build: dict[str, object],
    placement: str,
    *,
    allow_slot_reuse: bool = True,
) -> tuple[Path, Path, Path]:
    circuit = load_circuit({"circuit": {"kind": "qasm_file", "path": QASM}}, ROOT)
    network = build_tensor_network(circuit)
    source_graph = with_execution_identity(
        plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})
    )
    package_graph, package_network = route._lower_real_float32(source_graph, network)
    package = build_resident_graph_package(
        package_graph,
        package_network,
        case_id="adapter-parser",
        suite_id="adapter-parser",
        quantization_mode="none",
        allow_slot_reuse=allow_slot_reuse,
    ).write(
        tmp_path,
        dpu_binary=Path(str(build["dpu_binary"])),
        request_id=f"adapter-parser-{placement}",
    )
    assert package.manifest_path is not None
    plan = compile_plan(source_graph, package, placement_policy=placement)
    (tmp_path / "execution_plan.json").write_text(
        json.dumps(plan.to_json()), encoding="utf-8"
    )
    schedule_path = tmp_path / f"{placement}.bin"
    schedule_path.write_bytes(serialize_schedule(plan))
    return Path(str(build["host_binary"])), package.manifest_path, schedule_path


def _write_request(
    tmp_path: Path,
    build: dict[str, object],
    manifest_path: Path,
    schedule_path: Path,
    placement: str,
) -> Path:
    plan = json.loads((tmp_path / "execution_plan.json").read_text(encoding="utf-8"))
    request = {
        "schema_version": "upmem_execution_plan_request_v1",
        "manifest_kind": "upmem_execution_plan_request",
        "requested_dpu_count": plan["requested_dpu_count"],
        "tasklets_per_dpu": plan["tasklets_per_dpu"],
        "package_path": str(manifest_path),
        "schedule_path": str(schedule_path),
        "dpu_binary": str(build["dpu_binary"]),
        "package_file_sha256": plan["package_file_sha256"],
        "schedule_sidecar_sha256": plan["schedule_sidecar_sha256"],
        "execution_plan_hash": plan["execution_plan_hash"],
        "source_identity": plan["source_identity"],
        "package_identity": plan["package_identity"],
        "final_outputs": plan["final_outputs"],
        "request_id": f"adapter-parser-{placement}",
        "requested_warmups": 1,
        "requested_repetitions": 3,
    }
    for prefix in ("source", "package"):
        request.update(
            {
                f"{prefix}_{key}": value
                for key, value in request[f"{prefix}_identity"].items()
            }
        )
    request_path = tmp_path / "block2_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path


def test_execute_requires_explicit_hardware_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", raising=False)
    with pytest.raises(executor.NativeAdapterError, match="hardware_opt_in_missing"):
        executor.execute(request_path, timeout_s=1.0)


def test_native_parser_accepts_python_generated_schedule(
    tmp_path: Path,
) -> None:
    build = _build_fixture(tmp_path)
    host_binary, manifest_path, schedule_path = _native_validation_inputs(
        tmp_path, build, PLACEMENT_SINGLE
    )
    response_path = tmp_path / "single.response.json"
    completed = subprocess.run(
        [
            str(host_binary),
            "--validate-plan",
            "--resident-package",
            str(manifest_path),
            "--schedule",
            str(schedule_path),
            "--response",
            str(response_path),
        ],
        cwd=host_binary.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "validated"
    assert response["target_observed"] == "not_allocated"
    assert response["allocated_dpu_count"] == 0
    assert response["hardware_allocation_verified"] is False
    assert response["simulator_kernel_executed"] is False
    assert response["cpu_fallback_used"] is False


def test_native_parser_rejects_frontier_schedule_with_live_slot_alias(tmp_path: Path) -> None:
    build = _build_fixture(tmp_path)
    host_binary, manifest_path, schedule_path = _native_validation_inputs(
        tmp_path, build, PLACEMENT_FRONTIER
    )
    response_path = tmp_path / "frontier.response.json"
    completed = subprocess.run(
        [
            str(host_binary),
            "--validate-plan",
            "--resident-package",
            str(manifest_path),
            "--schedule",
            str(schedule_path),
            "--response",
            str(response_path),
        ],
        cwd=host_binary.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "failed"
    assert "same-wave output slot aliases" in response["error"]
    assert response["allocated_dpu_count"] == 0


def test_native_parser_rejects_malformed_python_schedule(tmp_path: Path) -> None:
    build = _build_fixture(tmp_path)
    host_binary, manifest_path, schedule_path = _native_validation_inputs(
        tmp_path, build, PLACEMENT_FRONTIER
    )
    malformed_path = tmp_path / "malformed.bin"
    malformed = bytearray(schedule_path.read_bytes())
    malformed[0] ^= 0x01
    malformed_path.write_bytes(malformed)
    response_path = tmp_path / "malformed.response.json"
    completed = subprocess.run(
        [
            str(host_binary),
            "--validate-plan",
            "--resident-package",
            str(manifest_path),
            "--schedule",
            str(malformed_path),
            "--response",
            str(response_path),
        ],
        cwd=host_binary.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "failed"
    assert response["allocated_dpu_count"] == 0
    assert response["simulator_kernel_executed"] is False


@pytest.mark.parametrize("placement", [PLACEMENT_SINGLE, PLACEMENT_FRONTIER])
def test_validate_generated_request_is_parser_only(
    tmp_path: Path, placement: str
) -> None:
    build = _build_fixture(tmp_path)
    host_binary, manifest_path, schedule_path = _native_validation_inputs(
        tmp_path, build, placement, allow_slot_reuse=False
    )
    request_path = _write_request(
        tmp_path, build, manifest_path, schedule_path, placement
    )

    result = executor.validate(request_path, timeout_s=10.0)

    assert result["status"] == "passed"
    assert result["native_validation_status"] == "validated"
    assert result["validation_status"] == "plan_valid"
    assert result["target_observed"] == "not_allocated"
    assert result["allocated_dpu_count"] == 0
    assert result["allocation_attempted"] is False
    assert result["launch_attempted"] is False
    assert result["dpu_allocation_attempted"] is False
    assert result["dpu_launch_attempted"] is False
    assert "--validate-plan" in result["native_command"]
    assert result["native_command"][0] == str(host_binary)
    route._validate_prepared_request(executor, request_path, 10.0)


def test_validate_rejects_malformed_generated_request(tmp_path: Path) -> None:
    build = _build_fixture(tmp_path)
    host_binary, manifest_path, schedule_path = _native_validation_inputs(
        tmp_path, build, PLACEMENT_SINGLE
    )
    request_path = _write_request(
        tmp_path, build, manifest_path, schedule_path, PLACEMENT_SINGLE
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["requested_dpu_count"] = 2
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(executor.NativeAdapterError, match="execution_plan_compile_failed"):
        executor.validate(request_path, timeout_s=10.0)


def test_real_normalized_session_passes_route_validator(tmp_path: Path) -> None:
    validator = getattr(route, "_validate_adapter_session", None)
    if not callable(validator):
        pytest.skip("route adapter-session validator is unavailable")
    build = _build_fixture(tmp_path)
    host_binary, manifest_path, schedule_path = _native_validation_inputs(
        tmp_path, build, PLACEMENT_SINGLE
    )
    request_path = _write_request(
        tmp_path, build, manifest_path, schedule_path, PLACEMENT_SINGLE
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    plan = executor.parse_plan_json(
        (tmp_path / "execution_plan.json").read_bytes()
    )
    output_path = tmp_path / "native_final_output.bin"
    output_values = [0.25] * plan.final_outputs[0].element_count
    output_path.write_bytes(
        struct.pack("<" + "f" * len(output_values), *output_values)
    )
    repetitions = 4
    metrics = {
        "descriptor_h2d_bytes": 64,
        "operand_h2d_bytes": 0,
        "reset_h2d_bytes": repetitions * 64,
        "cross_d2h_bytes": 0,
        "cross_h2d_bytes": 0,
        "final_d2h_bytes": repetitions * 8,
        "actual_h2d_bytes": 64 + repetitions * 64,
        "actual_d2h_bytes": repetitions * 8,
        "actual_transfer_bytes": 64 + repetitions * (64 + 8),
        "launch_count": plan.logical_task_count * repetitions,
        "synchronize_count": plan.logical_task_count * repetitions,
        "completion_reads": plan.logical_task_count * repetitions,
        "cross_dpu_edge_count": 0,
        "completed_per_dpu": [plan.logical_task_count * repetitions],
    }
    native_response = {
        "allocated_dpu_count": plan.requested_dpu_count,
        "allocation": {
            "attempted": True,
            "confirmed": False,
            "succeeded": True,
            "release_confirmed": True,
        },
        "hardware_allocation_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_speedup_applicable": False,
        "metrics": metrics,
        "timing": {
            "allocation_time_s": 0.1,
            "binary_load_time_s": 0.1,
            "descriptor_h2d_time_s": 0.1,
            "release_time_s": 0.1,
        },
    }
    session = executor._normalize_session(
        native_response,
        request=request,
        plan=plan,
        elapsed_s=0.5,
        response_path=tmp_path / "native.response.json",
        final_output_path=output_path,
        command=(str(host_binary), "--execute-plan"),
    )
    prepared = SimpleNamespace(
        request_id=request["request_id"],
        plan=plan,
        package_file_sha256=plan.package_file_sha256,
        schedule_sidecar_sha256=plan.schedule_sidecar_sha256,
        request_path=request_path,
        source_output=np.asarray(output_values, dtype=np.float32),
    )

    route._complete_session_validation(session, prepared)
    validator(session, prepared)
    assert session["allocation_succeeded"] is True
    assert session["persistent_allocation_observed"] is True
    assert session["session_validation"]["status"] == "passed"
    assert session["session_validation"]["scope"] == "final_session_output_only"
    assert session["session_validation"]["output"] == output_values
    assert session["repetitions"][0]["validation_id"] == session["session_validation"]["validation_id"]
    assert session["repetitions"][0]["repeat_output_validation_status"] == "not_individually_collected"
    assert "output" not in session["repetitions"][0]
    assert session["session_completion_scope"] == "aggregate_across_warmups_and_repetitions"
    assert session["aggregate_completed_per_dpu"] == [12]
    assert session["aggregate_total_task_completion_count"] == 12
    assert session["aggregate_session_completion_id"].startswith("aggregate_session_completion:")
    forbidden = {
        "completed_task_count", "completed_task_ids", "task_completion_counts",
        "exactly_once_execution_verified",
    }
    assert all(forbidden.isdisjoint(repetition) for repetition in session["repetitions"])
    assert all(repetition["scheduled_task_count"] == 3 for repetition in session["repetitions"])
    assert all(
        repetition["aggregate_session_completion_id"] == session["aggregate_session_completion_id"]
        for repetition in session["repetitions"]
    )


@pytest.mark.parametrize(
    ("dpu_ids", "expected_completed_per_dpu"),
    [([0, 0, 0], [12]), ([0, 1, 0], [8, 4])],
)
def test_native_completion_counts_are_aggregate_over_the_session(
    dpu_ids: list[int], expected_completed_per_dpu: list[int]
) -> None:
    plan = SimpleNamespace(
        assignments=[SimpleNamespace(dpu_id=dpu_id) for dpu_id in dpu_ids],
        requested_dpu_count=len(expected_completed_per_dpu),
        logical_task_count=3,
        transfer_edges=(),
    )
    request = {"requested_warmups": 1, "requested_repetitions": 3}
    metrics = {
        "completed_per_dpu": expected_completed_per_dpu,
        "launch_count": 12,
        "synchronize_count": 12,
        "completion_reads": 12,
        "cross_dpu_edge_count": 0,
        "descriptor_h2d_bytes": 0,
        "operand_h2d_bytes": 0,
        "reset_h2d_bytes": 0,
        "cross_d2h_bytes": 0,
        "cross_h2d_bytes": 0,
        "final_d2h_bytes": 0,
        "actual_h2d_bytes": 0,
        "actual_d2h_bytes": 0,
        "actual_transfer_bytes": 0,
    }

    executor._validate_native_metrics(metrics, plan, request)


def test_native_completion_count_mismatch_is_rejected() -> None:
    plan = SimpleNamespace(
        assignments=[SimpleNamespace(dpu_id=dpu_id) for dpu_id in (0, 1, 0)],
        requested_dpu_count=2,
        logical_task_count=3,
        transfer_edges=(),
    )
    request = {"requested_warmups": 1, "requested_repetitions": 3}
    metrics = {
        "completed_per_dpu": [2, 1],
        "launch_count": 12,
        "synchronize_count": 12,
        "completion_reads": 12,
        "cross_dpu_edge_count": 0,
        "descriptor_h2d_bytes": 0,
        "operand_h2d_bytes": 0,
        "reset_h2d_bytes": 0,
        "cross_d2h_bytes": 0,
        "cross_h2d_bytes": 0,
        "final_d2h_bytes": 0,
        "actual_h2d_bytes": 0,
        "actual_d2h_bytes": 0,
        "actual_transfer_bytes": 0,
    }

    with pytest.raises(executor.NativeAdapterError, match="completion counts differ"):
        executor._validate_native_metrics(metrics, plan, request)
