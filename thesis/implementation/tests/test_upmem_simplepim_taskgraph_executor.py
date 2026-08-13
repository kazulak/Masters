from __future__ import annotations

import json
import os
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
    CrossDpuTransfer,
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


def test_physical_environment_requires_a_valid_rank_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    monkeypatch.delenv("DPU_BACKEND", raising=False)
    monkeypatch.delenv("UPMEM_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("UPMEM_HW_RANK_PATH", raising=False)
    with pytest.raises(executor.NativeAdapterError, match="UPMEM_HW_RANK_PATH is required"):
        executor._require_physical_environment()

    monkeypatch.setenv("UPMEM_HW_RANK_PATH", "not-a-rank")
    with pytest.raises(executor.NativeAdapterError, match="UPMEM_HW_RANK_PATH"):
        executor._require_physical_environment()

    monkeypatch.setenv("UPMEM_HW_RANK_PATH", "/dev/dpu_rank1")
    assert executor._require_physical_environment() == "/dev/dpu_rank1"


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
    assert response["requested_rank_path"] is None
    assert response["observed_rank_count"] == 0
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


def test_native_execute_rejects_missing_rank_before_allocation(tmp_path: Path) -> None:
    build = _build_fixture(tmp_path)
    host_binary, manifest_path, schedule_path = _native_validation_inputs(
        tmp_path, build, PLACEMENT_SINGLE
    )
    response_path = tmp_path / "missing-rank.response.json"
    environment = dict(os.environ)
    environment["UPMEM_ALLOW_PHYSICAL_HARDWARE"] = "1"
    environment.pop("UPMEM_HW_RANK_PATH", None)
    environment.pop("DPU_BACKEND", None)
    completed = subprocess.run(
        [
            str(host_binary),
            "--execute-plan",
            "--resident-package",
            str(manifest_path),
            "--schedule",
            str(schedule_path),
            "--response",
            str(response_path),
        ],
        cwd=host_binary.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "failed"
    assert response["failure_stage"] == "hardware_profile_violation"
    assert response["requested_rank_path"] is None
    assert response["observed_rank_count"] == 0
    assert response["allocated_dpu_count"] == 0
    assert response["allocation"]["attempted"] is False


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
        "active_operation_h2d_bytes": plan.logical_task_count * repetitions * 8,
        "completion_d2h_bytes": plan.logical_task_count * repetitions * 120,
        "cross_d2h_bytes": 0,
        "cross_h2d_bytes": 0,
        "final_d2h_bytes": repetitions * 8,
        "actual_h2d_bytes": 64 + repetitions * (64 + plan.logical_task_count * 8),
        "actual_d2h_bytes": repetitions * (plan.logical_task_count * 120 + 8),
        "actual_transfer_bytes": 64 + repetitions * (64 + plan.logical_task_count * 8 + plan.logical_task_count * 120 + 8),
        "launch_count": plan.logical_task_count * repetitions,
        "synchronize_count": plan.logical_task_count * repetitions,
        "completion_reads": plan.logical_task_count * repetitions,
        "cross_dpu_edge_count": 0,
    }
    native_response = {
        "completed_per_dpu": [plan.logical_task_count * repetitions],
        "allocated_dpu_count": plan.requested_dpu_count,
        "requested_rank_path": "/dev/dpu_rank1",
        "observed_rank_count": 1,
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
        requested_rank_path="/dev/dpu_rank1",
    )
    prepared = SimpleNamespace(
        request_id=request["request_id"],
        plan=plan,
        package_file_sha256=plan.package_file_sha256,
        schedule_sidecar_sha256=plan.schedule_sidecar_sha256,
        request_path=request_path,
        source_output=np.asarray(output_values, dtype=np.float32),
    )

    route._complete_session_validation(session, prepared, route.M45_CONTRACT)
    validator(session, prepared, route.M45_CONTRACT)
    assert session["allocation_succeeded"] is True
    assert session["persistent_allocation_observed"] is True
    assert session["requested_rank_path"] == "/dev/dpu_rank1"
    assert session["observed_rank_count"] == 1
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
        total_cross_dpu_transfer_bytes=0,
    )
    request = {"requested_warmups": 1, "requested_repetitions": 3}
    response = {
        "completed_per_dpu": expected_completed_per_dpu,
        "metrics": {
            "launch_count": 12,
            "synchronize_count": 12,
            "completion_reads": 12,
            "cross_dpu_edge_count": 0,
            "descriptor_h2d_bytes": 0,
            "operand_h2d_bytes": 0,
            "reset_h2d_bytes": 0,
            "active_operation_h2d_bytes": 96,
            "completion_d2h_bytes": 1440,
            "cross_d2h_bytes": 0,
            "cross_h2d_bytes": 0,
            "final_d2h_bytes": 0,
            "actual_h2d_bytes": 96,
            "actual_d2h_bytes": 1440,
            "actual_transfer_bytes": 1536,
        },
    }

    executor._validate_native_metrics(response, plan, request)


def test_native_completion_count_mismatch_is_rejected() -> None:
    plan = SimpleNamespace(
        assignments=[SimpleNamespace(dpu_id=dpu_id) for dpu_id in (0, 1, 0)],
        requested_dpu_count=2,
        logical_task_count=3,
        transfer_edges=(),
        total_cross_dpu_transfer_bytes=0,
    )
    request = {"requested_warmups": 1, "requested_repetitions": 3}
    response = {
        "completed_per_dpu": [2, 1],
        "metrics": {
            "launch_count": 12,
            "synchronize_count": 12,
            "completion_reads": 12,
            "cross_dpu_edge_count": 0,
            "descriptor_h2d_bytes": 0,
            "operand_h2d_bytes": 0,
            "reset_h2d_bytes": 0,
            "active_operation_h2d_bytes": 96,
            "completion_d2h_bytes": 1440,
            "cross_d2h_bytes": 0,
            "cross_h2d_bytes": 0,
            "final_d2h_bytes": 0,
            "actual_h2d_bytes": 96,
            "actual_d2h_bytes": 1440,
            "actual_transfer_bytes": 1536,
        },
    }

    with pytest.raises(executor.NativeAdapterError, match="completion counts differ"):
        executor._validate_native_metrics(response, plan, request)


def test_native_completion_counts_must_not_be_nested_in_metrics() -> None:
    plan = SimpleNamespace(
        assignments=[SimpleNamespace(dpu_id=dpu_id) for dpu_id in (0, 0, 0)],
        requested_dpu_count=1,
        logical_task_count=3,
        transfer_edges=(),
        total_cross_dpu_transfer_bytes=0,
    )
    request = {"requested_warmups": 1, "requested_repetitions": 3}
    response = {
        "metrics": {
            "completed_per_dpu": [12],
            "launch_count": 12,
            "synchronize_count": 12,
            "completion_reads": 12,
            "cross_dpu_edge_count": 0,
            "descriptor_h2d_bytes": 0,
            "operand_h2d_bytes": 0,
            "reset_h2d_bytes": 0,
            "active_operation_h2d_bytes": 96,
            "completion_d2h_bytes": 1440,
            "cross_d2h_bytes": 0,
            "cross_h2d_bytes": 0,
            "final_d2h_bytes": 0,
            "actual_h2d_bytes": 96,
            "actual_d2h_bytes": 1440,
            "actual_transfer_bytes": 1536,
        }
    }

    with pytest.raises(executor.NativeAdapterError, match="top-level response fields"):
        executor._validate_native_metrics(response, plan, request)


def _control_and_completion_response() -> tuple[SimpleNamespace, dict[str, int], dict[str, object]]:
    edge = CrossDpuTransfer(
        producer_operation_id=1,
        consumer_operation_id=2,
        producer_task_id="task_1",
        consumer_task_id="task_2",
        producer_dpu_id=1,
        consumer_dpu_id=0,
        slot_id=3,
        element_count=1,
        transfer_bytes=4,
    )
    plan = SimpleNamespace(
        assignments=[SimpleNamespace(dpu_id=dpu_id) for dpu_id in (0, 1, 0)],
        requested_dpu_count=2,
        logical_task_count=3,
        transfer_edges=(edge,),
        total_cross_dpu_transfer_bytes=2 * edge.transfer_bytes,
    )
    request = {"requested_warmups": 1, "requested_repetitions": 3}
    response = {
        "completed_per_dpu": [8, 4],
        "metrics": {
            "launch_count": 12,
            "synchronize_count": 12,
            "completion_reads": 12,
            "cross_dpu_edge_count": 4,
            "descriptor_h2d_bytes": 64,
            "operand_h2d_bytes": 16,
            "reset_h2d_bytes": 32,
            "active_operation_h2d_bytes": 96,
            "completion_d2h_bytes": 1440,
            "cross_d2h_bytes": 16,
            "cross_h2d_bytes": 16,
            "final_d2h_bytes": 8,
            "actual_h2d_bytes": 224,
            "actual_d2h_bytes": 1464,
            "actual_transfer_bytes": 1688,
        },
    }
    return plan, request, response


def test_native_metrics_include_per_operation_control_and_completion_bytes() -> None:
    plan, request, response = _control_and_completion_response()

    executor._validate_native_metrics(response, plan, request)


@pytest.mark.parametrize("metric", ["cross_h2d_bytes", "cross_d2h_bytes"])
def test_native_metrics_reject_cross_dpu_bytes_not_bound_to_plan(metric: str) -> None:
    plan, request, response = _control_and_completion_response()
    metrics = dict(response["metrics"])
    metrics[metric] += 4
    response = {**response, "metrics": metrics}

    with pytest.raises(executor.NativeAdapterError, match="cross-DPU bytes differ"):
        executor._validate_native_metrics(response, plan, request)


def test_native_process_timeout_allows_native_cleanup_grace() -> None:
    assert executor._native_process_timeout(1.1) == 7


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("active_operation_h2d_bytes", 88, "active-operation control byte count"),
        ("completion_d2h_bytes", 1432, "completion byte count"),
        ("reduction_d2h_bytes", 8, "distributed reduction bytes"),
    ],
)
def test_native_metrics_reject_inconsistent_control_or_completion_bytes(
    metric: str, value: int, message: str
) -> None:
    plan, request, response = _control_and_completion_response()
    metrics = dict(response["metrics"])
    metrics[metric] = value
    response = {**response, "metrics": metrics}

    with pytest.raises(executor.NativeAdapterError, match=message):
        executor._validate_native_metrics(response, plan, request)


def test_native_rank_response_requires_exact_requested_rank_and_one_sdk_rank() -> None:
    response = {"requested_rank_path": "/dev/dpu_rank1", "observed_rank_count": 1}
    executor._validate_native_rank_response(response, "/dev/dpu_rank1")

    with pytest.raises(executor.NativeAdapterError, match="requested rank path differs"):
        executor._validate_native_rank_response(response, "/dev/dpu_rank2")
    with pytest.raises(executor.NativeAdapterError, match="exactly one SDK rank"):
        executor._validate_native_rank_response(
            {**response, "observed_rank_count": 2}, "/dev/dpu_rank1"
        )


def test_native_execution_plan_host_requires_rank_pinned_provider() -> None:
    source = (
        ROOT
        / "native/upmem/simplepim/upmem_sdk_execution_plan/host.c"
    ).read_text(encoding="ascii")

    assert 'getenv("UPMEM_HW_RANK_PATH")' in source
    assert "UPMEM_HW_RANK_PATH is required for the physical route" in source
    assert "execution_plan_provider_init_on_rank(" in source
    assert "execution_plan_provider_init(&provider" not in source
    assert "provider.observed_ranks == 1u" in source
    assert "requested_rank_path" in source
    assert "observed_rank_count" in source


@pytest.mark.parametrize("tasklets", [1, 2, 4, 8, 16])
def test_multi_tasklet_benchmark_scaling(tasklets: int) -> None:
    from dataclasses import replace
    from quantum_bench.targets.upmem.hardware_taskgraph_resident import load_hardware_taskgraph_resident_suite
    from quantum_bench.targets.upmem.runtime_evidence import _base_task_metric, _summary_payload, UPMEM_DPU_CLOCK_HZ
    from quantum_bench.tn.upmem_path_cost_v2 import calibrate_pim_cost_model

    suite_path = ROOT / "configs/suites/upmem_hardware_taskgraph_resident_m4_6_tasklet_scaling.yml"
    suite = load_hardware_taskgraph_resident_suite(suite_path)
    profile = replace(suite.profile, tasklets_per_dpu=tasklets)
    assert profile.tasklets_per_dpu == tasklets

    # 1. Assert metric dictionary structure contains dpu_run_time_cycles key
    task = SimpleNamespace(
        id="task_0",
        input_tensor_ids=("t_in0", "t_in1"),
        output_tensor_id="t_out",
        input_shapes=((2, 2), (2, 2)),
        output_shape=(2, 2),
        estimated_flops=1000,
        estimated_bytes=512,
    )
    metric = _base_task_metric(
        "case_test",
        0,
        task,
        "generic-only",
        "none",
        status="completed",
        reason=None,
        task_started=0.0,
        dpu_run_time_cycles=14000,
    )
    assert "dpu_run_time_cycles" in metric
    assert metric["dpu_run_time_cycles"] == 14000

    # 2. Assert summary payload contains dpu_run_time_cycles and dpu_compute_seconds
    summary = _summary_payload(
        case_id="case_test",
        policy="generic-only",
        quantization_mode="none",
        status="completed",
        reason=None,
        started=0.0,
        task_metrics=[metric],
        kernel_family_counts={"generic_loop_fallback": 1},
        backend_counts={"upmem_sdk_simulator_generic_loop": 1},
        final_validation={"passed": True},
        final_tensor_id="t0",
        final_tensor_labels=(0, 1),
        final_transpose_applied=False,
        total_bridge_time_s=0.01,
        total_kernel_time_s=0.005,
        total_build_time_s=0.001,
        peak_live_tensor_bytes=1024,
    )
    assert "dpu_run_time_cycles" in summary
    assert "dpu_compute_seconds" in summary
    assert summary["dpu_run_time_cycles"] == 14000
    assert summary["dpu_compute_seconds"] == pytest.approx(14000 / UPMEM_DPU_CLOCK_HZ)

    # 3. Test non-zero fallback guards for simulator & model calibration
    c_pim_sim = calibrate_pim_cost_model(0, 1024)
    assert c_pim_sim is None  # Software simulator returns 0 cycles, handled safely

    c_pim_hw = calibrate_pim_cost_model(14000, 1000)
    assert c_pim_hw == 14.0  # Hardware 14000 cycles / 1000 flops = 14.0
