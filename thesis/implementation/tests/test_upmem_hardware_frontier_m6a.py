from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from quantum_bench.bench import upmem_hardware_frontier_m6a as m6a
from quantum_bench.bench import upmem_simplepim_taskgraph as engine
import quantum_bench.targets.upmem.simplepim_taskgraph_executor as executor
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_EXECUTION_PLAN_BACKEND_ID,
    RESIDENT_EXECUTION_PLAN_PROFILE_VERSION,
    RESIDENT_EXECUTION_PLAN_ROUTE_ID,
    RESIDENT_PACKAGE_HEADER_BYTES,
    RESIDENT_PACKAGE_HEADER_FORMAT,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/suites/upmem_hardware_frontier_m6a.yml"
RANK_PATH = "/dev/dpu_rank1"
ACTIVE_OPERATION_BYTES = 8
COMPLETION_BYTES = 120
FLOAT32_BYTES = 4
DESCRIPTOR_CONTROL_BYTES = 16
INITIAL_OPERAND_BYTES_PER_DPU = 80
RESET_INPUT_BYTES_PER_DPU = 80
RESET_OUTPUT_BYTES_PER_DPU = 48


def _align8(byte_count: int) -> int:
    return (byte_count + 7) & ~7


def _package_transfer_accounting(request_path: Path, request: dict[str, object]) -> dict[str, int]:
    """Derive host-visible descriptor traffic from the generated package."""

    package_path = (request_path.parent / str(request["package_path"])).resolve()
    package = package_path.read_bytes()
    header = struct.unpack_from(RESIDENT_PACKAGE_HEADER_FORMAT, package, 0)
    header_bytes = int(header[3])
    file_bytes = int(header[5])
    slot_bytes = int(header[7])
    operation_bytes = int(header[9])
    assert header_bytes == RESIDENT_PACKAGE_HEADER_BYTES
    assert file_bytes == len(package)
    assert slot_bytes + operation_bytes == file_bytes - header_bytes
    return {
        "package_file_bytes": file_bytes,
        # The header is host-only. Native copy_package_to_dpu transfers only
        # slot and operation sections, then a separate resident control word.
        "descriptor_operation_h2d_bytes": slot_bytes + operation_bytes,
        "descriptor_control_h2d_bytes": DESCRIPTOR_CONTROL_BYTES,
    }


class FakeTarget:
    def __init__(self, *, observed_rank_count: int = 1) -> None:
        self.observed_rank_count = observed_rank_count
        self.build_calls = 0
        self.validate_calls: list[Path] = []
        self.execute_calls: list[Path] = []
        self.last_transfer_accounting: dict[str, int] | None = None

    def build(self, build_dir: Path, *, prepare_only: bool = True) -> dict[str, object]:
        self.build_calls += 1
        build_dir.mkdir(parents=True, exist_ok=True)
        host = build_dir / "host.bin"
        dpu = build_dir / "dpu.bin"
        host.write_bytes(b"host")
        dpu.write_bytes(b"dpu")
        return {
            "status": "built",
            "prepare_only": prepare_only,
            "allocation_attempted": False,
            "launch_attempted": False,
            "host_binary": str(host),
            "dpu_binary": str(dpu),
        }

    def validate(self, request_path: Path, *, timeout_s: float) -> dict[str, object]:
        self.validate_calls.append(request_path)
        return {
            "status": "passed",
            "allocation_attempted": False,
            "launch_attempted": False,
        }

    def execute(self, request_path: Path, *, timeout_s: float) -> dict[str, object]:
        self.execute_calls.append(request_path)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        plan = json.loads((request_path.parent / "execution_plan.json").read_text(encoding="utf-8"))
        reference = np.load(request_path.parent / "cpu_reference.npy", allow_pickle=False)
        package_accounting = _package_transfer_accounting(request_path, request)
        self.last_transfer_accounting = package_accounting
        total = int(request["requested_warmups"]) + int(request["requested_repetitions"])
        task_count = int(plan["logical_task_count"])
        dpu_count = int(request["requested_dpu_count"])
        cross_h2d = int(plan["transfer_summary"]["host_to_dpu_bytes"])
        cross_d2h = int(plan["transfer_summary"]["dpu_to_host_bytes"])
        final_output_bytes = _align8(
            int(plan["final_outputs"][0]["element_count"]) * FLOAT32_BYTES
        )
        # The fixed M6a package has six initial real slots (80 bytes) and five
        # produced slots (48 bytes). Native initialization transfers the
        # package sections plus one 16-byte resident-control structure per DPU.
        # Each repeat then resets completion/control, inputs, and outputs.
        reset_bytes_per_dpu = (
            COMPLETION_BYTES
            + ACTIVE_OPERATION_BYTES
            + RESET_INPUT_BYTES_PER_DPU
            + RESET_OUTPUT_BYTES_PER_DPU
        )
        reset_h2d = dpu_count * reset_bytes_per_dpu
        repeat_h2d = task_count * ACTIVE_OPERATION_BYTES + reset_h2d + cross_h2d
        repeat_d2h = task_count * COMPLETION_BYTES + cross_d2h + final_output_bytes
        initial_h2d = dpu_count * (
            package_accounting["descriptor_operation_h2d_bytes"]
            + package_accounting["descriptor_control_h2d_bytes"]
            + INITIAL_OPERAND_BYTES_PER_DPU
        )
        repetitions = []
        for index in range(total):
            warmup = index < request["requested_warmups"]
            repetitions.append(
                {
                    "repeat_id": index if warmup else index - request["requested_warmups"],
                    "warmup": warmup,
                    "status": "completed",
                    "scheduled_task_count": plan["logical_task_count"],
                    "wave_barrier_count": plan["wave_count"],
                    "launch_count": plan["logical_task_count"],
                    "synchronize_count": plan["logical_task_count"],
                    "device_launch_mode": engine.DEVICE_LAUNCH_MODE,
                    "synchronization_policy": engine.SYNCHRONIZATION_POLICY,
                    "fully_synchronous_kernel_launch": False,
                    "timing": {
                        "operand_h2d_time_s": None,
                        "cross_dpu_transfer_time_s": None,
                        "wave_launch_sync_time_s": None,
                        "final_d2h_time_s": None,
                        "total_repetition_time_s": None,
                    },
                    "transfer": {
                        "h2d_bytes": repeat_h2d,
                        "d2h_bytes": repeat_d2h,
                        "total_bytes": repeat_h2d + repeat_d2h,
                    },
                    "validation_id": "m6a-fake-final-output",
                    "repeat_output_validation_status": "not_individually_collected",
                    "session_completion_scope": "aggregate_across_warmups_and_repetitions",
                    "repeat_completion_observation_status": "not_individually_collected",
                    "aggregate_session_completion_id": "aggregate_session_completion:m6a-fake",
                    "aggregate_session_completion_status": "passed",
                }
            )
        per_dpu = [
            sum(item["dpu_id"] == dpu for item in plan["task_assignments"]) * total
            for dpu in range(request["requested_dpu_count"])
        ]
        return {
            "schema_version": engine.ADAPTER_SESSION_SCHEMA,
            "status": "completed",
            "returncode": 0,
            "request_id": request["request_id"],
            "backend_id": engine.NATIVE_BACKEND_ID,
            "target_requested": "hardware",
            "target_observed": "physical_hardware",
            "hardware_profile_version": executor.NATIVE_HARDWARE_PROFILE_VERSION,
            "native_hardware_profile_version": executor.NATIVE_HARDWARE_PROFILE_VERSION,
            "resident_manifest_route_id": request["resident_route_id"],
            "resident_manifest_backend_id": request["resident_backend_id"],
            "resident_manifest_hardware_profile_version": request["resident_hardware_profile_version"],
            "resident_manifest_requested_dpu_count": request["resident_requested_dpu_count"],
            "resident_manifest_tasklets_per_dpu": request["resident_tasklets_per_dpu"],
            "execution_plan_hash": request["execution_plan_hash"],
            "upmem_execution_plan_hash": request["execution_plan_hash"],
            "package_file_sha256": request["package_file_sha256"],
            "schedule_sidecar_sha256": request["schedule_sidecar_sha256"],
            "source_identity": request["source_identity"],
            "package_identity": request["package_identity"],
            "requested_dpu_count": 2,
            "allocated_dpu_count": 2,
            "tasklets_per_dpu": 1,
            "allocation_attempted": True,
            "allocation_count": 1,
            "allocation_succeeded": True,
            "persistent_allocation_observed": True,
            "release_confirmed": True,
            "hardware_allocation_verified": True,
            "native_kernel_executed": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_speedup_applicable": False,
            "device_launch_mode": engine.DEVICE_LAUNCH_MODE,
            "synchronization_policy": engine.SYNCHRONIZATION_POLICY,
            "fully_synchronous_kernel_launch": False,
            "requested_warmups": request["requested_warmups"],
            "requested_repetitions": request["requested_repetitions"],
            "native_session_count": 1,
            "logical_task_count": plan["logical_task_count"],
            "session_completion_scope": "aggregate_across_warmups_and_repetitions",
            "aggregate_completed_per_dpu": per_dpu,
            "aggregate_total_task_completion_count": sum(per_dpu),
            "aggregate_session_completion_id": "aggregate_session_completion:m6a-fake",
            "aggregate_session_completion_status": "passed",
            "total_task_completion_count": plan["logical_task_count"] * total,
            "exactly_once_execution_verified": True,
            "wave_barrier_count_total": plan["wave_count"] * total,
            "operation_assignments": plan["task_assignments"],
            "cross_dpu_transfer": {
                "count": plan["transfer_summary"]["edge_count"],
                "bytes": plan["transfer_summary"]["total_bytes"],
            },
            "schedule_h2d_bytes": 0,
            "session_timing": {
                "allocation_time_s": None,
                "binary_load_time_s": None,
                "descriptor_h2d_time_s": None,
                "release_time_s": None,
                "total_session_time_s": None,
            },
            "session_transfer": {
                "initial_h2d_bytes": initial_h2d,
                "actual_h2d_bytes": initial_h2d + total * repeat_h2d,
                "actual_d2h_bytes": total * repeat_d2h,
                "actual_transfer_bytes": initial_h2d + total * (repeat_h2d + repeat_d2h),
            },
            "requested_rank_path": request["requested_rank_path"],
            "observed_rank_count": self.observed_rank_count,
            "session_validation": {
                "validation_id": "m6a-fake-final-output",
                "status": "collected",
                "scope": "final_session_output_only",
                "output": reference.real.tolist(),
                "final_output_path": "native_final_output.bin",
                "output_sha256": "0" * 64,
                "output_provenance": "m6a_fake_final_output",
            },
            "repetitions": repetitions,
        }


def test_m6a_suite_and_contract_are_fixed() -> None:
    suite = m6a.load_upmem_hardware_frontier_m6a_suite(SUITE)
    assert suite["planner"] == {"engine": "opt_einsum", "optimize": "greedy"}
    assert suite["warmups"] == 1
    assert suite["repeats"] == 5
    assert m6a.M6A_CONTRACT.expected_path == ((0, 2), (0, 1), (0, 2), (0, 1), (0, 1))
    assert m6a.M6A_CONTRACT.expected_wave_widths == (2, 2, 1)
    assert m6a.M6A_CONTRACT.expected_dpu_task_counts == (3, 2)
    assert m6a.M6A_CONTRACT.expected_assignment_dpu_ids == (0, 1, 0, 1, 0)
    assert m6a.M6A_CONTRACT.expected_transfer_edges == (
        engine.TransferEdgeExpectation("task_3", "task_4", 1, 0, 9, 2, 8),
    )
    assert m6a.M6A_CONTRACT.allow_slot_reuse is False
    assert m6a.M6A_CONTRACT.include_failures_in_normalized_records is True


def test_m6a_rejects_profile_dpu_count_drift(tmp_path: Path) -> None:
    altered = tmp_path / "m6a_bad_dpus.yml"
    altered.write_text(
        SUITE.read_text(encoding="utf-8").replace("requested_dpu_count: 2", "requested_dpu_count: 1"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requested DPU count"):
        m6a.load_upmem_hardware_frontier_m6a_suite(altered)


def test_m6a_requires_explicit_cap_fields(tmp_path: Path) -> None:
    altered = tmp_path / "m6a_missing_cap.yml"
    altered.write_text(
        SUITE.read_text(encoding="utf-8").replace("    max_waves: 8\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_waves"):
        m6a.load_upmem_hardware_frontier_m6a_suite(altered)


def test_m6a_prepare_is_dry_and_enforces_frontier_shape(tmp_path: Path) -> None:
    target = FakeTarget()
    result = m6a.prepare_upmem_hardware_frontier_m6a(
        tmp_path, suite_path=SUITE, build=True, native_target=target
    )
    assert result["status"] == "prepared"
    assert result["dpu_allocation_attempted"] is False
    assert result["dpu_launch_attempted"] is False
    assert target.build_calls == 1
    assert len(target.validate_calls) == 1
    placement = result["placements"][0]
    assert placement["requested_dpu_count"] == 2
    assert placement["task_count"] == 5
    assert [len(wave) for wave in json.loads(Path(placement["plan_path"]).read_text())["waves"]] == [2, 2, 1]
    assert [sum(item["dpu_id"] == dpu for item in placement["assignments"]) for dpu in (0, 1)] == [3, 2]
    plan = json.loads(Path(placement["plan_path"]).read_text(encoding="utf-8"))
    manifests = list(Path(result["plan_dir"]).rglob("*_resident_request.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["route_id"] == RESIDENT_EXECUTION_PLAN_ROUTE_ID
    assert manifest["backend_id"] == RESIDENT_EXECUTION_PLAN_BACKEND_ID
    assert manifest["hardware_profile_version"] == RESIDENT_EXECUTION_PLAN_PROFILE_VERSION
    assert manifest["requested_dpu_count"] == 2
    assert manifest["tasklets"] == 1
    assert [item["dpu_id"] for item in plan["task_assignments"]] == [0, 1, 0, 1, 0]
    assert plan["transfer_summary"] == {
        "edge_count": 1,
        "host_to_dpu_bytes": 8,
        "dpu_to_host_bytes": 8,
        "total_bytes": 16,
    }
    assert plan["transfer_edges"] == [
        {
            "producer_operation_id": 3,
            "consumer_operation_id": 4,
            "producer_task_id": "task_3",
            "consumer_task_id": "task_4",
            "producer_dpu_id": 1,
            "consumer_dpu_id": 0,
            "slot_id": 9,
            "element_count": 2,
            "transfer_bytes": 8,
            "transport": "host_mediated_v1",
        }
    ]


def test_m6a_completed_execution_records_frontier_and_rank_evidence(tmp_path: Path) -> None:
    target = FakeTarget()
    result = m6a.run_upmem_hardware_frontier_m6a(
        tmp_path,
        suite_path=SUITE,
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_HW_RANK_PATH": RANK_PATH},
        native_target=target,
    )
    assert result["status"] == "completed"
    run_dir = Path(result["run_dir"])
    records = [json.loads(line) for line in (run_dir / "normalized_records.jsonl").read_text().splitlines()]
    assert len(records) == 5
    assert all(row["requested_rank_path"] == RANK_PATH for row in records)
    assert all(row["observed_rank_count"] == 1 for row in records)
    assert all(row["aggregate_exactly_once_execution_verified"] is True for row in records)
    assert all(row["cross_dpu_transfer_count"] == 1 for row in records)
    assert all(row["cross_dpu_transfer_bytes"] == 16 for row in records)
    assert all(row["cpu_fallback_used"] is False for row in records)
    assert all(row["target_requested"] == "hardware" for row in records)
    assert all(row["target_observed"] == "physical_hardware" for row in records)
    assert all(row["hardware_allocation_verified"] is True for row in records)
    assert all(row["route_hardware_profile_version"] == m6a.M6A_CONTRACT.profile_version for row in records)
    assert all(row["native_hardware_profile_version"] == executor.NATIVE_HARDWARE_PROFILE_VERSION for row in records)
    assert all(row["resident_manifest_route_id"] == RESIDENT_EXECUTION_PLAN_ROUTE_ID for row in records)
    assert all(row["resident_manifest_backend_id"] == RESIDENT_EXECUTION_PLAN_BACKEND_ID for row in records)
    assert all(row["resident_manifest_hardware_profile_version"] == RESIDENT_EXECUTION_PLAN_PROFILE_VERSION for row in records)
    assert all(row["resident_manifest_requested_dpu_count"] == 2 for row in records)
    assert all(row["resident_manifest_tasklets_per_dpu"] == 1 for row in records)
    assert all(row["hardware_kernel_executed"] is True for row in records)
    assert all(row["resident_slot_reuse_limited"] is True for row in records)
    assert {row["actual_h2d_bytes"] for row in records} == {560}
    assert {row["actual_d2h_bytes"] for row in records} == {624}
    assert {row["actual_transfer_bytes"] for row in records} == {1184}
    assert target.last_transfer_accounting == {
        "package_file_bytes": 4192,
        "descriptor_operation_h2d_bytes": 4096,
        "descriptor_control_h2d_bytes": 16,
    }


def test_m6a_rejects_wrong_graph_shape(tmp_path: Path) -> None:
    def wrong_plan(*args: object, **kwargs: object) -> object:
        plan = engine.compile_plan(*args, **kwargs)
        return replace(plan, waves=(plan.waves[0], plan.waves[1], ()))

    with pytest.raises(ValueError, match="wave widths"):
        m6a.prepare_upmem_hardware_frontier_m6a(
            tmp_path, suite_path=SUITE, native_target=FakeTarget(), plan_compiler=wrong_plan
        )


def test_m6a_rejects_assignment_or_transfer_topology_drift(tmp_path: Path) -> None:
    def wrong_assignment(*args: object, **kwargs: object) -> object:
        plan = engine.compile_plan(*args, **kwargs)
        assignments = tuple(
            replace(item, dpu_id=(1 if item.operation_id == 2 else 0 if item.operation_id == 3 else item.dpu_id))
            for item in plan.assignments
        )
        return replace(plan, assignments=assignments)

    with pytest.raises(ValueError, match="DPU assignment differs"):
        m6a.prepare_upmem_hardware_frontier_m6a(
            tmp_path, suite_path=SUITE, native_target=FakeTarget(), plan_compiler=wrong_assignment
        )

    def wrong_transfers(*args: object, **kwargs: object) -> object:
        plan = engine.compile_plan(*args, **kwargs)
        return replace(plan, transfer_edges=())

    with pytest.raises(ValueError, match="transfer edges differ"):
        m6a.prepare_upmem_hardware_frontier_m6a(
            tmp_path, suite_path=SUITE, native_target=FakeTarget(), plan_compiler=wrong_transfers
        )


def test_m6a_requires_hardware_opt_in(tmp_path: Path) -> None:
    with pytest.raises(engine.NativeExecutionFailure, match="UPMEM_ALLOW_PHYSICAL_HARDWARE"):
        m6a.run_upmem_hardware_frontier_m6a(
            tmp_path,
            suite_path=SUITE,
            environment={"UPMEM_HW_RANK_PATH": RANK_PATH},
            native_target=FakeTarget(),
        )


def test_m6a_rejects_invalid_rank_before_native_build(tmp_path: Path) -> None:
    target = FakeTarget()
    with pytest.raises(engine.NativeExecutionFailure, match="must match"):
        m6a.run_upmem_hardware_frontier_m6a(
            tmp_path,
            suite_path=SUITE,
            environment={
                "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
                "UPMEM_HW_RANK_PATH": "/tmp/not-a-rank",
            },
            native_target=target,
        )
    assert target.build_calls == 0
    assert target.execute_calls == []


def test_m6a_rejects_invalid_native_rank_evidence(tmp_path: Path) -> None:
    result = m6a.run_upmem_hardware_frontier_m6a(
        tmp_path,
        suite_path=SUITE,
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_HW_RANK_PATH": RANK_PATH},
        native_target=FakeTarget(observed_rank_count=2),
    )
    assert result["status"] == "failed"
    run_dir = Path(result["run_dir"])
    failure = json.loads((run_dir / "session_failures.jsonl").read_text().splitlines()[0])
    assert failure["failure_stage"] == "output_manifest_failed"
    normalized = [json.loads(line) for line in (run_dir / "normalized_records.jsonl").read_text().splitlines()]
    assert len(normalized) == 1
    assert normalized[0]["status"] == "failed"
    assert normalized[0]["failure_stage"] == failure["failure_stage"]
    assert normalized[0]["route_id"] == m6a.ROUTE_ID
    summary = json.loads((run_dir / "upmem_hardware_frontier_m6a_summary.json").read_text())
    assert summary["measured_row_count"] == 0
    assert summary["failure_record_count"] == 1
    assert summary["normalized_record_count"] == 1
