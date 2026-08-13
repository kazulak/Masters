from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from quantum_bench.bench import upmem_simplepim_taskgraph as route


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "suites" / "upmem_hardware_simplepim_taskgraph.yml"
RANK_PATH = "/dev/dpu_rank1"
PHYSICAL_ENVIRONMENT = {
    "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
    "UPMEM_HW_RANK_PATH": RANK_PATH,
}


class FakeNativeTarget:
    def __init__(
        self,
        *,
        failure: str | None = None,
        write_response: bool = True,
        bad_transfer: bool = False,
        bad_hash: bool = False,
        returncode: int | None = None,
        unsafe_prepare: bool = False,
        session_validation_status: str | None = "collected",
    ) -> None:
        self.failure = failure
        self.write_response = write_response
        self.bad_transfer = bad_transfer
        self.bad_hash = bad_hash
        self.returncode = returncode
        self.unsafe_prepare = unsafe_prepare
        self.session_validation_status = session_validation_status
        self.build_calls = 0
        self.validate_calls: list[tuple[Path, float]] = []
        self.execute_calls: list[tuple[Path, float]] = []

    def build(self, build_dir: Path, *, prepare_only: bool = True) -> dict[str, object]:
        self.build_calls += 1
        build_dir.mkdir(parents=True, exist_ok=True)
        dpu = build_dir / "dpu.bin"
        host = build_dir / "host.bin"
        dpu.write_bytes(b"dpu")
        host.write_bytes(b"host")
        return {
            "status": "built",
            "prepare_only": prepare_only,
            "allocation_attempted": self.unsafe_prepare,
            "launch_attempted": False,
            "dpu_binary": str(dpu),
            "host_binary": str(host),
        }

    def validate(self, request_path: Path, *, timeout_s: float) -> dict[str, object]:
        self.validate_calls.append((request_path, timeout_s))
        return {
            "status": "passed",
            "allocation_attempted": False,
            "launch_attempted": False,
        }

    def execute(self, request_path: Path, *, timeout_s: float) -> dict[str, object]:
        self.execute_calls.append((request_path, timeout_s))
        request = json.loads(request_path.read_text(encoding="utf-8"))
        plan = json.loads(
            (request_path.parent / "execution_plan.json").read_text(encoding="utf-8")
        )
        if self.failure:
            return {"status": "failed", "failure_stage": self.failure, "reason": self.failure}
        if not self.write_response:
            return {"status": "completed"}

        reference = np.load(request_path.parent / "cpu_reference.npy", allow_pickle=False)
        repetitions: list[dict[str, object]] = []
        total_repetitions = request["requested_warmups"] + request["requested_repetitions"]
        aggregate_completed_per_dpu = [
            sum(item["dpu_id"] == dpu for item in plan["task_assignments"])
            * total_repetitions
            for dpu in range(request["requested_dpu_count"])
        ]
        for index in range(total_repetitions):
            warmup = index < request["requested_warmups"]
            h2d_bytes = 64
            d2h_bytes = 8
            repetitions.append(
                {
                    "repeat_id": index if warmup else index - request["requested_warmups"],
                    "warmup": warmup,
                    "status": "completed",
                    "scheduled_task_count": plan["logical_task_count"],
                    "wave_barrier_count": plan["wave_count"],
                    "launch_count": plan["logical_task_count"],
                    "synchronize_count": plan["logical_task_count"],
                    "device_launch_mode": route.DEVICE_LAUNCH_MODE,
                    "synchronization_policy": route.SYNCHRONIZATION_POLICY,
                    "fully_synchronous_kernel_launch": False,
                    "timing": {
                        "operand_h2d_time_s": None,
                        "cross_dpu_transfer_time_s": None,
                        "wave_launch_sync_time_s": None,
                        "final_d2h_time_s": None,
                        "total_repetition_time_s": None,
                    },
                    "transfer": {
                        "h2d_bytes": h2d_bytes,
                        "d2h_bytes": d2h_bytes,
                        "total_bytes": h2d_bytes + d2h_bytes - int(self.bad_transfer),
                    },
                    "validation_id": "fake-final-session-output",
                    "repeat_output_validation_status": "not_individually_collected",
                    "session_completion_scope": "aggregate_across_warmups_and_repetitions",
                    "repeat_completion_observation_status": "not_individually_collected",
                    "aggregate_session_completion_id": "aggregate_session_completion:fake",
                    "aggregate_session_completion_status": "passed",
                }
            )
        return {
            "schema_version": route.ADAPTER_SESSION_SCHEMA,
            "status": "completed",
            "returncode": 0 if self.returncode is None else self.returncode,
            "request_id": request["request_id"],
            "backend_id": route.NATIVE_BACKEND_ID,
            "target_requested": "hardware",
            "target_observed": "physical_hardware",
            "execution_plan_hash": request["execution_plan_hash"],
            "upmem_execution_plan_hash": request["execution_plan_hash"],
            "package_file_sha256": "0" * 64 if self.bad_hash else request["package_file_sha256"],
            "schedule_sidecar_sha256": request["schedule_sidecar_sha256"],
            "source_identity": request["source_identity"],
            "package_identity": request["package_identity"],
            "requested_dpu_count": request["requested_dpu_count"],
            "allocated_dpu_count": request["requested_dpu_count"],
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
            "device_launch_mode": route.DEVICE_LAUNCH_MODE,
            "synchronization_policy": route.SYNCHRONIZATION_POLICY,
            "fully_synchronous_kernel_launch": False,
            "requested_warmups": request["requested_warmups"],
            "requested_repetitions": request["requested_repetitions"],
            "native_session_count": 1,
            "logical_task_count": plan["logical_task_count"],
            "session_completion_scope": "aggregate_across_warmups_and_repetitions",
            "aggregate_completed_per_dpu": aggregate_completed_per_dpu,
            "aggregate_total_task_completion_count": sum(aggregate_completed_per_dpu),
            "aggregate_session_completion_id": "aggregate_session_completion:fake",
            "aggregate_session_completion_status": "passed",
            "total_task_completion_count": plan["logical_task_count"] * total_repetitions,
            "exactly_once_execution_verified": True,
            "wave_barrier_count_total": plan["wave_count"] * total_repetitions,
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
                "initial_h2d_bytes": 64,
                "actual_h2d_bytes": 64 + total_repetitions * 64,
                "actual_d2h_bytes": total_repetitions * 8,
                "actual_transfer_bytes": 64 + total_repetitions * (64 + 8),
            },
            "requested_rank_path": request["requested_rank_path"],
            "observed_rank_count": 1,
            **(
                {}
                if self.session_validation_status is None
                else {
                    "session_validation": {
                        "validation_id": "fake-final-session-output",
                        "status": self.session_validation_status,
                        "scope": "final_session_output_only",
                        "output": reference.real.tolist(),
                        "final_output_path": "native_final_output.bin",
                        "output_sha256": "0" * 64,
                        "output_provenance": "fake_final_session_output",
                    }
                }
            ),
            "repetitions": repetitions,
        }


def test_suite_is_fixed_and_prepare_is_parser_only(tmp_path: Path) -> None:
    target = FakeNativeTarget()
    result = route.prepare(tmp_path, suite_path=SUITE, environment={}, native_target=target)
    assert result["status"] == "prepared"
    assert result["dpu_allocation_attempted"] is False
    assert result["dpu_launch_attempted"] is False
    assert result["preparation_mode"] == "parser_only_and_plan_validation"
    assert route.M45_CONTRACT.require_rank_evidence is True
    placements = result["placements"]
    assert [item["requested_dpu_count"] for item in placements] == [1, 2]
    assert placements[0]["package_file_sha256"] == placements[1]["package_file_sha256"]
    assert placements[0]["upmem_execution_plan_hash"] != placements[1]["upmem_execution_plan_hash"]
    assert placements[0]["schedule_sidecar_sha256"] != placements[1]["schedule_sidecar_sha256"]
    assert len(target.validate_calls) == 2
    assert all(timeout == 120.0 for _, timeout in target.validate_calls)
    package_paths = {
        (Path(item["request_path"]).parent / json.loads(Path(item["request_path"]).read_text())["package_path"]).resolve()
        for item in placements
    }
    assert len(package_paths) == 1
    for item in placements:
        request = json.loads(Path(item["request_path"]).read_text(encoding="utf-8"))
        plan = json.loads(Path(item["plan_path"]).read_text(encoding="utf-8"))
        assert request["requested_dpu_count"] == item["requested_dpu_count"]
        assert request["schedule_sidecar_h2d_bytes"] == 0
        assert request["schedule_h2d_bytes"] == 0
        assert set(plan["source_identity"]) == set(plan["package_identity"])
        assert plan["source_identity"] != plan["package_identity"]


def test_execute_keeps_warmups_out_of_normalized_records(tmp_path: Path) -> None:
    target = FakeNativeTarget()
    result = route.execute(
        tmp_path,
        suite_path=SUITE,
        environment=PHYSICAL_ENVIRONMENT,
        native_target=target,
    )
    assert result["status"] == "completed"
    run_dir = Path(result["run_dir"])
    measured = [json.loads(line) for line in (run_dir / "normalized_records.jsonl").read_text().splitlines()]
    warmups = [json.loads(line) for line in (run_dir / "warmup_records.jsonl").read_text().splitlines()]
    assert len(measured) == 6
    assert len(warmups) == 2
    assert all(row["warmup"] is False for row in measured)
    assert all(row["warmup"] is True for row in warmups)
    assert {row["requested_dpu_count"] for row in measured} == {1, 2}
    assert all(row["requested_rank_path"] == RANK_PATH for row in measured)
    assert all(row["observed_rank_count"] == 1 for row in measured)
    assert all(row["hardware_speedup_applicable"] is False for row in measured)
    assert all(row["validation_status"] == "not_individually_collected" for row in measured)
    assert all(row["session_validation_status"] == "passed" for row in measured)
    assert all(row["repeat_output_validation_status"] == "not_individually_collected" for row in measured)
    assert all(row["session_completion_scope"] == "aggregate_across_warmups_and_repetitions" for row in measured)
    assert all(row["repeat_completion_observation_status"] == "not_individually_collected" for row in measured)
    assert all(row["aggregate_session_completion_status"] == "passed" for row in measured)
    assert all(row["scheduled_task_count"] == 3 for row in measured)
    assert all(row["aggregate_session_completion_id"] == "aggregate_session_completion:fake" for row in measured)
    forbidden = {
        "completed_task_count", "completed_task_ids", "task_completion_counts",
        "exactly_once_execution_verified",
    }
    assert all(forbidden.isdisjoint(row) for row in measured)
    assert {tuple(row["aggregate_completed_per_dpu"]) for row in measured} == {(12,), (8, 4)}
    assert {row["aggregate_total_task_completion_count"] for row in measured} == {12}
    assert all("output" not in row for row in measured)
    assert target.build_calls == 1
    assert len(target.execute_calls) == 2
    assert all(timeout == 120.0 for _, timeout in target.execute_calls)


def test_hardware_opt_in_is_required_before_native_target_load(tmp_path: Path) -> None:
    with pytest.raises(route.NativeExecutionFailure, match="UPMEM_ALLOW_PHYSICAL_HARDWARE"):
        route.execute(tmp_path, suite_path=SUITE, environment={})


@pytest.mark.parametrize(
    "environment, message",
    [
        ({"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}, "UPMEM_HW_RANK_PATH is required"),
        (
            {
                "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
                "UPMEM_HW_RANK_PATH": "/tmp/not-a-rank",
            },
            "must match",
        ),
    ],
)
def test_m45_requires_valid_rank_before_native_target_build(
    tmp_path: Path, environment: dict[str, str], message: str
) -> None:
    target = FakeNativeTarget()
    with pytest.raises(route.NativeExecutionFailure, match=message):
        route.execute(
            tmp_path,
            suite_path=SUITE,
            environment=environment,
            native_target=target,
        )
    assert target.build_calls == 0
    assert target.execute_calls == []


def test_native_failure_stage_is_preserved_and_no_retry_occurs(tmp_path: Path) -> None:
    target = FakeNativeTarget(failure="kernel_timeout")
    result = route.execute(
        tmp_path,
        suite_path=SUITE,
        environment=PHYSICAL_ENVIRONMENT,
        native_target=target,
    )
    assert result["status"] == "failed"
    assert len(target.execute_calls) == 1
    run_dir = Path(result["run_dir"])
    failure = json.loads((run_dir / "session_failures.jsonl").read_text().splitlines()[0])
    assert failure["failure_stage"] == "kernel_timeout"
    assert (run_dir / "normalized_records.jsonl").read_text() == ""


def test_missing_response_is_failed_closed(tmp_path: Path) -> None:
    target = FakeNativeTarget(write_response=False)
    result = route.execute(
        tmp_path,
        suite_path=SUITE,
        environment=PHYSICAL_ENVIRONMENT,
        native_target=target,
    )
    run_dir = Path(result["run_dir"])
    failure = json.loads((run_dir / "session_failures.jsonl").read_text().splitlines()[0])
    assert result["status"] == "failed"
    assert failure["failure_stage"] == "output_manifest_failed"


def test_transfer_invariant_rejects_inconsistent_native_evidence(tmp_path: Path) -> None:
    result = route.execute(
        tmp_path,
        suite_path=SUITE,
        environment=PHYSICAL_ENVIRONMENT,
        native_target=FakeNativeTarget(bad_transfer=True),
    )
    failure = json.loads(
        (Path(result["run_dir"]) / "session_failures.jsonl").read_text().splitlines()[0]
    )
    assert result["status"] == "failed"
    assert failure["failure_stage"] == "output_manifest_failed"


def test_malformed_native_hash_is_not_admitted(tmp_path: Path) -> None:
    result = route.execute(
        tmp_path,
        suite_path=SUITE,
        environment=PHYSICAL_ENVIRONMENT,
        native_target=FakeNativeTarget(bad_hash=True),
    )
    failure = json.loads(
        (Path(result["run_dir"]) / "session_failures.jsonl").read_text().splitlines()[0]
    )
    assert result["status"] == "failed"
    assert failure["failure_stage"] == "output_manifest_failed"


def test_nonzero_native_returncode_is_not_admitted(tmp_path: Path) -> None:
    result = route.execute(
        tmp_path,
        suite_path=SUITE,
        environment=PHYSICAL_ENVIRONMENT,
        native_target=FakeNativeTarget(returncode=1),
    )
    failure = json.loads(
        (Path(result["run_dir"]) / "session_failures.jsonl").read_text().splitlines()[0]
    )
    assert result["status"] == "failed"
    assert failure["failure_stage"] == "output_manifest_failed"


@pytest.mark.parametrize("status", [None, "failed"])
def test_missing_or_failed_session_validation_is_not_admitted(
    tmp_path: Path, status: str | None
) -> None:
    result = route.execute(
        tmp_path,
        suite_path=SUITE,
        environment=PHYSICAL_ENVIRONMENT,
        native_target=FakeNativeTarget(session_validation_status=status),
    )
    failure = json.loads(
        (Path(result["run_dir"]) / "session_failures.jsonl").read_text().splitlines()[0]
    )
    assert result["status"] == "failed"
    assert failure["failure_stage"] in {"output_manifest_failed", "output_validation_failed"}


def test_prepare_build_requires_explicit_no_hardware_flags(tmp_path: Path) -> None:
    with pytest.raises(route.NativeExecutionFailure, match="must not allocate"):
        route.prepare(
            tmp_path,
            suite_path=SUITE,
            build=True,
            native_target=FakeNativeTarget(unsafe_prepare=True),
        )


def test_public_make_targets_have_dry_commands() -> None:
    completed = subprocess.run(
        ["make", "-n", "upmem-simplepim-plan", "upmem-simplepim-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "upmem-simplepim-taskgraph" in completed.stdout
