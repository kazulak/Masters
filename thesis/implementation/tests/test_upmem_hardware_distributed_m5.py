from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest
import numpy as np

from quantum_bench.bench import upmem_hardware_distributed_m5 as m5
from quantum_bench.targets.upmem.distributed_plan_v3 import UnsupportedPartitionError


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "suites" / "upmem_hardware_distributed_m5.yml"


class FakeM5NativeTarget:
    def __init__(self) -> None:
        self.environment: dict[str, str] | None = None
        self.build_calls = 0
        self.validate_calls = 0
        self.execute_calls = 0
        self.requests: list[Mapping[str, Any]] = []
        self.full_precision_passed = True

    def set_environment(self, environment: Mapping[str, str]) -> None:
        self.environment = dict(environment)

    def build(self, build_dir: Path, *, tasklets: int) -> Mapping[str, Any]:
        self.build_calls += 1
        build_dir.mkdir(parents=True)
        return {"host_binary": str(build_dir / f"host_v3_t{tasklets}")}

    def prepare_request(self, **kwargs: Any) -> Mapping[str, Any]:
        request = {
            "dpu_count": kwargs["dpu_count"],
            "tasklets": kwargs["tasklets"],
            "quantization_mode": kwargs["quantization_mode"],
            "partition_strategy": kwargs["partition_strategy"],
            "response_path": str(kwargs["root"] / "response.json"),
            "distributed_plan": str(kwargs["root"] / "distributed_plan_v3.bin"),
            "package_sha256": "c" * 64,
            "sidecar_sha256": "d" * 64,
            "host_binary_sha256": "e" * 64,
            "dpu_binary_sha256": "f" * 64,
            "initialization_binary": str(kwargs["root"] / "dpu_simplepim_management_init"),
            "initialization_binary_sha256": "g" * 64,
            "execution_plan_hash": f"plan-{kwargs['dpu_count']}-{kwargs['partition_strategy']}",
            "policy_reference": {
                "path": str(kwargs["root"] / "policy_reference_f32.bin"),
                "sha256": "a" * 64,
                "max_abs_tolerance": 1.0e-5,
            },
            "full_precision_reference": {
                "path": str(kwargs["root"] / "full_precision_reference_f32.bin"),
                "sha256": "b" * 64,
                "max_abs_tolerance": 1.0e-5,
                "required": kwargs["quantization_mode"] == "none",
            },
        }
        self.requests.append(request)
        return request

    def validate(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]:
        self.validate_calls += 1
        return {
            "schema_version": m5.NATIVE_RESPONSE_SCHEMA,
            "status": "validated",
            "target_observed": "not_allocated",
            "requested_dpu_count": request["dpu_count"],
            "allocated_dpu_count": 0,
            "tasklets_per_dpu": request["tasklets"],
        }

    def execute(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]:
        self.execute_calls += 1
        contracted = request["partition_strategy"] == "contracted"
        int8_requantization = request["quantization_mode"] == "per_task_resident_requantize"
        repetitions = [
            {
                "warmup": True,
                "repeat_id": index,
                "total_time_s": 0.01,
                "transfers": {"h2d_bytes": 10 + index, "d2h_bytes": 2},
            }
            for index in range(m5.WARMUPS)
        ] + [
            {
                "warmup": False,
                "repeat_id": index,
                "total_time_s": 0.02 + index / 1000,
                "transfers": {"h2d_bytes": 20 + index, "d2h_bytes": 4},
            }
            for index in range(m5.REPEATS)
        ]
        return {
            "schema_version": m5.NATIVE_RESPONSE_SCHEMA,
            "status": "completed",
            "failure_stage": None,
            "error": None,
            "target_requested": "hardware",
            "target_observed": "physical_hardware",
            "requested_dpu_count": request["dpu_count"],
            "allocated_dpu_count": request["dpu_count"],
            "observed_rank_count": 1,
            "tasklets_per_dpu": request["tasklets"],
            "requested_warmups": m5.WARMUPS,
            "requested_repetitions": m5.REPEATS,
            "cpu_fallback_used": False,
            "simulator_kernel_executed": False,
            "fallback_used": False,
            "partition_strategy": request["partition_strategy"],
            "numeric_mode": (
                "per_task_resident_requantize" if int8_requantization else "float32"
            ),
            "numeric_arithmetic": "int8_requantized" if int8_requantization else "float32",
            "numeric_transport": "float32_mram",
            "requantization_scope": "per_task_on_dpu" if int8_requantization else "none",
            "packed_int8_transfer": False,
            "allocation_provider": "upmem_sdk_rank_profile_v1",
            "simplepim_role": "initialization_binary_and_management_state_only",
            "kernel_provider": "thesis_resident_generic_c_v3",
            "transfer_provider": "upmem_sdk_synchronous_v1",
            "collective_provider": "host_mediated_sum_v1" if contracted else "none",
            "reconstruction_provider": (
                "host_float64_reduction_v1"
                if contracted
                else "host_owned_range_assembly_v1"
            ),
            "hardware_allocation_verified": True,
            "native_kernel_executed": True,
            "hardware_kernel_executed": True,
            "hardware_release_verified": True,
            "allocation": {
                "attempted": True,
                "confirmed": True,
                "release_attempted": True,
                "release_confirmed": True,
            },
            "persistent_session_reused": True,
            "repetitions": repetitions,
            "run_total_transfers": {"h2d_bytes": 100, "d2h_bytes": 20},
            "load_balance": {"ratio": 1.0},
            "validation": {"reference_error": 0.0, "output_error": 0.0},
            "policy_reference_validation": {
                "passed": True,
                "status": "passed",
                "reference_sha256": request["policy_reference"]["sha256"],
                "max_abs_error": 0.0,
            },
            "full_precision_accuracy": {
                "passed": self.full_precision_passed,
                "status": "passed" if self.full_precision_passed else "failed",
                "reference_kind": "cpu_full_precision_float32_reference",
                "max_abs_error": 0.0,
                "l2_error": 0.0,
                "relative_l2_error": 0.0,
            },
            "package_file_sha256": request["package_sha256"],
            "distributed_plan_v3_sha256": request["sidecar_sha256"],
            "host_binary_sha256": request["host_binary_sha256"],
            "staged_dpu_binary_sha256": request["dpu_binary_sha256"],
            "initialization_binary_sha256": request["initialization_binary_sha256"],
            "launch_attempted": True,
            "launch_count": 1,
        }


def _selection(case: Mapping[str, Any], **_: Any) -> Mapping[str, Any]:
    return {
        "status": "materialized",
        "circuit_semantics_hash": f"circuit-{case['case_id']}",
        "tensor_network_hash": f"network-{case['case_id']}",
        "contraction_plan_hash": f"contraction-{case['case_id']}",
        "task_hash": f"task-{case['case_id']}",
    }


def _physical_env() -> dict[str, str]:
    return {
        "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
        "UPMEM_HW_RANK_PATH": "/dev/dpu_rank1",
    }


def test_default_executor_binds_prepared_policy_reference_and_computes_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "policy_reference_f32.bin"
    full_precision_path = tmp_path / "full_precision_reference_f32.bin"
    output_path = tmp_path / "output.bin"
    response_path = tmp_path / "response.json"
    np.asarray([1.0, 2.0], dtype="<f4").tofile(policy_path)
    np.asarray([1.0, 2.0], dtype="<f4").tofile(full_precision_path)
    np.asarray([1.0, 2.0], dtype="<f4").tofile(output_path)
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    full_precision_sha256 = hashlib.sha256(full_precision_path.read_bytes()).hexdigest()
    captured: dict[str, list[str]] = {}

    def fake_run(command: list[str], **_: Any) -> Any:
        captured["command"] = command
        response_path.write_text(json.dumps({
            "status": "completed",
            "policy_reference_validation": {
                "status": "passed",
                "passed": True,
                "reference_sha256": policy_sha256,
            },
        }), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(m5.subprocess, "run", fake_run)
    request = {
        "host_binary": str(tmp_path / "host"),
        "resident_manifest": str(tmp_path / "manifest.json"),
        "distributed_plan": str(tmp_path / "plan.bin"),
        "response_path": str(response_path),
        "output_path": str(output_path),
        "quantization_mode": "none",
        "policy_reference": {
            "path": str(policy_path), "sha256": policy_sha256, "max_abs_tolerance": 1.0e-5,
        },
        "full_precision_reference": {
            "path": str(full_precision_path), "sha256": full_precision_sha256,
            "max_abs_tolerance": 1.0e-5, "required": True,
        },
    }
    target = m5._DefaultNativeTarget()
    target.set_environment(_physical_env())
    response = target.execute(request, timeout_s=1.0)
    command = captured["command"]
    assert command[command.index("--policy-reference") + 1] == str(policy_path)
    assert command[command.index("--policy-reference-sha256") + 1] == policy_sha256
    assert command[command.index("--policy-tolerance") + 1] == "1e-05"
    assert response["full_precision_accuracy"]["max_abs_error"] == 0.0


def test_default_executor_rejects_mismatched_policy_reference_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_path = tmp_path / "response.json"

    def fake_run(*_: Any, **__: Any) -> Any:
        response_path.write_text(json.dumps({
            "status": "completed",
            "policy_reference_validation": {
                "status": "passed",
                "passed": True,
                "reference_sha256": "0" * 64,
            },
        }), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(m5.subprocess, "run", fake_run)
    target = m5._DefaultNativeTarget()
    request = {
        "host_binary": str(tmp_path / "host"),
        "resident_manifest": str(tmp_path / "manifest.json"),
        "distributed_plan": str(tmp_path / "plan.bin"),
        "response_path": str(response_path),
        "policy_reference": {
            "path": str(tmp_path / "policy.bin"), "sha256": "a" * 64, "max_abs_tolerance": 1.0e-5,
        },
    }
    with pytest.raises(m5.NativeExecutionError, match="did not pass") as failure:
        target.execute(request, timeout_s=1.0)
    assert failure.value.response["policy_reference_validation"]["reference_sha256"] == "0" * 64


def test_default_validator_accepts_not_run_policy_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_path = tmp_path / "response.json"

    def fake_run(*_: Any, **__: Any) -> Any:
        response_path.write_text(json.dumps({
            "schema_version": m5.NATIVE_RESPONSE_SCHEMA,
            "status": "validated",
            "target_observed": "not_allocated",
            "requested_dpu_count": 3,
            "allocated_dpu_count": 0,
            "tasklets_per_dpu": 3,
            "policy_reference_validation": {
                "status": "not_run",
                "passed": False,
                "reference_sha256": "",
            },
        }), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(m5.subprocess, "run", fake_run)
    target = m5._DefaultNativeTarget()
    request = {
        "host_binary": str(tmp_path / "host"),
        "resident_manifest": str(tmp_path / "manifest.json"),
        "distributed_plan": str(tmp_path / "plan.bin"),
        "response_path": str(response_path),
        "policy_reference": {
            "path": str(tmp_path / "policy.bin"),
            "sha256": "a" * 64,
            "max_abs_tolerance": 1.0e-5,
        },
    }

    response = target.validate(request, timeout_s=1.0)

    assert response["status"] == "validated"
    assert response["policy_reference_validation"]["status"] == "not_run"


def test_suite_defaults_and_overrides() -> None:
    default = m5.load_m5_suite(SUITE)
    assert default.dpu_counts == m5.DEFAULT_DPU_COUNTS
    assert default.tasklets == 8
    assert (default.warmups, default.repeats) == (2, 7)
    custom = m5.load_m5_suite(SUITE, dpu_counts="3,5,12", tasklets=24)
    assert custom.dpu_counts == (3, 5, 12)
    assert custom.tasklets == 24


def test_physical_admission_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hardware_opt_in_missing"):
        m5.execute(tmp_path, suite_path=SUITE, environment={})
    with pytest.raises(ValueError, match="hardware_rank_path_missing"):
        m5.execute(tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"})
    with pytest.raises(ValueError, match="physical hardware"):
        m5.execute(
            tmp_path,
            suite_path=SUITE,
            environment={**_physical_env(), "DPU_BACKEND": "simulator"},
        )


def test_prepare_has_no_native_execute_or_allocation(tmp_path: Path) -> None:
    target = FakeM5NativeTarget()
    result = m5.prepare(
        tmp_path,
        suite_path=SUITE,
        build=True,
        dpu_counts=(3,),
        native_target=target,
        task_selector=_selection,
    )
    assert result["status"] == "prepared"
    assert set(result) == {
        "plan_dir",
        "artifact",
        "status",
        "prepared_count",
        "unsupported_count",
        "failed_count",
        "dpu_allocation_attempted",
        "dpu_launch_attempted",
    }
    assert "plans" not in result
    assert "preparation_rows" not in result
    assert target.build_calls == 1
    assert target.validate_calls > 0
    assert target.execute_calls == 0
    payload = json.loads(Path(result["artifact"]).read_text(encoding="utf-8"))
    assert payload["dpu_allocation_attempted"] is False
    assert payload["dpu_launch_attempted"] is False
    assert payload["prepared_count"] > 0
    assert payload["unsupported_count"] == 0
    assert payload["failed_count"] == 0


def test_all_unsupported_execution_keeps_native_attempt_flags_false(tmp_path: Path) -> None:
    result = m5.execute(
        tmp_path,
        suite_path=SUITE,
        dpu_counts=(65,),
        environment=_physical_env(),
        native_target=FakeM5NativeTarget(),
        task_selector=_selection,
    )
    summary = json.loads(Path(result["artifact"]).read_text(encoding="utf-8"))

    assert summary["dpu_allocation_attempted"] is False
    assert summary["dpu_launch_attempted"] is False
    assert result["dpu_allocation_attempted"] is False
    assert result["dpu_launch_attempted"] is False


def test_fake_execution_writes_repeat_rows_and_preserves_identity(tmp_path: Path) -> None:
    target = FakeM5NativeTarget()
    result = m5.execute(
        tmp_path,
        suite_path=SUITE,
        dpu_counts=(3, 5),
        tasklets=12,
        environment=_physical_env(),
        native_target=target,
        task_selector=_selection,
    )
    assert result["status"] == "completed"
    rows = [
        json.loads(line)
        for line in (Path(result["run_dir"]) / "normalized_records.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 5 * 2 * 2 * 2 * 7
    assert {row["requested_dpu_count"] for row in rows} == {3, 5}
    assert {row["quantization_mode"] for row in rows} == set(m5.QUANTIZATION_MODES)
    assert {row["partition_strategy"] for row in rows} == set(m5.PARTITION_STRATEGIES)
    assert all(row["cpu_fallback_used"] is False for row in rows)
    assert all(row["claims"]["speedup"] is False for row in rows)
    assert all(row["circuit_semantics_hash"] for row in rows)
    assert all(row["tensor_network_hash"] for row in rows)
    assert all(row["contraction_plan_hash"] for row in rows)
    assert all(row["task_hash"] for row in rows)
    assert all(row["per_repeat_timing"]["total_time_s"] > 0 for row in rows)
    assert {row["scaling_kind"] for row in rows} == {"strong_scaling", "weak_scaling"}
    assert all(row["transfers"]["h2d_bytes"] != 100 for row in rows)
    assert all(row["run_metadata"]["transfers"]["h2d_bytes"] == 100 for row in rows)
    assert all(row["policy_reference_validation"]["passed"] is True for row in rows)
    assert all(isinstance(row["policy_reference_validation"]["max_abs_error"], (float, int)) for row in rows)
    float32 = [row for row in rows if row["quantization_mode"] == "none"]
    int8 = [row for row in rows if row["quantization_mode"] == "per_task_resident_requantize"]
    assert float32 and all(isinstance(row["full_precision_accuracy"]["max_abs_error"], (float, int)) for row in float32)
    assert all(row["quantization_error_vs_float32"] is None for row in float32)
    assert int8 and all(isinstance(row["quantization_error_vs_float32"]["max_abs_error"], (float, int)) for row in int8)
    assert all(row["observed_rank_count"] == 1 for row in rows)


@pytest.mark.parametrize(
    ("quantization_mode", "expected"),
    (
        (
            "per_task_resident_requantize",
            {
                "quantization_mode": "per_task_resident_requantize",
                "numeric_arithmetic": "int8_requantized",
                "numeric_transport": "float32_mram",
                "requantization_scope": "per_task_on_dpu",
                "packed_int8_transfer": False,
            },
        ),
        (
            "none",
            {
                "quantization_mode": "none",
                "numeric_arithmetic": "float32",
                "numeric_transport": "float32_mram",
                "requantization_scope": "none",
                "packed_int8_transfer": False,
            },
        ),
    ),
)
def test_completed_rows_have_canonical_numeric_evidence(
    quantization_mode: str,
    expected: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    target = FakeM5NativeTarget()
    request = target.prepare_request(
        dpu_count=3,
        tasklets=m5.DEFAULT_TASKLETS,
        quantization_mode=quantization_mode,
        partition_strategy="output",
        root=tmp_path,
    )
    response = target.execute(request, timeout_s=1.0)
    config = m5.load_m5_suite(SUITE)
    m5._validate_execute_response(response, request, config)
    plan = {
        "case_id": "numeric-evidence",
        "workload_id": "numeric-evidence",
        "benchmark_role": "m5_distributed_hardware",
        "quantum_case": "real_circuit",
        "partition_strategy": "output",
        "quantization_mode": quantization_mode,
        "requested_dpu_count": 3,
        "tasklets_per_dpu": m5.DEFAULT_TASKLETS,
        "request": request,
    }

    row = m5._normalized_record(
        plan,
        response,
        response["repetitions"][m5.WARMUPS],
        0,
        _physical_env(),
    )

    assert row["status"] == "completed"
    assert {field: row[field] for field in expected} == expected
    assert row["initialization_binary"] == request["initialization_binary"]
    assert row["initialization_binary_sha256"] == request["initialization_binary_sha256"]


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("numeric_mode", "float32"),
        ("numeric_arithmetic", "float32"),
        ("numeric_transport", "packed_int8"),
        ("requantization_scope", "none"),
        ("packed_int8_transfer", True),
    ),
)
def test_success_acceptance_rejects_mismatched_numeric_evidence(
    field: str,
    invalid: Any,
    tmp_path: Path,
) -> None:
    target = FakeM5NativeTarget()
    request = target.prepare_request(
        dpu_count=3,
        tasklets=m5.DEFAULT_TASKLETS,
        quantization_mode="per_task_resident_requantize",
        partition_strategy="output",
        root=tmp_path,
    )
    response = dict(target.execute(request, timeout_s=1.0))
    response[field] = invalid

    with pytest.raises(ValueError, match=field):
        m5._validate_execute_response(response, request, m5.load_m5_suite(SUITE))


def test_failed_native_response_is_preserved_as_structured_evidence(tmp_path: Path) -> None:
    class FailingTarget(FakeM5NativeTarget):
        def execute(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]:
            self.execute_calls += 1
            raise m5.NativeExecutionError(
                "native kernel launch failed",
                response={
                    "schema_version": m5.NATIVE_RESPONSE_SCHEMA,
                    "status": "failed",
                    "failure_stage": "kernel_launch_failed",
                    "error": "sdk launch error",
                    "allocated_dpu_count": 2,
                    "observed_rank_count": 1,
                    "hardware_allocation_verified": False,
                    "native_kernel_executed": False,
                    "hardware_kernel_executed": False,
                    "hardware_release_verified": True,
                    "cpu_fallback_used": False,
                    "simulator_kernel_executed": False,
                    "fallback_used": False,
                    "allocation": {
                        "attempted": True,
                        "confirmed": False,
                        "release_attempted": True,
                        "release_confirmed": True,
                    },
                    "launch_attempted": True,
                    "launch_count": 0,
                },
                returncode=1,
            )

    result = m5.execute(
        tmp_path,
        suite_path=SUITE,
        dpu_counts=(3,),
        environment=_physical_env(),
        native_target=FailingTarget(),
        task_selector=_selection,
    )
    rows = [
        json.loads(line)
        for line in (Path(result["run_dir"]) / "normalized_records.jsonl").read_text().splitlines()
    ]

    assert rows
    assert all(row["failure_stage"] == "kernel_launch_failed" for row in rows)
    assert all(row["allocated_dpu_count"] == 2 for row in rows)
    assert all(row["allocation"]["release_attempted"] is True for row in rows)
    assert all(row["hardware_release_verified"] is True for row in rows)
    assert result["dpu_allocation_attempted"] is True
    assert result["dpu_launch_attempted"] is True
    assert all(row["native_response"]["error"] == "sdk launch error" for row in rows)


@pytest.mark.parametrize(
    "response_field",
    (
        "package_file_sha256",
        "distributed_plan_v3_sha256",
        "host_binary_sha256",
        "staged_dpu_binary_sha256",
        "initialization_binary_sha256",
    ),
)
def test_success_acceptance_requires_prepared_native_artifact_hashes(
    response_field: str, tmp_path: Path,
) -> None:
    target = FakeM5NativeTarget()
    m5.prepare(
        tmp_path,
        suite_path=SUITE,
        build=True,
        dpu_counts=(3,),
        native_target=target,
        task_selector=_selection,
    )
    request = target.requests[0]
    response = dict(target.execute(request, timeout_s=1.0))
    response[response_field] = "0" * 64

    with pytest.raises(ValueError, match=response_field):
        m5._validate_execute_response(response, request, m5.load_m5_suite(SUITE))


def test_capacity_is_an_explicit_unsupported_row(tmp_path: Path) -> None:
    target = FakeM5NativeTarget()
    result = m5.execute(
        tmp_path,
        suite_path=SUITE,
        dpu_counts=(65,),
        environment=_physical_env(),
        native_target=target,
        task_selector=_selection,
    )
    rows = [
        json.loads(line)
        for line in (Path(result["run_dir"]) / "normalized_records.jsonl").read_text().splitlines()
    ]
    assert rows
    assert all(row["status"] == "unsupported" for row in rows)
    assert all(row["failure_stage"] == "capacity" for row in rows)
    assert target.execute_calls == 0


def test_prepare_keeps_mixed_capacity_results_prepared(tmp_path: Path) -> None:
    result = m5.prepare(
        tmp_path,
        suite_path=SUITE,
        build=True,
        dpu_counts=(3, 65),
        native_target=FakeM5NativeTarget(),
        task_selector=_selection,
    )
    assert result["status"] == "prepared"
    assert result["prepared_count"] > 0
    assert result["unsupported_count"] > 0
    assert result["failed_count"] == 0


def test_prepare_fails_when_every_plan_is_unsupported(tmp_path: Path) -> None:
    result = m5.prepare(
        tmp_path,
        suite_path=SUITE,
        build=True,
        dpu_counts=(65,),
        native_target=FakeM5NativeTarget(),
        task_selector=_selection,
    )
    assert result["status"] == "failed"
    assert result["prepared_count"] == 0
    assert result["unsupported_count"] > 0
    assert result["failed_count"] == 0


def test_partition_incompatibility_is_unsupported(tmp_path: Path) -> None:
    class PartitionLimitedTarget(FakeM5NativeTarget):
        def prepare_request(self, **kwargs: Any) -> Mapping[str, Any]:
            if kwargs["partition_strategy"] == "contracted":
                raise UnsupportedPartitionError(
                    "unsupported preparation: total work cannot give every requested DPU positive aligned work"
                )
            return super().prepare_request(**kwargs)

    result = m5.prepare(
        tmp_path,
        suite_path=SUITE,
        build=True,
        dpu_counts=(3,),
        native_target=PartitionLimitedTarget(),
        task_selector=_selection,
    )
    payload = json.loads(Path(result["artifact"]).read_text(encoding="utf-8"))
    rows = payload["preparation_rows"]
    unsupported = [row for row in rows if row["partition_strategy"] == "contracted"]
    assert unsupported
    assert all(row["status"] == "unsupported" for row in unsupported)
    assert all(row["failure_stage"] == "partition_incompatible" for row in unsupported)


def test_prepare_fails_closed_on_mixed_unexpected_error(tmp_path: Path) -> None:
    class BrokenTarget(FakeM5NativeTarget):
        def prepare_request(self, **kwargs: Any) -> Mapping[str, Any]:
            if kwargs["partition_strategy"] == "contracted":
                raise RuntimeError("native_contract_bug: descriptor identity drifted")
            return super().prepare_request(**kwargs)

    result = m5.prepare(
        tmp_path,
        suite_path=SUITE,
        build=True,
        dpu_counts=(3,),
        native_target=BrokenTarget(),
        task_selector=_selection,
    )

    assert result["status"] == "failed"
    assert result["prepared_count"] > 0
    assert result["failed_count"] > 0
    assert result["unsupported_count"] == 0


def test_full_precision_failure_is_mandatory_only_for_float32(tmp_path: Path) -> None:
    target = FakeM5NativeTarget()
    target.full_precision_passed = False
    result = m5.execute(
        tmp_path,
        suite_path=SUITE,
        dpu_counts=(3,),
        environment=_physical_env(),
        native_target=target,
        task_selector=_selection,
    )
    assert result["status"] == "failed"
    rows = [
        json.loads(line)
        for line in (Path(result["run_dir"]) / "normalized_records.jsonl").read_text().splitlines()
    ]
    float32 = [row for row in rows if row["quantization_mode"] == "none"]
    int8 = [row for row in rows if row["quantization_mode"] == "per_task_resident_requantize"]
    assert float32 and all(row["status"] == "failed" for row in float32)
    assert all(row["full_precision_accuracy"]["passed"] is False for row in float32)
    assert all(row["scientific_validation_status"] == "failed" for row in float32)
    assert int8 and all(row["status"] == "completed" for row in int8)
    assert all(row["full_precision_accuracy_status"] == "descriptive" for row in int8)
    assert all(row["scientific_validation_status"] == "passed_with_descriptive_quantization_difference" for row in int8)


def test_cli_and_make_targets_are_wired() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "quantum_bench.bench", "--help"],
        cwd=ROOT,
        env={"PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "upmem-hardware-distributed-m5" in completed.stdout
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("upmem-hw-m5-plan", "upmem-hw-m5", "upmem-hw-m5-report"):
        assert f"{target}:" in makefile
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-m5-plan", "UPMEM_HW_M5_DPU_COUNTS=3,5,12", "UPMEM_HW_M5_TASKLETS=12"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plan.returncode == 0
    assert "--dpu-counts 3,5,12" in plan.stdout
    assert "--tasklets 12" in plan.stdout
