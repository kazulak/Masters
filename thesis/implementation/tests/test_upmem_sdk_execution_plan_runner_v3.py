from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "native/upmem/simplepim/upmem_sdk_execution_plan_runner.py"
SPEC = importlib.util.spec_from_file_location("upmem_v3_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _units(dpu_count: int, *, output_elements: int = 128) -> list[tuple[int, ...]]:
    base = output_elements // dpu_count
    remainder = output_elements % dpu_count
    offset = 0
    result = []
    for dpu_id in range(dpu_count):
        elements = base + (dpu_id < remainder)
        result.append((0, 0, 1, dpu_id, offset, elements, 0, 4))
        offset += elements
    return result


def test_v3_pack_parse_supports_64_dpus_and_24_tasklets() -> None:
    package = b"package-v3"
    operation = b"operation-v3"
    payload = runner.pack_v3_sidecar(
        package_bytes=package,
        operation_bytes=operation,
        dpu_count=64,
        tasklets_per_dpu=24,
        partition_mode=runner.V3_PARTITION_OUTPUT,
        numeric_mode=runner.V3_NUMERIC_FLOAT32,
        output_elements=128,
        contracted_elements=4,
        output_slot=7,
        work_units=_units(64),
    )
    assert len(payload) == 136 + 64 * 32
    plan = runner.parse_v3_sidecar(payload, expected_tasklets=24)
    assert plan.dpu_count == 64
    assert plan.tasklets_per_dpu == 24
    assert plan.work_units[-1].dpu_id == 63
    assert plan.package_sha256 == hashlib.sha256(package).digest()


@pytest.mark.parametrize("dpu_count", (3, 5))
@pytest.mark.parametrize("tasklets", (3, 24))
def test_v3_pack_parse_allows_arbitrary_bounded_dpus_and_tasklets(dpu_count: int, tasklets: int) -> None:
    output_elements = dpu_count * 8
    payload = runner.pack_v3_sidecar(
        package_bytes=b"package", operation_bytes=b"operation", dpu_count=dpu_count,
        tasklets_per_dpu=tasklets, partition_mode=runner.V3_PARTITION_OUTPUT,
        numeric_mode=runner.V3_NUMERIC_FLOAT32, output_elements=output_elements,
        contracted_elements=4, output_slot=0, work_units=_units(dpu_count, output_elements=output_elements),
    )
    plan = runner.parse_v3_sidecar(payload, expected_tasklets=tasklets)
    assert plan.dpu_count == dpu_count
    assert plan.tasklets_per_dpu == tasklets


def test_v3_contracted_partition_and_int8_mode_are_distinct() -> None:
    units = [
        (0, 0, 2, 0, 0, 16, 0, 2),
        (0, 0, 2, 1, 0, 16, 2, 2),
    ]
    payload = runner.pack_v3_sidecar(
        package_bytes=b"p",
        operation_bytes=b"o",
        dpu_count=2,
        tasklets_per_dpu=1,
        partition_mode=runner.V3_PARTITION_CONTRACTED,
        numeric_mode=runner.V3_NUMERIC_INT8_REQUANTIZE,
        output_elements=16,
        contracted_elements=4,
        output_slot=3,
        work_units=units,
    )
    plan = runner.parse_v3_sidecar(payload)
    assert plan.partition_mode == runner.V3_PARTITION_CONTRACTED
    assert plan.numeric_mode == runner.V3_NUMERIC_INT8_REQUANTIZE
    assert runner.V3_NUMERIC_NAMES[plan.numeric_mode] == "per_task_resident_requantize"


def test_v3_rejects_odd_nonterminal_output_boundary() -> None:
    with pytest.raises(runner.V3RunnerError, match="aligned"):
        runner.pack_v3_sidecar(
            package_bytes=b"p", operation_bytes=b"o", dpu_count=2, tasklets_per_dpu=1,
            partition_mode=runner.V3_PARTITION_OUTPUT, numeric_mode=runner.V3_NUMERIC_FLOAT32,
            output_elements=5, contracted_elements=1, output_slot=0,
            work_units=[(0, 0, 1, 0, 0, 3, 0, 1), (0, 0, 1, 1, 3, 2, 0, 1)],
        )


def test_execute_fails_closed_without_physical_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    with pytest.raises(runner.V3RunnerError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE"):
        runner.execute(
            tmp_path / "host", resident_manifest=tmp_path / "manifest.json",
            sidecar=tmp_path / "plan.bin", response=tmp_path / "response.json",
            tasklets_per_dpu=1, warmups=2, environment={},
        )


def test_execute_accepts_two_warmups_and_sixteen_repetitions_with_passed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = tmp_path / "response.json"
    reference = tmp_path / "reference_f32.bin"
    reference.write_bytes(b"reference")
    captured: dict[str, object] = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> object:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        response.write_text(json.dumps({
            "status": "completed", "hardware_kernel_executed": True, "native_kernel_executed": True,
            "hardware_release_verified": True, "hardware_allocation_verified": True,
            "simulator_kernel_executed": False, "cpu_fallback_used": False,
            "requested_rank_path": "/dev/dpu_rank1", "observed_rank_count": 1,
            "partition_strategy": "output",
            "allocation_provider": "upmem_sdk_rank_profile_v1",
            "simplepim_role": "initialization_binary_and_management_state_only",
            "kernel_provider": "thesis_resident_generic_c_v3",
            "transfer_provider": "upmem_sdk_synchronous_v1",
            "collective_provider": "none",
            "reconstruction_provider": "host_owned_range_assembly_v1",
            "policy_reference_validation": {
                "status": "passed", "passed": True, "max_abs_error": 0.0,
                "tolerance": 1.0e-5, "finite": True,
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            },
            "host_binary_sha256": "host", "staged_dpu_binary_sha256": "dpu",
        }), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    payload = runner.execute(
        tmp_path / "host", resident_manifest=tmp_path / "manifest.json",
        sidecar=tmp_path / "plan.bin", response=response, tasklets_per_dpu=3,
        warmups=2, repetitions=16, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_HW_RANK_PATH": "/dev/dpu_rank1"},
        policy_reference=reference,
    )
    command = captured["command"]
    assert isinstance(command, tuple)
    assert "--warmups" in command and command[command.index("--warmups") + 1] == "2"
    assert "--repetitions" in command and command[command.index("--repetitions") + 1] == "16"
    assert captured["env"] == {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_HW_RANK_PATH": "/dev/dpu_rank1"}
    assert payload["hardware_kernel_executed"] is True


def test_execute_rejects_completed_response_without_policy_passed_boolean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = tmp_path / "response.json"
    reference = tmp_path / "reference_f32.bin"
    reference.write_bytes(b"reference")

    def fake_run(*args: object, **kwargs: object) -> object:
        response.write_text(json.dumps({
            "status": "completed", "hardware_kernel_executed": True, "native_kernel_executed": True,
            "hardware_release_verified": True, "hardware_allocation_verified": True,
            "simulator_kernel_executed": False, "cpu_fallback_used": False,
            "requested_rank_path": "/dev/dpu_rank1", "observed_rank_count": 1,
            "partition_strategy": "output",
            "allocation_provider": "upmem_sdk_rank_profile_v1",
            "simplepim_role": "initialization_binary_and_management_state_only",
            "kernel_provider": "thesis_resident_generic_c_v3",
            "transfer_provider": "upmem_sdk_synchronous_v1",
            "collective_provider": "none",
            "reconstruction_provider": "host_owned_range_assembly_v1",
            "policy_reference_validation": {
                "status": "passed", "max_abs_error": 0.0, "tolerance": 1.0e-5,
                "finite": True, "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            },
        }), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(runner.V3RunnerError, match="policy reference"):
        runner.execute(
            tmp_path / "host", resident_manifest=tmp_path / "manifest.json", sidecar=tmp_path / "plan.bin",
            response=response, tasklets_per_dpu=1, policy_reference=reference,
            environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_HW_RANK_PATH": "/dev/dpu_rank1"},
        )


def test_execute_rejects_completed_response_without_policy_or_rank_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = tmp_path / "response.json"
    reference = tmp_path / "reference_f32.bin"
    reference.write_bytes(b"reference")

    def fake_run(*args: object, **kwargs: object) -> object:
        response.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(runner.V3RunnerError, match="simulator or CPU fallback"):
        runner.execute(
            tmp_path / "host", resident_manifest=tmp_path / "manifest.json", sidecar=tmp_path / "plan.bin",
            response=response, tasklets_per_dpu=1, policy_reference=reference,
            environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_HW_RANK_PATH": "/dev/dpu_rank1"},
        )
