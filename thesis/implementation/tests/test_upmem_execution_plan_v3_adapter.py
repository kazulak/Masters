from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.targets.upmem import execution_plan_v3 as v3
from quantum_bench.targets.upmem.m5_task_selection import select_highest_work_supported_task
from quantum_bench.tn import build_tensor_network, plan_task_graph


def _build_metadata(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    host = root / "host_v3_t3"
    dpu = root / "dpu_resident_v3_t3"
    initialization = root / "dpu_simplepim_management_init"
    host.write_bytes(b"fake-host")
    dpu.write_bytes(b"fake-dpu")
    initialization.write_bytes(b"fake-initialization")
    return {
        "host_binary": str(host),
        "dpu_binary": str(dpu),
        "initialization_binary": str(initialization),
        "initialization_binary_sha256": hashlib.sha256(initialization.read_bytes()).hexdigest(),
    }


def test_v3_native_runner_path_loads_from_implementation_root() -> None:
    runner = v3._native_runner()
    expected = Path(v3.__file__).resolve().parents[4] / "native/upmem/simplepim/upmem_sdk_execution_plan_runner.py"
    assert Path(runner.__file__).resolve() == expected


def test_real_selection_builds_without_retained_network(tmp_path: Path) -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    graph = plan_task_graph(network)
    selection = select_highest_work_supported_task(graph, network)
    request = v3.prepare_request(
        case={"case_id": "bell_2q_real", "quantum_case": "real_circuit"},
        materialized={"selection_object": selection},
        dpu_count=1,
        tasklets=3,
        quantization_mode="none",
        partition_strategy="output",
        build=_build_metadata(tmp_path / "build"),
        root=tmp_path / "real",
    )
    manifest = json.loads(Path(request["resident_manifest"]).read_text(encoding="utf-8"))
    assert manifest["requested_dpus"] == 1
    assert manifest["tasklets"] == 3
    assert manifest["hardware_profile_version"] == v3.NATIVE_V3_MANIFEST_PROFILE_VERSION
    assert manifest["resident_v3_profile_version"] == v3.RESIDENT_V3_PROFILE_VERSION
    assert request["task_id"] == selection.task_id
    assert request["simplepim_role"] == "initialization_binary_and_management_state_only"
    assert Path(request["initialization_binary"]).name == "dpu_simplepim_management_init"
    assert request["initialization_binary_sha256"] == hashlib.sha256(
        Path(request["initialization_binary"]).read_bytes()
    ).hexdigest()
    for field in (
        "package_circuit_semantics_hash",
        "package_tensor_network_hash",
        "package_contraction_plan_hash",
    ):
        assert len(request[field]) == 64


def test_synthetic_weak_resolves_shape_and_int8_code_without_allocation(tmp_path: Path) -> None:
    case = {
        "case_id": "synthetic_weak",
        "quantum_case": "non_quantum",
        "non_quantum": True,
        "diagnostic": "weak_scaling",
        "matrix_shapes": [["4*dpu_count", 256], [256, 64]],
    }
    request = v3.prepare_request(
        case=case,
        materialized={"status": "selected"},
        dpu_count=3,
        tasklets=3,
        quantization_mode="per_task_resident_requantize",
        partition_strategy="output",
        build=_build_metadata(tmp_path / "build"),
        root=tmp_path,
    )
    manifest = json.loads(Path(request["resident_manifest"]).read_text(encoding="utf-8"))
    plan = json.loads(Path(request["distributed_plan_json"]).read_text(encoding="utf-8"))
    assert manifest["operation_abi_version"] == 2
    assert manifest["requested_dpus"] == 3
    assert manifest["tasklets"] == 3
    assert plan["total_output_elements"] == 12 * 64
    assert plan["numeric_mode"] == "per_task_resident_requantize"
    assert "allocation" not in manifest
    assert manifest["package_parse_timing_boundary"].endswith("before_dpu_alloc")
    assert request["numeric_mode"] == "per_task_resident_requantize"
    assert request["dpu_count"] == request["requested_dpus"] == 3
    assert request["tasklets"] == request["tasklets_per_dpu"] == 3
    assert request["non_quantum"] is True
    assert v3.validate_request(request)["dpu_count"] == 3
    assert request["output_path"]
    assert request["response_path"]
    policy_reference = request["policy_reference"]
    full_precision_reference = request["full_precision_reference"]
    assert policy_reference["path"] != full_precision_reference["path"]
    assert policy_reference["sha256"] == hashlib.sha256(
        Path(policy_reference["path"]).read_bytes()
    ).hexdigest()
    assert full_precision_reference["sha256"] == hashlib.sha256(
        Path(full_precision_reference["path"]).read_bytes()
    ).hexdigest()
    assert request["policy_reference_metadata"]["quantization_mode"] == "per_task_resident_requantize"
    assert request["full_precision_reference_metadata"]["quantization_mode"] == "none"
    assert policy_reference["max_abs_tolerance"] == pytest.approx(1.0e-5)
    assert request["full_precision_reference"]["required"] is False
    assert request["scaling_kind"] == "weak_scaling"
    assert request["host_binary_hash"] == request["host_binary_sha256"]
    assert request["dpu_binary_hash"] == request["dpu_binary_sha256"]
    assert request["simplepim_role"] == "initialization_binary_and_management_state_only"
    assert request["initialization_binary_sha256"] == hashlib.sha256(
        Path(request["initialization_binary"]).read_bytes()
    ).hexdigest()
    assert request["collective_provider"] == "none"


def test_missing_dpu_binary_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        v3._stage_dpu_binary(tmp_path / "missing", tmp_path)


def test_stale_initialization_binary_hash_is_rejected(tmp_path: Path) -> None:
    build = _build_metadata(tmp_path / "build")
    build["initialization_binary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="initialization binary SHA-256"):
        v3.prepare_request(
            case={"case_id": "stale_init", "quantum_case": "non_quantum", "non_quantum": True,
                  "matrix_shapes": [[4, 4], [4, 2]]},
            materialized={"status": "selected"},
            dpu_count=1,
            tasklets=3,
            quantization_mode="none",
            partition_strategy="output",
            build=build,
            root=tmp_path / "request",
        )


def test_v3_build_wrapper_loads_runner_from_file(monkeypatch, tmp_path: Path) -> None:
    class FakeRunner:
        def build(self, build_dir: Path, *, tasklets_per_dpu: int, environment: object) -> dict[str, str]:
            host = build_dir / "host"
            dpu = build_dir / "dpu"
            initialization = build_dir / "dpu_simplepim_management_init"
            build_dir.mkdir(parents=True, exist_ok=True)
            host.write_bytes(b"host")
            dpu.write_bytes(b"dpu")
            initialization.write_bytes(b"initialization")
            return {
                "host_binary": str(host),
                "dpu_binary": str(dpu),
                "initialization_binary": str(initialization),
            }

    monkeypatch.setattr(v3, "_native_runner", lambda: FakeRunner())
    result = v3.build(tmp_path, tasklets=3, environment={})
    assert result["tasklets_per_dpu"] == 3
    assert result["max_elements"] == 65536
    assert result["initialization_binary_sha256"] == hashlib.sha256(
        (tmp_path / "dpu_simplepim_management_init").read_bytes()
    ).hexdigest()
