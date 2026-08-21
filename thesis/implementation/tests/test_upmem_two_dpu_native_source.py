from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    build_resident_graph_package,
)
from quantum_bench.targets.upmem.hardware_taskgraph_sliced_resident import (
    build_two_slice_resident_graph_packages,
    build_two_slice_resident_plan,
    write_two_slice_resident_graph_packages,
)
from quantum_bench.tn.execution_bundle import with_execution_identity
from quantum_bench.tn.network import TensorNetworkValue


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
SIMPLEPIM_ROOT = IMPLEMENTATION_ROOT / "native" / "upmem" / "simplepim"
TWO_DPU_ROOT = SIMPLEPIM_ROOT / "upmem_sdk_generic_loop_resident_two_dpu"
ONE_DPU_ROOT = SIMPLEPIM_ROOT / "upmem_sdk_generic_loop_resident"
HOST_BINARY = TWO_DPU_ROOT / "bin" / "host_two_dpu"
FNV1A64_OFFSET = 14695981039346656037
FNV1A64_PRIME = 1099511628211


def _canonical_fnv1a64(payload: bytes) -> str:
    value = FNV1A64_OFFSET
    for byte in payload:
        value ^= byte
        value = (value * FNV1A64_PRIME) & ((1 << 64) - 1)
    return f"{value:016x}"


def _rewrite_canonical_fnv1a64(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = manifest["slice_execution"]
    execution["resident_descriptor_fnv1a64"] = _canonical_fnv1a64(
        (manifest_path.parent / manifest["package_path"]).read_bytes()
    )
    execution["restricted_input_fnv1a64"] = {
        entry["input_path"]: _canonical_fnv1a64(
            (manifest_path.parent / entry["input_path"]).read_bytes()
        )
        for entry in manifest["initial_slots"]
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _one_operation_real_package() -> tuple[TaskGraph, TensorNetworkValue]:
    left = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    right = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    circuit = CircuitSpec("two_dpu_native_fixture", 0, (), {"kind": "fixture"})
    left_spec = TensorSpec("left", (0, 1), left.shape, "dense", dtype="float64")
    right_spec = TensorSpec("right", (1, 2), right.shape, "dense", dtype="float64")
    network_spec = TensorNetworkSpec(
        circuit, (left_spec, right_spec), (0, 2), "ab,bc->ac"
    )
    task = ContractionTask(
        id="one_contract",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=(left.shape, right.shape),
        output_shape=(2, 2),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=2,
        gemm_k=2,
        gemm_n=2,
        structure="dense",
        estimated_flops=16,
        estimated_bytes=0,
    )
    graph = with_execution_identity(
        TaskGraph(
            network_spec,
            (task,),
            ((0, 1),),
            PathSummary("fixture", "greedy", 1, 1, None, None, "fixture"),
            0.0,
        )
    )
    return graph, TensorNetworkValue(
        network_spec, [TensorValue(left_spec, left), TensorValue(right_spec, right)]
    )


def test_two_dpu_native_path_uses_two_distinct_contraction_packages() -> None:
    host = (TWO_DPU_ROOT / "host.c").read_text(encoding="utf-8")
    makefile = (TWO_DPU_ROOT / "Makefile").read_text(encoding="utf-8")

    assert (
        "dpu_alloc(RESIDENT_TWO_DPU_COUNT, RESIDENT_TWO_DPU_ALLOCATION_PROFILE, &set)"
        in host
    )
    assert '"backend=hw"' in host
    assert "allocated_dpus != RESIDENT_TWO_DPU_COUNT" in host
    assert "two_dpu_validate_slice_pair" in host
    assert "slice_packages_must_be_distinct" in host
    assert "dpu_copy_to" in host
    assert "dpu_broadcast_to" not in host
    assert "dpu_launch(set, DPU_ASYNCHRONOUS)" in host
    assert host.count("dpu_sync(set)") == 1
    assert "for (uint32_t operation_index = 0; operation_index < operation_count; operation_index++)" in host
    assert "completed_operation_count" in host
    assert "observed_operation_completion_count" in host
    assert "observed_operation_completion_counts" in host
    assert "operation_completion_confirmed" in host
    assert "device_launch_mode" in host
    assert "host_completion_mode" in host
    assert "dpu_written_completion_sentinel_read_after_each_sync" in host
    assert "RESIDENT_COMPLETION" in host
    assert "completion_sentinel_read_count" in host
    assert "clock_gettime(CLOCK_MONOTONIC" in host
    assert "sync_wait_is_not_pure_kernel_time" in host
    assert "python_sum_partials" in host
    assert "native_reconstruction_performed" in host
    assert "two_dpu_allocation" in host
    assert "RESIDENT_SLICE_CONTROL" not in host
    assert "slice_execution" in host
    assert "upmem_sliced_resident_execution_v1" in host
    assert "resident_descriptor_sha256" in host
    assert "restricted_input_sha256" in host
    assert "resident_descriptor_fnv1a64" in host
    assert "restricted_input_fnv1a64" in host
    assert "slice_descriptor_fingerprint_mismatch" in host
    assert "slice_restricted_input_fingerprint_mismatch" in host
    assert "14695981039346656037ULL" in host
    assert "resident_request_load" in host
    assert "number_end != end && *number_end != ','" in host
    assert "cpu_fallback_used" in host
    assert "requested_dpus" in host
    assert "asynchronous" in host
    assert "partial_output_transfer_bytes" in host
    assert "partial_output_raw_bytes" in host
    assert "completion_confirmed" in host
    assert "hardware_execution" in host
    assert "dpu_alloc(RESIDENT_TWO_DPU_COUNT" in host
    assert not (TWO_DPU_ROOT / "dpu.c").exists()
    assert not (TWO_DPU_ROOT / "session_protocol.c").exists()
    assert not (TWO_DPU_ROOT / "session_protocol.h").exists()
    assert "../upmem_sdk_generic_loop_resident/dpu.c" in makefile
    assert "../upmem_sdk_generic_loop_resident/session_protocol.c" in makefile
    assert (ONE_DPU_ROOT / "host.c").is_file()
    assert (ONE_DPU_ROOT / "dpu.c").is_file()


@pytest.mark.skipif(
    shutil.which("dpu-upmem-dpurte-clang") is None
    or shutil.which("dpu-pkg-config") is None,
    reason="UPMEM SDK compiler is unavailable",
)
@pytest.mark.parametrize(
    "quantization_mode", ("none", "per_task_resident_requantize")
)
def test_two_dpu_validate_slice_packages_accepts_genuine_one_contract_pair_via_shared_parser(
    tmp_path: Path, quantization_mode: str,
) -> None:
    subprocess.run(["make", "clean", "all"], cwd=TWO_DPU_ROOT, check=True)

    graph, network = _one_operation_real_package()
    plan = build_two_slice_resident_plan(graph, network)
    packages = build_two_slice_resident_graph_packages(
        plan,
        case_id="two_dpu_native_fixture",
        suite_id="two_dpu_native_fixture",
        quantization_mode=quantization_mode,
    )
    dpu_binary = tmp_path / "dpu_resident"
    dpu_binary.write_bytes(b"fixture")
    written = write_two_slice_resident_graph_packages(
        packages,
        tmp_path,
        dpu_binary=dpu_binary,
        request_id_prefix="long-input-path-" + "x" * 80,
    )
    first, second = (item.package for item in written)
    assert first.manifest_path is not None
    assert second.manifest_path is not None
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert len(first_manifest["initial_slots"][0]["input_path"]) > 95
    _rewrite_canonical_fnv1a64(first.manifest_path)
    _rewrite_canonical_fnv1a64(second.manifest_path)

    completed = subprocess.run(
        [
            str(HOST_BINARY),
            "--validate-slice-packages",
            str(first.manifest_path),
            str(second.manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    validation = json.loads(completed.stdout)
    assert validation == {
        "status": "valid",
        "reason": None,
        "slice_package_paths_distinct": True,
        "slice_package_hashes_distinct": True,
        "native_reconstruction_performed": False,
        "reconstruction_contract": "python_sum_partials",
    }
    assert first.manifest_path != second.manifest_path
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["slice_execution"]["slice_id"] == 0
    assert second_manifest["slice_execution"]["slice_id"] == 1
    assert (
        first_manifest["slice_execution"]["restricted_input_sha256"]
        != second_manifest["slice_execution"]["restricted_input_sha256"]
    )
    assert (
        first_manifest["slice_execution"]["resident_descriptor_sha256"]
        == second_manifest["slice_execution"]["resident_descriptor_sha256"]
    )
    assert first.package_path is not None
    assert second.package_path is not None
    assert (
        first_manifest["slice_execution"]["resident_descriptor_sha256"]
        == hashlib.sha256(first.package_path.read_bytes()).hexdigest()
    )
    assert (
        second_manifest["slice_execution"]["resident_descriptor_sha256"]
        == hashlib.sha256(second.package_path.read_bytes()).hexdigest()
    )
    assert first.final_output_paths["real"] != second.final_output_paths["real"]
    assert not first.final_output_paths["real"].exists()
    assert not second.final_output_paths["real"].exists()

    duplicate = subprocess.run(
        [
            str(HOST_BINARY),
            "--validate-slice-packages",
            str(first.manifest_path),
            str(first.manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode == 1
    assert json.loads(duplicate.stdout)["reason"] == "slice_packages_must_be_distinct"

    def validate(left: Path, right: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(HOST_BINARY), "--validate-slice-packages", str(left), str(right)],
            check=False,
            capture_output=True,
            text=True,
        )

    original_second = second.manifest_path.read_text(encoding="utf-8")
    if quantization_mode == "per_task_resident_requantize":
        mismatched_second = json.loads(original_second)
        mismatched_second["quantization_mode"] = "none"
        second.manifest_path.write_text(
            json.dumps(mismatched_second), encoding="utf-8"
        )
        mismatch = validate(first.manifest_path, second.manifest_path)
        assert mismatch.returncode == 1
        mismatch_result = json.loads(mismatch.stdout)
        assert mismatch_result["status"] == "invalid"
        assert mismatch_result["reason"] == "slice_manifest_parse_failed"
        second.manifest_path.write_text(original_second, encoding="utf-8")

    for mutate in (
        lambda manifest: manifest.pop("slice_execution"),
        lambda manifest: manifest["slice_execution"].update(
            {"slice_id": 0, "dpu_id": 0, "assignment_value": 0}
        ),
        lambda manifest: manifest["slice_execution"]["restrictions"].__setitem__(
            0, first_manifest["slice_execution"]["restrictions"][0]
        ),
        lambda manifest: manifest["slice_execution"]["source_hashes"].update(
            {"contraction_plan_hash": "0" * 64}
        ),
    ):
        mutated = json.loads(original_second)
        mutate(mutated)
        second.manifest_path.write_text(json.dumps(mutated), encoding="utf-8")
        assert validate(first.manifest_path, second.manifest_path).returncode == 1
    second.manifest_path.write_text(original_second, encoding="utf-8")

    missing_execution = json.loads(original_second)
    missing_execution.pop("slice_execution")
    second.manifest_path.write_text(json.dumps(missing_execution), encoding="utf-8")
    parse_failure = validate(first.manifest_path, second.manifest_path)
    assert parse_failure.returncode == 1
    assert json.loads(parse_failure.stdout)["reason"] == "slice_execution_parse_failed"
    second.manifest_path.write_text(original_second, encoding="utf-8")

    malformed_final_component = json.loads(original_second)
    malformed_final_component["final_outputs"][0]["component"] = "imag"
    second.manifest_path.write_text(
        json.dumps(malformed_final_component), encoding="utf-8"
    )
    assert validate(first.manifest_path, second.manifest_path).returncode == 1

    malformed_final_bytes = json.loads(original_second)
    malformed_final_bytes["final_outputs"][0]["transfer_bytes"] += 8
    second.manifest_path.write_text(json.dumps(malformed_final_bytes), encoding="utf-8")
    assert validate(first.manifest_path, second.manifest_path).returncode == 1
    second.manifest_path.write_text(original_second, encoding="utf-8")

    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(original_second)

    input_path = (
        first.manifest_path.parent / first_manifest["initial_slots"][0]["input_path"]
    )
    original_input = input_path.read_bytes()
    mutated_input = bytearray(original_input)
    mutated_input[0] ^= 0x01
    input_path.write_bytes(mutated_input)
    assert validate(first.manifest_path, second.manifest_path).returncode == 1
    input_path.write_bytes(original_input)

    stale_descriptor = json.loads(original_second)
    stale_descriptor["slice_execution"]["resident_descriptor_fnv1a64"] = "0" * 16
    second.manifest_path.write_text(json.dumps(stale_descriptor), encoding="utf-8")
    assert validate(first.manifest_path, second.manifest_path).returncode == 1

    swapped_assignment = json.loads(original_second)
    swapped_assignment["slice_execution"].update({"dpu_id": 0, "assignment_value": 0})
    for restriction in swapped_assignment["slice_execution"]["restrictions"]:
        restriction["value"] = 0
    second.manifest_path.write_text(json.dumps(swapped_assignment), encoding="utf-8")
    assert validate(first.manifest_path, second.manifest_path).returncode == 1

    overlapping = json.loads(original_second)
    overlapping_restrictions = overlapping["slice_execution"]["restrictions"]
    overlapping_restrictions[1] = dict(overlapping_restrictions[0])
    second.manifest_path.write_text(json.dumps(overlapping), encoding="utf-8")
    assert validate(first.manifest_path, second.manifest_path).returncode == 1

    malformed_uint = original_second.replace(
        '"assignment_value": 1', '"assignment_value": 0junk'
    )
    assert malformed_uint != original_second
    second.manifest_path.write_text(malformed_uint, encoding="utf-8")
    assert validate(first.manifest_path, second.manifest_path).returncode == 1
    second.manifest_path.write_text(original_second, encoding="utf-8")

    unsliced = build_resident_graph_package(
        graph,
        network,
        case_id="two_dpu_native_fixture",
        suite_id="two_dpu_native_fixture",
        quantization_mode="none",
    )
    unsliced_first = unsliced.write(
        tmp_path, dpu_binary=dpu_binary, request_id="unsliced-0"
    )
    unsliced_second = unsliced.write(
        tmp_path, dpu_binary=dpu_binary, request_id="unsliced-1"
    )
    assert unsliced_first.manifest_path is not None
    assert unsliced_second.manifest_path is not None
    assert (
        validate(unsliced_first.manifest_path, unsliced_second.manifest_path).returncode
        == 1
    )


def test_two_dpu_native_makefile_has_a_static_sdk_build_contract() -> None:
    dry_run = subprocess.run(
        ["make", "-Bn"],
        cwd=TWO_DPU_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dpu-upmem-dpurte-clang" in dry_run.stdout
    assert "dpu-pkg-config --cflags --libs dpu" in dry_run.stdout
