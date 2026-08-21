from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from quantum_bench.targets.upmem.generic_boundary import build_generic_boundary_workload
import quantum_bench.targets.upmem.hardware_taskgraph_resident as resident
from quantum_bench.targets.upmem.hardware_taskgraph_resident import build_resident_graph_package
import quantum_bench.targets.upmem.simplepim_taskgraph_executor as executor


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_HEADER_FORMAT = "<8s4I5Q8I"
SCHEDULE_HEADER_FORMAT = "<8s10I32s"
SCHEDULE_RECORD_FORMAT = "<8I"
V2_HEADER_FORMAT = "<8s15I32s32s"
V2_RECORD_FORMAT = "<8I"


def _native_sdk_available() -> bool:
    return all(
        shutil.which(name) is not None
        for name in ("make", "dpu-pkg-config", "dpu-upmem-dpurte-clang")
    )


def _sidecar(package, path: Path, dpu_count: int) -> Path:
    assert len(package.operations) == 1
    operation = package.operations[0]
    operation_bytes = operation.to_bytes(
        operation_abi_version=package.operation_abi_version
    )
    package_bytes = package.package_path.read_bytes()
    output_elements = int(operation.output_elements)
    contracted_elements = int(operation.args["contracted_combination_count"])
    base, remainder = divmod(output_elements, dpu_count)
    header = struct.pack(
        V2_HEADER_FORMAT,
        b"UPXDPV2\0",
        2,
        struct.calcsize(V2_HEADER_FORMAT),
        dpu_count,
        dpu_count,
        1,
        1,
        1,
        0,
        int(operation.operation_id),
        output_elements,
        contracted_elements,
        int(operation.slot_out_real),
        struct.calcsize(V2_RECORD_FORMAT),
        0,
        0,
        hashlib.sha256(package_bytes).digest(),
        hashlib.sha256(operation_bytes).digest(),
    )
    records = []
    offset = 0
    for dpu_id in range(dpu_count):
        elements = base + (1 if dpu_id < remainder else 0)
        records.append(
            struct.pack(
                V2_RECORD_FORMAT,
                0,
                int(operation.operation_id),
                1,
                dpu_id,
                offset,
                elements,
                0,
                contracted_elements,
            )
        )
        offset += elements
    path.write_bytes(header + b"".join(records))
    return path


def _execution_plan_manifest(path: Path, dpu_count: int) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "route_id": "upmem_tn_hardware_execution_plan_resident",
            "backend_id": "upmem_sdk_hardware_execution_plan_resident",
            "hardware_profile_version": "hardware_taskgraph_execution_plan_resident_v1",
            "requested_dpus": dpu_count,
            "requested_dpu_count": dpu_count,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _v1_schedule(package, path: Path) -> Path:
    operation = package.operations[0]
    package_hash = hashlib.sha256(package.package_path.read_bytes()).digest()
    header = struct.pack(
        SCHEDULE_HEADER_FORMAT,
        b"UPXPLAN1",
        1,
        struct.calcsize(SCHEDULE_HEADER_FORMAT),
        1,
        1,
        1,
        1,
        3,
        struct.calcsize(SCHEDULE_RECORD_FORMAT),
        0,
        0,
        package_hash,
    )
    record = struct.pack(
        SCHEDULE_RECORD_FORMAT,
        0,
        int(operation.operation_id),
        0,
        0,
        0,
        int(operation.slot_a),
        int(operation.slot_b),
        int(operation.slot_out_real),
    )
    path.write_bytes(header + record)
    return path


def _native_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    if not _native_sdk_available():
        pytest.skip("UPMEM SDK compiler tools are unavailable")
    build = executor.build(tmp_path / "native_build", prepare_only=True)
    workload = build_generic_boundary_workload()
    package = build_resident_graph_package(
        workload.graph,
        workload.network,
        case_id="m51-output-partition",
        suite_id="m51-output-partition",
        quantization_mode="none",
        operation_abi_version=2,
    ).write(
        tmp_path,
        dpu_binary=Path(str(build["dpu_binary"])).with_name("dpu_resident_v2"),
        request_id="m51-output-partition",
    )
    assert package.manifest_path is not None
    assert package.package_path is not None
    _execution_plan_manifest(package.manifest_path, 4)
    sidecar = _sidecar(package, tmp_path / "output_partition_v2.bin", 4)
    return Path(str(build["host_binary"])).with_name("host_upmem_execution_plan_v2"), package.manifest_path, sidecar


def _native_v1_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    if not _native_sdk_available():
        pytest.skip("UPMEM SDK compiler tools are unavailable")
    build = executor.build(tmp_path / "native_build", prepare_only=True)
    workload = build_generic_boundary_workload()
    package = build_resident_graph_package(
        workload.graph,
        workload.network,
        case_id="m51-v1-compatibility",
        suite_id="m51-v1-compatibility",
        quantization_mode="none",
    ).write(
        tmp_path,
        dpu_binary=Path(str(build["dpu_binary"])),
        request_id="m51-v1-compatibility",
    )
    assert package.manifest_path is not None
    assert package.package_path is not None
    _execution_plan_manifest(package.manifest_path, 1)
    schedule = _v1_schedule(package, tmp_path / "execution_plan_v1.bin")
    v2_sidecar = _sidecar(package, tmp_path / "cross_version_v2.bin", 4)
    return Path(str(build["host_binary"])), package.manifest_path, schedule, v2_sidecar


def _run_validate(host: Path, manifest: Path, sidecar: Path, response: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(host),
            "--validate-plan",
            "--resident-package",
            str(manifest),
            "--distributed-plan-v2",
            str(sidecar),
            "--response",
            str(response),
        ],
        cwd=host.parent,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_validate_v1(host: Path, manifest: Path, schedule: Path, response: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(host),
            "--validate-plan",
            "--resident-package",
            str(manifest),
            "--schedule",
            str(schedule),
            "--response",
            str(response),
        ],
        cwd=host.parent,
        capture_output=True,
        text=True,
        check=False,
    )


def test_native_abi_sizes_offsets_and_completion_versions(tmp_path: Path) -> None:
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("gcc is unavailable")
    source = tmp_path / "abi_probe.c"
    source.write_text(
        "#include <stddef.h>\n"
        "#include <stdio.h>\n"
        "#include \"common.h\"\n"
        "int main(void) {\n"
        "    printf(\"%zu %zu %zu %zu %zu %zu %zu\\n\",\n"
        "        sizeof(upmem_generic_args_t), sizeof(resident_operation_t),\n"
        "        offsetof(resident_operation_t, args), sizeof(resident_completion_t),\n"
        "        sizeof(upmem_generic_args_v1_t), sizeof(upmem_generic_args_v2_t),\n"
        "        offsetof(upmem_generic_args_v2_t, dpu_slice_offset));\n"
        "    return 0;\n"
        "}\n",
        encoding="ascii",
    )
    common = ROOT / "native/upmem/simplepim/upmem_sdk_generic_loop_resident"
    for operation_abi, operation_bytes in ((1, 784), (2, 800)):
        for completion_abi, completion_bytes in ((1, 40), (2, 120)):
            binary = tmp_path / f"probe_{operation_abi}_{completion_abi}"
            completed = subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    f"-DRESIDENT_OPERATION_ABI_VERSION={operation_abi}",
                    f"-DRESIDENT_COMPLETION_VERSION={completion_abi}",
                    "-I",
                    str(common),
                    str(source),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            values = [int(item) for item in subprocess.check_output([str(binary)], text=True).split()]
            assert values == [740 if operation_abi == 1 else 756, operation_bytes, 44, completion_bytes, 740, 756, 740]


def test_v2_protocol_layout_and_source_contract() -> None:
    assert struct.calcsize(V2_HEADER_FORMAT) == 132
    assert struct.calcsize(V2_RECORD_FORMAT) == 32
    assert resident.RESIDENT_OPERATION_BYTES == 784
    assert resident.RESIDENT_OPERATION_BYTES_V2 == 800

    common = (ROOT / "native/upmem/simplepim/upmem_sdk_execution_plan/execution_plan_v2_common.h").read_text()
    host = (ROOT / "native/upmem/simplepim/upmem_sdk_execution_plan/host.c").read_text()
    raw_host = (ROOT / "native/upmem/simplepim/upmem_sdk_generic_loop_resident/host.c").read_text()
    assert '#define EXECUTION_PLAN_V2_MAGIC "UPXDPV2"' in common
    assert "--distributed-plan-v2" in host
    assert "operation.args.dpu_slice_elements = unit->output_elements" in host
    assert "dpu_alloc(1," in raw_host
    assert "dpu_alloc(requested_dpus)" not in raw_host


def test_python_resident_producers_emit_versioned_packages_without_rewrite(tmp_path: Path) -> None:
    workload = build_generic_boundary_workload()
    for abi_version, magic, operation_bytes, binary_name in (
        (1, b"UPRGPCK1", 784, "dpu_resident"),
        (2, b"UPRGPCK2", 800, "dpu_resident_v2"),
    ):
        root = tmp_path / f"abi_{abi_version}"
        binary = root / binary_name
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"placeholder")
        package = build_resident_graph_package(
            workload.graph,
            workload.network,
            case_id=f"abi-{abi_version}",
            suite_id=f"abi-{abi_version}",
            quantization_mode="none",
            operation_abi_version=abi_version,
        ).write(root, dpu_binary=binary, request_id=f"abi-{abi_version}")
        assert package.package_path is not None
        assert package.manifest_path is not None
        payload = package.package_path.read_bytes()
        header = struct.unpack_from(PACKAGE_HEADER_FORMAT, payload, 0)
        assert header[0] == magic
        assert header[1] == abi_version
        assert header[9] == header[11] * operation_bytes
        metadata = resident.validate_resident_graph_package_bytes(
            payload, operation_abi_version=abi_version
        )
        assert metadata["operation_bytes"] == operation_bytes
        manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
        assert manifest["package_magic"] == magic.decode("ascii")
        assert manifest["package_version"] == abi_version
        assert manifest["operation_abi_version"] == abi_version
        assert manifest["operation_bytes"] == operation_bytes
        assert manifest["dpu_binary_abi"] == binary_name
        if abi_version == 2:
            operation_offset = int(header[8])
            values = struct.unpack_from(
                resident._resident_operation_format(
                    operation_abi_version=abi_version
                ),
                payload,
                operation_offset,
            )
            assert values[-4:] == (0, values[17], 0, values[18])


def test_native_v2_validate_only_accepts_four_output_slices(tmp_path: Path) -> None:
    host, manifest, sidecar = _native_fixture(tmp_path)
    response_path = tmp_path / "valid.response.json"
    completed = _run_validate(host, manifest, sidecar, response_path)

    assert completed.returncode == 0, completed.stderr
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["schema_version"] == "upmem_execution_plan_native_v2"
    assert response["status"] == "validated"
    assert response["requested_dpu_count"] == 4
    assert response["allocated_dpu_count"] == 0
    assert response["schedule_sidecar_sha256"] is None
    assert response["distributed_plan_v2_sha256"]
    assignments = response["operation_assignments"]
    assert len(assignments) == 4
    assert [item["dpu_id"] for item in assignments] == [0, 1, 2, 3]
    assert sum(item["output_elements"] for item in assignments) == 60
    assert all(item["contracted_offset"] == 0 for item in assignments)
    assert all(item["contracted_elements"] == 4 for item in assignments)


def test_native_v2_validate_only_rejects_output_coverage_gap(tmp_path: Path) -> None:
    host, manifest, sidecar = _native_fixture(tmp_path)
    malformed = bytearray(sidecar.read_bytes())
    record_offset = struct.calcsize(V2_HEADER_FORMAT) + struct.calcsize(V2_RECORD_FORMAT) + 16
    output_offset = struct.unpack_from("<I", malformed, record_offset)[0]
    struct.pack_into("<I", malformed, record_offset, output_offset + 1)
    malformed_path = tmp_path / "malformed_output_partition_v2.bin"
    malformed_path.write_bytes(malformed)
    response_path = tmp_path / "invalid.response.json"

    completed = _run_validate(host, manifest, malformed_path, response_path)

    assert completed.returncode != 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "failed"
    assert "coverage" in response["error"] or "range" in response["error"]
    assert response["allocated_dpu_count"] == 0


def test_native_v2_validate_only_rejects_manifest_dpu_count_conflict_before_allocation(tmp_path: Path) -> None:
    host, manifest, sidecar = _native_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["requested_dpu_count"] = payload["requested_dpus"] + 1
    conflicting_manifest = manifest.parent / "conflicting_dpu_count_request.json"
    conflicting_manifest.write_text(json.dumps(payload), encoding="utf-8")
    response_path = tmp_path / "conflicting-dpu-count.response.json"

    completed = _run_validate(host, conflicting_manifest, sidecar, response_path)

    assert completed.returncode != 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "failed"
    assert response["failure_stage"] == "hardware_profile_violation"
    assert response["allocated_dpu_count"] == 0
    assert "DPU counts conflict" in response["error"]


def test_native_v2_validate_only_rejects_missing_manifest_dpu_count_before_allocation(tmp_path: Path) -> None:
    host, manifest, sidecar = _native_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["requested_dpu_count"]
    incomplete_manifest = manifest.parent / "missing_dpu_count_request.json"
    incomplete_manifest.write_text(json.dumps(payload), encoding="utf-8")
    response_path = tmp_path / "missing-dpu-count.response.json"

    completed = _run_validate(host, incomplete_manifest, sidecar, response_path)

    assert completed.returncode != 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "failed"
    assert response["failure_stage"] == "hardware_profile_violation"
    assert response["allocated_dpu_count"] == 0
    assert "identity missing" in response["error"]


def test_native_v1_fixture_remains_validate_only_compatible(tmp_path: Path) -> None:
    host, manifest, schedule, _ = _native_v1_fixture(tmp_path)
    response_path = tmp_path / "v1.response.json"
    completed = _run_validate_v1(host, manifest, schedule, response_path)

    assert completed.returncode == 0, completed.stderr
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "validated"
    assert response["allocated_dpu_count"] == 0


def test_native_cross_version_descriptors_are_rejected(tmp_path: Path) -> None:
    v2_host, v2_manifest, v2_sidecar = _native_fixture(tmp_path / "v2")
    v1_host, v1_manifest, _, v1_as_v2_sidecar = _native_v1_fixture(tmp_path / "v1")

    v2_as_v1_payload = json.loads(v2_manifest.read_text(encoding="utf-8"))
    v2_as_v1_payload.update(
        {
            "dpu_binary": "dpu_resident",
            "dpu_binary_abi": "dpu_resident",
            "package_magic": "UPRGPCK1",
            "package_version": 1,
            "operation_abi_version": 1,
            "operation_bytes": 784,
        }
    )
    v2_as_v1_manifest = v2_manifest.parent / "v2_as_v1_request.json"
    v2_as_v1_manifest.write_text(json.dumps(v2_as_v1_payload), encoding="utf-8")

    v1_as_v2_payload = json.loads(v1_manifest.read_text(encoding="utf-8"))
    v1_as_v2_payload.update(
        {
            "dpu_binary": "dpu_resident_v2",
            "dpu_binary_abi": "dpu_resident_v2",
            "package_magic": "UPRGPCK2",
            "package_version": 2,
            "operation_abi_version": 2,
            "operation_bytes": 800,
        }
    )
    v1_as_v2_manifest = v1_manifest.parent / "v1_as_v2_request.json"
    v1_as_v2_manifest.write_text(json.dumps(v1_as_v2_payload), encoding="utf-8")

    v1_response_path = tmp_path / "v1-rejects-v2.response.json"
    v1_completed = _run_validate(v1_host, v2_as_v1_manifest, v2_sidecar, v1_response_path)
    assert v1_completed.returncode != 0
    v1_response = json.loads(v1_response_path.read_text(encoding="utf-8"))
    assert "v2" in v1_response["error"] or "abi" in v1_response["error"].lower()

    v2_response_path = tmp_path / "v2-rejects-v1.response.json"
    v2_completed = _run_validate(v2_host, v1_as_v2_manifest, v1_as_v2_sidecar, v2_response_path)
    assert v2_completed.returncode != 0
    v2_response = json.loads(v2_response_path.read_text(encoding="utf-8"))
    assert any(item in v2_response["error"].lower() for item in ("abi", "magic", "version"))
